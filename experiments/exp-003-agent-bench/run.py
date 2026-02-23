#!/usr/bin/env python3
"""
EXP-003: Agent Communication Latency Benchmark
BlackRoad Labs — Benchmarks the RFC-0002 in-process message bus latency.
"""
import hashlib
import json
import time
import statistics
from dataclasses import dataclass, field, asdict
from typing import Literal
import uuid

MessageType = Literal["request", "response", "event", "broadcast"]


@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    type: MessageType
    topic: str
    payload: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    timestamp_ns: int = field(default_factory=time.time_ns)


class MessageBus:
    def __init__(self):
        self._subs: dict[str, list] = {}
        self._delivered = 0

    def subscribe(self, topic: str, handler) -> None:
        self._subs.setdefault(topic, []).append(handler)

    def publish(self, msg: AgentMessage) -> None:
        for h in self._subs.get(msg.topic, []):
            h(msg)
            self._delivered += 1


def run_benchmark(n_messages: int = 10_000) -> dict:
    """Benchmark message publish/receive latency."""
    bus = MessageBus()
    latencies_ns = []

    def handler(msg: AgentMessage):
        latency = time.time_ns() - msg.timestamp_ns
        latencies_ns.append(latency)

    bus.subscribe("bench.ping", handler)

    # Warmup
    for _ in range(100):
        msg = AgentMessage("agent/sender", "agent/receiver", "event", "bench.ping",
                           {"seq": -1})
        bus.publish(msg)
    latencies_ns.clear()

    # Benchmark
    start = time.time()
    for i in range(n_messages):
        msg = AgentMessage("agent/sender", "agent/receiver", "event", "bench.ping",
                           {"seq": i, "data": "x" * 64})
        bus.publish(msg)
    elapsed = time.time() - start

    lat_us = [x / 1000 for x in latencies_ns]
    return {
        "n_messages": n_messages,
        "elapsed_s": round(elapsed, 4),
        "msgs_per_sec": round(n_messages / elapsed),
        "latency_median_us": round(statistics.median(lat_us), 2),
        "latency_p99_us": round(sorted(lat_us)[int(len(lat_us) * 0.99)], 2),
        "latency_max_us": round(max(lat_us), 2),
    }


def run_fanout_benchmark(n_subs: int = 100, n_messages: int = 1000) -> dict:
    """Benchmark 1-to-N fanout performance."""
    bus = MessageBus()
    received = [0]

    for _ in range(n_subs):
        bus.subscribe("bench.fanout", lambda m: received.__setitem__(0, received[0] + 1))

    start = time.time()
    for i in range(n_messages):
        msg = AgentMessage("agent/broadcaster", "broadcast", "broadcast", "bench.fanout")
        bus.publish(msg)
    elapsed = time.time() - start

    return {
        "subscribers": n_subs,
        "n_messages": n_messages,
        "total_deliveries": received[0],
        "elapsed_s": round(elapsed, 4),
        "deliveries_per_sec": round(received[0] / elapsed),
    }


if __name__ == "__main__":
    print("🔬 EXP-003: Agent Communication Latency Benchmark")
    print("=" * 60)

    print("\n📊 Point-to-point (10K messages)")
    r1 = run_benchmark(10_000)
    for k, v in r1.items():
        print(f"  {k}: {v}")

    print("\n📊 1-to-100 fanout (1K messages)")
    r2 = run_fanout_benchmark(100, 1000)
    for k, v in r2.items():
        print(f"  {k}: {v}")

    print(f"\n✅ Benchmark complete. Throughput: {r1[\"msgs_per_sec\"]:,} msg/s")

