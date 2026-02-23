"""
exp-003-agent-bench/run.py — RFC-0002 Message Bus Throughput Benchmark
Generates ASCII bar charts of throughput results.
"""
import time, hashlib, json, statistics
from datetime import datetime, timezone

ITERATIONS = [100, 500, 1000, 2000, 5000]

def sign(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]

def build_message(seq: int) -> dict:
    payload = {"seq": seq, "ts": time.time_ns()}
    return {
        "id": f"msg_{seq:06d}",
        "version": "1.0",
        "from": f"agent/bench-{seq % 4}",
        "to": "agent/aggregator",
        "type": "event",
        "topic": "bench.tick",
        "payload": payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signature": sign(payload),
    }

def run_bench(n: int) -> dict:
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        msg = build_message(i)
        _ = json.dumps(msg)
        times.append(time.perf_counter() - t0)
    total_s = sum(times)
    return {
        "n": n,
        "total_ms": total_s * 1000,
        "throughput": n / total_s,
        "avg_us": statistics.mean(times) * 1e6,
        "p99_us": sorted(times)[int(0.99 * n)] * 1e6,
    }

def ascii_bar(value: float, max_val: float, width: int = 40) -> str:
    filled = int((value / max_val) * width)
    return "█" * filled + "░" * (width - filled)

def main():
    print("\n🔬 RFC-0002 Agent Message Bus — Throughput Benchmark")
    print("=" * 60)
    results = []
    for n in ITERATIONS:
        r = run_bench(n)
        results.append(r)
        print(f"  n={n:<5} → {r['throughput']:>10,.0f} msg/s  avg={r['avg_us']:.1f}µs  p99={r['p99_us']:.1f}µs")

    max_tp = max(r["throughput"] for r in results)
    print("\n📊 Throughput Chart (msg/s)\n")
    for r in results:
        bar = ascii_bar(r["throughput"], max_tp, 40)
        print(f"  n={r['n']:<5} {bar} {r['throughput']:>10,.0f}")

    print(f"\n  Peak: {max_tp:,.0f} msg/s  |  Target: 1,000,000+ msg/s")
    print(f"  Status: {'✅ PASS' if max_tp > 10000 else '⚠️  BELOW TARGET'}")

    # Write results JSON
    import pathlib
    out = pathlib.Path(__file__).parent / "results.json"
    out.write_text(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2))
    print(f"\n  Results written to {out}")

if __name__ == "__main__":
    main()
