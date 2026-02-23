"""
BlackRoad Labs — Async Agent Metrics Ingestion Pipeline
Collects metrics from all gateway nodes → PS-SHA∞ → local SQLite.
"""
from __future__ import annotations
import asyncio, hashlib, json, sqlite3, time
from dataclasses import dataclass, asdict
from typing import AsyncIterator
from pathlib import Path
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

DB_PATH = Path.home() / ".blackroad" / "labs-pipeline.db"
NODES = [
    {"id": "aria64",      "url": "http://192.168.4.38:8787/health"},
    {"id": "blackroad-pi","url": "http://192.168.4.64:8787/health"},
    {"id": "lucidia-alt", "url": "http://192.168.4.99:8787/health"},
]


@dataclass
class MetricEntry:
    node_id: str
    agents_active: int
    latency_ms: float
    status: str
    hash: str
    prev_hash: str
    timestamp_ns: int


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def init_db(db: sqlite3.Connection) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id TEXT, agents_active INTEGER, latency_ms REAL,
        status TEXT, hash TEXT UNIQUE, prev_hash TEXT, timestamp_ns INTEGER
    )""")
    db.commit()


def _prev_hash(db: sqlite3.Connection) -> str:
    row = db.execute("SELECT hash FROM metrics ORDER BY timestamp_ns DESC LIMIT 1").fetchone()
    return row[0] if row else "GENESIS"


async def poll_node(node: dict, timeout: float = 5.0) -> dict:
    """Poll a single gateway node for health metrics."""
    if not HAS_HTTPX:
        return {"node_id": node["id"], "status": "httpx_missing", "agents": 0, "latency_ms": 0}
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.get(node["url"])
            latency = (time.monotonic() - start) * 1000
            data = res.json()
            return {
                "node_id": node["id"],
                "status": "ok",
                "agents_active": data.get("agents_active", 0),
                "latency_ms": round(latency, 2),
            }
    except Exception as e:
        return {"node_id": node["id"], "status": f"error:{type(e).__name__}", "agents_active": 0,
                "latency_ms": (time.monotonic() - start) * 1000}


async def collect_all() -> list[dict]:
    return list(await asyncio.gather(*[poll_node(n) for n in NODES]))


def store_metrics(db: sqlite3.Connection, readings: list[dict]) -> list[MetricEntry]:
    entries = []
    for r in readings:
        ts_ns = time.time_ns()
        prev = _prev_hash(db)
        content = json.dumps(r, sort_keys=True)
        h = _sha256(f"{prev}:{r['node_id']}:{content}:{ts_ns}")
        entry = MetricEntry(
            node_id=r["node_id"], agents_active=r.get("agents_active", 0),
            latency_ms=r.get("latency_ms", 0), status=r.get("status", "unknown"),
            hash=h, prev_hash=prev, timestamp_ns=ts_ns,
        )
        db.execute(
            "INSERT OR IGNORE INTO metrics (node_id, agents_active, latency_ms, status, hash, prev_hash, timestamp_ns) VALUES (?,?,?,?,?,?,?)",
            [entry.node_id, entry.agents_active, entry.latency_ms, entry.status, entry.hash, entry.prev_hash, entry.timestamp_ns]
        )
        entries.append(entry)
    db.commit()
    return entries


async def run_pipeline(interval_s: float = 30.0) -> AsyncIterator[list[MetricEntry]]:
    """Run continuously, yielding batches of stored entries."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    init_db(db)
    print(f"Pipeline started — polling {len(NODES)} nodes every {interval_s}s")
    while True:
        readings = await collect_all()
        entries = store_metrics(db, readings)
        for e in entries:
            status_icon = "✅" if e.status == "ok" else "⚠️"
            print(f"  {status_icon} {e.node_id:15s} agents={e.agents_active:5d}  latency={e.latency_ms:6.1f}ms  [{e.hash[:8]}]")
        yield entries
        await asyncio.sleep(interval_s)


if __name__ == "__main__":
    async def main():
        async for _ in run_pipeline(30):
            pass
    asyncio.run(main())
