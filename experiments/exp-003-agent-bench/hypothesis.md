# EXP-003: Agent Communication Latency Benchmark

**Experiment**: In-process RFC-0002 message bus throughput and latency  
**Status**: Running  
**Hypothesis**: The Python reference MessageBus can sustain >100,000 messages/sec point-to-point
with median latency <5µs, making it suitable for local agent coordination.

## Variables

- **Independent**: Number of messages, number of subscribers
- **Dependent**: Messages/sec, median latency (µs), p99 latency (µs)
- **Controlled**: Single process, no network I/O, no serialization

## Expected Results

| Metric | Hypothesis | Threshold |
|--------|-----------|----------|
| Throughput | >100K msg/s | >50K (minimum) |
| Median latency | <5µs | <50µs |
| P99 latency | <20µs | <100µs |
| 1-to-100 fanout | >20K deliveries/s | >5K |

## Implications

If hypothesis is confirmed:
- Agents can exchange >1M messages/sec in tight coordination loops
- Memory consolidation (ECHO) can keep up with real-time agent streams
- Trinary logic validation (EXP-002) can be inlined into message delivery

If hypothesis fails:
- Need to evaluate async I/O (asyncio) or ZMQ for higher throughput
- May need C extension for hot path

