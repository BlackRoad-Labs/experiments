# EXP-004: Provider-Free Local LLM Inference via Ollama

**Experiment**: Validate that Ollama can replace all cloud LLM providers for local research workflows  
**Status**: Running  
**Hypothesis**: Running Ollama locally achieves acceptable response quality and throughput for
BlackRoad Labs experiments without any dependency on external API providers or secrets.

## Motivation

External LLM providers (OpenAI, Anthropic, etc.) introduce:
- API key management overhead and cost
- Network round-trip latency
- External data exposure risk
- Hard dependency on third-party availability

Ollama runs models fully on local hardware — no keys, no cloud, no limits.

## Variables

- **Independent**: Model size (7B / 13B / 70B), prompt length, temperature
- **Dependent**: Response latency (ms), tokens/sec, response quality score
- **Controlled**: Same hardware, single-process, no batching

## Expected Results

| Metric | Hypothesis | Threshold |
|--------|-----------|----------|
| Availability check | <100ms | <500ms |
| Model list latency | <200ms | <1s |
| Generate latency (7B) | <5s | <30s |
| Chat latency (7B) | <5s | <30s |
| Throughput (7B) | >10 tok/s | >5 tok/s |

## Implications

If hypothesis is confirmed:
- All BlackRoad Labs experiments can run fully offline
- No API keys or secrets need to be managed
- Inference cost drops to zero (electricity only)
- Privacy: prompts never leave the local machine

If hypothesis fails:
- Evaluate quantised models (Q4_K_M, Q8_0) for speed
- Consider GPU acceleration (`OLLAMA_NUM_GPU` env var)
- Benchmark smaller models (phi3-mini, gemma2:2b)

## Running

```bash
# 1. Install Ollama  →  https://ollama.com/download
# 2. Start the server
ollama serve

# 3. Pull a model
ollama pull llama3

# 4. Run the experiment
python experiments/exp-004-ollama/run.py

# 5. Or use the standalone client CLI
python -m src.ollama_client check
python -m src.ollama_client list
python -m src.ollama_client generate --model llama3 --prompt "Hello, world"
python -m src.ollama_client chat     --model llama3 --message "What is trinary logic?"
```
