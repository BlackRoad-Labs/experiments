#!/usr/bin/env python3
"""
EXP-004: Provider-Free Local LLM Inference via Ollama
BlackRoad Labs — validates local Ollama inference as a drop-in replacement
for cloud LLM providers.

Run:
    python experiments/exp-004-ollama/run.py [--model MODEL] [--host HOST]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running directly or via `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ollama_client import ChatMessage, OllamaClient, OllamaError

DEFAULT_MODEL = "llama3"
PROMPTS = [
    "What is trinary logic? Answer in one sentence.",
    "Explain hash-chain memory integrity in two sentences.",
    "What makes a good agent communication protocol? Answer briefly.",
]


def _probe_availability(client: OllamaClient) -> dict:
    t0 = time.perf_counter()
    available = client.is_available()
    latency_ms = (time.perf_counter() - t0) * 1000
    return {"available": available, "latency_ms": round(latency_ms, 2)}


def _probe_list_models(client: OllamaClient) -> dict:
    t0 = time.perf_counter()
    models = client.list_models()
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "model_count": len(models),
        "names": [m.name for m in models],
        "latency_ms": round(latency_ms, 2),
    }


def _probe_generate(client: OllamaClient, model: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    resp = client.generate(model, prompt, options={"temperature": 0.0})
    latency_ms = (time.perf_counter() - t0) * 1000
    tokens_per_sec = (
        resp.eval_count / (resp.total_duration_ns / 1e9)
        if resp.total_duration_ns > 0 and resp.eval_count > 0
        else None
    )
    return {
        "model": resp.model,
        "prompt_chars": len(prompt),
        "response_chars": len(resp.response),
        "eval_count": resp.eval_count,
        "latency_ms": round(latency_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
        "response_preview": resp.response[:120],
    }


def _probe_chat(client: OllamaClient, model: str) -> dict:
    messages = [
        ChatMessage.system("You are a concise research assistant for BlackRoad Labs."),
        ChatMessage.user("In one sentence, what is the purpose of Ollama?"),
    ]
    t0 = time.perf_counter()
    resp = client.chat(model, messages, options={"temperature": 0.0})
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "model": resp.model,
        "response_chars": len(resp.message.content),
        "latency_ms": round(latency_ms, 2),
        "response_preview": resp.message.content[:120],
    }


def run_experiment(model: str = DEFAULT_MODEL, host: str = "http://localhost:11434") -> dict:
    client = OllamaClient(host=host)

    print("=" * 62)
    print("EXP-004: Provider-Free Local LLM Inference via Ollama")
    print("=" * 62)

    results: dict = {"model": model, "host": host, "probes": {}}

    # 1. Availability
    print("\n[1] Availability check …", end=" ", flush=True)
    avail = _probe_availability(client)
    results["probes"]["availability"] = avail
    print(f"{'✅ OK' if avail['available'] else '❌ UNREACHABLE'}  ({avail['latency_ms']:.1f}ms)")

    if not avail["available"]:
        print(
            "\n  ⚠  Ollama is not running. Start it with: ollama serve\n"
            "  ⚠  Then pull a model with:             ollama pull llama3\n"
        )
        results["status"] = "skipped"
        return results

    # 2. List models
    print("[2] List models …", end=" ", flush=True)
    try:
        list_result = _probe_list_models(client)
        results["probes"]["list_models"] = list_result
        names_str = ", ".join(list_result["names"][:5]) or "(none pulled)"
        print(f"{list_result['model_count']} model(s)  ({list_result['latency_ms']:.1f}ms)  [{names_str}]")
    except OllamaError as e:
        print(f"❌ {e}")
        results["probes"]["list_models"] = {"error": str(e)}

    # 3. Generate
    print(f"\n[3] Generate — model={model}")
    gen_results = []
    for prompt in PROMPTS:
        print(f"  ↓ {prompt[:60]!r} …", end=" ", flush=True)
        try:
            g = _probe_generate(client, model, prompt)
            gen_results.append(g)
            tps = f"{g['tokens_per_sec']:.1f} tok/s" if g["tokens_per_sec"] else "n/a tok/s"
            print(f"✅  {g['latency_ms']:.0f}ms  {tps}")
            print(f"     → {g['response_preview']!r}")
        except OllamaError as e:
            print(f"❌ {e}")
            gen_results.append({"error": str(e), "prompt": prompt})
    results["probes"]["generate"] = gen_results

    # 4. Chat
    print(f"\n[4] Chat — model={model}")
    try:
        chat_result = _probe_chat(client, model)
        results["probes"]["chat"] = chat_result
        print(f"  ✅  {chat_result['latency_ms']:.0f}ms")
        print(f"   → {chat_result['response_preview']!r}")
    except OllamaError as e:
        print(f"  ❌ {e}")
        results["probes"]["chat"] = {"error": str(e)}

    # Summary
    results["status"] = "completed"
    gen_ok    = sum(1 for g in gen_results if "error" not in g)
    total_gen = len(gen_results)
    avg_lat   = (
        sum(g["latency_ms"] for g in gen_results if "error" not in g) / gen_ok
        if gen_ok else 0
    )
    print(f"\n{'─' * 62}")
    print(f"  Generate : {gen_ok}/{total_gen} succeeded  avg={avg_lat:.0f}ms")
    print(f"  Chat     : {'✅' if 'error' not in results['probes'].get('chat', {}) else '❌'}")
    print(f"{'=' * 62}\n")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-004: Ollama local LLM inference")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host",  default="http://localhost:11434")
    parser.add_argument("--output", default=None,
                        help="Path to write JSON results (default: exp-004 results dir)")
    args = parser.parse_args()

    results = run_experiment(model=args.model, host=args.host)

    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / "results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
