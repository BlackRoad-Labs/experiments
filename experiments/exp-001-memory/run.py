#!/usr/bin/env python3
"""
EXP-001: PS-SHA∞ Memory Integrity
Tests the integrity of hash-chain memory journals under various conditions.
"""

import hashlib
import time
import json
import random
from pathlib import Path


def ps_sha_hash(prev_hash: str, content: str, timestamp_ns: int) -> str:
    """PS-SHA∞ hash function: sha256(prev_hash:content:timestamp_ns)"""
    payload = f"{prev_hash}:{content}:{timestamp_ns}"
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_chain(length: int, tamper_at: int = -1) -> list[dict]:
    """Generate a memory chain of given length, optionally tampered."""
    chain = []
    prev_hash = "GENESIS"
    
    for i in range(length):
        content = f"Memory entry {i}: agent={random.choice(['LUCIDIA','ALICE','ECHO','CIPHER'])} action=test"
        ts = time.time_ns()
        h = ps_sha_hash(prev_hash, content, ts)
        
        entry = {
            "index": i,
            "hash": h,
            "prev": prev_hash,
            "content": content,
            "timestamp_ns": ts,
            "truth_state": random.choice([1, 0, -1])
        }
        
        # Tamper with a specific entry to test detection
        if i == tamper_at:
            entry["content"] = "TAMPERED: " + content
        
        chain.append(entry)
        prev_hash = h
    
    return chain


def verify_chain(chain: list[dict]) -> tuple[bool, list[int]]:
    """Verify chain integrity. Returns (valid, list of tampered indices)."""
    corrupted = []
    
    for i, entry in enumerate(chain):
        expected_prev = "GENESIS" if i == 0 else chain[i-1]["hash"]
        
        if entry["prev"] != expected_prev:
            corrupted.append(i)
            continue
        
        expected_hash = ps_sha_hash(
            entry["prev"],
            entry["content"],
            entry["timestamp_ns"]
        )
        
        if entry["hash"] != expected_hash:
            corrupted.append(i)
    
    return len(corrupted) == 0, corrupted


def run_experiment(iterations: int = 100, chain_length: int = 50) -> dict:
    """Run the integrity experiment."""
    print(f"EXP-001: PS-SHA∞ Memory Integrity")
    print(f"  Iterations: {iterations}, Chain length: {chain_length}")
    print()
    
    results = {
        "iterations": iterations,
        "chain_length": chain_length,
        "clean_chains_detected": 0,
        "tampered_chains_detected": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    
    for i in range(iterations):
        tamper = random.randint(0, chain_length - 1) if random.random() > 0.5 else -1
        chain = generate_chain(chain_length, tamper_at=tamper)
        valid, corrupted = verify_chain(chain)
        
        if tamper == -1:
            # Should be valid
            if valid:
                results["clean_chains_detected"] += 1
            else:
                results["false_positives"] += 1
        else:
            # Should be invalid
            if not valid:
                results["tampered_chains_detected"] += 1
            else:
                results["false_negatives"] += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{iterations}")
    
    print()
    print("Results:")
    total_clean = results["clean_chains_detected"] + results["false_positives"]
    total_tampered = results["tampered_chains_detected"] + results["false_negatives"]
    
    print(f"  Clean chains: {results['clean_chains_detected']}/{total_clean} detected correctly")
    print(f"  Tampered chains: {results['tampered_chains_detected']}/{total_tampered} detected correctly")
    print(f"  False positives: {results['false_positives']}")
    print(f"  False negatives: {results['false_negatives']}")
    
    accuracy = (results["clean_chains_detected"] + results["tampered_chains_detected"]) / iterations
    print(f"  Overall accuracy: {accuracy:.1%}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EXP-001: Memory Integrity Test")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--chain-length", type=int, default=50)
    args = parser.parse_args()
    
    results = run_experiment(args.iterations, args.chain_length)
    
    # Save results
    Path("results/exp-001").mkdir(parents=True, exist_ok=True)
    with open("results/exp-001/data.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/exp-001/data.json")
