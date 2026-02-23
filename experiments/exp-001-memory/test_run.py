"""Tests for EXP-001: PS-SHA∞ Memory Integrity"""
import hashlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.exp_001_memory.run import ps_sha_hash, generate_chain, verify_chain


def test_hash_deterministic():
    h1 = ps_sha_hash("GENESIS", "test content", 12345678)
    h2 = ps_sha_hash("GENESIS", "test content", 12345678)
    assert h1 == h2


def test_hash_changes_on_content():
    h1 = ps_sha_hash("GENESIS", "content A", 12345678)
    h2 = ps_sha_hash("GENESIS", "content B", 12345678)
    assert h1 != h2


def test_chain_generation():
    chain = generate_chain(5)
    assert len(chain) == 5
    for entry in chain:
        assert "hash" in entry
        assert "content" in entry
        assert "prev_hash" in entry


def test_chain_integrity_passes():
    chain = generate_chain(10)
    valid, issues = verify_chain(chain)
    assert valid is True
    assert len(issues) == 0


def test_tampered_chain_detected():
    chain = generate_chain(10, tamper_at=5)
    valid, issues = verify_chain(chain)
    assert valid is False
    assert len(issues) > 0


def test_genesis_prev_hash():
    chain = generate_chain(1)
    assert chain[0]["prev_hash"] == "GENESIS"

