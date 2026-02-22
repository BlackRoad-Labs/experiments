# EXP-001: PS-SHA∞ Memory Integrity

**Hypothesis:** The PS-SHA∞ hash-chain memory format provides tamper-evident storage
with 100% detection rate for single-entry modifications.

**Why this matters:** If any AI agent (or adversary) modifies a past memory entry,
the entire chain from that point forward becomes invalid — making history immutable.

**Method:**
1. Generate random memory chains of N entries
2. Randomly tamper with ~50% of chains
3. Run `verify_chain()` on all chains
4. Measure detection accuracy

**Expected outcome:** 100% detection with zero false negatives.

**Dependencies:** None (pure Python, stdlib only)

**Status:** 🟢 Running
