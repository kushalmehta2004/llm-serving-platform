"""
Correctness oracle — run after Phases 3, 4, and 5.

First run (no baseline file): generates and saves the golden token ids.
Every subsequent run: asserts the output is token-for-token identical.

Usage:
    python -m pytest tests/test_correctness.py -v
    # or directly:
    python tests/test_correctness.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from server.model import load_model, generate

BASELINE_FILE = os.path.join(os.path.dirname(__file__), "baseline.json")

# Fixed inputs — never change these after Phase 1.
PROMPT = "The key insight behind continuous batching in LLM serving is"
MAX_TOKENS = 30
TEMPERATURE = 0.0   # greedy — fully deterministic
SEED = 42


def _get_or_create_baseline() -> list[int]:
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            data = json.load(f)
        return data["token_ids"]

    # First run: generate and save
    load_model()
    _text, token_ids, _p, _c = generate(PROMPT, MAX_TOKENS, TEMPERATURE, SEED)
    with open(BASELINE_FILE, "w") as f:
        json.dump({"prompt": PROMPT, "token_ids": token_ids}, f, indent=2)
    print(f"Baseline saved to {BASELINE_FILE}: {token_ids}")
    return token_ids


def test_greedy_output_matches_baseline():
    """Single-request greedy output must be token-for-token identical to the saved baseline."""
    load_model()
    baseline = _get_or_create_baseline()
    _text, token_ids, _p, _c = generate(PROMPT, MAX_TOKENS, TEMPERATURE, SEED)
    assert token_ids == baseline, (
        f"Output drifted from baseline!\n"
        f"Expected: {baseline}\n"
        f"Got:      {token_ids}"
    )


if __name__ == "__main__":
    load_model()
    baseline = _get_or_create_baseline()
    _text, token_ids, _p, _c = generate(PROMPT, MAX_TOKENS, TEMPERATURE, SEED)
    assert token_ids == baseline, (
        f"FAIL — output drifted!\nExpected: {baseline}\nGot: {token_ids}"
    )
    print("PASS — output matches baseline.")
