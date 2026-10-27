"""
Correctness oracle — run after Phases 3, 4, and 5.

First run (no baseline file): generates and saves the golden token ids.
Every subsequent run: asserts the output is token-for-token identical.

Tests:
  1. Single-request path (generate()) vs saved baseline.
  2. Batched path (batch_forward_step loop, batch_size=1) vs saved baseline.

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
from server.model import load_model, generate, get_tokenizer, batch_generate, decode_step, get_model_device
from transformers import DynamicCache
import torch

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

    load_model()
    _text, token_ids, _p, _c = generate(PROMPT, MAX_TOKENS, TEMPERATURE, SEED)
    with open(BASELINE_FILE, "w") as f:
        json.dump({"prompt": PROMPT, "token_ids": token_ids}, f, indent=2)
    print(f"Baseline saved to {BASELINE_FILE}: {token_ids}")
    return token_ids


def _batched_greedy(prompt: str, max_tokens: int) -> list[int]:
    """
    Runs a batch of 1 through batch_generate (the Phase 3 scheduler path).
    Must produce token-for-token identical output to the single-request baseline.
    """
    results = batch_generate([prompt], max_tokens, temperature=0.0)
    token_ids = results[0]
    # Strip trailing EOS if present (matches scheduler behaviour)
    tokenizer = get_tokenizer()
    if token_ids and token_ids[-1] == tokenizer.eos_token_id:
        token_ids = token_ids[:-1]
    return token_ids


# ── tests ─────────────────────────────────────────────────────────────────────

def test_greedy_output_matches_baseline():
    """Single-request path must be token-for-token identical to the saved baseline."""
    load_model()
    baseline = _get_or_create_baseline()
    _text, token_ids, _p, _c = generate(PROMPT, MAX_TOKENS, TEMPERATURE, SEED)
    assert token_ids == baseline, (
        f"Single-request output drifted!\nExpected: {baseline}\nGot: {token_ids}"
    )


def test_batched_path_matches_baseline():
    """Batch-of-1 through batch_forward_step must match the baseline token-for-token."""
    load_model()
    baseline = _get_or_create_baseline()
    token_ids = _batched_greedy(PROMPT, MAX_TOKENS)
    assert token_ids == baseline, (
        f"Batched path output drifted!\nExpected: {baseline}\nGot: {token_ids}"
    )


def test_kv_cache_decode_matches_baseline():
    """Phase 4 KV-cache decode loop must match baseline token-for-token."""
    load_model()
    baseline = _get_or_create_baseline()
    tokenizer = get_tokenizer()
    device = get_model_device()

    enc = tokenizer(PROMPT, return_tensors="pt").to(device)
    ids  = enc["input_ids"]
    mask = enc["attention_mask"]
    L = ids.shape[1]

    past_kv = DynamicCache()
    cache_pos = torch.arange(L, dtype=torch.long, device=device)
    logits, past_kv = decode_step(ids, mask, past_kv, cache_pos)

    token_ids = []
    tok = int(logits[0].argmax().item())
    token_ids.append(tok)

    for step in range(1, MAX_TOKENS):
        if tok == tokenizer.eos_token_id:
            break
        inp = torch.tensor([[tok]], dtype=torch.long, device=device)
        cur_mask = torch.ones((1, L + step), dtype=torch.long, device=device)
        cache_pos = torch.tensor([L + step - 1], dtype=torch.long, device=device)
        logits, past_kv = decode_step(inp, cur_mask, past_kv, cache_pos)
        tok = int(logits[0].argmax().item())
        token_ids.append(tok)

    assert token_ids == baseline, (
        f"KV-cache decode path drifted!\nExpected: {baseline}\nGot: {token_ids}"
    )


if __name__ == "__main__":
    load_model()
    baseline = _get_or_create_baseline()

    _text, token_ids, _p, _c = generate(PROMPT, MAX_TOKENS, TEMPERATURE, SEED)
    assert token_ids == baseline, f"FAIL (single-request) — got: {token_ids}"
    print("PASS — single-request matches baseline.")

    token_ids = _batched_greedy(PROMPT, MAX_TOKENS)
    assert token_ids == baseline, f"FAIL (batched path) — got: {token_ids}"
    print("PASS — batched path matches baseline.")
