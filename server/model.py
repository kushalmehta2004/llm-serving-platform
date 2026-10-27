"""
Model loading and inference. Keep this free of batching/scheduling logic —
it just loads the model and runs forward passes on whatever it's handed.

repetition_penalty is set to 1.0 throughout so that:
  - model.generate() produces logits identical to a raw model() forward pass
  - the manual KV-cache decode loop in the scheduler matches exactly
  - the correctness baseline is stable regardless of call path
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from dotenv import load_dotenv

load_dotenv()

_tokenizer = None
_model = None
_device = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    global _tokenizer, _model, _device

    model_id = os.environ["MODEL_ID"]
    _device = get_device()

    print(f"Loading model {model_id!r} on {_device} ...")
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32 if _device.type == "cpu" else torch.float16,
    ).to(_device)
    _model.eval()
    print("Model loaded.")


def get_tokenizer():
    if _tokenizer is None:
        load_model()
    return _tokenizer


def get_model():
    if _model is None:
        load_model()
    return _model


def get_model_device() -> torch.device:
    if _device is None:
        load_model()
    return _device


# ── single-request paths (Phases 0–2) ────────────────────────────────────────

def generate_one(prompt: str, max_new_tokens: int = 20) -> str:
    """Greedy-decode up to max_new_tokens from prompt. Used by smoke test."""
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_model_device()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def generate(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    seed: int | None = None,
) -> tuple[str, list[int], int, int]:
    """
    Full generation. Returns (text, output_token_ids, prompt_tokens, completion_tokens).
    Correctness oracle for all later phases.
    """
    token_ids: list[int] = []
    prompt_tokens = 0
    for token_id, p_len in generate_stream(prompt, max_tokens, temperature, seed):
        token_ids.append(token_id)
        prompt_tokens = p_len

    tokenizer = get_tokenizer()
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text, token_ids, prompt_tokens, len(token_ids)


def generate_stream(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    seed: int | None = None,
):
    """Yields (token_id, prompt_len) for each new token via model.generate()."""
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_model_device()

    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            repetition_penalty=1.0,
        )

    new_ids = output_ids[0][prompt_len:].tolist()
    for token_id in new_ids:
        yield token_id, prompt_len


# ── batched path (Phase 3) ────────────────────────────────────────────────────

def batch_generate(
    prompts: list[str],
    max_tokens: int,
    temperature: float = 0.0,
) -> list[list[int]]:
    """
    Generate tokens for a batch via model.generate() (left-padded).
    Returns one list of token ids per prompt (prompt tokens excluded, EOS stripped).
    Correctness: batch-of-1 == generate() for same prompt.
    """
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_model_device()

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(device)

    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=pad_id,
            repetition_penalty=1.0,
        )

    prompt_len = enc["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id
    results = []
    for i in range(len(prompts)):
        tokens = out[i][prompt_len:].tolist()
        if eos_id in tokens:
            tokens = tokens[: tokens.index(eos_id)]
        results.append(tokens)
    return results


# ── step-by-step KV-cache path (Phase 4) ─────────────────────────────────────

def decode_step(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: DynamicCache,
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor, DynamicCache]:
    """
    One forward pass using the model's own KV cache.

    Prefill call  (step 0): input_ids = (B, prompt_len), cache_position = arange(prompt_len)
    Decode call   (step k): input_ids = (B, 1),          cache_position = [prompt_len + k - 1]

    Returns:
        logits:          (B, vocab_size) at the last position
        past_key_values: updated DynamicCache for the next step
    """
    model = get_model()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
    return out.logits[:, -1, :], out.past_key_values


def decode_step_paged(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    seq_kv,          # server.kv_cache.SequenceKV
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    """
    One decode step using the paged KV cache.

    Reconstructs a DynamicCache from paged blocks, runs the model forward,
    writes the new token's KV back into the paged store, then discards the
    temporary DynamicCache.

    Returns:
        logits  : (1, vocab_size) at the last position
        None    : caller should use seq_kv for subsequent steps, not a cache object
    """
    past_kv = seq_kv.to_dynamic_cache()
    logits, updated_kv = decode_step(input_ids, attention_mask, past_kv, cache_position)
    # Write the newly computed token's KV (last position in updated_kv)
    new_pos = seq_kv.num_tokens        # 0-indexed position of the token just processed
    seq_kv.write_token(updated_kv, new_pos)
    return logits, None


def batch_forward_step(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Full-sequence forward, no KV cache.
    Returns logits (B, vocab) at the last position.
    Used only in test_correctness.py for the batched-path check.
    NOTE: diverges from model.generate() when repetition_penalty != 1.0;
    since we now use penalty=1.0 everywhere this matches correctly.
    """
    model = get_model()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits[:, -1, :]
