"""
Model loading and inference. Keep this free of batching/scheduling logic —
it just loads the model and runs forward passes on whatever it's handed.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    Full generation with greedy (temperature=0) or sampled decoding.
    Returns (text, output_token_ids, prompt_tokens, completion_tokens).
    This is the single-request path and the correctness oracle for all later phases.
    Internally delegates to generate_stream and collects all tokens.
    """
    token_ids: list[int] = []
    prompt_tokens = 0
    for token_id, p_len in generate_stream(prompt, max_tokens, temperature, seed):
        token_ids.append(token_id)
        prompt_tokens = p_len  # same every iteration

    tokenizer = get_tokenizer()
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text, token_ids, prompt_tokens, len(token_ids)


def generate_stream(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    seed: int | None = None,
):
    """
    Token-by-token generator. Yields (token_id, prompt_len) for each new token.
    Used by the streaming endpoint; generate() collects these into a full result.

    Phase 2: uses model.generate() then streams the result token-by-token —
    identical sequence to generate(), only the delivery changes.
    Phase 3/4 will replace this with a true step-by-step loop for the scheduler.
    """
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
        )

    new_ids = output_ids[0][prompt_len:].tolist()
    for token_id in new_ids:
        yield token_id, prompt_len


def batch_generate(
    prompts: list[str],
    max_tokens: int,
    temperature: float = 0.0,
) -> list[list[int]]:
    """
    Generate tokens for a batch of prompts using model.generate().
    Left-pads shorter prompts so all sequences have the same length.
    Returns one list of token ids per prompt (prompt tokens excluded).

    Correctness guarantee: for batch_size=1 with no padding the output is
    token-for-token identical to generate() — same model.generate() codepath.
    Phase 4 replaces this with a continuous step-by-step scheduler loop.
    """
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_model_device()

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,          # left-pads by default for decoder-only models
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
        )

    prompt_len = enc["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id
    results = []
    for i in range(len(prompts)):
        tokens = out[i][prompt_len:].tolist()
        # Truncate at the first EOS — model.generate pads the rest with pad/eos
        if eos_id in tokens:
            tokens = tokens[: tokens.index(eos_id)]
        results.append(tokens)
    return results


def batch_forward_step(
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> "torch.Tensor":
    """
    One forward pass over a batch, no KV cache.
    Returns logits (B, vocab) for the last position of each sequence.
    Used in test_correctness.py; Phase 4 scheduler will use this directly.
    NOTE: in transformers 5.x this diverges from model.generate() due to
    internal cache_position handling — use batch_generate() for correctness.
    """
    model = get_model()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits[:, -1, :]  # (B, vocab)
