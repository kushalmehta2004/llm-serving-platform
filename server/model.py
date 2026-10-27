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
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, new_ids, prompt_len, len(new_ids)
