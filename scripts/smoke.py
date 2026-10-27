"""
Phase 0 smoke test — loads the model and runs a 20-token greedy generation.
Usage: python scripts/smoke.py
"""

import sys
import os

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.model import load_model, generate_one, get_model_device

if __name__ == "__main__":
    load_model()
    prompt = "The key insight behind continuous batching in LLM serving is"
    print(f"Device : {get_model_device()}")
    print(f"Prompt : {prompt!r}")
    output = generate_one(prompt, max_new_tokens=20)
    print(f"Output : {output!r}")
    print("Smoke test passed.")
