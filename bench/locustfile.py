"""
Phase 6 — Locust load profiles with mixed prompt lengths.

Sends chat-completion requests to an OpenAI-compatible endpoint.
Three prompt length buckets (short / medium / long) weighted 50/30/20.

Usage:
    locust -f bench/locustfile.py --host http://localhost:8000 \
           --headless -u 8 -r 2 --run-time 60s
"""

import random
from locust import HttpUser, task, constant_pacing

# Fixed prompt pool — same distribution every run (deterministic seed below)
_rng = random.Random(42)

SHORT_PROMPTS = [
    "What is 2+2?",
    "Name the capital of France.",
    "Define 'entropy'.",
    "What color is the sky?",
    "Who wrote Hamlet?",
]

MEDIUM_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in three sentences.",
    "Describe how a transformer attention mechanism works at a high level.",
    "What are the main trade-offs between batch size and learning rate during training?",
    "Summarize the key ideas behind reinforcement learning from human feedback (RLHF).",
]

LONG_PROMPTS = [
    (
        "You are an expert distributed systems engineer. "
        "Explain in detail how continuous batching improves GPU utilization in LLM inference, "
        "contrasting it with static batching, covering: (1) how padding waste arises in static batching, "
        "(2) how continuous batching reclaims idle slots mid-sequence, "
        "(3) how paged KV caching reduces memory fragmentation, "
        "and (4) what scheduling heuristics (e.g. FCFS vs. priority) affect fairness and throughput."
    ),
    (
        "You are a senior ML researcher. Describe the full training pipeline for a large language model: "
        "data collection and deduplication, tokenization choices, model architecture decisions "
        "(number of layers, heads, hidden size), pre-training objectives, learning rate schedules, "
        "mixed-precision training, gradient checkpointing, and final alignment steps such as SFT and RLHF."
    ),
]


def _random_prompt() -> str:
    r = _rng.random()
    if r < 0.50:
        return _rng.choice(SHORT_PROMPTS)
    elif r < 0.80:
        return _rng.choice(MEDIUM_PROMPTS)
    else:
        return _rng.choice(LONG_PROMPTS)


class LLMUser(HttpUser):
    # Pacing: each user sends ~1 request per second on average.
    wait_time = constant_pacing(1)

    @task
    def chat_completion(self):
        prompt = _random_prompt()
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.0,
        }
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                resp.success()
                # Attach token count for throughput calculation in run_bench.py
                resp.meta = {"completion_tokens": tokens}
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
