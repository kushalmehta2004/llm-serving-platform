"""
Phase 6 — Benchmark sweep: naive (static) vs continuous+paged vs vLLM.

For each target × concurrency level it:
  1. Sends a warmup burst (excluded from metrics).
  2. Fires N concurrent chat-completion requests using asyncio.
  3. Measures throughput (tokens/s), TTFT, and total latency p50/p99.
  4. Logs config + results to W&B.

Usage:
    # Start each server first, then:
    python bench/run_bench.py \
        --static-url  http://localhost:8000 \
        --paged-url   http://localhost:8000 \
        --vllm-url    http://localhost:8001 \
        --concurrency 1 8 32 \
        --max-tokens  64 \
        --wandb-project llm-serving-platform

    # To skip a target, omit its --*-url flag.
    # To skip W&B, omit --wandb-project.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from typing import Optional

import aiohttp

# ── prompts (same pool as locustfile.py) ─────────────────────────────────────

_RNG = random.Random(42)

_PROMPTS = [
    "What is 2+2?",
    "Name the capital of France.",
    "Explain the difference between supervised and unsupervised learning in three sentences.",
    "Describe how a transformer attention mechanism works at a high level.",
    (
        "You are an expert distributed systems engineer. "
        "Explain in detail how continuous batching improves GPU utilization in LLM inference, "
        "contrasting it with static batching, covering: (1) padding waste, "
        "(2) continuous admission, (3) paged KV caching, and (4) scheduling heuristics."
    ),
]

_WEIGHTS = [0.25, 0.25, 0.20, 0.20, 0.10]


def _sample_prompt() -> str:
    return _RNG.choices(_PROMPTS, weights=_WEIGHTS, k=1)[0]


# ── single request ────────────────────────────────────────────────────────────

async def _one_request(
    session: aiohttp.ClientSession,
    url: str,
    max_tokens: int,
) -> dict:
    """Returns {ttft, total_latency, completion_tokens, ok}."""
    prompt = _sample_prompt()
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            t_first = time.perf_counter()   # non-streaming: first byte ≈ TTFT
            data = await resp.json()
            t_end = time.perf_counter()

        if resp.status != 200:
            return {"ok": False, "error": f"HTTP {resp.status}"}

        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {
            "ok": True,
            "ttft": t_first - t0,
            "total_latency": t_end - t0,
            "completion_tokens": tokens,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── benchmark one (url, concurrency) pair ────────────────────────────────────

async def _benchmark(
    url: str,
    concurrency: int,
    n_requests: int,
    max_tokens: int,
    warmup: int = 4,
) -> dict:
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        warmup_tasks = [_one_request(session, url, max_tokens) for _ in range(warmup)]
        await asyncio.gather(*warmup_tasks)

        # Timed run
        sem = asyncio.Semaphore(concurrency)

        async def _guarded():
            async with sem:
                return await _one_request(session, url, max_tokens)

        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*[_guarded() for _ in range(n_requests)])
        t_wall_end = time.perf_counter()

    wall_time = t_wall_end - t_wall_start
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return {"error": "all requests failed"}

    ttfts         = [r["ttft"]           for r in ok]
    latencies     = [r["total_latency"]  for r in ok]
    total_tokens  = sum(r["completion_tokens"] for r in ok)
    throughput    = total_tokens / wall_time

    def _p(vals, pct):
        vals_sorted = sorted(vals)
        idx = int(len(vals_sorted) * pct / 100)
        return vals_sorted[min(idx, len(vals_sorted) - 1)]

    return {
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "wall_time_s": round(wall_time, 2),
        "throughput_tps": round(throughput, 2),
        "ttft_p50": round(_p(ttfts, 50), 3),
        "ttft_p99": round(_p(ttfts, 99), 3),
        "latency_p50": round(_p(latencies, 50), 3),
        "latency_p99": round(_p(latencies, 99), 3),
        "total_tokens": total_tokens,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def _log_wandb(project: str, run_name: str, config: dict, summary: dict):
    try:
        import wandb
        with wandb.init(project=project, name=run_name, config=config, reinit=True) as run:
            run.summary.update(summary)
        print(f"  [W&B] logged → {project}/{run_name}")
    except Exception as exc:
        print(f"  [W&B] skipped ({exc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-url",  default=None)
    ap.add_argument("--paged-url",   default=None)
    ap.add_argument("--vllm-url",    default=None)
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--n-requests",  type=int, default=50)
    ap.add_argument("--max-tokens",  type=int, default=64)
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    targets = {}
    if args.static_url:  targets["static"]  = args.static_url
    if args.paged_url:   targets["paged"]   = args.paged_url
    if args.vllm_url:    targets["vllm"]    = args.vllm_url

    if not targets:
        print("No targets specified. Pass at least one of --static-url / --paged-url / --vllm-url.")
        return

    all_results: list[dict] = []

    for name, url in targets.items():
        for conc in args.concurrency:
            print(f"\n>> {name}  concurrency={conc}  n={args.n_requests}")
            metrics = asyncio.run(
                _benchmark(url, conc, args.n_requests, args.max_tokens)
            )
            row = {"target": name, "concurrency": conc, **metrics}
            all_results.append(row)
            print(f"  throughput={metrics.get('throughput_tps')} tok/s  "
                  f"lat_p50={metrics.get('latency_p50')}s  "
                  f"lat_p99={metrics.get('latency_p99')}s  "
                  f"ttft_p50={metrics.get('ttft_p50')}s")

            if args.wandb_project:
                cfg = {
                    "target": name,
                    "concurrency": conc,
                    "n_requests": args.n_requests,
                    "max_tokens": args.max_tokens,
                }
                _log_wandb(
                    args.wandb_project,
                    f"{name}_c{conc}",
                    cfg,
                    metrics,
                )

    # Save results JSON for plot.py
    out_path = "bench/results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print("Run  python bench/plot.py  to generate charts.")


if __name__ == "__main__":
    main()
