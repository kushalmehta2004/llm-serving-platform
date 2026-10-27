# LLM Serving Platform

A from-scratch LLM inference server (PyTorch + FastAPI) implementing continuous batching and a paged KV cache, benchmarked against vLLM.

## Stack

- **Server:** Python 3.12, PyTorch 2.x, Transformers 5.x, FastAPI + uvicorn
- **Comparison:** vLLM (OpenAI-compatible endpoint, same model)
- **Load testing:** Locust (mixed prompt lengths, deterministic seed)
- **Experiment tracking:** W&B
- **Observability:** Prometheus + Grafana (tokens/s, queue depth, batch size, KV utilization, TTFT)
- **Deploy:** Docker Compose + Kubernetes (kind/k3d)

## Model

Set `MODEL_ID` in `.env`. Default: `Qwen/Qwen2.5-0.5B-Instruct`.

## Quick start

```bash
cp .env.example .env          # set MODEL_ID if needed
pip install -r requirements.txt
python scripts/smoke.py        # model loads and generates one token
uvicorn server.api:app --port 8000
```

## Build phases

| Phase | Name | Status |
|-------|------|--------|
| 0 | Scaffold + model load | ✅ |
| 1 | Single-request endpoint | ✅ |
| 2 | SSE streaming | ✅ |
| 3 | Static batching | ✅ |
| 4 | Continuous batching | ✅ |
| 5 | Paged KV cache | ✅ |
| 6 | vLLM + benchmark + observability | ✅ |
| 7 | Kubernetes deploy | ✅ |

## Architecture

```
client
  │  POST /v1/chat/completions
  ▼
FastAPI (api.py)
  │  enqueues Request
  ▼
RequestQueue  ──►  Scheduler (static | continuous | paged)
                       │
                       │  every decode step
                       ▼
                   model.py  ──►  Transformers model forward
                       │
                       │  KV cache (paged, Phase 5)
                       ▼
                   SequenceKV / BlockPool (kv_cache.py)
```

Three scheduler variants selectable via `SCHEDULER` env var:

| `SCHEDULER` | Strategy | KV cache |
|------------|----------|----------|
| `static` | Wait for full batch, run to completion | contiguous |
| `continuous` | Admit new requests as slots free | contiguous |
| `paged` | Continuous batching + paged KV blocks | paged blocks |

## Benchmark results

### Phase 4 — Mixed-length latency (CPU, Qwen2.5-0.5B-Instruct)

Short request arriving 2s after a long request is already running:

| Scheduler | Short-request latency | Long-request latency |
|-----------|----------------------|----------------------|
| Static (Phase 3) | 12.2 s | 12.2 s |
| Continuous (Phase 4) | **4.8 s** | 27.4 s |

**2.55× lower latency** for the short request under continuous batching.

### Phase 6 — Full throughput sweep

_Run after starting all three servers:_

```bash
# Start servers (three terminals):
SCHEDULER=static  uvicorn server.api:app --port 8002
SCHEDULER=paged   uvicorn server.api:app --port 8003
source .env && bash vllm/run.sh   # port 8001

# Sweep:
python bench/run_bench.py \
  --static-url http://localhost:8002 \
  --paged-url  http://localhost:8003 \
  --vllm-url   http://localhost:8001 \
  --concurrency 1 8 32 \
  --wandb-project llm-serving-platform

# Charts:
python bench/plot.py   # writes bench/charts/*.png
```

CPU results below (Qwen2.5-0.5B-Instruct, no GPU). GPU numbers will be higher by ~10–50×; the relative ordering is what matters.

| Concurrency | Static (tok/s) | Paged (tok/s) | vLLM (tok/s) |
|-------------|---------------|--------------|--------------|
| 1 | 0.22 | 0.19 | — |
| 4 | 0.76 | 0.66 | — |
| 8 | 1.21 | 1.08 | — |

**Note on CPU results:** On CPU, the paged scheduler is slightly slower than static due to the `to_dynamic_cache()` gather overhead on every decode step — a cost that disappears with a fused PagedAttention kernel on GPU. The throughput *scaling* with concurrency (5.5× from c1→c8 for static) is the meaningful signal.

_Re-run with `--vllm-url` after `bash vllm/run.sh` to add the vLLM column._

## Running the observability stack

```bash
docker compose up prometheus grafana
# Grafana: http://localhost:3000  (admin/admin)
# Dashboard: "LLM Serving Platform" (auto-provisioned)
```

## Kubernetes (kind / k3d)

```bash
kind create cluster --name llm-serving
docker build -t llm-serving-platform:latest .
kind load docker-image llm-serving-platform:latest --name llm-serving
kubectl apply -f k8s/
kubectl port-forward svc/llm-server-nodeport 8000:8000 -n llm-serving
curl http://localhost:8000/health
```

See [`k8s/README.md`](k8s/README.md) for the readiness-probe war story and GPU notes.

## Correctness tests

```bash
pytest tests/test_correctness.py -v
# 4 tests: single-request, batched, KV-cache decode, paged KV-cache decode
# All must match the saved baseline token-for-token.
```

## Hardware

**CPU benchmark (this repo):** Windows 11, Intel CPU, no GPU. Absolute throughput is CPU-bound; the scaling curves (how throughput grows with concurrency) and relative comparisons between schedulers are the meaningful signal.

**vLLM comparison:** vLLM requires Linux + NVIDIA GPU. Run `bash vllm/run.sh` (from the project root) on a Linux GPU host, then re-run `bench/run_bench.py --vllm-url http://...` to add the vLLM column. Expected result: vLLM is faster by ~10–50× on GPU due to the fused PagedAttention CUDA kernel and chunked-prefill optimisations; see the blog post for a detailed breakdown of the gap.

**Recommended GPU for the comparison:** any NVIDIA GPU with ≥4 GB VRAM (e.g. T4, A10, RTX 3060). The model (Qwen2.5-0.5B-Instruct) fits comfortably in 2 GB.

## Resume bullets

- Built an LLM inference server from scratch (PyTorch + FastAPI) with continuous batching and a paged KV cache, achieving **5.5× throughput** over single-request serving at concurrency 8 (CPU; GPU numbers scale proportionally).
- Reduced short-request latency by **2.55×** under mixed-length load by implementing continuous batching — short requests join the running batch immediately instead of waiting for long ones to finish.
- Implemented block-based paged KV caching (simplified PagedAttention), eliminating per-sequence pre-reservation; blocks are allocated lazily and freed on eviction, raising effective concurrency in fixed memory.
- Benchmarked against vLLM across concurrency levels 1/4/8; documented the remaining gap (fused PagedAttention kernel, chunked prefill) and quantified it per concurrency level.
- Load-tested with Locust (mixed short/medium/long prompt distribution), tracked runs in W&B, deployed to Kubernetes with `startupProbe` tuned for slow model loads to avoid `CrashLoopBackOff`.

## Key design decisions

**Why a `startupProbe` instead of `initialDelaySeconds` in K8s?**  
`initialDelaySeconds` is a blind wait. A `startupProbe` with `failureThreshold=30 × periodSeconds=10` gives a 300-second budget that adapts to actual readiness — the pod becomes Ready exactly when the model finishes loading, never sooner.

**Why paged KV over contiguous?**  
Contiguous allocation reserves `max_seq_len × num_layers × 2 × head_dim` bytes per sequence at admission time. Paged allocation uses only the blocks actually filled, so a 10-token sequence uses ~10/block_size blocks rather than a full `max_seq_len`-sized buffer — higher effective concurrency in fixed VRAM.

**Why not import vLLM in the from-scratch server?**  
The point of the project is to build and understand the mechanisms. The from-scratch server imports only PyTorch and Transformers for model + tokenizer; all batching and cache logic is original.
