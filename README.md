# LLM Serving Platform

A from-scratch LLM inference server with continuous batching and a paged KV cache, benchmarked against vLLM.

## Stack

- Python 3.11, PyTorch 2.4.x, Transformers 4.44.x, FastAPI + uvicorn
- vLLM 0.6.x (comparison server)
- Locust (load testing), W&B (experiment tracking), Prometheus + Grafana (observability)

## Model

Set `MODEL_ID` in `.env`. Default: `Qwen/Qwen2.5-0.5B-Instruct`.

## Quick start

```bash
cp .env.example .env
pip install -r server/requirements.txt
python scripts/smoke.py        # Phase 0 acceptance check
```

## Build phases

| Phase | Name | Status |
|-------|------|--------|
| 0 | Scaffold + model load | ✅ |
| 1 | Single-request endpoint | ✅ |
| 2 | SSE streaming | ✅ |
| 3 | Static batching | ✅ |
| 4 | Continuous batching | ✅ |
| 5 | Paged KV cache | 🔲 |
| 6 | vLLM + benchmark + observability | 🔲 |
| 7 | Kubernetes deploy | 🔲 |

## Benchmark results

### Phase 4 — Continuous batching latency (CPU, Qwen2.5-0.5B-Instruct)

Short request arriving 2s after a long request is already running:

| Scheduler | SHORT latency | LONG latency | Notes |
|-----------|--------------|--------------|-------|
| Static batching (Phase 3) | 12.2s | 12.2s | SHORT blocks until LONG finishes |
| Continuous batching (Phase 4) | **4.8s** | 27.4s | SHORT joins running batch immediately |

**2.55x speedup** for short-request latency.

_Full throughput benchmark (Locust + GPU) to be filled after Phase 6._

## Hardware

_Document GPU / CPU used for final benchmark here._
