"""
Phase 6 — Prometheus metrics for the from-scratch LLM server.

Exposes a /metrics endpoint (via prometheus_client) with:
  llm_tokens_generated_total   — counter
  llm_queue_depth              — gauge
  llm_active_batch_size        — gauge
  llm_kv_utilization           — gauge (paged scheduler only)
  llm_request_latency_seconds  — histogram (TTFT + total)

Usage: import and call update_*() from the scheduler; mount the /metrics
route in api.py.
"""

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# ── counters / gauges ─────────────────────────────────────────────────────────

tokens_generated = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated across all requests",
)

queue_depth = Gauge(
    "llm_queue_depth",
    "Number of requests waiting in the queue",
)

active_batch_size = Gauge(
    "llm_active_batch_size",
    "Number of sequences currently being decoded",
)

kv_utilization = Gauge(
    "llm_kv_utilization",
    "Fraction of KV-cache blocks in use (paged scheduler only)",
)

request_latency = Histogram(
    "llm_request_latency_seconds",
    "End-to-end request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

ttft = Histogram(
    "llm_ttft_seconds",
    "Time-to-first-token in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# ── ASGI app (mount at /metrics in api.py) ────────────────────────────────────

metrics_app = make_asgi_app()
