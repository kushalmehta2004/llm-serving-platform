"""
Phase 6 — Throughput-vs-concurrency and latency-vs-concurrency charts.

Reads bench/results.json produced by run_bench.py.

Usage:
    python bench/plot.py [--input bench/results.json] [--out-dir bench/charts]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Color scheme per target (color-blind friendly)
_COLORS = {
    "static": "#e07b39",   # orange
    "paged":  "#4a90d9",   # blue
    "vllm":   "#5cb85c",   # green
}
_LABELS = {
    "static": "Static batching (Phase 3)",
    "paged":  "Continuous + Paged KV (Phase 5)",
    "vllm":   "vLLM",
}


def _group(results: list[dict]) -> dict[str, dict[int, dict]]:
    """Returns {target: {concurrency: metrics_row}}."""
    out: dict[str, dict] = {}
    for row in results:
        t, c = row["target"], row["concurrency"]
        out.setdefault(t, {})[c] = row
    return out


def _plot_metric(
    grouped: dict,
    metric: str,
    ylabel: str,
    title: str,
    out_path: str,
    log_y: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    for target, by_conc in sorted(grouped.items()):
        xs = sorted(by_conc)
        ys = [by_conc[c].get(metric) for c in xs]
        if any(y is None for y in ys):
            continue
        ax.plot(
            xs, ys,
            marker="o",
            linewidth=2,
            color=_COLORS.get(target, "#888"),
            label=_LABELS.get(target, target),
        )

    ax.set_xlabel("Concurrency (simultaneous requests)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default="bench/results.json")
    ap.add_argument("--out-dir", default="bench/charts")
    args = ap.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    grouped = _group(results)

    _plot_metric(
        grouped,
        metric="throughput_tps",
        ylabel="Throughput (tokens / second)",
        title="Throughput vs Concurrency",
        out_path=os.path.join(args.out_dir, "throughput_vs_concurrency.png"),
    )
    _plot_metric(
        grouped,
        metric="latency_p50",
        ylabel="P50 Total Latency (s)",
        title="P50 Latency vs Concurrency",
        out_path=os.path.join(args.out_dir, "latency_p50_vs_concurrency.png"),
    )
    _plot_metric(
        grouped,
        metric="latency_p99",
        ylabel="P99 Total Latency (s)",
        title="P99 Latency vs Concurrency",
        out_path=os.path.join(args.out_dir, "latency_p99_vs_concurrency.png"),
        log_y=True,
    )
    _plot_metric(
        grouped,
        metric="ttft_p50",
        ylabel="P50 Time-to-First-Token (s)",
        title="TTFT P50 vs Concurrency",
        out_path=os.path.join(args.out_dir, "ttft_p50_vs_concurrency.png"),
    )

    print(f"\nAll charts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
