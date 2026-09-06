#!/usr/bin/env python3
"""Generates the result charts committed under results/charts/ for the project
README and site/. Reads exclusively from the already-committed comparison
reports in results/reports/*.json — no generator/judge/guardrail calls,
nothing paid, no dependency on local gitignored artifacts/*.db state, fully
reproducible by anyone who has cloned this repo alone.

The guardrail v1-vs-v2 numbers are the one exception: they come from
scripts/compare_guardrail_versions.py's directly-verified output (Phase 8.2),
never persisted as a Comparison row, and are hardcoded here with that
citation (identical to the constants in frontend/data/safety.ts).

Usage:
    python scripts/generate_result_charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "charts"
REPORTS_DIR = REPO_ROOT / "results" / "reports"

# Phase 8.2, scripts/compare_guardrail_versions.py: 120 dev cases, both guardrail
# versions tested via real ApplyGuardrail calls (not a Comparison row).
GUARDRAIL_V1_SUITE_FP = 0.5556
GUARDRAIL_V2_SUITE_FP = 0.1111
GUARDRAIL_V1_OVERALL_FP = 0.1212
GUARDRAIL_V2_OVERALL_FP = 0.0606

_COLORS = {
    "v1": "#6b7280",
    "v2": "#94a3b8",
    "v3": "#f59e0b",
    "v3.1": "#fbbf24",
    "v3.2": "#facc15",
    "v3.3": "#a3e635",
    "v3.4": "#10b981",
}

# Dev-split versions in chronological/narrative order, each compared against
# the V1 baseline. Every filename here must exist in results/reports/.
DEV_VERSIONS = ["v2", "v3", "v3.1", "v3.2", "v3.3", "v3.4"]


def _load(name: str) -> dict:
    return json.loads((REPORTS_DIR / f"{name}.json").read_text())


def chart_quality_progression(dev_reports: dict[str, dict]) -> None:
    """V1 -> V2 -> V3 -> V3.1 -> V3.2 -> V3.3 -> V3.4 on the three headline
    quality metrics (dev). A line chart, since seven versions as grouped bars
    would be unreadable."""
    metrics = ["answer_correctness", "faithfulness", "abstention_accuracy"]
    labels = ["V1"] + [v.upper() for v in DEV_VERSIONS]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    v1_baseline = dev_reports["v2"]["baseline_values"]
    for metric, marker in zip(metrics, ["o", "s", "^"], strict=True):
        values = [v1_baseline[metric]]
        for v in DEV_VERSIONS:
            values.append(dev_reports[v]["candidate_values"][metric])
        ax.plot(labels, values, marker=marker, label=metric.replace("_", " "), linewidth=2)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Quality progression V1 -> V3.4 (120 dev cases)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "quality_progression_dev.png", dpi=150)
    plt.close(fig)


def chart_dev_vs_holdout(dev_reports: dict[str, dict]) -> None:
    """The headline finding, kept intact: V3 looked like an improvement on dev
    but was a real, CI-confirmed regression on the sealed holdout split. V3.4
    is the contrast -- a genuine win confirmed on *both* splits."""
    v3_holdout = _load("v1_vs_v3_holdout")
    v34_holdout = _load("v1_vs_v3.4_holdout")

    rows = [
        ("V3\ndev", dev_reports["v3"]["delta"]["answer_correctness"], "#f59e0b"),
        ("V3\nholdout", v3_holdout["delta"]["answer_correctness"], "#ef4444"),
        ("V3.4\ndev", dev_reports["v3.4"]["delta"]["answer_correctness"], "#34d399"),
        ("V3.4\nholdout", v34_holdout["delta"]["answer_correctness"], "#10b981"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("answer_correctness delta (candidate - V1)")
    ax.set_title("The generalization test: V3's warning sign vs. V3.4's confirmed win")
    for bar, (_, val, _) in zip(bars, rows, strict=True):
        ax.annotate(
            f"{val:+.3f}",
            (bar.get_x() + bar.get_width() / 2, val),
            textcoords="offset points",
            xytext=(0, 6 if val >= 0 else -14),
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dev_vs_holdout_correctness.png", dpi=150)
    plt.close(fig)


def chart_cost_latency(dev_reports: dict[str, dict]) -> None:
    labels = ["V1"] + [v.upper() for v in DEV_VERSIONS]
    v1_baseline = dev_reports["v2"]["baseline_values"]

    p95 = [v1_baseline["p95_latency"]] + [
        dev_reports[v]["candidate_values"]["p95_latency"] for v in DEV_VERSIONS
    ]
    cost = [v1_baseline["cost_per_query"] * 1000] + [
        dev_reports[v]["candidate_values"]["cost_per_query"] * 1000 for v in DEV_VERSIONS
    ]
    colors = [_COLORS["v1"]] + [_COLORS[v] for v in DEV_VERSIONS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, p95, color=colors)
    axes[0].set_title("p95 latency (ms)")
    axes[0].set_ylabel("ms")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(labels, cost, color=colors)
    axes[1].set_title("Cost per query (m$, thousandths of a cent)")
    axes[1].set_ylabel("m$ per query")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Cost / latency across prompt versions (120 dev cases)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost_latency_dev.png", dpi=150)
    plt.close(fig)


def chart_guardrail_improvement() -> None:
    """Unchanged from the original: the guardrail PROMPT_ATTACK threshold
    (v1=HIGH, v2=MEDIUM) is a separate axis from the prompt versions above and
    was never stale -- regenerated here only for a consistent look."""
    labels = ["Guardrail suite\nbenign cases (9)", "Overall\n(all 99 non-attack cases)"]
    v1_vals = [GUARDRAIL_V1_SUITE_FP, GUARDRAIL_V1_OVERALL_FP]
    v2_vals = [GUARDRAIL_V2_SUITE_FP, GUARDRAIL_V2_OVERALL_FP]

    x = range(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [i - width / 2 for i in x],
        v1_vals,
        width,
        label="Guardrail v1 (PROMPT_ATTACK=HIGH)",
        color=_COLORS["v1"],
    )
    ax.bar(
        [i + width / 2 for i in x],
        v2_vals,
        width,
        label="Guardrail v2 (PROMPT_ATTACK=MEDIUM)",
        color=_COLORS["v2"],
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Benign false-positive rate")
    ax.set_title("Guardrail calibration: benign false positives (all 120 dev cases)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "guardrail_improvement.png", dpi=150)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dev_reports = {v: _load(f"v1_vs_{v}_dev") for v in DEV_VERSIONS}

    chart_quality_progression(dev_reports)
    chart_dev_vs_holdout(dev_reports)
    chart_cost_latency(dev_reports)
    chart_guardrail_improvement()

    print(f"Wrote 4 charts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
