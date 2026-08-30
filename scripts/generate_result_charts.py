#!/usr/bin/env python3
"""Generates the result charts committed under results/charts/ for the project
README. Pure post-hoc analysis over already-persisted comparisons in
artifacts/dev_eval.db and artifacts/holdout_eval.db — no generator/judge/guardrail
calls, nothing paid, fully reproducible from data already on disk.

The guardrail v1-vs-v2 numbers are the one exception: they come from
scripts/compare_guardrail_versions.py's directly-verified output (Phase 8.2), not a
persisted Comparison row, and are hardcoded here with that citation.

Usage:
    python scripts/generate_result_charts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evalguard.db.store import get_comparison, get_connection  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "charts"
DEV_DB = REPO_ROOT / "artifacts" / "dev_eval.db"
HOLDOUT_DB = REPO_ROOT / "artifacts" / "holdout_eval.db"

# Phase 8.2, scripts/compare_guardrail_versions.py: 120 dev cases, both guardrail
# versions tested via real ApplyGuardrail calls (not a Comparison row).
GUARDRAIL_V1_SUITE_FP = 0.5556
GUARDRAIL_V2_SUITE_FP = 0.1111
GUARDRAIL_V1_OVERALL_FP = 0.1212
GUARDRAIL_V2_OVERALL_FP = 0.0606

_COLORS = {"v1": "#6b7280", "v2": "#3b82f6", "v3": "#10b981"}


def _delta(conn, cid: int) -> dict:
    comparison = get_comparison(conn, cid)
    return {d.metric.value: (d.baseline_value, d.candidate_value) for d in comparison.deltas}


def chart_quality_progression(dev_v1v2: dict, dev_v1v3: dict) -> None:
    """V1 -> V2 -> V3 on the three headline quality metrics (dev)."""
    v1_correctness = dev_v1v2["answer_correctness"][0]
    v2_correctness = dev_v1v2["answer_correctness"][1]
    v3_correctness = dev_v1v3["answer_correctness"][1]
    v1_faith = dev_v1v2["faithfulness"][0]
    v2_faith = dev_v1v2["faithfulness"][1]
    v3_faith = dev_v1v3["faithfulness"][1]
    v1_abst = dev_v1v2["abstention_accuracy"][0]
    v2_abst = dev_v1v2["abstention_accuracy"][1]
    v3_abst = dev_v1v3["abstention_accuracy"][1]

    metrics = ["answer_correctness", "faithfulness", "abstention_accuracy"]
    v1_vals = [v1_correctness, v1_faith, v1_abst]
    v2_vals = [v2_correctness, v2_faith, v2_abst]
    v3_vals = [v3_correctness, v3_faith, v3_abst]

    x = range(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width for i in x], v1_vals, width, label="V1 (production)", color=_COLORS["v1"])
    ax.bar(x, v2_vals, width, label="V2 (candidate)", color=_COLORS["v2"])
    ax.bar([i + width for i in x], v3_vals, width, label="V3 (candidate)", color=_COLORS["v3"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Answer\ncorrectness", "Faithfulness", "Abstention\naccuracy"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Quality progression V1 -> V2 -> V3 (120 dev cases)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "quality_progression_dev.png", dpi=150)
    plt.close(fig)


def chart_dev_vs_holdout(dev_v1v3: dict, holdout_v1v3: dict) -> None:
    """The headline finding: answer_correctness looked like an improvement on dev,
    but was a real, CI-confirmed regression on the sealed holdout split."""
    dev_delta = dev_v1v3["answer_correctness"][1] - dev_v1v3["answer_correctness"][0]
    holdout_delta = holdout_v1v3["answer_correctness"][1] - holdout_v1v3["answer_correctness"][0]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Dev\n(120 cases)", "Holdout\n(40 cases, sealed)"],
        [dev_delta, holdout_delta],
        color=["#10b981", "#ef4444"],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("answer_correctness delta (V3 - V1)")
    ax.set_title("The generalization gap: V3 vs V1 answer_correctness")
    for bar, val in zip(bars, [dev_delta, holdout_delta], strict=True):
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


def chart_cost_latency(dev_v1v2: dict, dev_v1v3: dict) -> None:
    labels = ["V1", "V2", "V3"]
    p95 = [
        dev_v1v2["p95_latency"][0],
        dev_v1v2["p95_latency"][1],
        dev_v1v3["p95_latency"][1],
    ]
    cost = [
        dev_v1v2["cost_per_query"][0] * 1000,
        dev_v1v2["cost_per_query"][1] * 1000,
        dev_v1v3["cost_per_query"][1] * 1000,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, p95, color=[_COLORS["v1"], _COLORS["v2"], _COLORS["v3"]])
    axes[0].set_title("p95 latency (ms)")
    axes[0].set_ylabel("ms")
    axes[1].bar(labels, cost, color=[_COLORS["v1"], _COLORS["v2"], _COLORS["v3"]])
    axes[1].set_title("Cost per query (m$, thousandths of a cent)")
    axes[1].set_ylabel("m$ per query")
    fig.suptitle("Cost / latency across prompt versions (120 dev cases)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost_latency_dev.png", dpi=150)
    plt.close(fig)


def chart_guardrail_improvement() -> None:
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

    dev_conn = get_connection(DEV_DB)
    dev_v1v2 = _delta(dev_conn, 1)
    dev_v1v3 = _delta(dev_conn, 2)

    holdout_conn = get_connection(HOLDOUT_DB)
    holdout_row = holdout_conn.execute("SELECT id FROM comparisons").fetchone()
    holdout_v1v3 = _delta(holdout_conn, holdout_row["id"])

    chart_quality_progression(dev_v1v2, dev_v1v3)
    chart_dev_vs_holdout(dev_v1v3, holdout_v1v3)
    chart_cost_latency(dev_v1v2, dev_v1v3)
    chart_guardrail_improvement()

    print(f"Wrote 4 charts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
