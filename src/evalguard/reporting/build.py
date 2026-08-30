"""Builds an ExperimentReport from a real, already-computed Comparison — pure
reshaping, no new computation, no fabricated values. Every field traces back to
either the Comparison itself (deltas, gate results, decision, n_paired,
created_at) or the PolicyConfig that produced its gates (config/policy.yaml,
loaded the same way regression/compare.py loads it for compare_runs).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from evalguard.config import GateConfig, PolicyConfig
from evalguard.db.store import (
    get_guardrail_results_for_response,
    get_responses_for_run,
    get_scores_for_response,
)
from evalguard.models import Comparison

from .models import CostBreakdown, ExperimentReport

# GateConfig fields that represent a configured threshold (as opposed to `metric`/
# `direction`, which describe the gate rather than bound it).
_THRESHOLD_FIELDS = (
    "hold_if_delta_worse_than",
    "review_if_delta_worse_than",
    "hold_if_delta_worse_than_pct",
    "review_if_delta_worse_than_pct",
    "absolute_min",
    "absolute_max",
)


def _configured_thresholds(gate: GateConfig) -> dict[str, float]:
    return {
        field: value for field in _THRESHOLD_FIELDS if (value := getattr(gate, field)) is not None
    }


def _compute_cost_breakdown(conn: sqlite3.Connection, run_id: int) -> CostBreakdown:
    """Same logic as the ad-hoc `_run_costs()` helper duplicated in
    scripts/run_v1_v3_dev_eval.py and scripts/run_holdout_eval.py — generator cost
    from Response.cost_usd (successful responses only), guardrail cost from
    GuardrailResult.cost_usd, judge cost from Score.judge_cost_usd. None (not 0.0)
    when a service has no priced rows for this run at all."""
    responses = get_responses_for_run(conn, run_id)
    successful = [r for r in responses if r.error_class is None]
    generator_cost = (
        sum(r.cost_usd for r in successful)
        if successful and all(r.cost_usd is not None for r in successful)
        else None
    )

    guardrail_results = [
        gr for r in responses for gr in get_guardrail_results_for_response(conn, r.id)
    ]
    priced_guardrail = [gr.cost_usd for gr in guardrail_results if gr.cost_usd is not None]
    guardrail_cost = sum(priced_guardrail) if priced_guardrail else None

    scores = [s for r in responses for s in get_scores_for_response(conn, r.id)]
    priced_judge = [s.judge_cost_usd for s in scores if s.judge_cost_usd is not None]
    judge_cost = sum(priced_judge) if priced_judge else None

    components = (generator_cost, guardrail_cost, judge_cost)
    total_evaluation_cost = sum(components) if all(c is not None for c in components) else None

    return CostBreakdown(
        generator_cost=generator_cost,
        guardrail_cost=guardrail_cost,
        judge_cost=judge_cost,
        total_evaluation_cost=total_evaluation_cost,
    )


def build_experiment_report(
    conn: sqlite3.Connection,
    comparison: Comparison,
    policy: PolicyConfig,
    *,
    experiment_id: str,
    baseline_version: str,
    candidate_version: str,
    dataset: str,
) -> ExperimentReport:
    metrics = [d.metric.value for d in comparison.deltas]
    baseline_values = {d.metric.value: d.baseline_value for d in comparison.deltas}
    candidate_values = {d.metric.value: d.candidate_value for d in comparison.deltas}
    delta = {d.metric.value: d.delta for d in comparison.deltas}
    confidence_interval = {
        d.metric.value: ([d.ci_low, d.ci_high] if d.ci_low is not None else None)
        for d in comparison.deltas
    }
    effect_size = {d.metric.value: d.effect_size for d in comparison.deltas}
    threshold = {gate.metric: _configured_thresholds(gate) for gate in policy.gates}

    decision_reasons = [
        f"{gate.gate_name}: {gate.reason}"
        for gate in comparison.release.gates
        if gate.status.value != "GO"
    ]
    if not decision_reasons:
        decision_reasons = [comparison.release.summary or "all gates passed"]

    baseline_cost_breakdown = _compute_cost_breakdown(conn, comparison.baseline_run_id)
    candidate_cost_breakdown = _compute_cost_breakdown(conn, comparison.candidate_run_id)

    return ExperimentReport(
        experiment_id=experiment_id,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        dataset=dataset,
        valid_cases=comparison.n_paired,
        metrics=metrics,
        baseline_values=baseline_values,
        candidate_values=candidate_values,
        delta=delta,
        confidence_interval=confidence_interval,
        effect_size=effect_size,
        threshold=threshold,
        gate_results=comparison.release.gates,
        decision=comparison.release.status.value,
        decision_reasons=decision_reasons,
        timestamp=comparison.created_at,
        baseline_cost_breakdown=baseline_cost_breakdown,
        candidate_cost_breakdown=candidate_cost_breakdown,
    )


def write_report(report: ExperimentReport, path: str | Path) -> None:
    """Writes the report as pretty-printed JSON. Callers own directory creation."""
    Path(path).write_text(report.model_dump_json(indent=2) + "\n")
