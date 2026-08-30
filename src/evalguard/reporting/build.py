"""Builds an ExperimentReport from a real, already-computed Comparison — pure
reshaping, no new computation, no fabricated values. Every field traces back to
either the Comparison itself (deltas, gate results, decision, n_paired,
created_at) or the PolicyConfig that produced its gates (config/policy.yaml,
loaded the same way regression/compare.py loads it for compare_runs).
"""

from __future__ import annotations

from pathlib import Path

from evalguard.config import GateConfig, PolicyConfig
from evalguard.models import Comparison

from .models import ExperimentReport

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


def build_experiment_report(
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
    )


def write_report(report: ExperimentReport, path: str | Path) -> None:
    """Writes the report as pretty-printed JSON. Callers own directory creation."""
    Path(path).write_text(report.model_dump_json(indent=2) + "\n")
