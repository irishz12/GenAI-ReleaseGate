"""Pydantic contract for one structured, JSON-serializable experiment report.

An `ExperimentReport` is a flattened, frontend-friendly view of one already-computed
`Comparison` (regression/compare.py) plus the `PolicyConfig` (config/policy.yaml)
that decided its gates — nothing here computes a new statistic or a new decision.
Every field is sourced from real, already-persisted data (see build.py); this model
is the on-the-wire shape a future Vercel/Next.js frontend consumes, and the shape
test_reporting.py's tests hold every field to.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from evalguard.models import GateResult


class ExperimentReport(BaseModel):
    """One baseline-vs-candidate comparison, reshaped for external consumption."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    baseline_version: str
    candidate_version: str
    dataset: str
    valid_cases: int
    metrics: list[str]
    baseline_values: dict[str, float]
    candidate_values: dict[str, float]
    delta: dict[str, float]
    confidence_interval: dict[str, list[float] | None]
    effect_size: dict[str, float | None]
    # Per-metric configured thresholds straight from config/policy.yaml's GateConfig
    # (whichever of hold_if_delta_worse_than(_pct)/review_if_delta_worse_than(_pct)/
    # absolute_min/absolute_max are actually set) — not the gate's *triggered*
    # threshold (that's already in gate_results), but every threshold that governs
    # the metric regardless of whether it fired this time.
    threshold: dict[str, dict[str, float]]
    gate_results: list[GateResult] = Field(default_factory=list)
    decision: str
    decision_reasons: list[str]
    timestamp: datetime
