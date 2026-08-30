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


class CostBreakdown(BaseModel):
    """One run's real cost, split by which independently-billed service incurred
    it. `cost_per_query` (the release-gate metric in `delta`/`baseline_values`/
    `candidate_values` above) is generator cost only, unchanged by this model's
    existence — this is purely additional visibility into the other two real costs
    (guardrail, judge) that were always tracked in the database but never exposed
    through the reporting layer.

    Each field is `None` — never a fabricated 0.0 — when that service has no priced
    data for this run (e.g. a DETERMINISTIC-method guardrail check has no cost_usd
    at all), mirroring `compute_cost_per_query`'s existing "unverified, not zero"
    convention. `total_evaluation_cost` is itself `None` unless all three components
    are present — the same "never a silently-partial aggregate" discipline, so a
    reader never mistakes a partial sum (e.g. guardrail + judge, generator missing)
    for the genuine total.
    """

    model_config = ConfigDict(extra="forbid")

    generator_cost: float | None
    guardrail_cost: float | None
    judge_cost: float | None
    total_evaluation_cost: float | None


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
    baseline_cost_breakdown: CostBreakdown
    candidate_cost_breakdown: CostBreakdown
