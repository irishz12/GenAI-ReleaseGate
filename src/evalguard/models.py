"""Pydantic contracts for every row shape in the results store (docs/ARCHITECTURE.md §9).

Each model corresponds to one of the 7 SQLite tables, except `MetricDelta`, `GateResult`,
and `ReleaseDecision`, which are structured sub-objects stored as JSON inside `Comparison`
rather than their own tables (see §9's rationale for not normalizing those out further).

`id` is `Optional[int] = None` on every stored model: `None` means "not yet persisted",
set means "this is the SQLite rowid" once `db/store.py` has inserted it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evalguard.enums import (
    AttackType,
    CaseCategory,
    CaseSplit,
    ErrorClass,
    FinishReason,
    GateStatus,
    GuardrailMethod,
    GuardrailOutcome,
    GuardrailStage,
    MetricName,
    PromptRole,
    ReleaseStatus,
    RunStatus,
    ScoreMethod,
)


def _utcnow() -> datetime:
    """Timezone-aware replacement for the deprecated `datetime.utcnow()`."""
    return datetime.now(UTC)


class _Base(BaseModel):
    """Shared config for every contract in this module."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# prompts
# ─────────────────────────────────────────────────────────────────────────────


class Prompt(_Base):
    """A single named, versioned prompt template (production, candidate, or judge)."""

    id: int | None = None
    name: str
    version: str
    role: PromptRole
    template: str
    content_hash: str
    created_at: datetime = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# evaluation_cases
# ─────────────────────────────────────────────────────────────────────────────


class EvaluationCase(_Base):
    """A single row of the evaluation dataset.

    Fields are a superset across all four categories; most are `None` outside their
    relevant category (e.g. `expected_block` only applies to `category == GUARDRAIL`).
    """

    id: int | None = None
    external_id: str
    category: CaseCategory
    split: CaseSplit
    input: str
    context: str | None = None
    reference_answer: str | None = None
    constraints: dict[str, Any] | None = None
    expected_abstention: bool | None = None
    attack_type: AttackType | None = None
    canary: str | None = None
    expected_block: bool | None = None
    dataset_hash: str


# ─────────────────────────────────────────────────────────────────────────────
# runs
# ─────────────────────────────────────────────────────────────────────────────


class Run(_Base):
    """One prompt version executed over one dataset split under a frozen config."""

    id: int | None = None
    prompt_id: int
    model_id: str
    model_params: dict[str, Any] = Field(default_factory=dict)
    judge_model_id: str
    guardrail_config_hash: str
    code_sha: str
    config_hash: str
    status: RunStatus = RunStatus.RUNNING
    case_count: int | None = None
    failure_count: int = 0
    total_cost_usd: float | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None


# ─────────────────────────────────────────────────────────────────────────────
# responses
# ─────────────────────────────────────────────────────────────────────────────


class Response(_Base):
    """One generated completion for one case within one run. Immutable once written."""

    id: int | None = None
    run_id: int
    case_id: int
    output_text: str | None = None
    finish_reason: FinishReason
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    error_class: ErrorClass | None = None
    created_at: datetime = Field(default_factory=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# scores
# ─────────────────────────────────────────────────────────────────────────────


class Score(_Base):
    """One quality metric value for one response (deterministic check or judge call).

    The judge_* fields track the judge's own tokens/cost/latency, separate from
    `Response.cost_usd` (the generator's cost) — a metric derived from another
    metric's judge call (e.g. `hallucination` from `faithfulness`) leaves these unset
    rather than double-counting the same call's cost twice.
    """

    id: int | None = None
    response_id: int
    metric: MetricName
    method: ScoreMethod
    value: float
    rationale: str | None = None
    judge_input_tokens: int | None = None
    judge_output_tokens: int | None = None
    judge_cost_usd: float | None = None
    judge_latency_ms: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# guardrail_results
# ─────────────────────────────────────────────────────────────────────────────


class GuardrailResult(_Base):
    """One guardrail check outcome for one response, at a given stage.

    `method` defaults to DETERMINISTIC (Phase 4's post-hoc regex classifier) so
    existing call sites that predate Phase 6's live Bedrock guardrail keep working
    unchanged.
    """

    id: int | None = None
    response_id: int
    stage: GuardrailStage
    triggered: bool
    expected_block: bool
    outcome: GuardrailOutcome
    detail: str | None = None
    method: GuardrailMethod = GuardrailMethod.DETERMINISTIC
    # Only ever set for method=BEDROCK — a live ApplyGuardrail call measures both;
    # DETERMINISTIC's post-hoc regex scoring has neither a call latency nor a cost.
    latency_ms: int | None = None
    cost_usd: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# comparisons — plus its nested JSON payload shapes
# ─────────────────────────────────────────────────────────────────────────────


class MetricDelta(_Base):
    """Baseline-vs-candidate delta for a single metric, with an optional simple CI.

    `ci_low`/`ci_high` are populated only where a paired CI is computed (see
    docs/ARCHITECTURE.md §10) — not every metric gets one in V1.

    Phase 2 adds four more optional, additive statistics — all descriptive or
    multiple-comparison-adjusted *evidence*, never inputs to the release decision
    itself (policy/engine.py reads only `delta`/`ci_low`/`ci_high`, unchanged):
    `median_delta` (robust to a single outlier case, unlike the mean), `effect_size`
    (paired Cohen's d), `p_value` (bootstrap two-sided, H0: true mean diff is zero),
    and `holm_significant` (this metric's significance after Holm-Bonferroni
    correction across every metric tested in the same comparison). All four are
    `None` when not computed (e.g. percentile metrics, or fewer than 2 paired
    samples) — see regression/statistics.py.
    """

    metric: MetricName
    baseline_value: float
    candidate_value: float
    delta: float
    ci_low: float | None = None
    ci_high: float | None = None
    median_delta: float | None = None
    effect_size: float | None = None
    p_value: float | None = None
    holm_significant: bool | None = None


class GateResult(_Base):
    """Outcome of evaluating one release gate from config/policy.yaml against the deltas."""

    gate_name: str
    status: GateStatus
    threshold: float | None = None
    observed: float | None = None
    reason: str


class ReleaseDecision(_Base):
    """The release-policy engine's verdict: overall status plus the gates that produced it."""

    status: ReleaseStatus
    gates: list[GateResult] = Field(default_factory=list)
    summary: str | None = None


class Comparison(_Base):
    """A single V1-vs-V2 comparison: the paired deltas and the resulting release decision.

    Maps onto one `comparisons` row. `deltas` and `release.gates` are stored as the
    `deltas_json`/`gates_json` columns; `release.status` is stored as the plain-text
    `decision` column (see db/store.py for the mapping).
    """

    id: int | None = None
    baseline_run_id: int
    candidate_run_id: int
    n_paired: int
    deltas: list[MetricDelta] = Field(default_factory=list)
    release: ReleaseDecision
    created_at: datetime = Field(default_factory=_utcnow)
