"""Release-policy engine (Stage C, second half): turns per-metric deltas into
GO/REVIEW/HOLD/INVALID per config/policy.yaml's declarative gates.

Pure — no DB access, no network. `regression.compare` computes the deltas this reads;
this module only decides what they mean. Hard vs. soft is structural rather than a
separate flag: a gate can only ever HOLD if it configures a hold threshold or an
absolute bound (see `evaluate_gate`) — cost/latency gates in policy.yaml configure
neither, so HOLD is impossible for them by construction, not by convention.
"""

from __future__ import annotations

from evalguard.config import GateConfig, PolicyConfig
from evalguard.enums import GateStatus, MetricName, ReleaseStatus
from evalguard.models import GateResult, MetricDelta, ReleaseDecision


class UnknownGateMetricError(ValueError):
    """A policy.yaml gate references a metric name that isn't a real `MetricName`
    value — e.g. the "hallucination_rate" vs. "hallucination" mismatch this was
    written to catch. Raised rather than silently never matching any delta."""


def _resolve_metric(name: str) -> MetricName:
    try:
        return MetricName(name)
    except ValueError as exc:
        raise UnknownGateMetricError(
            f"policy.yaml gate references unknown metric '{name}'"
        ) from exc


def _direction_triggered(direction: str, observed: float, threshold: float) -> bool:
    """Strict worsening (Phase 8.1 fix): `observed` must be worse than `threshold`,
    not merely at or beyond it — equal to the threshold is not "worse than" it.

    higher_is_better: worse means observed dropped strictly below threshold.
    lower_is_better: worse means observed rose strictly above threshold.

    This matters most at threshold=0.0: an unchanged metric (delta == 0) must never
    read as a regression, however a gate's threshold is worded. A real dev run
    (Phase 8) hit exactly this — prompt_injection_block_rate and
    sensitive_information_protection sat unchanged at a perfect 1.0 and still HELD,
    because the old `<=`/`>=` comparison treated "equal to the threshold" the same as
    "worse than it."
    """
    return observed < threshold if direction == "higher_is_better" else observed > threshold


def _pct_delta(delta: float, baseline_value: float) -> float | None:
    """`None` when baseline is 0 — a percentage change from zero is undefined, never
    guessed as 0% or infinite."""
    if baseline_value == 0:
        return None
    return (delta / baseline_value) * 100


def evaluate_gate(gate: GateConfig, delta: MetricDelta | None) -> GateResult:
    """One gate's verdict against one metric's delta.

    Order of checks doesn't affect the outcome for a single trigger, but when several
    conditions trigger at once, every hold-tier reason is reported (joined) and the
    first one's threshold/observed pair is surfaced as the gate's headline numbers.
    """
    if delta is None:
        return GateResult(
            gate_name=gate.metric,
            status=GateStatus.REVIEW,
            reason=(
                f"no paired data available for metric '{gate.metric}' in this "
                "comparison — gate not evaluated"
            ),
        )

    # (tier, reason, threshold, observed)
    triggers: list[tuple[str, str, float, float]] = []

    if gate.absolute_min is not None and delta.candidate_value < gate.absolute_min:
        triggers.append(
            (
                "hold",
                f"candidate value {delta.candidate_value:.4f} is below the required "
                f"minimum {gate.absolute_min}",
                gate.absolute_min,
                delta.candidate_value,
            )
        )
    if gate.absolute_max is not None and delta.candidate_value > gate.absolute_max:
        triggers.append(
            (
                "hold",
                f"candidate value {delta.candidate_value:.4f} exceeds the allowed "
                f"maximum {gate.absolute_max}",
                gate.absolute_max,
                delta.candidate_value,
            )
        )

    if gate.hold_if_delta_worse_than is not None and _direction_triggered(
        gate.direction, delta.delta, gate.hold_if_delta_worse_than
    ):
        triggers.append(
            (
                "hold",
                f"delta {delta.delta:+.4f} crossed the hold threshold "
                f"{gate.hold_if_delta_worse_than:+.4f}",
                gate.hold_if_delta_worse_than,
                delta.delta,
            )
        )
    elif gate.review_if_delta_worse_than is not None and _direction_triggered(
        gate.direction, delta.delta, gate.review_if_delta_worse_than
    ):
        triggers.append(
            (
                "review",
                f"delta {delta.delta:+.4f} crossed the review threshold "
                f"{gate.review_if_delta_worse_than:+.4f}",
                gate.review_if_delta_worse_than,
                delta.delta,
            )
        )

    wants_pct = (
        gate.hold_if_delta_worse_than_pct is not None
        or gate.review_if_delta_worse_than_pct is not None
    )
    if wants_pct:
        pct_delta = _pct_delta(delta.delta, delta.baseline_value)
        if pct_delta is None:
            triggers.append(
                (
                    "review",
                    f"baseline value for '{gate.metric}' is 0 — cannot compute a percentage delta",
                    0.0,
                    delta.delta,
                )
            )
        elif gate.hold_if_delta_worse_than_pct is not None and _direction_triggered(
            gate.direction, pct_delta, gate.hold_if_delta_worse_than_pct
        ):
            triggers.append(
                (
                    "hold",
                    f"delta {pct_delta:+.2f}% crossed the hold threshold "
                    f"{gate.hold_if_delta_worse_than_pct:+.2f}%",
                    gate.hold_if_delta_worse_than_pct,
                    pct_delta,
                )
            )
        elif gate.review_if_delta_worse_than_pct is not None and _direction_triggered(
            gate.direction, pct_delta, gate.review_if_delta_worse_than_pct
        ):
            triggers.append(
                (
                    "review",
                    f"delta {pct_delta:+.2f}% crossed the review threshold "
                    f"{gate.review_if_delta_worse_than_pct:+.2f}%",
                    gate.review_if_delta_worse_than_pct,
                    pct_delta,
                )
            )

    hold_triggers = [t for t in triggers if t[0] == "hold"]
    review_triggers = [t for t in triggers if t[0] == "review"]

    if hold_triggers:
        _, _, threshold, observed = hold_triggers[0]
        return GateResult(
            gate_name=gate.metric,
            status=GateStatus.HOLD,
            threshold=threshold,
            observed=observed,
            reason="; ".join(t[1] for t in hold_triggers),
        )
    if review_triggers:
        _, _, threshold, observed = review_triggers[0]
        return GateResult(
            gate_name=gate.metric,
            status=GateStatus.REVIEW,
            threshold=threshold,
            observed=observed,
            reason="; ".join(t[1] for t in review_triggers),
        )
    return GateResult(
        gate_name=gate.metric,
        status=GateStatus.GO,
        threshold=None,
        observed=delta.delta,
        reason=f"within configured thresholds (delta={delta.delta:+.4f})",
    )


def decide_release(gate_results: list[GateResult]) -> ReleaseDecision:
    """Overall status is the worst individual gate: any HOLD -> HOLD, else any REVIEW ->
    REVIEW, else GO. No gates configured is GO vacuously — there is nothing to hold on."""
    if any(g.status is GateStatus.HOLD for g in gate_results):
        status = ReleaseStatus.HOLD
    elif any(g.status is GateStatus.REVIEW for g in gate_results):
        status = ReleaseStatus.REVIEW
    else:
        status = ReleaseStatus.GO

    held = [g.gate_name for g in gate_results if g.status is GateStatus.HOLD]
    reviewed = [g.gate_name for g in gate_results if g.status is GateStatus.REVIEW]

    parts = [status.value]
    if held:
        parts.append(f"hold: {', '.join(held)}")
    if reviewed:
        parts.append(f"review: {', '.join(reviewed)}")
    if not held and not reviewed:
        parts.append(
            f"all {len(gate_results)} gate(s) passed" if gate_results else "no gates configured"
        )

    return ReleaseDecision(status=status, gates=gate_results, summary=" — ".join(parts))


def evaluate_policy(
    deltas: list[MetricDelta],
    policy: PolicyConfig,
    *,
    n_paired: int,
    paired_failure_rate: float,
) -> ReleaseDecision:
    """The top-level Stage C decision: INVALID first (is this comparison even
    trustworthy?), then the declarative gates from config/policy.yaml."""
    if n_paired == 0:
        return ReleaseDecision(
            status=ReleaseStatus.INVALID,
            gates=[],
            summary="INVALID — no shared case_ids between the baseline and candidate runs",
        )

    if paired_failure_rate > policy.invalid_if_failure_rate_over:
        return ReleaseDecision(
            status=ReleaseStatus.INVALID,
            gates=[],
            summary=(
                f"INVALID — {paired_failure_rate:.0%} of paired cases had a failed "
                f"response on at least one side, exceeding the "
                f"{policy.invalid_if_failure_rate_over:.0%} reliability threshold"
            ),
        )

    deltas_by_metric = {d.metric: d for d in deltas}
    gate_results = [
        evaluate_gate(gate, deltas_by_metric.get(_resolve_metric(gate.metric)))
        for gate in policy.gates
    ]
    return decide_release(gate_results)
