"""Release-policy engine: turns metric deltas into GO/REVIEW/HOLD/INVALID per
config/policy.yaml's declarative gates. Pure — no DB, no I/O; every test constructs
its own GateConfig/MetricDelta directly.

The bulk of this file is unit tests for evaluate_gate/decide_release/evaluate_policy
in isolation. The final section, "Decision Engine Validation Scenarios," runs the
same engine end to end against the REAL config/policy.yaml with deterministic
synthetic fixtures — see that section's header comment for why these are fixtures,
not real experiment results.
"""

from __future__ import annotations

import pytest

from evalguard.config import GateConfig, PolicyConfig
from evalguard.enums import GateStatus, ReleaseStatus
from evalguard.models import MetricDelta
from evalguard.policy.engine import (
    UnknownGateMetricError,
    decide_release,
    evaluate_gate,
    evaluate_policy,
)


def _delta(
    metric="answer_correctness", baseline=0.8, candidate=0.8, delta=None, ci=None
) -> MetricDelta:
    d = candidate - baseline if delta is None else delta
    ci_low, ci_high = ci if ci is not None else (None, None)
    return MetricDelta(
        metric=metric,
        baseline_value=baseline,
        candidate_value=candidate,
        delta=d,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _gate(**overrides) -> GateConfig:
    defaults = dict(metric="answer_correctness", direction="higher_is_better")
    defaults.update(overrides)
    return GateConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_gate — raw-delta thresholds, higher_is_better / lower_is_better
# ─────────────────────────────────────────────────────────────────────────────


def test_improvement_goes_on_a_higher_is_better_gate() -> None:
    gate = _gate(review_if_delta_worse_than=0.0, hold_if_delta_worse_than=-0.05)
    delta = _delta(baseline=0.80, candidate=0.85, delta=0.05)
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.GO


def test_regression_past_hold_threshold_holds() -> None:
    gate = _gate(review_if_delta_worse_than=0.0, hold_if_delta_worse_than=-0.05)
    delta = _delta(baseline=0.80, candidate=0.70, delta=-0.10)
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.HOLD
    assert "delta" in result.reason and "-0.1" in result.reason


def test_regression_past_review_but_not_hold_threshold_reviews() -> None:
    gate = _gate(review_if_delta_worse_than=0.0, hold_if_delta_worse_than=-0.05)
    delta = _delta(baseline=0.80, candidate=0.78, delta=-0.02)
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.REVIEW


def test_lower_is_better_regression_holds_when_delta_increases() -> None:
    gate = _gate(metric="hallucination", direction="lower_is_better", hold_if_delta_worse_than=0.02)
    delta = _delta(metric="hallucination", baseline=0.01, candidate=0.05, delta=0.04)
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.HOLD


def test_lower_is_better_improvement_goes() -> None:
    gate = _gate(metric="hallucination", direction="lower_is_better", hold_if_delta_worse_than=0.02)
    delta = _delta(metric="hallucination", baseline=0.05, candidate=0.01, delta=-0.04)
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.GO


# ─────────────────────────────────────────────────────────────────────────────
# boundary values — exact threshold equality
# ─────────────────────────────────────────────────────────────────────────────


def test_delta_exactly_at_hold_threshold_does_not_hold_but_still_reviews() -> None:
    """Strict-worsening semantics (Phase 8.1 fix): a delta exactly equal to a
    threshold is not *worse than* that threshold, so it must not trigger HOLD —
    but -0.05 is still worse than the 0.0 review threshold, so REVIEW still fires."""
    gate = _gate(review_if_delta_worse_than=0.0, hold_if_delta_worse_than=-0.05)
    delta = _delta(baseline=0.80, candidate=0.75, delta=-0.05)
    assert evaluate_gate(gate, delta).status is GateStatus.REVIEW


def test_delta_strictly_worse_than_hold_threshold_holds() -> None:
    gate = _gate(review_if_delta_worse_than=0.0, hold_if_delta_worse_than=-0.05)
    delta = _delta(baseline=0.80, candidate=0.74, delta=-0.06)
    assert evaluate_gate(gate, delta).status is GateStatus.HOLD


def test_delta_just_short_of_hold_threshold_only_reviews() -> None:
    gate = _gate(review_if_delta_worse_than=0.0, hold_if_delta_worse_than=-0.05)
    delta = _delta(baseline=0.80, candidate=0.7501, delta=-0.0499)
    assert evaluate_gate(gate, delta).status is GateStatus.REVIEW


def test_delta_exactly_zero_does_not_trigger_hold_or_review() -> None:
    """Phase 8.1 fix: an unchanged metric (delta == 0) is never a regression, however
    a threshold is worded — this is the exact bug a real dev run surfaced (both
    prompt_injection_block_rate and sensitive_information_protection sat unchanged at
    a perfect 1.0 and still HELD, because the old `<=`/`>=` comparison treated
    "equal to the threshold" as "worse than the threshold")."""
    gate = _gate(review_if_delta_worse_than=0.0, hold_if_delta_worse_than=-0.05)
    delta = _delta(baseline=0.80, candidate=0.80, delta=0.0)
    assert evaluate_gate(gate, delta).status is GateStatus.GO


def test_delta_exactly_zero_does_not_trigger_even_with_a_zero_hold_threshold() -> None:
    """The specific real-world shape of the bug: a hard gate whose hold threshold
    is itself 0.0 (e.g. prompt_injection_block_rate in config/policy.yaml) must not
    HOLD on a perfectly unchanged value."""
    gate = _gate(hold_if_delta_worse_than=0.0)
    delta = _delta(baseline=1.0, candidate=1.0, delta=0.0)
    assert evaluate_gate(gate, delta).status is GateStatus.GO


def test_absolute_min_boundary_exactly_at_minimum_goes() -> None:
    gate = _gate(
        metric="prompt_injection_block_rate", direction="higher_is_better", absolute_min=0.95
    )
    delta = _delta(metric="prompt_injection_block_rate", baseline=0.95, candidate=0.95, delta=0.0)
    assert evaluate_gate(gate, delta).status is GateStatus.GO


def test_absolute_min_boundary_just_below_minimum_holds() -> None:
    gate = _gate(
        metric="prompt_injection_block_rate", direction="higher_is_better", absolute_min=0.95
    )
    delta = _delta(
        metric="prompt_injection_block_rate", baseline=0.96, candidate=0.9499, delta=-0.0101
    )
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.HOLD
    assert "minimum" in result.reason


def test_absolute_max_boundary_exactly_at_maximum_goes() -> None:
    gate = _gate(
        metric="benign_false_positive_rate", direction="lower_is_better", absolute_max=0.05
    )
    delta = _delta(metric="benign_false_positive_rate", baseline=0.05, candidate=0.05, delta=0.0)
    assert evaluate_gate(gate, delta).status is GateStatus.GO


def test_absolute_max_boundary_just_above_maximum_holds() -> None:
    gate = _gate(
        metric="benign_false_positive_rate", direction="lower_is_better", absolute_max=0.05
    )
    delta = _delta(
        metric="benign_false_positive_rate", baseline=0.03, candidate=0.0501, delta=0.0201
    )
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.HOLD
    assert "maximum" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# percentage thresholds, incl. the baseline==0 boundary
# ─────────────────────────────────────────────────────────────────────────────


def test_pct_threshold_triggers_review_for_a_soft_latency_gate() -> None:
    gate = _gate(
        metric="p95_latency", direction="lower_is_better", review_if_delta_worse_than_pct=20.0
    )
    delta = _delta(metric="p95_latency", baseline=1000.0, candidate=1250.0, delta=250.0)
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.REVIEW


def test_pct_threshold_not_triggered_below_the_percentage() -> None:
    gate = _gate(
        metric="p95_latency", direction="lower_is_better", review_if_delta_worse_than_pct=20.0
    )
    delta = _delta(metric="p95_latency", baseline=1000.0, candidate=1100.0, delta=100.0)
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.GO


def test_soft_gate_can_never_hold_no_matter_how_large_the_regression() -> None:
    """Structural guarantee: a gate with no hold_if_delta_worse_than(_pct)/absolute_*
    configured cannot produce HOLD, however large the delta."""
    gate = _gate(
        metric="cost_per_query", direction="lower_is_better", review_if_delta_worse_than_pct=15.0
    )
    delta = _delta(metric="cost_per_query", baseline=0.001, candidate=1.0, delta=0.999)
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.REVIEW


def test_zero_baseline_value_cannot_compute_percentage_delta() -> None:
    gate = _gate(
        metric="cost_per_query", direction="lower_is_better", review_if_delta_worse_than_pct=15.0
    )
    delta = _delta(metric="cost_per_query", baseline=0.0, candidate=0.01, delta=0.01)
    result = evaluate_gate(gate, delta)
    assert result.status is GateStatus.REVIEW
    assert "baseline" in result.reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# missing metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_delta_reviews_rather_than_silently_going_or_holding() -> None:
    gate = _gate(hold_if_delta_worse_than=-0.05)
    result = evaluate_gate(gate, None)
    assert result.status is GateStatus.REVIEW
    assert "no" in result.reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# decide_release — aggregate gate results into one overall status
# ─────────────────────────────────────────────────────────────────────────────


def test_any_hold_makes_the_overall_release_hold() -> None:
    gates = [
        evaluate_gate(_gate(hold_if_delta_worse_than=-0.05), _delta(delta=-0.10)),
        evaluate_gate(
            _gate(
                metric="cost_per_query",
                direction="lower_is_better",
                review_if_delta_worse_than_pct=15.0,
            ),
            _delta(delta=0.0),
        ),
    ]
    decision = decide_release(gates)
    assert decision.status is ReleaseStatus.HOLD
    assert "answer_correctness" in decision.summary


def test_only_review_makes_the_overall_release_review() -> None:
    gates = [
        evaluate_gate(
            _gate(review_if_delta_worse_than=0.0, hold_if_delta_worse_than=-0.05),
            _delta(delta=-0.02),
        ),
    ]
    decision = decide_release(gates)
    assert decision.status is ReleaseStatus.REVIEW


def test_all_go_makes_the_overall_release_go() -> None:
    gates = [evaluate_gate(_gate(hold_if_delta_worse_than=-0.05), _delta(delta=0.02))]
    decision = decide_release(gates)
    assert decision.status is ReleaseStatus.GO


def test_no_gates_configured_goes_by_default() -> None:
    decision = decide_release([])
    assert decision.status is ReleaseStatus.GO


def test_decide_release_is_deterministic() -> None:
    gates = [evaluate_gate(_gate(hold_if_delta_worse_than=-0.05), _delta(delta=-0.10))]
    assert decide_release(gates) == decide_release(gates)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_policy — the INVALID checks
# ─────────────────────────────────────────────────────────────────────────────


def _policy(**overrides) -> PolicyConfig:
    defaults = dict(gates=[_gate(hold_if_delta_worse_than=-0.05)], invalid_if_failure_rate_over=0.2)
    defaults.update(overrides)
    return PolicyConfig(**defaults)


def test_zero_paired_cases_is_invalid() -> None:
    decision = evaluate_policy([], _policy(), n_paired=0, paired_failure_rate=0.0)
    assert decision.status is ReleaseStatus.INVALID
    assert decision.gates == []


def test_failure_rate_over_threshold_is_invalid() -> None:
    decision = evaluate_policy(
        [_delta(delta=0.02)], _policy(), n_paired=10, paired_failure_rate=0.3
    )
    assert decision.status is ReleaseStatus.INVALID


def test_failure_rate_at_or_under_threshold_is_not_invalid() -> None:
    decision = evaluate_policy(
        [_delta(delta=0.02)], _policy(), n_paired=10, paired_failure_rate=0.2
    )
    assert decision.status is not ReleaseStatus.INVALID


def test_unknown_gate_metric_raises_rather_than_silently_skipping() -> None:
    policy = _policy(gates=[_gate(metric="not_a_real_metric", hold_if_delta_worse_than=-0.05)])
    with pytest.raises(UnknownGateMetricError):
        evaluate_policy([], policy, n_paired=5, paired_failure_rate=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Decision Engine Validation Scenarios
#
# Everything below is a DETERMINISTIC GATE-VALIDATION FIXTURE, not a real prompt
# experiment result. Each scenario runs evaluate_policy() against the REAL,
# production config/policy.yaml (loaded via load_policy_config(), the exact same
# config compare_runs() uses) with hand-constructed, clearly synthetic deltas — so
# these tests prove the release-policy engine reaches and correctly aggregates
# every one of the four decision states under the actual thresholds the system
# runs in production, without depending on which real V1/V2/V3 comparison happens
# to land where. Real experiment results live in results/reports/*.json (Phase 3)
# and comparisons 1/2/5 in artifacts/dev_eval.db — never reproduced or referenced
# here. In particular: as of Phase 1, no real comparison has ever produced GO (V1
# vs V2 = HOLD, V1 vs V3 dev/holdout = REVIEW, V2 vs V3 = REVIEW) — the GO scenario
# below exists precisely to prove the engine supports GO without fabricating one.
# ─────────────────────────────────────────────────────────────────────────────


def _real_policy() -> PolicyConfig:
    from evalguard.config import load_policy_config

    return load_policy_config()


def _synthetic_deltas(**overrides: float) -> list[MetricDelta]:
    """One delta per metric gated in the real policy, all deep inside every
    threshold by default — `overrides` replaces individual metrics' delta (and,
    for benign_false_positive_rate, its candidate_value, since that gate reads an
    absolute bound rather than a delta)."""
    safe = {
        "answer_correctness": _delta(metric="answer_correctness", baseline=0.80, candidate=0.85),
        "faithfulness": _delta(metric="faithfulness", baseline=0.80, candidate=0.85),
        "hallucination": _delta(metric="hallucination", baseline=0.05, candidate=0.04),
        "prompt_injection_block_rate": _delta(
            metric="prompt_injection_block_rate", baseline=1.0, candidate=1.0, delta=0.0
        ),
        "sensitive_information_protection": _delta(
            metric="sensitive_information_protection", baseline=1.0, candidate=1.0, delta=0.0
        ),
        "benign_false_positive_rate": _delta(
            metric="benign_false_positive_rate", baseline=0.05, candidate=0.05, delta=0.0
        ),
        "p95_latency": _delta(metric="p95_latency", baseline=1000.0, candidate=1000.0, delta=0.0),
        "cost_per_query": _delta(
            metric="cost_per_query", baseline=0.001, candidate=0.001, delta=0.0
        ),
    }
    for metric, delta in overrides.items():
        base = safe[metric]
        safe[metric] = _delta(
            metric=metric,
            baseline=base.baseline_value,
            candidate=base.baseline_value + delta,
            delta=delta,
        )
    return list(safe.values())


def test_scenario_go_deterministic_fixture_every_gate_within_threshold() -> None:
    """GO: every gate comfortably clears its real policy.yaml threshold. This is
    the clearly-labeled synthetic GO fixture — no real comparison has produced GO
    (see module comment above), so this scenario proves GO is reachable without
    fabricating or forcing a real result to look like one."""
    decision = evaluate_policy(
        _synthetic_deltas(), _real_policy(), n_paired=120, paired_failure_rate=0.0
    )
    assert decision.status is ReleaseStatus.GO
    assert all(g.status is GateStatus.GO for g in decision.gates)


def test_scenario_hold_a_single_hard_gate_violation() -> None:
    """HOLD: faithfulness alone crosses its -0.08 hold threshold; every other gate
    is fine. Overall status must still be HOLD (worst-gate-wins)."""
    decision = evaluate_policy(
        _synthetic_deltas(faithfulness=-0.15), _real_policy(), n_paired=120, paired_failure_rate=0.0
    )
    assert decision.status is ReleaseStatus.HOLD
    faithfulness_gate = next(g for g in decision.gates if g.gate_name == "faithfulness")
    assert faithfulness_gate.status is GateStatus.HOLD
    other_gates = [g for g in decision.gates if g.gate_name != "faithfulness"]
    assert all(g.status is GateStatus.GO for g in other_gates)


def test_scenario_review_a_single_soft_gate_flagged() -> None:
    """REVIEW: cost_per_query rises 20% (past its 15% review threshold), a soft
    gate that structurally cannot HOLD. Overall status is REVIEW, not HOLD."""
    decision = evaluate_policy(
        _synthetic_deltas(cost_per_query=0.0002),  # +20% of the 0.001 baseline
        _real_policy(),
        n_paired=120,
        paired_failure_rate=0.0,
    )
    assert decision.status is ReleaseStatus.REVIEW
    cost_gate = next(g for g in decision.gates if g.gate_name == "cost_per_query")
    assert cost_gate.status is GateStatus.REVIEW


def test_scenario_invalid_failure_rate_overrides_every_gate() -> None:
    """INVALID: the reliability check runs before any gate is even evaluated —
    a comparison too unreliable to trust is INVALID regardless of how good its
    (synthetic, otherwise all-GO) deltas look."""
    decision = evaluate_policy(
        _synthetic_deltas(), _real_policy(), n_paired=10, paired_failure_rate=0.5
    )
    assert decision.status is ReleaseStatus.INVALID
    assert decision.gates == []


def test_scenario_multiple_hard_gates_hold_simultaneously() -> None:
    """Precedence scenario: two independent hard gates (faithfulness and
    benign_false_positive_rate) violate their thresholds at the same time. Overall
    status is still exactly HOLD (not some more-severe fifth state), and both
    violations are named in the summary — worst-gate-wins aggregates, it doesn't
    pick just one and hide the other."""
    decision = evaluate_policy(
        _synthetic_deltas(faithfulness=-0.15, benign_false_positive_rate=0.20),
        _real_policy(),
        n_paired=120,
        paired_failure_rate=0.0,
    )
    assert decision.status is ReleaseStatus.HOLD
    held = {g.gate_name for g in decision.gates if g.status is GateStatus.HOLD}
    assert held == {"faithfulness", "benign_false_positive_rate"}
    assert "faithfulness" in decision.summary
    assert "benign_false_positive_rate" in decision.summary


def test_scenario_hold_and_review_together_hold_still_wins() -> None:
    """Precedence scenario: one gate HOLDs (faithfulness) while a different,
    independent gate only REVIEWs (cost_per_query) in the same comparison. Overall
    status must be HOLD — REVIEW never outranks a HOLD, however many gates only
    review."""
    decision = evaluate_policy(
        _synthetic_deltas(faithfulness=-0.15, cost_per_query=0.0002),
        _real_policy(),
        n_paired=120,
        paired_failure_rate=0.0,
    )
    assert decision.status is ReleaseStatus.HOLD
    statuses = {g.gate_name: g.status for g in decision.gates}
    assert statuses["faithfulness"] is GateStatus.HOLD
    assert statuses["cost_per_query"] is GateStatus.REVIEW


def test_scenario_review_outranks_go_when_both_present() -> None:
    """Precedence scenario: cost_per_query REVIEWs while every other gate GOes.
    Overall status is REVIEW, not GO — a single review-tier gate is enough to keep
    the whole release from reading as GO."""
    decision = evaluate_policy(
        _synthetic_deltas(cost_per_query=0.0002),
        _real_policy(),
        n_paired=120,
        paired_failure_rate=0.0,
    )
    assert decision.status is ReleaseStatus.REVIEW
    assert sum(1 for g in decision.gates if g.status is GateStatus.HOLD) == 0
    assert sum(1 for g in decision.gates if g.status is GateStatus.REVIEW) == 1
