"""Tests for evalguard.quality.aggregate — pure rollup functions over Response/Score/
GuardrailResult lists. Every function's "empty input" / "nothing measurable" edge case
is tested explicitly, since that's exactly where a silent wrong-guess bug would hide.
"""

from __future__ import annotations

import pytest

from evalguard.enums import (
    ErrorClass,
    FinishReason,
    GuardrailOutcome,
    GuardrailStage,
)
from evalguard.models import GuardrailResult, Response, Score
from evalguard.quality.aggregate import (
    compute_cost_per_query,
    compute_failure_rate,
    compute_guardrail_rate,
    compute_latency_percentiles,
    compute_pass_rate,
    compute_token_usage,
    percentile,
)


def _response(**overrides) -> Response:
    defaults = dict(run_id=1, case_id=1, finish_reason=FinishReason.STOP)
    defaults.update(overrides)
    return Response(**defaults)


def _guardrail_result(outcome: GuardrailOutcome, **overrides) -> GuardrailResult:
    defaults = dict(
        response_id=1,
        stage=GuardrailStage.OUTPUT,
        triggered=True,
        expected_block=True,
        outcome=outcome,
    )
    defaults.update(overrides)
    return GuardrailResult(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# percentile
# ─────────────────────────────────────────────────────────────────────────────


def test_percentile_of_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


def test_percentile_of_single_value() -> None:
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 0) == 42.0
    assert percentile([42.0], 100) == 42.0


def test_percentile_p0_is_the_minimum() -> None:
    assert percentile([3.0, 1.0, 2.0], 0) == 1.0


def test_percentile_p100_is_the_maximum() -> None:
    assert percentile([3.0, 1.0, 2.0], 100) == 3.0


def test_percentile_p50_of_odd_length_is_the_middle_value() -> None:
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_percentile_interpolates_between_ranks() -> None:
    # [1,2,3,4]: rank = (4-1)*0.5 = 1.5 -> interpolate between index 1 (2.0) and 2 (3.0)
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_is_insensitive_to_input_order() -> None:
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 95) == percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95)


def test_percentile_rejects_out_of_range_p() -> None:
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], -1)
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101)


# ─────────────────────────────────────────────────────────────────────────────
# compute_failure_rate
# ─────────────────────────────────────────────────────────────────────────────


def test_failure_rate_of_empty_list_is_zero() -> None:
    assert compute_failure_rate([]) == 0.0


def test_failure_rate_with_no_failures() -> None:
    responses = [_response(error_class=None) for _ in range(4)]
    assert compute_failure_rate(responses) == 0.0


def test_failure_rate_with_some_failures() -> None:
    responses = [_response(error_class=None)] * 3 + [_response(error_class=ErrorClass.PERMANENT)]
    assert compute_failure_rate(responses) == pytest.approx(0.25)


def test_failure_rate_all_failed() -> None:
    responses = [_response(error_class=ErrorClass.TRANSIENT) for _ in range(2)]
    assert compute_failure_rate(responses) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_latency_percentiles
# ─────────────────────────────────────────────────────────────────────────────


def test_latency_percentiles_none_when_no_latency_data() -> None:
    responses = [_response(latency_ms=None, error_class=ErrorClass.PERMANENT)]
    assert compute_latency_percentiles(responses) is None


def test_latency_percentiles_none_for_empty_list() -> None:
    assert compute_latency_percentiles([]) is None


def test_latency_percentiles_computed_over_available_values() -> None:
    responses = [_response(latency_ms=ms) for ms in (100, 200, 300, 400, 500)]
    result = compute_latency_percentiles(responses)
    assert result["p50"] == 300
    assert result["p95"] == pytest.approx(480.0)


def test_latency_percentiles_excludes_failed_responses_without_latency() -> None:
    responses = [
        _response(latency_ms=100),
        _response(latency_ms=200),
        _response(latency_ms=None, error_class=ErrorClass.PERMANENT),
    ]
    result = compute_latency_percentiles(responses)
    assert result["p50"] == 150.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_token_usage
# ─────────────────────────────────────────────────────────────────────────────


def test_token_usage_of_empty_list_is_zeros() -> None:
    usage = compute_token_usage([])
    assert usage == {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "avg_input_tokens": 0.0,
        "avg_output_tokens": 0.0,
    }


def test_token_usage_totals_and_averages() -> None:
    responses = [
        _response(input_tokens=100, output_tokens=10),
        _response(input_tokens=200, output_tokens=20),
    ]
    usage = compute_token_usage(responses)
    assert usage["total_input_tokens"] == 300
    assert usage["total_output_tokens"] == 30
    assert usage["avg_input_tokens"] == 150.0
    assert usage["avg_output_tokens"] == 15.0


def test_token_usage_ignores_responses_without_token_counts() -> None:
    responses = [
        _response(input_tokens=100, output_tokens=10),
        _response(input_tokens=None, output_tokens=None),
    ]
    usage = compute_token_usage(responses)
    assert usage["total_input_tokens"] == 100
    assert usage["avg_input_tokens"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_cost_per_query — the "never guess" edge cases matter most here
# ─────────────────────────────────────────────────────────────────────────────


def test_cost_per_query_none_for_empty_list() -> None:
    assert compute_cost_per_query([]) is None


def test_cost_per_query_none_when_all_responses_failed() -> None:
    responses = [_response(error_class=ErrorClass.PERMANENT, cost_usd=None)]
    assert compute_cost_per_query(responses) is None


def test_cost_per_query_none_when_pricing_unverified() -> None:
    """A successful response with cost_usd=None means pricing wasn't configured —
    the aggregate must not silently treat that as $0."""
    responses = [_response(error_class=None, cost_usd=None)]
    assert compute_cost_per_query(responses) is None


def test_cost_per_query_averages_only_successful_responses() -> None:
    responses = [
        _response(error_class=None, cost_usd=0.01),
        _response(error_class=None, cost_usd=0.03),
        _response(error_class=ErrorClass.PERMANENT, cost_usd=None),  # excluded
    ]
    assert compute_cost_per_query(responses) == pytest.approx(0.02)


def test_cost_per_query_handles_verified_zero_cost() -> None:
    """A verified $0 (e.g. FakeClient) is a real answer, not 'unverified'."""
    responses = [
        _response(error_class=None, cost_usd=0.0),
        _response(error_class=None, cost_usd=0.0),
    ]
    assert compute_cost_per_query(responses) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_pass_rate
# ─────────────────────────────────────────────────────────────────────────────


def _score(value: float) -> Score:
    from evalguard.enums import MetricName, ScoreMethod

    return Score(
        response_id=1,
        metric=MetricName.INSTRUCTION_FOLLOWING,
        method=ScoreMethod.DETERMINISTIC,
        value=value,
    )


def test_pass_rate_none_for_empty_list() -> None:
    assert compute_pass_rate([]) is None


def test_pass_rate_all_passing() -> None:
    assert compute_pass_rate([_score(1.0), _score(1.0)]) == 1.0


def test_pass_rate_mixed() -> None:
    assert compute_pass_rate([_score(1.0), _score(0.0), _score(1.0), _score(0.0)]) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# compute_guardrail_rate
# ─────────────────────────────────────────────────────────────────────────────


def test_guardrail_rate_none_for_empty_list() -> None:
    assert compute_guardrail_rate([], numerator_outcome=GuardrailOutcome.TP) is None


def test_guardrail_rate_block_rate_all_resisted() -> None:
    results = [_guardrail_result(GuardrailOutcome.TP), _guardrail_result(GuardrailOutcome.TP)]
    assert compute_guardrail_rate(results, numerator_outcome=GuardrailOutcome.TP) == 1.0


def test_guardrail_rate_block_rate_mixed() -> None:
    results = [
        _guardrail_result(GuardrailOutcome.TP),
        _guardrail_result(GuardrailOutcome.FN),
        _guardrail_result(GuardrailOutcome.TP),
        _guardrail_result(GuardrailOutcome.FN),
    ]
    assert compute_guardrail_rate(results, numerator_outcome=GuardrailOutcome.TP) == 0.5


def test_guardrail_rate_benign_false_positive_rate() -> None:
    results = [
        _guardrail_result(GuardrailOutcome.TN),
        _guardrail_result(GuardrailOutcome.FP),
        _guardrail_result(GuardrailOutcome.TN),
        _guardrail_result(GuardrailOutcome.TN),
    ]
    assert compute_guardrail_rate(results, numerator_outcome=GuardrailOutcome.FP) == 0.25
