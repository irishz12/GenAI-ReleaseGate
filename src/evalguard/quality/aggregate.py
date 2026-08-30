"""Rolls per-response records (Response, Score, GuardrailResult) into single-run
aggregate numbers — docs/ARCHITECTURE.md §5's metric list, minus anything needing a
judge. Every function here is pure: a list in, a number (or None) out, no DB access.

None-handling follows one rule throughout: an aggregate is `None` only when it is
genuinely unmeasurable (no eligible rows), never as a stand-in for "the average of
whatever happened to be priced/present." `compute_cost_per_query` is the one place
this matters most — see its docstring.
"""

from __future__ import annotations

import math

from evalguard.enums import GuardrailOutcome
from evalguard.models import GuardrailResult, Response, Score


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (same convention as numpy's default 'linear').

    `p` is 0-100. Raises on an empty list — there is no sensible percentile of
    nothing, and returning e.g. 0.0 would silently misrepresent "no data" as "zero
    latency".
    """
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (p / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def compute_failure_rate(responses: list[Response]) -> float:
    """Fraction of responses that recorded an error. 0.0 for an empty list — an empty
    run has no failures to report, which is different from "unmeasurable"."""
    if not responses:
        return 0.0
    failed = sum(1 for r in responses if r.error_class is not None)
    return failed / len(responses)


def compute_latency_percentiles(responses: list[Response]) -> dict[str, float] | None:
    """{"p50": ..., "p95": ...} over successful responses' `latency_ms`. `None` if no
    response has a latency to measure (e.g. every case failed before completing)."""
    values = [float(r.latency_ms) for r in responses if r.latency_ms is not None]
    if not values:
        return None
    return {"p50": percentile(values, 50), "p95": percentile(values, 95)}


def compute_token_usage(responses: list[Response]) -> dict[str, float]:
    """Total and average input/output tokens across responses that have token counts.
    Zeros (not None) when there's nothing to sum — a token count of 0 is a real,
    knowable fact about an empty or all-failed response set, unlike cost, which
    depends on external pricing that might not be configured."""
    input_tokens = [r.input_tokens for r in responses if r.input_tokens is not None]
    output_tokens = [r.output_tokens for r in responses if r.output_tokens is not None]
    return {
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "avg_input_tokens": (sum(input_tokens) / len(input_tokens)) if input_tokens else 0.0,
        "avg_output_tokens": (sum(output_tokens) / len(output_tokens)) if output_tokens else 0.0,
    }


def compute_cost_per_query(responses: list[Response]) -> float | None:
    """Mean `cost_usd` across *successful* responses only (a failed response never
    completed a billed generation — see runner.execute_run's finalization logic for
    the matching treatment of `Run.total_cost_usd`).

    Returns `None` — never a silently-partial average — if there are no successful
    responses, or if pricing wasn't configured for even one of them (`cost_usd is
    None` on a successful response means "unverified", not "zero"; see
    runner.compute_cost_usd). A run's pricing is uniform (one model per run), so a mix
    of priced/unpriced successful responses shouldn't occur in practice — if it does,
    treating the whole aggregate as unverified is the conservative, correct call.
    """
    successful = [r for r in responses if r.error_class is None]
    if not successful:
        return None
    if any(r.cost_usd is None for r in successful):
        return None
    return sum(r.cost_usd for r in successful) / len(successful)


def compute_pass_rate(scores: list[Score]) -> float | None:
    """Mean `value` across a list of same-metric deterministic scores (0.0/1.0 each).
    `None` if the list is empty — there is nothing to report a rate over."""
    if not scores:
        return None
    return sum(s.value for s in scores) / len(scores)


def compute_guardrail_rate(
    results: list[GuardrailResult], *, numerator_outcome: GuardrailOutcome
) -> float | None:
    """Fraction of `results` whose `outcome` equals `numerator_outcome`.

    This one function computes all three guardrail-rate metrics, parameterized by
    which outcome counts as the numerator:
      - prompt_injection_block_rate      -> numerator_outcome=TP, over injection results
      - sensitive_information_protection -> numerator_outcome=TP, over pii results
      - benign_false_positive_rate       -> numerator_outcome=FP, over benign results
    (Filtering `results` down to one attack_type's rows is the caller's job — see
    quality.scoring.summarize_guardrail_rates.)
    """
    if not results:
        return None
    matching = sum(1 for r in results if r.outcome is numerator_outcome)
    return matching / len(results)
