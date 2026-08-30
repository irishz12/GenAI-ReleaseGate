"""Regression engine (Stage C, first half): inner-joins Prompt V1/V2 responses on
case_id, computes per-metric deltas with a simple paired bootstrap 95% CI where
meaningful, and hands the result to the release-policy engine.

Pure reads over already-persisted responses/scores/guardrail_results — this module
never calls a generator, judge, or guardrail; Stage A/B are assumed complete. Dev-split
data only, same as every other Stage B/C module (holdout is never read here).
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass

from evalguard.config import PolicyConfig
from evalguard.db.store import (
    get_case,
    get_guardrail_results_for_response,
    get_responses_for_run,
    get_scores_for_response,
)
from evalguard.enums import AttackType, GuardrailOutcome, MetricName
from evalguard.guardrails.gate import resolve_expected_block
from evalguard.models import Comparison, EvaluationCase, MetricDelta, Response
from evalguard.policy.engine import evaluate_policy
from evalguard.quality.aggregate import percentile
from evalguard.regression.statistics import (
    bootstrap_two_sided_p_value,
    holm_bonferroni,
    paired_cohens_d,
    paired_median_delta,
)

# Bootstrap is deterministic (fixed seed): the same DB state must always produce the
# same CI — a comparison's release decision cannot depend on which run happened to
# draw a wider interval.
_CI_SEED = 42
_CI_RESAMPLES = 2000

# metric -> (attack_type the metric is scored over, outcome that counts as its numerator)
_GUARDRAIL_RATE_METRICS = {
    MetricName.PROMPT_INJECTION_BLOCK_RATE: (AttackType.INJECTION, GuardrailOutcome.TP),
    MetricName.SENSITIVE_INFORMATION_PROTECTION: (AttackType.PII_EXTRACTION, GuardrailOutcome.TP),
    MetricName.BENIGN_FALSE_POSITIVE_RATE: (AttackType.BENIGN, GuardrailOutcome.FP),
}
_SCORE_METRICS = {
    MetricName.ANSWER_CORRECTNESS,
    MetricName.FAITHFULNESS,
    MetricName.HALLUCINATION,
    MetricName.INSTRUCTION_FOLLOWING,
    MetricName.ABSTENTION_ACCURACY,
}
# metric -> percentile — these aggregate a whole distribution, not a per-case mean, so
# a paired-diff CI doesn't apply to them ("simple paired 95% CI where meaningful").
_PERCENTILE_METRICS = {MetricName.P50_LATENCY: 50, MetricName.P95_LATENCY: 95}


@dataclass(frozen=True)
class PairedCase:
    case_id: int
    baseline: Response
    candidate: Response


def pair_by_case_id(baseline: list[Response], candidate: list[Response]) -> list[PairedCase]:
    """Inner join on case_id — only cases both variants actually produced a response
    for. Sorted for determinism: iteration order feeds the bootstrap resampling."""
    baseline_by_case = {r.case_id: r for r in baseline}
    candidate_by_case = {r.case_id: r for r in candidate}
    shared_case_ids = sorted(set(baseline_by_case) & set(candidate_by_case))
    return [
        PairedCase(cid, baseline_by_case[cid], candidate_by_case[cid]) for cid in shared_case_ids
    ]


def paired_bootstrap_ci_95(
    diffs: list[float], *, n_resamples: int = _CI_RESAMPLES, seed: int = _CI_SEED
) -> tuple[float, float] | None:
    """Simple paired bootstrap 95% CI for the mean of paired differences (candidate -
    baseline per case). `None` below 2 samples — nothing to resample."""
    n = len(diffs)
    if n < 2:
        return None
    rng = random.Random(seed)
    resample_means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    lo = resample_means[int(0.025 * n_resamples)]
    hi = resample_means[min(int(0.975 * n_resamples), n_resamples - 1)]
    return lo, hi


def _score_value(conn: sqlite3.Connection, response: Response, metric: MetricName) -> float | None:
    if response.output_text is None:
        return None
    matching = [s for s in get_scores_for_response(conn, response.id) if s.metric is metric]
    return matching[0].value if matching else None


def _guardrail_value(
    conn: sqlite3.Connection, response: Response, case: EvaluationCase, metric: MetricName
) -> float | None:
    attack_type, numerator_outcome = _GUARDRAIL_RATE_METRICS[metric]
    if case.attack_type is not attack_type:
        return None
    results = get_guardrail_results_for_response(conn, response.id)
    if not results:
        return None
    return 1.0 if results[0].outcome is numerator_outcome else 0.0


def _overall_benign_fp_value(
    conn: sqlite3.Connection, response: Response, case: EvaluationCase
) -> float | None:
    """Phase 8.1: false positives across *every* case never expected to be blocked —
    not just the guardrail suite's own benign subset (see BENIGN_FALSE_POSITIVE_RATE).
    `None` (excluded, never guessed) for a case that *was* expected to be blocked, or
    for a response with no guardrail check at all."""
    if resolve_expected_block(case):
        return None
    results = get_guardrail_results_for_response(conn, response.id)
    if not results:
        return None
    return 1.0 if results[0].triggered else 0.0


def _mean_metric_value(
    conn: sqlite3.Connection, metric: MetricName, response: Response, case: EvaluationCase
) -> float | None:
    if metric in _SCORE_METRICS:
        return _score_value(conn, response, metric)
    if metric in _GUARDRAIL_RATE_METRICS:
        return _guardrail_value(conn, response, case, metric)
    if metric is MetricName.OVERALL_BENIGN_FALSE_POSITIVE_RATE:
        return _overall_benign_fp_value(conn, response, case)
    if metric is MetricName.COST_PER_QUERY:
        return None if response.error_class is not None else response.cost_usd
    if metric is MetricName.FAILURE_RATE:
        return 1.0 if response.error_class is not None else 0.0
    raise ValueError(f"'{metric.value}' is not a mean-style metric")


def _mean_metric_delta(
    conn: sqlite3.Connection, metric: MetricName, paired: list[PairedCase]
) -> MetricDelta | None:
    pairs: list[tuple[float, float]] = []
    for pc in paired:
        case = get_case(conn, pc.case_id)
        baseline_value = _mean_metric_value(conn, metric, pc.baseline, case)
        candidate_value = _mean_metric_value(conn, metric, pc.candidate, case)
        if baseline_value is not None and candidate_value is not None:
            pairs.append((baseline_value, candidate_value))

    if not pairs:
        return None

    baseline_mean = sum(b for b, _ in pairs) / len(pairs)
    candidate_mean = sum(c for _, c in pairs) / len(pairs)
    diffs = [c - b for b, c in pairs]
    ci = paired_bootstrap_ci_95(diffs)
    return MetricDelta(
        metric=metric,
        baseline_value=baseline_mean,
        candidate_value=candidate_mean,
        delta=candidate_mean - baseline_mean,
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        median_delta=paired_median_delta(diffs),
        effect_size=paired_cohens_d(diffs),
        p_value=bootstrap_two_sided_p_value(diffs),
    )


def _percentile_metric_delta(metric: MetricName, paired: list[PairedCase]) -> MetricDelta | None:
    p = _PERCENTILE_METRICS[metric]
    baseline_latencies = [
        float(pc.baseline.latency_ms) for pc in paired if pc.baseline.latency_ms is not None
    ]
    candidate_latencies = [
        float(pc.candidate.latency_ms) for pc in paired if pc.candidate.latency_ms is not None
    ]
    if not baseline_latencies or not candidate_latencies:
        return None
    baseline_value = percentile(baseline_latencies, p)
    candidate_value = percentile(candidate_latencies, p)
    return MetricDelta(
        metric=metric,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=candidate_value - baseline_value,
    )


def compute_metric_delta(
    conn: sqlite3.Connection, metric: MetricName, paired: list[PairedCase]
) -> MetricDelta | None:
    if metric in _PERCENTILE_METRICS:
        return _percentile_metric_delta(metric, paired)
    return _mean_metric_delta(conn, metric, paired)


def _apply_holm_correction(deltas: list[MetricDelta]) -> None:
    """Holm-Bonferroni correction across every metric in this comparison that has a
    p-value — the family is "all metrics tested in this one comparison," which is
    the multiple-comparison exposure Phase 2 is meant to surface. Mutates each
    delta's `holm_significant` in place (MetricDelta allows attribute assignment);
    never touches `delta`/`ci_low`/`ci_high`, so the release decision this
    comparison's gates were already computed from cannot change."""
    p_values = {delta.metric.value: delta.p_value for delta in deltas if delta.p_value is not None}
    if not p_values:
        return
    significance = holm_bonferroni(p_values)
    for delta in deltas:
        if delta.metric.value in significance:
            delta.holm_significant = significance[delta.metric.value]


def compare_runs(
    conn: sqlite3.Connection, baseline_run_id: int, candidate_run_id: int, policy: PolicyConfig
) -> Comparison:
    """The full Stage C pipeline for one baseline/candidate run pair: pair by
    case_id, compute every metric's delta, then hand off to the policy engine for
    the GO/REVIEW/HOLD/INVALID decision."""
    baseline_responses = get_responses_for_run(conn, baseline_run_id)
    candidate_responses = get_responses_for_run(conn, candidate_run_id)
    paired = pair_by_case_id(baseline_responses, candidate_responses)

    deltas = [
        delta
        for metric in MetricName
        if (delta := compute_metric_delta(conn, metric, paired)) is not None
    ]
    _apply_holm_correction(deltas)

    if paired:
        failed_pairs = sum(
            1
            for pc in paired
            if pc.baseline.error_class is not None or pc.candidate.error_class is not None
        )
        paired_failure_rate = failed_pairs / len(paired)
    else:
        paired_failure_rate = 0.0  # irrelevant: evaluate_policy short-circuits on n_paired==0

    release = evaluate_policy(
        deltas, policy, n_paired=len(paired), paired_failure_rate=paired_failure_rate
    )

    return Comparison(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        n_paired=len(paired),
        deltas=deltas,
        release=release,
    )
