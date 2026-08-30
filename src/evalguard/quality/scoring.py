"""Stage B orchestration: score every response already stored for one run, persisting
per-response Score/GuardrailResult rows, then roll them up into a summary.

Hard constraints enforced here, not just documented:
- **Dev-only.** Every case scored must have `split == CaseSplit.DEV`; a holdout case
  slipping in raises rather than being silently scored.
- **No new generation, no judge.** This module only ever reads `responses` that
  already exist (via `get_responses_for_run`) and calls deterministic checkers — it
  never calls an `LLMClient`.
- **Idempotent.** Re-running `score_run` on the same run does not duplicate rows or
  re-derive scores that already exist; it fills in whatever's missing (e.g. after a
  checker bug fix that only affected some cases) and leaves the rest untouched.
- **Failed responses are counted, not scored.** A response with `output_text is None`
  contributes to `failure_rate` only — there is nothing to check it against.
"""

from __future__ import annotations

import sqlite3

from pydantic import BaseModel, ConfigDict

from evalguard.config import load_guardrails_config
from evalguard.db.store import (
    get_case,
    get_guardrail_results_for_response,
    get_response,
    get_responses_for_run,
    get_run,
    get_scores_for_response,
    insert_guardrail_result,
    insert_score,
)
from evalguard.enums import (
    AttackType,
    CaseCategory,
    CaseSplit,
    GuardrailOutcome,
    MetricName,
    ScoreMethod,
)
from evalguard.guardrails.checks import score_guardrail_response
from evalguard.models import EvaluationCase, GuardrailResult, Response, Score
from evalguard.quality.aggregate import (
    compute_cost_per_query,
    compute_failure_rate,
    compute_guardrail_rate,
    compute_latency_percentiles,
    compute_pass_rate,
    compute_token_usage,
)
from evalguard.quality.deterministic import check_abstention, check_instruction_following


class HoldoutAccessError(RuntimeError):
    """A case slated for scoring is not in the dev split. Scoring must never touch
    holdout — this is a hard stop, not a warning."""


class RunScoringSummary(BaseModel):
    """Everything computed for one run: what got scored just now, and the resulting
    aggregate metrics. Not persisted as its own table — it's a report over the
    `responses`/`scores`/`guardrail_results` rows, which are the persisted source of
    truth (docs/ARCHITECTURE.md §9's 7-table schema stays 7 tables)."""

    model_config = ConfigDict(extra="forbid")

    run_id: int
    case_count: int
    newly_scored: int
    already_scored: int
    skipped_failed: int
    skipped_unscoreable: int

    failure_rate: float
    latency_percentiles: dict[str, float] | None = None
    token_usage: dict[str, float]
    cost_per_query: float | None = None

    instruction_following_rate: float | None = None
    abstention_accuracy_rate: float | None = None
    prompt_injection_block_rate: float | None = None
    sensitive_information_protection: float | None = None
    benign_false_positive_rate: float | None = None


def require_dev_case(case: EvaluationCase, *, allow_holdout: bool = False) -> None:
    """Refuses a holdout case by default — the standing guard against accidental
    holdout access during development. `allow_holdout=True` is the one deliberate,
    explicit exception: the sealed holdout acceptance run (Phase 9), which must
    actually score holdout to make the real promotion decision. The default stays
    False everywhere else; every other caller is unaffected."""
    if case.split is not CaseSplit.DEV and not allow_holdout:
        raise HoldoutAccessError(
            f"case {case.external_id} is split={case.split.value}, not dev — "
            "scoring must never access holdout."
        )


def _score_instruction_following(response: Response, case: EvaluationCase) -> Score:
    passed = check_instruction_following(response.output_text, case.constraints or {})
    return Score(
        response_id=response.id,
        metric=MetricName.INSTRUCTION_FOLLOWING,
        method=ScoreMethod.DETERMINISTIC,
        value=1.0 if passed else 0.0,
        rationale=f"constraints={case.constraints}",
    )


def _score_abstention(response: Response, case: EvaluationCase) -> Score:
    passed = check_abstention(response.output_text, expected_abstention=case.expected_abstention)
    return Score(
        response_id=response.id,
        metric=MetricName.ABSTENTION_ACCURACY,
        method=ScoreMethod.DETERMINISTIC,
        value=1.0 if passed else 0.0,
        rationale=f"expected_abstention={case.expected_abstention}",
    )


def score_run(
    conn: sqlite3.Connection, run_id: int, *, allow_holdout: bool = False
) -> RunScoringSummary:
    """Score every not-yet-scored response for `run_id`, persist the results, and
    return the resulting aggregate summary (computed fresh over ALL responses for the
    run, scored just now or previously). `allow_holdout` defaults to False — see
    `require_dev_case`."""
    run = get_run(conn, run_id)
    if run is None:
        raise ValueError(f"no run with id={run_id}")

    responses = get_responses_for_run(conn, run_id)

    newly_scored = 0
    already_scored = 0
    skipped_failed = 0
    skipped_unscoreable = 0

    guardrail_patterns = load_guardrails_config().patterns

    for response in responses:
        if response.output_text is None:
            skipped_failed += 1
            continue

        case = get_case(conn, response.case_id)
        if case is None:
            raise ValueError(f"response {response.id} references a case that no longer exists")
        require_dev_case(case, allow_holdout=allow_holdout)

        if case.category is CaseCategory.INSTRUCTION_FOLLOWING:
            if _already_has_score(conn, response.id, MetricName.INSTRUCTION_FOLLOWING):
                already_scored += 1
                continue
            insert_score(conn, _score_instruction_following(response, case))
            newly_scored += 1

        elif case.category is CaseCategory.ABSTENTION:
            if case.expected_abstention is None:
                skipped_unscoreable += 1
                continue
            if _already_has_score(conn, response.id, MetricName.ABSTENTION_ACCURACY):
                already_scored += 1
                continue
            insert_score(conn, _score_abstention(response, case))
            newly_scored += 1

        elif case.category is CaseCategory.GUARDRAIL:
            if _already_has_guardrail_result(conn, response.id):
                already_scored += 1
                continue
            insert_guardrail_result(
                conn, score_guardrail_response(response, case, guardrail_patterns)
            )
            newly_scored += 1

        else:
            # grounded_qa: answer_correctness/faithfulness need a judge — out of scope
            # for Phase 4 (deterministic-only).
            skipped_unscoreable += 1

    return _summarize(
        conn,
        run_id,
        responses,
        newly_scored=newly_scored,
        already_scored=already_scored,
        skipped_failed=skipped_failed,
        skipped_unscoreable=skipped_unscoreable,
    )


def _already_has_score(conn: sqlite3.Connection, response_id: int, metric: MetricName) -> bool:
    return any(s.metric is metric for s in get_scores_for_response(conn, response_id))


def _already_has_guardrail_result(conn: sqlite3.Connection, response_id: int) -> bool:
    return bool(get_guardrail_results_for_response(conn, response_id))


def _summarize(
    conn: sqlite3.Connection,
    run_id: int,
    responses: list[Response],
    *,
    newly_scored: int,
    already_scored: int,
    skipped_failed: int,
    skipped_unscoreable: int,
) -> RunScoringSummary:
    all_scores: list[Score] = []
    all_guardrail_results: list[GuardrailResult] = []
    for response in responses:
        all_scores.extend(get_scores_for_response(conn, response.id))
        all_guardrail_results.extend(get_guardrail_results_for_response(conn, response.id))

    instruction_following_scores = [
        s for s in all_scores if s.metric is MetricName.INSTRUCTION_FOLLOWING
    ]
    abstention_scores = [s for s in all_scores if s.metric is MetricName.ABSTENTION_ACCURACY]

    guardrail_by_case: dict[int, tuple[GuardrailResult, EvaluationCase]] = {}
    for gr in all_guardrail_results:
        response = get_response(conn, gr.response_id)
        case = get_case(conn, response.case_id)
        guardrail_by_case[gr.response_id] = (gr, case)

    injection_results = [
        gr for gr, case in guardrail_by_case.values() if case.attack_type is AttackType.INJECTION
    ]
    pii_results = [
        gr
        for gr, case in guardrail_by_case.values()
        if case.attack_type is AttackType.PII_EXTRACTION
    ]
    benign_results = [
        gr for gr, case in guardrail_by_case.values() if case.attack_type is AttackType.BENIGN
    ]

    return RunScoringSummary(
        run_id=run_id,
        case_count=len(responses),
        newly_scored=newly_scored,
        already_scored=already_scored,
        skipped_failed=skipped_failed,
        skipped_unscoreable=skipped_unscoreable,
        failure_rate=compute_failure_rate(responses),
        latency_percentiles=compute_latency_percentiles(responses),
        token_usage=compute_token_usage(responses),
        cost_per_query=compute_cost_per_query(responses),
        instruction_following_rate=compute_pass_rate(instruction_following_scores),
        abstention_accuracy_rate=compute_pass_rate(abstention_scores),
        prompt_injection_block_rate=compute_guardrail_rate(
            injection_results, numerator_outcome=GuardrailOutcome.TP
        ),
        sensitive_information_protection=compute_guardrail_rate(
            pii_results, numerator_outcome=GuardrailOutcome.TP
        ),
        benign_false_positive_rate=compute_guardrail_rate(
            benign_results, numerator_outcome=GuardrailOutcome.FP
        ),
    )
