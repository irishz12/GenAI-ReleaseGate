"""Tests for evalguard.quality.scoring.score_run — the Stage B orchestration that ties
deterministic checks + guardrail checks + persistence together over a real (temp)
SQLite DB. No LLMClient is ever constructed here — only pre-inserted responses.
"""

from __future__ import annotations

import sqlite3

import pytest

from evalguard.db.store import (
    get_guardrail_results_for_response,
    get_scores_for_response,
    insert_case,
    insert_prompt,
    insert_response,
    insert_run,
)
from evalguard.enums import (
    AttackType,
    CaseCategory,
    CaseSplit,
    ErrorClass,
    FinishReason,
    GuardrailOutcome,
    MetricName,
    PromptRole,
)
from evalguard.models import EvaluationCase, Prompt, Response, Run
from evalguard.quality.scoring import HoldoutAccessError, score_run


def _prompt(conn: sqlite3.Connection) -> Prompt:
    return insert_prompt(
        conn,
        Prompt(
            name="support_agent",
            version="v1",
            role=PromptRole.PRODUCTION,
            template="x",
            content_hash="hash-v1",
        ),
    )


def _run(conn: sqlite3.Connection, prompt_id: int) -> Run:
    return insert_run(
        conn,
        Run(
            prompt_id=prompt_id,
            model_id="fake-generator-v1",
            judge_model_id="fake-judge-v1",
            guardrail_config_hash="ghash",
            code_sha="csha",
            config_hash="chash",
        ),
    )


def _case(conn: sqlite3.Connection, **overrides) -> EvaluationCase:
    defaults = dict(
        external_id=f"case-{overrides.get('_n', 0)}",
        category=CaseCategory.GROUNDED_QA,
        split=CaseSplit.DEV,
        input="question",
        dataset_hash="hash",
    )
    overrides.pop("_n", None)
    defaults.update(overrides)
    return insert_case(conn, EvaluationCase(**defaults))


def _response(conn: sqlite3.Connection, run_id: int, case_id: int, **overrides) -> Response:
    defaults = dict(
        run_id=run_id, case_id=case_id, finish_reason=FinishReason.STOP, output_text="some answer"
    )
    defaults.update(overrides)
    return insert_response(conn, Response(**defaults))


_MAX_THREE_WORDS = {"type": "max_words", "max_words": 3}


# ─────────────────────────────────────────────────────────────────────────────
# per-category scoring
# ─────────────────────────────────────────────────────────────────────────────


def test_scores_instruction_following_case(conn: sqlite3.Connection) -> None:
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    case = _case(
        conn,
        _n=1,
        category=CaseCategory.INSTRUCTION_FOLLOWING,
        constraints={"type": "max_words", "max_words": 3},
    )
    response = _response(conn, run.id, case.id, output_text="one two three")

    summary = score_run(conn, run.id)

    scores = get_scores_for_response(conn, response.id)
    assert len(scores) == 1
    assert scores[0].metric is MetricName.INSTRUCTION_FOLLOWING
    assert scores[0].value == 1.0
    assert summary.newly_scored == 1
    assert summary.instruction_following_rate == 1.0


def test_scores_abstention_case(conn: sqlite3.Connection) -> None:
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    case = _case(conn, _n=1, category=CaseCategory.ABSTENTION, expected_abstention=True)
    response = _response(
        conn, run.id, case.id, output_text="I don't have enough information to answer that."
    )

    summary = score_run(conn, run.id)

    scores = get_scores_for_response(conn, response.id)
    assert scores[0].metric is MetricName.ABSTENTION_ACCURACY
    assert scores[0].value == 1.0
    assert summary.abstention_accuracy_rate == 1.0


def test_scores_guardrail_injection_case(conn: sqlite3.Connection) -> None:
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    case = _case(
        conn,
        _n=1,
        category=CaseCategory.GUARDRAIL,
        attack_type=AttackType.INJECTION,
        expected_block=True,
    )
    response = _response(
        conn, run.id, case.id, output_text="I can't do that, but happy to help otherwise."
    )

    summary = score_run(conn, run.id)

    results = get_guardrail_results_for_response(conn, response.id)
    assert len(results) == 1
    assert results[0].outcome is GuardrailOutcome.TP
    assert summary.prompt_injection_block_rate == 1.0


def test_skips_grounded_qa_cases_as_unscoreable(conn: sqlite3.Connection) -> None:
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    case = _case(conn, _n=1, category=CaseCategory.GROUNDED_QA)
    response = _response(conn, run.id, case.id)

    summary = score_run(conn, run.id)

    assert get_scores_for_response(conn, response.id) == []
    assert get_guardrail_results_for_response(conn, response.id) == []
    assert summary.skipped_unscoreable == 1
    assert summary.newly_scored == 0


def test_skips_failed_responses_without_scoring(conn: sqlite3.Connection) -> None:
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    case = _case(
        conn, _n=1, category=CaseCategory.INSTRUCTION_FOLLOWING, constraints=_MAX_THREE_WORDS
    )
    response = _response(
        conn,
        run.id,
        case.id,
        output_text=None,
        finish_reason=FinishReason.ERROR,
        error_class=ErrorClass.PERMANENT,
    )

    summary = score_run(conn, run.id)

    assert get_scores_for_response(conn, response.id) == []
    assert summary.skipped_failed == 1
    assert summary.failure_rate == 1.0


def test_skips_abstention_case_without_expected_abstention_label(conn: sqlite3.Connection) -> None:
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    abstention_case = _case(conn, _n=1, category=CaseCategory.ABSTENTION, expected_abstention=None)
    response = _response(conn, run.id, abstention_case.id)

    summary = score_run(conn, run.id)
    assert get_scores_for_response(conn, response.id) == []
    assert summary.skipped_unscoreable == 1


# ─────────────────────────────────────────────────────────────────────────────
# idempotency
# ─────────────────────────────────────────────────────────────────────────────


def test_score_run_is_idempotent(conn: sqlite3.Connection) -> None:
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    case = _case(
        conn, _n=1, category=CaseCategory.INSTRUCTION_FOLLOWING, constraints=_MAX_THREE_WORDS
    )
    response = _response(conn, run.id, case.id, output_text="one two three")

    first = score_run(conn, run.id)
    second = score_run(conn, run.id)

    assert first.newly_scored == 1
    assert second.newly_scored == 0
    assert second.already_scored == 1
    assert len(get_scores_for_response(conn, response.id)) == 1  # not duplicated


def test_score_run_fills_in_missing_scores_on_rerun(conn: sqlite3.Connection) -> None:
    """Simulates: score a run, then a second case gets a response later (e.g. resume) —
    re-scoring should score only the new one, not touch the first."""
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    case1 = _case(
        conn, _n=1, category=CaseCategory.INSTRUCTION_FOLLOWING, constraints=_MAX_THREE_WORDS
    )
    _response(conn, run.id, case1.id, output_text="one two three")
    score_run(conn, run.id)

    case2 = _case(
        conn, _n=2, category=CaseCategory.INSTRUCTION_FOLLOWING, constraints=_MAX_THREE_WORDS
    )
    _response(conn, run.id, case2.id, output_text="four five six")

    summary = score_run(conn, run.id)
    assert summary.newly_scored == 1
    assert summary.already_scored == 1


# ─────────────────────────────────────────────────────────────────────────────
# holdout safety
# ─────────────────────────────────────────────────────────────────────────────


def test_score_run_raises_on_holdout_case(conn: sqlite3.Connection) -> None:
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    case = _case(
        conn,
        _n=1,
        category=CaseCategory.INSTRUCTION_FOLLOWING,
        split=CaseSplit.HOLDOUT,
        constraints=_MAX_THREE_WORDS,
    )
    _response(conn, run.id, case.id, output_text="one two three")

    with pytest.raises(HoldoutAccessError):
        score_run(conn, run.id)


def test_score_run_allows_holdout_when_explicitly_opted_in(conn: sqlite3.Connection) -> None:
    """The one sanctioned exception (Phase 9's holdout acceptance run): scoring
    holdout must still be refused by default (see the test above), but must succeed
    when the caller explicitly opts in — the guard is a deliberate default, not an
    unconditional wall with no legitimate way through it."""
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)
    case = _case(
        conn,
        _n=1,
        category=CaseCategory.INSTRUCTION_FOLLOWING,
        split=CaseSplit.HOLDOUT,
        constraints=_MAX_THREE_WORDS,
    )
    response = _response(conn, run.id, case.id, output_text="one two three")

    summary = score_run(conn, run.id, allow_holdout=True)

    assert summary.newly_scored == 1
    assert get_scores_for_response(conn, response.id)[0].value == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# operational metrics fold into the same summary
# ─────────────────────────────────────────────────────────────────────────────


def test_summary_includes_operational_metrics_across_mixed_categories(
    conn: sqlite3.Connection,
) -> None:
    prompt = _prompt(conn)
    run = _run(conn, prompt.id)

    gqa_case = _case(conn, _n=1, category=CaseCategory.GROUNDED_QA)
    _response(
        conn,
        run.id,
        gqa_case.id,
        input_tokens=100,
        output_tokens=20,
        latency_ms=300,
        cost_usd=0.01,
    )

    fail_case = _case(conn, _n=2, category=CaseCategory.GROUNDED_QA)
    _response(
        conn,
        run.id,
        fail_case.id,
        output_text=None,
        finish_reason=FinishReason.ERROR,
        error_class=ErrorClass.PERMANENT,
    )

    summary = score_run(conn, run.id)

    assert summary.case_count == 2
    assert summary.failure_rate == 0.5
    assert summary.token_usage["total_input_tokens"] == 100
    assert summary.cost_per_query == 0.01  # only the successful response counts


def test_run_not_found_raises(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="no run"):
        score_run(conn, 9999)
