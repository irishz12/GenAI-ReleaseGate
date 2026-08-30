"""Tests for evalguard.quality.judge.score_grounded_qa_case_with_judge — the
DB-backed orchestration: two judge calls per case, idempotent, cost tracked
separately per metric, dev-only, blind to run/prompt identity.
"""

from __future__ import annotations

import sqlite3

import pytest

from evalguard.db.store import (
    get_scores_for_response,
    insert_case,
    insert_prompt,
    insert_response,
    insert_run,
)
from evalguard.enums import CaseCategory, CaseSplit, MetricName, PromptRole
from evalguard.models import EvaluationCase, Prompt, Response, Run
from evalguard.providers.base import Completion, CompletionParams, Usage
from evalguard.quality.judge import score_grounded_qa_case_with_judge
from evalguard.quality.scoring import HoldoutAccessError

_CORRECTNESS_TEMPLATE = (
    "Question: {question}\nReference: {reference_answer}\nCandidate: {candidate_answer}\n"
    'Respond: {{"score": <0-1>, "verdict": "correct"|"partially_correct"|"incorrect", '
    '"rationale": "<text>"}}'
)
_FAITHFULNESS_TEMPLATE = (
    "Context: {context}\nCandidate: {candidate_answer}\n"
    'Respond: {{"claims": [{{"claim": "<text>", '
    '"label": "SUPPORTED"|"UNSUPPORTED"|"CONTRADICTED"}}]}}'
)


class _RecordingJudgeClient:
    """Test-only LLMClient: returns scripted JSON responses in order and records every
    prompt it was asked to complete, so tests can assert on what the judge actually saw."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts_seen: list[str] = []
        self.params_seen: list[CompletionParams] = []

    def complete(self, prompt: str, params: CompletionParams) -> Completion:
        self.prompts_seen.append(prompt)
        self.params_seen.append(params)
        text = self._responses.pop(0)
        return Completion(
            text=text,
            finish_reason="stop",
            usage=Usage(input_tokens=100, output_tokens=30),
            latency_ms=250,
        )


def _case(conn: sqlite3.Connection, **overrides) -> EvaluationCase:
    defaults = dict(
        external_id="gqa-0001",
        category=CaseCategory.GROUNDED_QA,
        split=CaseSplit.DEV,
        input="What is the refund window?",
        context="Refunds are accepted within 30 days.",
        reference_answer="30 days",
        dataset_hash="hash",
    )
    defaults.update(overrides)
    return insert_case(conn, EvaluationCase(**defaults))


def _run(conn: sqlite3.Connection) -> Run:
    prompt = insert_prompt(
        conn,
        Prompt(
            name="support_agent",
            version="v1",
            role=PromptRole.PRODUCTION,
            template="x",
            content_hash="hash-v1",
        ),
    )
    return insert_run(
        conn,
        Run(
            prompt_id=prompt.id,
            model_id="qwen.qwen3-next-80b-a3b-instruct",
            judge_model_id="openai.gpt-oss-120b",
            guardrail_config_hash="ghash",
            code_sha="csha",
            config_hash="chash",
        ),
    )


def _response(conn: sqlite3.Connection, case_id: int, **overrides) -> Response:
    run = _run(conn)
    defaults = dict(
        run_id=run.id, case_id=case_id, finish_reason="stop", output_text="You have 30 days."
    )
    defaults.update(overrides)
    return insert_response(conn, Response(**defaults))


def _score_kwargs(**overrides):
    defaults = dict(
        correctness_template=_CORRECTNESS_TEMPLATE,
        faithfulness_template=_FAITHFULNESS_TEMPLATE,
        correctness_params=CompletionParams(temperature=0.0, max_tokens=512),
        faithfulness_params=CompletionParams(temperature=0.0, max_tokens=1024),
        price_per_1k_input=0.00018,
        price_per_1k_output=0.00071,
    )
    defaults.update(overrides)
    return defaults


_CORRECT_JSON = '{"score": 1.0, "verdict": "correct", "rationale": "matches reference"}'
_FAITHFUL_JSON = '{"claims": [{"claim": "30 day refund window", "label": "SUPPORTED"}]}'


def test_scores_correctness_faithfulness_and_hallucination(conn: sqlite3.Connection) -> None:
    case = _case(conn)
    response = _response(conn, case.id)
    client = _RecordingJudgeClient([_CORRECT_JSON, _FAITHFUL_JSON])

    scores = score_grounded_qa_case_with_judge(conn, response, case, client, **_score_kwargs())

    metrics = {s.metric for s in scores}
    assert metrics == {
        MetricName.ANSWER_CORRECTNESS,
        MetricName.FAITHFULNESS,
        MetricName.HALLUCINATION,
    }

    persisted = {s.metric: s for s in get_scores_for_response(conn, response.id)}
    assert persisted[MetricName.ANSWER_CORRECTNESS].value == 1.0
    assert persisted[MetricName.FAITHFULNESS].value == 1.0
    assert persisted[MetricName.HALLUCINATION].value == 0.0


def test_correctness_and_faithfulness_calls_use_their_own_frozen_budget(
    conn: sqlite3.Connection,
) -> None:
    """Phase 8.2: correctness is frozen at 512, faithfulness at 1024 (Phase 8.1
    evidence — the reasoning judge needs more room to finish faithfulness's
    claim-extraction JSON). The two calls must never share one CompletionParams."""
    case = _case(conn)
    response = _response(conn, case.id)
    client = _RecordingJudgeClient([_CORRECT_JSON, _FAITHFUL_JSON])

    score_grounded_qa_case_with_judge(
        conn,
        response,
        case,
        client,
        **_score_kwargs(
            correctness_params=CompletionParams(temperature=0.0, max_tokens=512),
            faithfulness_params=CompletionParams(temperature=0.0, max_tokens=1024),
        ),
    )

    assert len(client.params_seen) == 2
    assert client.params_seen[0].max_tokens == 512  # correctness call
    assert client.params_seen[1].max_tokens == 1024  # faithfulness call


def test_hallucination_score_has_no_separate_cost_fields(conn: sqlite3.Connection) -> None:
    """hallucination is derived from the faithfulness call, not a separate API call —
    its cost fields must stay None so summing judge cost never double-counts."""
    case = _case(conn)
    response = _response(conn, case.id)
    client = _RecordingJudgeClient([_CORRECT_JSON, _FAITHFUL_JSON])

    score_grounded_qa_case_with_judge(conn, response, case, client, **_score_kwargs())

    persisted = {s.metric: s for s in get_scores_for_response(conn, response.id)}
    faithfulness_score = persisted[MetricName.FAITHFULNESS]
    hallucination_score = persisted[MetricName.HALLUCINATION]

    assert faithfulness_score.judge_cost_usd is not None
    assert faithfulness_score.judge_input_tokens == 100
    assert hallucination_score.judge_cost_usd is None
    assert hallucination_score.judge_input_tokens is None


def test_correctness_and_faithfulness_costs_are_tracked_separately(
    conn: sqlite3.Connection,
) -> None:
    case = _case(conn)
    response = _response(conn, case.id)
    client = _RecordingJudgeClient([_CORRECT_JSON, _FAITHFUL_JSON])

    score_grounded_qa_case_with_judge(conn, response, case, client, **_score_kwargs())

    persisted = {s.metric: s for s in get_scores_for_response(conn, response.id)}
    correctness_cost = persisted[MetricName.ANSWER_CORRECTNESS].judge_cost_usd
    faithfulness_cost = persisted[MetricName.FAITHFULNESS].judge_cost_usd

    assert correctness_cost is not None
    assert faithfulness_cost is not None
    assert correctness_cost == faithfulness_cost  # same token counts in this fixture
    assert correctness_cost > 0


def test_is_idempotent_and_makes_no_second_judge_call(conn: sqlite3.Connection) -> None:
    case = _case(conn)
    response = _response(conn, case.id)
    client = _RecordingJudgeClient([_CORRECT_JSON, _FAITHFUL_JSON])

    first = score_grounded_qa_case_with_judge(conn, response, case, client, **_score_kwargs())
    second = score_grounded_qa_case_with_judge(conn, response, case, client, **_score_kwargs())

    assert len(first) == 3
    assert second == []  # nothing new to score
    assert len(client.prompts_seen) == 2  # not re-called
    assert len(get_scores_for_response(conn, response.id)) == 3  # not duplicated


def test_raises_on_holdout_case(conn: sqlite3.Connection) -> None:
    case = _case(conn, split=CaseSplit.HOLDOUT)
    response = _response(conn, case.id)
    client = _RecordingJudgeClient([_CORRECT_JSON, _FAITHFUL_JSON])

    with pytest.raises(HoldoutAccessError):
        score_grounded_qa_case_with_judge(conn, response, case, client, **_score_kwargs())


def test_allows_holdout_case_when_explicitly_opted_in(conn: sqlite3.Connection) -> None:
    """The one sanctioned exception (Phase 9's holdout acceptance run): judging a
    holdout case must still be refused by default (see the test above), but must
    succeed when the caller explicitly opts in."""
    case = _case(conn, split=CaseSplit.HOLDOUT)
    response = _response(conn, case.id)
    client = _RecordingJudgeClient([_CORRECT_JSON, _FAITHFUL_JSON])

    scores = score_grounded_qa_case_with_judge(
        conn, response, case, client, allow_holdout=True, **_score_kwargs()
    )

    assert len(scores) == 3


def test_raises_on_non_grounded_qa_category(conn: sqlite3.Connection) -> None:
    case = _case(conn, category=CaseCategory.ABSTENTION, expected_abstention=True)
    response = _response(conn, case.id)
    client = _RecordingJudgeClient([_CORRECT_JSON, _FAITHFUL_JSON])

    with pytest.raises(ValueError, match="grounded_qa"):
        score_grounded_qa_case_with_judge(conn, response, case, client, **_score_kwargs())


def test_raises_on_missing_output_text(conn: sqlite3.Connection) -> None:
    case = _case(conn)
    response = _response(conn, case.id, output_text=None, finish_reason="error")
    client = _RecordingJudgeClient([])

    with pytest.raises(ValueError, match="output_text"):
        score_grounded_qa_case_with_judge(conn, response, case, client, **_score_kwargs())


def test_judge_never_sees_run_or_prompt_identity(conn: sqlite3.Connection) -> None:
    """The rendered prompts the judge actually receives must never mention a run id,
    model id, prompt name, or version/role string."""
    from evalguard.db.store import get_run

    case = _case(conn)
    response = _response(conn, case.id)
    run = get_run(conn, response.run_id)
    client = _RecordingJudgeClient([_CORRECT_JSON, _FAITHFUL_JSON])

    score_grounded_qa_case_with_judge(conn, response, case, client, **_score_kwargs())

    # Check the *actual* identity values this run/prompt carry — not a guessed word
    # list — so this test fails if blindness breaks regardless of naming choices.
    forbidden = [run.model_id, run.judge_model_id, "production", "PromptRole"]
    for prompt in client.prompts_seen:
        for term in forbidden:
            assert term not in prompt, f"blindness violated: {term!r} found in judge prompt"
