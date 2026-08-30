"""Tests for evalguard.quality.judge — rendering, JSON parsing with bounded retry,
claim-wise faithfulness/hallucination derivation, and blindness.

The fake client used here is test-only (mirrors the pattern in
test_bedrock_mantle_client.py) — FakeClient itself stays untouched.
"""

from __future__ import annotations

import pytest

from evalguard.enums import CaseCategory, CaseSplit
from evalguard.models import EvaluationCase, Response
from evalguard.providers.base import Completion, CompletionParams, Usage
from evalguard.quality.judge import (
    JudgeRetryExhaustedError,
    MalformedJudgeOutputError,
    build_blind_judge_case,
    call_judge,
    compute_faithfulness_and_hallucination,
    parse_judge_json,
    render_judge_prompt,
)
from evalguard.quality.judge_schemas import ClaimVerdict, CorrectnessVerdict


def _case(**overrides) -> EvaluationCase:
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
    return EvaluationCase(**defaults)


def _response(output_text: str, **overrides) -> Response:
    defaults = dict(id=1, run_id=1, case_id=1, output_text=output_text, finish_reason="stop")
    defaults.update(overrides)
    return Response(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# render_judge_prompt
# ─────────────────────────────────────────────────────────────────────────────


def test_render_judge_prompt_substitutes_all_four_fields() -> None:
    from evalguard.quality.judge_schemas import BlindJudgeCase

    case = BlindJudgeCase(
        question="What is the refund window?",
        context="Refunds within 30 days.",
        reference_answer="30 days",
        candidate_answer="You get 30 days.",
    )
    template = "Q:{question}\nC:{context}\nR:{reference_answer}\nA:{candidate_answer}"
    rendered = render_judge_prompt(template, case)
    assert "What is the refund window?" in rendered
    assert "Refunds within 30 days." in rendered
    assert "30 days" in rendered
    assert "You get 30 days." in rendered
    assert "{question}" not in rendered


def test_render_judge_prompt_handles_none_context_and_reference() -> None:
    from evalguard.quality.judge_schemas import BlindJudgeCase

    case = BlindJudgeCase(question="q", context=None, reference_answer=None, candidate_answer="a")
    rendered = render_judge_prompt("{context}|{reference_answer}", case)
    assert rendered == "|"


# ─────────────────────────────────────────────────────────────────────────────
# parse_judge_json
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_judge_json_parses_clean_json() -> None:
    text = '{"score": 0.9, "verdict": "correct", "rationale": "matches"}'
    result = parse_judge_json(text, CorrectnessVerdict)
    assert result.score == 0.9


def test_parse_judge_json_extracts_json_wrapped_in_prose() -> None:
    text = (
        'Sure, here is my judgment:\n{"score": 1.0, "verdict": "correct", "rationale": "yes"}'
        "\nHope that helps!"
    )
    result = parse_judge_json(text, CorrectnessVerdict)
    assert result.score == 1.0


def test_parse_judge_json_extracts_json_in_code_fence() -> None:
    text = '```json\n{"score": 0.5, "verdict": "partially_correct", "rationale": "half"}\n```'
    result = parse_judge_json(text, CorrectnessVerdict)
    assert result.score == 0.5


def test_parse_judge_json_raises_on_no_json_found() -> None:
    with pytest.raises(MalformedJudgeOutputError):
        parse_judge_json("I refuse to answer in JSON.", CorrectnessVerdict)


def test_parse_judge_json_raises_on_invalid_json_syntax() -> None:
    with pytest.raises(MalformedJudgeOutputError):
        parse_judge_json('{"score": 0.9, "verdict": "correct", "rationale":}', CorrectnessVerdict)


def test_parse_judge_json_raises_on_schema_mismatch() -> None:
    with pytest.raises(MalformedJudgeOutputError):
        parse_judge_json(
            '{"score": 2.0, "verdict": "correct", "rationale": "x"}', CorrectnessVerdict
        )


# ─────────────────────────────────────────────────────────────────────────────
# call_judge — bounded retry
# ─────────────────────────────────────────────────────────────────────────────


class _ScriptedJudgeClient:
    """Test-only LLMClient returning one scripted text response per call, in order."""

    def __init__(self, texts: list[str], *, input_tokens=10, output_tokens=5, latency_ms=100):
        self._texts = list(texts)
        self.call_count = 0
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._latency_ms = latency_ms

    def complete(self, prompt: str, params: CompletionParams) -> Completion:
        self.call_count += 1
        text = self._texts.pop(0)
        return Completion(
            text=text,
            finish_reason="stop",
            usage=Usage(input_tokens=self._input_tokens, output_tokens=self._output_tokens),
            latency_ms=self._latency_ms,
        )


def test_call_judge_succeeds_on_first_attempt() -> None:
    client = _ScriptedJudgeClient(['{"score": 1.0, "verdict": "correct", "rationale": "x"}'])
    result, in_tok, out_tok, latency = call_judge(
        client, "prompt", CompletionParams(), CorrectnessVerdict
    )
    assert result.score == 1.0
    assert client.call_count == 1
    assert in_tok == 10
    assert out_tok == 5
    assert latency == 100


def test_call_judge_retries_once_on_malformed_then_succeeds() -> None:
    client = _ScriptedJudgeClient(
        ["not json at all", '{"score": 1.0, "verdict": "correct", "rationale": "x"}']
    )
    result, in_tok, out_tok, latency = call_judge(
        client, "prompt", CompletionParams(), CorrectnessVerdict, max_attempts=2
    )
    assert result.score == 1.0
    assert client.call_count == 2
    # Cost tracking must include the failed attempt too — real tokens were spent on it.
    assert in_tok == 20
    assert out_tok == 10
    assert latency == 200


def test_call_judge_raises_after_exhausting_retries() -> None:
    client = _ScriptedJudgeClient(["not json", "still not json"])
    with pytest.raises(JudgeRetryExhaustedError):
        call_judge(client, "prompt", CompletionParams(), CorrectnessVerdict, max_attempts=2)
    assert client.call_count == 2


def test_call_judge_default_max_attempts_is_bounded() -> None:
    """A never-ending stream of bad output must not retry forever."""
    client = _ScriptedJudgeClient(["bad"] * 10)
    with pytest.raises(JudgeRetryExhaustedError):
        call_judge(client, "prompt", CompletionParams(), CorrectnessVerdict)
    assert client.call_count <= 5  # some small bound, not unlimited


# ─────────────────────────────────────────────────────────────────────────────
# compute_faithfulness_and_hallucination
# ─────────────────────────────────────────────────────────────────────────────


def test_faithfulness_all_claims_supported() -> None:
    claims = [
        ClaimVerdict(claim="a", label="SUPPORTED"),
        ClaimVerdict(claim="b", label="SUPPORTED"),
    ]
    faithfulness, hallucinated = compute_faithfulness_and_hallucination(claims)
    assert faithfulness == 1.0
    assert hallucinated is False


def test_faithfulness_one_contradicted_claim_marks_hallucinated() -> None:
    claims = [
        ClaimVerdict(claim="a", label="SUPPORTED"),
        ClaimVerdict(claim="b", label="CONTRADICTED"),
    ]
    faithfulness, hallucinated = compute_faithfulness_and_hallucination(claims)
    assert faithfulness == 0.5
    assert hallucinated is True


def test_faithfulness_unsupported_claim_lowers_score_without_hallucination() -> None:
    """UNSUPPORTED (not addressed by context) is different from CONTRADICTED (context
    says otherwise) — only CONTRADICTED counts as a hallucination."""
    claims = [
        ClaimVerdict(claim="a", label="SUPPORTED"),
        ClaimVerdict(claim="b", label="UNSUPPORTED"),
    ]
    faithfulness, hallucinated = compute_faithfulness_and_hallucination(claims)
    assert faithfulness == 0.5
    assert hallucinated is False


def test_faithfulness_empty_claims_list_is_vacuously_faithful() -> None:
    faithfulness, hallucinated = compute_faithfulness_and_hallucination([])
    assert faithfulness == 1.0
    assert hallucinated is False


# ─────────────────────────────────────────────────────────────────────────────
# build_blind_judge_case — blindness
# ─────────────────────────────────────────────────────────────────────────────


def test_build_blind_judge_case_carries_only_the_four_fields() -> None:
    case = _case()
    response = _response("You have 30 days.")
    blind_case = build_blind_judge_case(case, response)

    assert blind_case.question == case.input
    assert blind_case.context == case.context
    assert blind_case.reference_answer == case.reference_answer
    assert blind_case.candidate_answer == response.output_text


def test_build_blind_judge_case_signature_cannot_accept_run_or_prompt_identity() -> None:
    """Structural safeguard: the function only takes (case, response) — there is no
    parameter through which a run's model_id or a prompt's version/role could flow in,
    even by accident."""
    import inspect

    params = list(inspect.signature(build_blind_judge_case).parameters)
    assert params == ["case", "response"]
