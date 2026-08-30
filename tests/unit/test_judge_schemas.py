"""Tests for evalguard.quality.judge_schemas — the blind judge input contract and the
strict Pydantic schemas judge output must validate against.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalguard.quality.judge_schemas import (
    BlindJudgeCase,
    ClaimVerdict,
    CorrectnessVerdict,
    FaithfulnessVerdict,
)


def test_blind_judge_case_holds_only_the_four_allowed_fields() -> None:
    case = BlindJudgeCase(
        question="What is the refund window?",
        context="Refunds are accepted within 30 days.",
        reference_answer="30 days",
        candidate_answer="You have 30 days to request a refund.",
    )
    assert set(case.model_dump().keys()) == {
        "question",
        "context",
        "reference_answer",
        "candidate_answer",
    }


def test_blind_judge_case_rejects_extra_fields() -> None:
    """The whole point of this model is that nothing else can be smuggled in —
    e.g. a prompt_version or model_id field would silently break blindness."""
    with pytest.raises(ValidationError):
        BlindJudgeCase(
            question="q",
            context="c",
            reference_answer="r",
            candidate_answer="a",
            prompt_version="v2",
        )


def test_correctness_verdict_accepts_valid_score() -> None:
    verdict = CorrectnessVerdict(score=0.8, verdict="partially_correct", rationale="close")
    assert verdict.score == 0.8


def test_correctness_verdict_rejects_score_above_one() -> None:
    with pytest.raises(ValidationError):
        CorrectnessVerdict(score=1.5, verdict="correct", rationale="x")


def test_correctness_verdict_rejects_score_below_zero() -> None:
    with pytest.raises(ValidationError):
        CorrectnessVerdict(score=-0.1, verdict="incorrect", rationale="x")


def test_correctness_verdict_rejects_unknown_verdict_label() -> None:
    with pytest.raises(ValidationError):
        CorrectnessVerdict(score=1.0, verdict="sort_of", rationale="x")


def test_claim_verdict_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        ClaimVerdict(claim="X applies", label="MAYBE")


def test_faithfulness_verdict_holds_a_list_of_claims() -> None:
    verdict = FaithfulnessVerdict(
        claims=[
            ClaimVerdict(claim="X applies", label="SUPPORTED"),
            ClaimVerdict(claim="Y also applies", label="CONTRADICTED"),
        ]
    )
    assert len(verdict.claims) == 2


def test_faithfulness_verdict_allows_empty_claims_list() -> None:
    verdict = FaithfulnessVerdict(claims=[])
    assert verdict.claims == []
