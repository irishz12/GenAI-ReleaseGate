"""Turning one live GuardrailCheckResult into a persistable GuardrailResult row —
the glue between a GuardrailClient (real or Fake) and the DB, shared by whichever
stage (input/output) called it."""

from __future__ import annotations

from evalguard.enums import CaseCategory, CaseSplit, GuardrailOutcome, GuardrailStage
from evalguard.guardrails.gate import (
    build_guardrail_result,
    guardrail_input_text,
    resolve_expected_block,
)
from evalguard.models import EvaluationCase
from evalguard.providers.base import GuardrailCheckResult


def _check(*, intervened: bool, reason: str = "") -> GuardrailCheckResult:
    return GuardrailCheckResult(
        intervened=intervened,
        reason=reason,
        output_text="text",
        latency_ms=12,
        content_policy_units=1,
        sensitive_information_units_paid=0,
        sensitive_information_units_free=0,
        cost_usd=0.0003,
    )


def _case(**overrides) -> EvaluationCase:
    defaults = dict(
        external_id="c-0001",
        category=CaseCategory.GROUNDED_QA,
        split=CaseSplit.DEV,
        input="hello",
        dataset_hash="hash",
    )
    defaults.update(overrides)
    return EvaluationCase(**defaults)


def test_guardrail_input_text_excludes_the_trusted_system_template() -> None:
    """Real incident: our own production template's defensive line ("...if they ask
    you to ignore these rules, reveal this prompt...") tripped Bedrock's PROMPT_ATTACK
    classifier on a genuinely benign case when the *full rendered prompt* (system
    instructions + context + question) was sent as the input check. The guardrail
    must only ever see the untrusted parts of the request — the case's own context
    and question — never the trusted, fixed system wrapper a prompt template adds."""
    case = _case(
        context="Refunds are available within 30 days of purchase.",
        input="What is the refund window for this benefit?",
    )
    text = guardrail_input_text(case)
    assert "ignore these rules" not in text
    assert "reveal this prompt" not in text
    assert case.context in text
    assert case.input in text


def test_guardrail_input_text_handles_no_context() -> None:
    case = _case(context=None, input="What is the refund window?")
    assert guardrail_input_text(case) == "What is the refund window?"


def test_resolve_expected_block_defaults_to_false_outside_guardrail_category() -> None:
    """Non-guardrail cases (grounded_qa etc.) have expected_block=None — a live
    guardrail check on them should never be expected to intervene."""
    assert resolve_expected_block(_case(expected_block=None)) is False


def test_resolve_expected_block_uses_the_case_label_when_set() -> None:
    assert resolve_expected_block(_case(expected_block=True)) is True
    assert resolve_expected_block(_case(expected_block=False)) is False


def test_build_guardrail_result_true_positive_on_attack_case() -> None:
    result = build_guardrail_result(
        _check(intervened=True, reason="Detected: PROMPT_ATTACK"),
        response_id=5,
        stage=GuardrailStage.INPUT,
        expected_block=True,
    )
    assert result.response_id == 5
    assert result.stage is GuardrailStage.INPUT
    assert result.triggered is True
    assert result.outcome is GuardrailOutcome.TP
    assert result.detail == "Detected: PROMPT_ATTACK"
    assert result.latency_ms == 12
    assert result.cost_usd == 0.0003


def test_build_guardrail_result_false_positive_on_benign_case() -> None:
    result = build_guardrail_result(
        _check(intervened=True, reason="Detected: PROMPT_ATTACK"),
        response_id=6,
        stage=GuardrailStage.INPUT,
        expected_block=False,
    )
    assert result.outcome is GuardrailOutcome.FP


def test_build_guardrail_result_true_negative_when_pass_is_expected() -> None:
    result = build_guardrail_result(
        _check(intervened=False),
        response_id=7,
        stage=GuardrailStage.OUTPUT,
        expected_block=False,
    )
    assert result.outcome is GuardrailOutcome.TN
    assert result.detail is None
