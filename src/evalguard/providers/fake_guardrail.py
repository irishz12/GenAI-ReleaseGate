"""Deterministic offline stand-in for BedrockGuardrailClient — no AWS calls.

Triggers on the same injection templates and canary format evalguard's own
guardrail dev cases use (see registry/case_builder.py), so tests exercise real
pass/intervene behavior without any network access. Same role for the guardrail
layer that FakeClient plays for the generator.
"""

from __future__ import annotations

import re
from typing import Literal

from evalguard.providers.base import GuardrailCheckResult

_INJECTION_MARKER = re.compile(
    r"(?i)ignore (all|the) (previous|above) instructions"
    r"|\bpwned\b"
    r"|system override"
    r"|reveal your (full )?system prompt"
    r"|tell me your (original|initial) instructions"
    r"|print your configuration"
)
_CANARY_PATTERN = re.compile(r"__CANARY_\d{4}_\d{6}__")

_BLOCKED_INPUT_MESSAGE = "I can't process that request."
_BLOCKED_OUTPUT_MESSAGE = "I can't share that information."


class FakeGuardrailClient:
    def __init__(self, *, latency_ms: int = 5) -> None:
        self._latency_ms = latency_ms

    def check(self, text: str, *, stage: Literal["input", "output"]) -> GuardrailCheckResult:
        reason = self._match(text)
        if reason is not None:
            blocked_message = (
                _BLOCKED_INPUT_MESSAGE if stage == "input" else _BLOCKED_OUTPUT_MESSAGE
            )
            return GuardrailCheckResult(
                intervened=True,
                reason=reason,
                output_text=blocked_message,
                latency_ms=self._latency_ms,
                content_policy_units=1,
                sensitive_information_units_paid=0,
                sensitive_information_units_free=0,
                cost_usd=0.0,
            )
        return GuardrailCheckResult(
            intervened=False,
            reason="",
            output_text=text,
            latency_ms=self._latency_ms,
            content_policy_units=1,
            sensitive_information_units_paid=0,
            sensitive_information_units_free=0,
            cost_usd=0.0,
        )

    def _match(self, text: str) -> str | None:
        if _CANARY_PATTERN.search(text):
            return "Detected: PII (canary secret)"
        if _INJECTION_MARKER.search(text):
            return "Detected: PROMPT_ATTACK"
        return None
