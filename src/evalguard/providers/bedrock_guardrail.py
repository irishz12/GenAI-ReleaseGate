"""Real client for Amazon Bedrock's native ApplyGuardrail API.

Auth is deliberately different from BedrockMantleClient: ApplyGuardrail is a native
AWS Bedrock Runtime API requiring SigV4-signed requests, not a bearer token. This
client never reads an AWS access key or secret from .env or a parameter — boto3's
default credential chain (an ambient AWS CLI/SSO session, an instance role, etc.)
supplies it. `guardrail_id`/`guardrail_version` are account identifiers, not secrets;
they come from config/guardrails.yaml (see config.load_bedrock_guardrail_config).
"""

from __future__ import annotations

import time
from typing import Any, Literal, Protocol

import boto3

from evalguard.config import BedrockGuardrailPricing
from evalguard.providers.base import GuardrailCheckResult


class _BotoBedrockRuntimeClient(Protocol):
    def apply_guardrail(self, **kwargs: Any) -> dict[str, Any]: ...


def _compute_guardrail_cost(
    content_policy_units: int,
    sensitive_information_units_paid: int,
    sensitive_information_units_free: int,
    pricing: BedrockGuardrailPricing,
) -> float | None:
    """Never guesses: any missing price makes the total None, not a silently-partial
    sum — matching runner.compute_cost_usd's rule for generator/judge cost."""
    prices = (
        pricing.price_per_1k_content_policy_units,
        pricing.price_per_1k_sensitive_information_units_paid,
        pricing.price_per_1k_sensitive_information_units_free,
    )
    if any(p is None for p in prices):
        return None
    return (
        (content_policy_units / 1000) * prices[0]
        + (sensitive_information_units_paid / 1000) * prices[1]
        + (sensitive_information_units_free / 1000) * prices[2]
    )


class BedrockGuardrailClient:
    """Implements one method, `check`, against the real `bedrock-runtime`
    ApplyGuardrail API — or a fake boto3 client injected for tests."""

    def __init__(
        self,
        guardrail_id: str,
        guardrail_version: str,
        region: str,
        *,
        boto_client: _BotoBedrockRuntimeClient | None = None,
        pricing: BedrockGuardrailPricing | None = None,
    ) -> None:
        self._guardrail_id = guardrail_id
        self._guardrail_version = guardrail_version
        self._client = boto_client or boto3.client("bedrock-runtime", region_name=region)
        self._pricing = pricing or BedrockGuardrailPricing()

    def check(self, text: str, *, stage: Literal["input", "output"]) -> GuardrailCheckResult:
        source = "INPUT" if stage == "input" else "OUTPUT"
        start = time.monotonic()
        response = self._client.apply_guardrail(
            guardrailIdentifier=self._guardrail_id,
            guardrailVersion=self._guardrail_version,
            source=source,
            content=[{"text": {"text": text}}],
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        intervened = response.get("action") == "GUARDRAIL_INTERVENED"
        outputs = response.get("outputs") or []
        output_text = outputs[0]["text"] if outputs else text

        usage = response.get("usage") or {}
        content_units = usage.get("contentPolicyUnits", 0)
        sens_paid = usage.get("sensitiveInformationPolicyUnits", 0)
        sens_free = usage.get("sensitiveInformationPolicyFreeUnits", 0)

        cost = _compute_guardrail_cost(content_units, sens_paid, sens_free, self._pricing)

        return GuardrailCheckResult(
            intervened=intervened,
            reason=response.get("actionReason") or "",
            output_text=output_text,
            latency_ms=latency_ms,
            content_policy_units=content_units,
            sensitive_information_units_paid=sens_paid,
            sensitive_information_units_free=sens_free,
            cost_usd=cost,
        )
