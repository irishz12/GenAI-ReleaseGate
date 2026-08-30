"""The provider-agnostic interface every LLM client (generator or judge) implements.

Nothing above this module — the runner, quality engine, etc. — is allowed to know
whether it's talking to Bedrock Mantle or the offline `FakeClient`. The Bedrock Mantle
adapter (`bedrock_mantle.py`) lands in Phase 3; Phase 0 only defines the contract and
the offline implementation used by every test in this repo.

Kept synchronous for now: Phase 0 has no runner yet, and forcing async here today
would mean pulling in pytest-asyncio for no present benefit. The runner (Phase 2) is
free to run `LLMClient.complete` inside `asyncio.to_thread` for its bounded-concurrency
pool, or this Protocol can grow an async variant then — whichever the real Bedrock SDK
call shape ends up wanting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from evalguard.enums import FinishReason


class CompletionParams(BaseModel):
    """Inference parameters, mirroring config.ModelParams but owned by the provider layer."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    output_tokens: int


class Completion(BaseModel):
    """A single provider response, normalized across backends."""

    model_config = ConfigDict(extra="forbid")

    text: str
    finish_reason: FinishReason
    usage: Usage
    latency_ms: int
    raw: dict[str, Any] | None = None


@runtime_checkable
class LLMClient(Protocol):
    """One method. Both the generator and the judge are instances of this Protocol,
    pointed at different models/params — see docs/ARCHITECTURE.md §7.
    """

    def complete(self, prompt: str, params: CompletionParams) -> Completion: ...


@dataclass(frozen=True)
class GuardrailCheckResult:
    """One `check()` call's outcome, normalized across the real Bedrock client and
    `FakeGuardrailClient` — same role `Completion` plays for `LLMClient`."""

    intervened: bool
    reason: str
    output_text: str
    latency_ms: int
    content_policy_units: int
    sensitive_information_units_paid: int
    sensitive_information_units_free: int
    cost_usd: float | None


@runtime_checkable
class GuardrailClient(Protocol):
    """One method. Both `BedrockGuardrailClient` and `FakeGuardrailClient` are
    instances of this Protocol — the runner never knows which one it's holding."""

    def check(self, text: str, *, stage: Literal["input", "output"]) -> GuardrailCheckResult: ...
