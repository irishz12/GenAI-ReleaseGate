"""Deterministic offline LLM client for tests and CI — no network, no cost, no credentials.

Same seed + same prompt + same params always produces the same completion. That
determinism is the point: it lets the runner, quality engine, and regression engine be
tested end to end without ever calling Bedrock Mantle, and it lets a test assert on an
exact output rather than just "some string came back".
"""

from __future__ import annotations

from evalguard.enums import FinishReason
from evalguard.hashing import sha256_json
from evalguard.providers.base import Completion, CompletionParams, Usage


class FakeClient:
    """Implements the `LLMClient` protocol (structurally — no inheritance needed)."""

    def __init__(self, seed: int = 0, latency_ms: int = 5) -> None:
        self.seed = seed
        self.latency_ms = latency_ms

    def complete(self, prompt: str, params: CompletionParams) -> Completion:
        digest = sha256_json({"seed": self.seed, "prompt": prompt, "params": params.model_dump()})
        text = f"[fake:{digest[:12]}]"

        return Completion(
            text=text,
            finish_reason=FinishReason.STOP,
            usage=Usage(
                input_tokens=max(1, len(prompt.split())),
                output_tokens=max(1, len(text.split())),
            ),
            latency_ms=self.latency_ms,
            raw={"digest": digest},
        )
