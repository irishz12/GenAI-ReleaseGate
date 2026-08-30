"""FakeClient must be deterministic offline: no network, same input -> same output."""

from __future__ import annotations

from evalguard.enums import FinishReason
from evalguard.providers.base import CompletionParams
from evalguard.providers.fake import FakeClient


def test_same_prompt_and_params_produce_identical_completion() -> None:
    client = FakeClient(seed=42)
    params = CompletionParams(temperature=0.0, max_tokens=256)

    first = client.complete("What is the refund window?", params)
    second = client.complete("What is the refund window?", params)

    assert first.text == second.text
    assert first == second


def test_different_prompt_produces_different_completion() -> None:
    client = FakeClient(seed=42)
    params = CompletionParams()

    a = client.complete("prompt A", params)
    b = client.complete("prompt B", params)

    assert a.text != b.text


def test_different_seed_produces_different_completion_for_same_prompt() -> None:
    params = CompletionParams()
    a = FakeClient(seed=1).complete("same prompt", params)
    b = FakeClient(seed=2).complete("same prompt", params)

    assert a.text != b.text


def test_different_params_produce_different_completion() -> None:
    client = FakeClient(seed=0)
    a = client.complete("same prompt", CompletionParams(temperature=0.0))
    b = client.complete("same prompt", CompletionParams(temperature=0.7))

    assert a.text != b.text


def test_completion_shape_is_well_formed() -> None:
    client = FakeClient(seed=0, latency_ms=7)
    completion = client.complete("hello there", CompletionParams())

    assert completion.finish_reason is FinishReason.STOP
    assert completion.usage.input_tokens >= 1
    assert completion.usage.output_tokens >= 1
    assert completion.latency_ms == 7
    assert completion.raw is not None and "digest" in completion.raw


def test_two_independent_client_instances_agree_given_same_seed() -> None:
    params = CompletionParams()
    a = FakeClient(seed=99).complete("consistency check", params)
    b = FakeClient(seed=99).complete("consistency check", params)

    assert a.text == b.text
