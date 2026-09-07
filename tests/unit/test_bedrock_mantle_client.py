"""BedrockMantleClient tests — entirely offline via an injected fake transport.

No network access happens here (FakeClient stays the only thing CI/tests actually call
over "the wire", per the Phase 3 constraint to keep it unchanged). `_FakeTransport`
below is BedrockMantleClient's own test double: it returns scripted
(status_code, response_body_bytes) tuples exactly as `_real_transport` would, so the
retry/parsing/error-classification logic in `_post_with_retry` is exercised for real —
only the actual socket is swapped out.
"""

from __future__ import annotations

import json
import warnings

import pytest

from evalguard.enums import FinishReason
from evalguard.errors import PermanentError, TransientError
from evalguard.providers.base import CompletionParams
from evalguard.providers.bedrock_mantle import (
    BedrockMantleClient,
    MantleConfig,
    MantleConfigError,
    _full_jitter_backoff,
    _map_finish_reason,
)


def _openai_response(
    text: str, *, finish_reason: str = "stop", prompt_tokens=10, completion_tokens=4
) -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    ).encode("utf-8")


class _ScriptedTransport:
    """Returns one scripted (status, body) per call, in order. Raises OSError if the
    script is exhausted before instructed to, or if told to simulate a connection
    failure directly."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls: list[tuple[str, str, object, dict, float]] = []

    def __call__(self, method, url, payload, headers, timeout):
        self.calls.append((method, url, payload, headers, timeout))
        if not self._script:
            raise AssertionError("transport called more times than scripted")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _client(script: list, *, model_id="gpt-4o-mini", max_retries=3, sleep_calls=None):
    config = MantleConfig(
        base_url="https://example.test/v1", api_key="secret-key-xyz", max_retries=max_retries
    )
    transport = _ScriptedTransport(script)
    sleep_calls = sleep_calls if sleep_calls is not None else []
    client = BedrockMantleClient(
        model_id,
        config,
        transport=transport,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    return client, transport


# ─────────────────────────────────────────────────────────────────────────────
# happy path
# ─────────────────────────────────────────────────────────────────────────────


def test_complete_parses_text_tokens_and_finish_reason() -> None:
    client, transport = _client([(200, _openai_response("Paris is the capital of France."))])
    completion = client.complete("What is the capital of France?", CompletionParams())

    assert completion.text == "Paris is the capital of France."
    assert completion.finish_reason is FinishReason.STOP
    assert completion.usage.input_tokens == 10
    assert completion.usage.output_tokens == 4
    assert completion.latency_ms >= 0
    assert len(transport.calls) == 1


def test_complete_sends_model_and_params_in_request_body() -> None:
    client, transport = _client([(200, _openai_response("ok"))])
    client.complete("hello", CompletionParams(temperature=0.3, max_tokens=50, top_p=0.9))

    method, url, payload, headers, _ = transport.calls[0]
    assert method == "POST"
    assert url == "https://example.test/v1/chat/completions"
    body = json.loads(payload.decode("utf-8"))
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 50
    assert body["top_p"] == 0.9
    assert headers["Authorization"] == "Bearer secret-key-xyz"
    assert headers["Content-Type"] == "application/json"


def test_check_connectivity_makes_exactly_one_call() -> None:
    client, transport = _client([(200, _openai_response("OK"))])
    completion = client.check_connectivity()
    assert completion.text == "OK"
    assert len(transport.calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# list_models — the pre-inference availability check
# ─────────────────────────────────────────────────────────────────────────────


def _models_response(ids: list[str]) -> bytes:
    payload = {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}
    return json.dumps(payload).encode()


def test_list_models_returns_ids_from_openai_shaped_response() -> None:
    client, transport = _client(
        [(200, _models_response(["qwen.qwen3-next-80b-a3b-instruct", "openai.gpt-oss-120b"]))]
    )
    models = client.list_models()
    assert models == ["qwen.qwen3-next-80b-a3b-instruct", "openai.gpt-oss-120b"]


def test_list_models_sends_a_get_request_with_no_body() -> None:
    client, transport = _client([(200, _models_response(["m1"]))])
    client.list_models()

    method, url, payload, headers, _ = transport.calls[0]
    assert method == "GET"
    assert url == "https://example.test/v1/models"
    assert payload is None
    assert "Content-Type" not in headers  # no body -> no content-type header
    assert headers["Authorization"] == "Bearer secret-key-xyz"


def test_list_models_returns_empty_list_for_unexpected_response_shape() -> None:
    client, _ = _client([(200, json.dumps({"unexpected": "shape"}).encode())])
    assert client.list_models() == []


def test_list_models_raises_permanent_error_on_401() -> None:
    client, transport = _client([(401, b"unauthorized")])
    with pytest.raises(PermanentError):
        client.list_models()
    assert len(transport.calls) == 1


def test_list_models_retries_on_429() -> None:
    client, transport = _client([(429, b"rate limited"), (200, _models_response(["m1"]))])
    models = client.list_models()
    assert models == ["m1"]
    assert len(transport.calls) == 2


def test_complete_coerces_null_content_to_empty_string() -> None:
    """A reasoning model can exhaust max_tokens while still 'thinking', returning
    `content: null` with `finish_reason: length` — its answer never reached the
    content field. This is a real (if degenerate) empty completion, not something
    that should crash deep inside Completion's own pydantic validation with an
    opaque error; downstream (e.g. judge JSON parsing) already has a real, specific
    error path for empty/malformed output."""
    body = json.dumps(
        {
            "choices": [{"message": {"content": None}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 512},
        }
    ).encode("utf-8")
    client, _ = _client([(200, body)])
    completion = client.complete("summarize this", CompletionParams())

    assert completion.text == ""
    assert completion.finish_reason is FinishReason.LENGTH


def test_finish_reason_length_is_mapped() -> None:
    client, _ = _client([(200, _openai_response("truncated...", finish_reason="length"))])
    completion = client.complete("x", CompletionParams())
    assert completion.finish_reason is FinishReason.LENGTH


def test_unrecognized_finish_reason_maps_to_error() -> None:
    client, _ = _client([(200, _openai_response("blocked", finish_reason="content_filter"))])
    completion = client.complete("x", CompletionParams())
    assert completion.finish_reason is FinishReason.ERROR


# ─────────────────────────────────────────────────────────────────────────────
# retry / backoff on transient failures
# ─────────────────────────────────────────────────────────────────────────────


def test_retries_on_429_then_succeeds() -> None:
    sleep_calls = []
    client, transport = _client(
        [(429, b"rate limited"), (200, _openai_response("ok"))], sleep_calls=sleep_calls
    )
    completion = client.complete("x", CompletionParams())
    assert completion.text == "ok"
    assert len(transport.calls) == 2
    assert len(sleep_calls) == 1  # backed off exactly once, between the two attempts


def test_retries_on_503_then_succeeds() -> None:
    client, transport = _client([(503, b"unavailable"), (200, _openai_response("ok"))])
    client.complete("x", CompletionParams())
    assert len(transport.calls) == 2


def test_exhausting_retries_on_persistent_429_raises_transient_error() -> None:
    client, transport = _client([(429, b"1"), (429, b"2"), (429, b"3")], max_retries=3)
    with pytest.raises(TransientError):
        client.complete("x", CompletionParams())
    assert len(transport.calls) == 3


def test_connection_level_failure_is_retried_then_raises_transient() -> None:
    client, transport = _client(
        [OSError("connection refused"), OSError("connection refused")], max_retries=2
    )
    with pytest.raises(TransientError):
        client.complete("x", CompletionParams())
    assert len(transport.calls) == 2


def test_backoff_is_only_called_between_attempts_not_after_final_failure() -> None:
    sleep_calls = []
    client, transport = _client([(429, b"1"), (429, b"2")], max_retries=2, sleep_calls=sleep_calls)
    with pytest.raises(TransientError):
        client.complete("x", CompletionParams())
    assert len(sleep_calls) == 1  # one sleep between attempt 1 and 2, none after 2


# ─────────────────────────────────────────────────────────────────────────────
# permanent (non-retryable) failures
# ─────────────────────────────────────────────────────────────────────────────


def test_401_raises_permanent_error_without_retrying() -> None:
    client, transport = _client([(401, b"unauthorized")], max_retries=3)
    with pytest.raises(PermanentError):
        client.complete("x", CompletionParams())
    assert len(transport.calls) == 1  # no retry on a 401


def test_400_raises_permanent_error_without_retrying() -> None:
    client, transport = _client([(400, b"bad request")], max_retries=3)
    with pytest.raises(PermanentError):
        client.complete("x", CompletionParams())
    assert len(transport.calls) == 1


def test_permanent_error_message_never_contains_the_api_key() -> None:
    """Security requirement: never leak the key, including through exception text."""
    client, _ = _client([(401, b"unauthorized")])
    with pytest.raises(PermanentError) as exc_info:
        client.complete("x", CompletionParams())
    assert "secret-key-xyz" not in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# config / security
# ─────────────────────────────────────────────────────────────────────────────


def test_mantle_config_error_message_never_contains_key_or_url_values(monkeypatch) -> None:
    monkeypatch.delenv("MANTLE_BASE_URL", raising=False)
    monkeypatch.delenv("MANTLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MantleConfigError) as exc_info:
        MantleConfig.from_env()
    message = str(exc_info.value)
    assert "MANTLE_BASE_URL" in message  # names the missing variable...
    assert "MANTLE_API_KEY" in message
    # ...but there's no secret value to leak here since none was ever set.


def test_mantle_config_falls_back_to_openai_api_key(monkeypatch) -> None:
    monkeypatch.delenv("MANTLE_BASE_URL", raising=False)
    monkeypatch.delenv("MANTLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fallback")

    with pytest.warns(UserWarning, match="MANTLE_API_KEY not set"):
        config = MantleConfig.from_env()
    assert config.api_key == "sk-test-fallback"
    assert config.base_url == "https://api.openai.com/v1"


def test_mantle_config_fallback_warning_never_contains_the_key_value(monkeypatch) -> None:
    monkeypatch.delenv("MANTLE_BASE_URL", raising=False)
    monkeypatch.delenv("MANTLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")

    with pytest.warns(UserWarning) as record:
        MantleConfig.from_env()
    assert len(record) == 1
    assert "sk-super-secret-value" not in str(record[0].message)


def test_mantle_config_prefers_explicit_mantle_vars_over_openai_fallback(monkeypatch) -> None:
    monkeypatch.setenv("MANTLE_BASE_URL", "https://mantle.internal/v1")
    monkeypatch.setenv("MANTLE_API_KEY", "mantle-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = MantleConfig.from_env()  # must NOT warn when a real Mantle key is set
    assert config.api_key == "mantle-key"
    assert config.base_url == "https://mantle.internal/v1"


def test_repr_of_client_does_not_expose_api_key() -> None:
    """A stray print(client) or log of the object itself must not leak the key."""
    config = MantleConfig(base_url="https://example.test/v1", api_key="super-secret-value")
    client = BedrockMantleClient("gpt-4o-mini", config)
    assert "super-secret-value" not in repr(client)
    assert "super-secret-value" not in str(client)


# ─────────────────────────────────────────────────────────────────────────────
# pure helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_full_jitter_backoff_is_bounded_and_nonnegative() -> None:
    for attempt in range(1, 6):
        value = _full_jitter_backoff(attempt, cap_seconds=20.0)
        assert 0 <= value <= 20.0


def test_map_finish_reason_handles_none() -> None:
    assert _map_finish_reason(None) is FinishReason.ERROR
