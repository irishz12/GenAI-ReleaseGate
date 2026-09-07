"""Real LLMClient implementation for an OpenAI-compatible Bedrock Mantle endpoint.

Configuration is deliberately split in two places, per docs/ARCHITECTURE.md:
- Connection secrets (endpoint URL, API key) come from the environment (.env) via
  `MantleConfig.from_env()` — never from a function argument a caller could log.
- Model identity (which model to call) is not a secret and comes from
  config/models.yaml's `bedrock_mantle` section instead (see config.py).

This project has no live Bedrock Mantle deployment to test against, so
`MantleConfig.from_env()` falls back to `OPENAI_API_KEY` / the real OpenAI API base URL
when `MANTLE_API_KEY` / `MANTLE_BASE_URL` aren't set. The real OpenAI chat-completions
API *is* a genuine OpenAI-compatible endpoint, so this proves the client's HTTP,
retry, parsing, and cost-tracking logic end to end against a real network call — not a
substitute implementation, just a different (real) OpenAI-compatible backend than a
production Mantle deployment would be. Swapping to a real Mantle endpoint later is a
config change (MANTLE_BASE_URL + model id), not a code change.

Uses stdlib `urllib.request` rather than adding an HTTP dependency: one JSON POST, a
couple of headers, a JSON response — `httpx`/`requests` would add a dependency for
functionality this doesn't need.

Security: nothing in this module ever logs, prints, or includes the API key or request
headers in an exception message. `Completion.raw` stores the parsed *response* body
only — the request (which carries the Authorization header) is never captured.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from evalguard.enums import FinishReason
from evalguard.errors import PermanentError, TransientError
from evalguard.providers.base import Completion, CompletionParams, Usage

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# HTTP statuses worth retrying: rate limiting, request timeout/conflict, server errors.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# (status_code, response_body_bytes) on a completed HTTP exchange, however it went.
TransportResponse = tuple[int, bytes]
# (method, url, payload_bytes_or_None, headers, timeout_seconds) -> TransportResponse.
# Raises OSError/TimeoutError for connection-level failures (DNS, refused, timed out).
TransportFn = Callable[[str, str, bytes | None, dict, float], TransportResponse]


class MantleConfigError(PermanentError):
    """Required connection config is missing from the environment. Never includes the
    key/URL values themselves — just which variable names are missing."""


@dataclass(frozen=True)
class MantleConfig:
    base_url: str
    api_key: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    max_retries: int = _DEFAULT_MAX_RETRIES

    @classmethod
    def from_env(cls) -> MantleConfig:
        """Read connection config from the environment. `.env` is loaded via
        `evalguard.config`'s `load_dotenv()` call, which runs at import time — importing
        this module without also having imported `evalguard.config` first will only see
        real process environment variables, not `.env`'s contents.

        Falls back to `OPENAI_API_KEY` / the real OpenAI base URL when the Mantle-
        specific variables aren't set (see module docstring) — this is the one place
        that fallback lives, so a real Mantle deployment is a pure config change.
        """
        base_url = os.environ.get("MANTLE_BASE_URL", "").strip()
        api_key = os.environ.get("MANTLE_API_KEY", "").strip()

        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if api_key:
                warnings.warn(
                    "MANTLE_API_KEY not set — falling back to OPENAI_API_KEY. This "
                    "will make live, billed calls to the real OpenAI API, not Bedrock "
                    "Mantle, and results/reports will be attributed to Mantle even "
                    "though they came from OpenAI. Set MANTLE_API_KEY explicitly "
                    "(see .env.example) to use Mantle, or to silence this warning "
                    "if the OpenAI fallback is intentional for this run.",
                    stacklevel=2,
                )
            if not base_url:
                base_url = _DEFAULT_OPENAI_BASE_URL

        if not base_url or not api_key:
            raise MantleConfigError(
                "No Mantle (or fallback OpenAI) credentials found. Set MANTLE_BASE_URL "
                "and MANTLE_API_KEY (or OPENAI_API_KEY) in .env — see .env.example."
            )

        timeout = float(os.environ.get("MANTLE_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS))
        max_retries = int(os.environ.get("MANTLE_MAX_RETRIES", _DEFAULT_MAX_RETRIES))
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout_seconds=timeout,
            max_retries=max_retries,
        )


def _real_transport(
    method: str, url: str, payload: bytes | None, headers: dict, timeout: float
) -> TransportResponse:
    """The actual network call. Connection-level failures (DNS, refused, timeout)
    raise naturally; HTTP-level failures (4xx/5xx) are returned as a status code
    rather than raised, so `_request_with_retry`'s classification logic is identical
    whether it's looking at a real response or a test double's canned one."""
    request = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _full_jitter_backoff(attempt: int, *, cap_seconds: float = 20.0) -> float:
    """AWS's "full jitter" exponential backoff: uniform(0, min(cap, base * 2**attempt))."""
    return random.uniform(0, min(cap_seconds, 2**attempt))


def _map_finish_reason(raw: str | None) -> FinishReason:
    return {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
    }.get(raw or "", FinishReason.ERROR)


class BedrockMantleClient:
    """Implements the `LLMClient` protocol (structurally) against an OpenAI-compatible
    `/chat/completions` endpoint."""

    def __init__(
        self,
        model_id: str,
        config: MantleConfig | None = None,
        *,
        transport: TransportFn = _real_transport,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model_id = model_id
        self._config = config or MantleConfig.from_env()
        self._transport = transport
        self._sleep_fn = sleep_fn

    @property
    def base_url(self) -> str:
        """The endpoint being called — safe to print/log. Never exposes `api_key`;
        there is no public accessor for that on purpose."""
        return self._config.base_url

    def complete(self, prompt: str, params: CompletionParams) -> Completion:
        body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
        }
        data, latency_ms = self._request_with_retry("POST", "/chat/completions", body)

        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return Completion(
            # A reasoning model can exhaust max_tokens while still "thinking",
            # returning content: null (finish_reason usually "length") — its answer
            # never reached this field. That's a real, if degenerate, empty
            # completion, not a malformed response: coerce to "" rather than let
            # `None` reach Completion's own str-typed field and fail validation with
            # an opaque error far from the actual cause.
            text=choice["message"]["content"] or "",
            finish_reason=_map_finish_reason(choice.get("finish_reason")),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            latency_ms=latency_ms,
            raw=data,
        )

    def check_connectivity(self) -> Completion:
        """One minimal real call, used only by scripts/smoke_test_mantle.py to prove
        the endpoint/credentials work before spending on real eval cases."""
        return self.complete(
            "Reply with exactly one word: OK",
            CompletionParams(temperature=0.0, max_tokens=8, top_p=1.0),
        )

    def list_models(self) -> list[str]:
        """`GET /models` — the OpenAI-compatible model-listing endpoint. Used before
        any inference call to confirm a frozen model id is actually available on this
        deployment, rather than discovering that from a failed generation call.

        Assumes the standard OpenAI-compatible shape (`{"data": [{"id": "..."}, ...]}`).
        If Mantle's `/models` response doesn't match that shape, this returns an empty
        list rather than raising — the caller (the smoke test) treats "empty" the same
        as "id not found" and stops, which is the safe behavior either way.
        """
        data, _latency_ms = self._request_with_retry("GET", "/models")
        return [entry["id"] for entry in data.get("data", []) if "id" in entry]

    def _request_with_retry(
        self, method: str, path: str, body: dict | None = None
    ) -> tuple[dict, int]:
        url = f"{self._config.base_url}{path}"
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"

        attempts = max(1, self._config.max_retries)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            start = time.monotonic()
            try:
                status, body_bytes = self._transport(
                    method, url, payload, headers, self._config.timeout_seconds
                )
            except (OSError, TimeoutError) as exc:
                last_error = exc
                if attempt < attempts:
                    self._sleep_fn(_full_jitter_backoff(attempt))
                    continue
                raise TransientError(
                    f"Mantle request failed after {attempt} attempt(s): connection error"
                ) from exc

            latency_ms = int((time.monotonic() - start) * 1000)

            if status < 400:
                return json.loads(body_bytes.decode("utf-8")), latency_ms

            if status in _RETRYABLE_STATUS_CODES and attempt < attempts:
                last_error = RuntimeError(f"HTTP {status}")
                self._sleep_fn(_full_jitter_backoff(attempt))
                continue

            error_kind = TransientError if status in _RETRYABLE_STATUS_CODES else PermanentError
            raise error_kind(f"Mantle request failed: HTTP {status} after {attempt} attempt(s)")

        # Unreachable in practice (the loop always returns or raises), but keeps the
        # function's control flow provably total rather than implicitly falling off
        # the end if `attempts` were ever 0.
        raise TransientError(f"Mantle request failed: {last_error}")
