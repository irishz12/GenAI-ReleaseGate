"""evalguard's error taxonomy — two classes, per docs/ARCHITECTURE.md §11.

`classify_error` is what the runner calls when a provider call raises; it's the only
thing tying this taxonomy to a stored `ErrorClass` value. Phase 2 records the
classification but does not retry — `TransientError` exists now so the classification
is meaningful and testable, but actual retry/backoff is a Phase 3 concern once a real
network-bound provider (Bedrock Mantle) exists to make retrying worth the complexity.
"""

from __future__ import annotations

from evalguard.enums import ErrorClass


class EvalGuardError(Exception):
    """Base class for every error evalguard raises deliberately (not a bare Exception)."""


class TransientError(EvalGuardError):
    """A retryable failure: throttling, timeout, connection reset, 5xx, and similar."""


class PermanentError(EvalGuardError):
    """A non-retryable failure: validation, content filter, auth, and similar."""


def classify_error(exc: BaseException) -> ErrorClass:
    """Map any exception raised by a provider call onto the two-class taxonomy.

    Anything not explicitly a `TransientError` is treated as permanent — the safe
    default, since retrying an error we don't recognize risks masking a real bug
    behind repeated attempts rather than surfacing it.
    """
    if isinstance(exc, TransientError):
        return ErrorClass.TRANSIENT
    return ErrorClass.PERMANENT
