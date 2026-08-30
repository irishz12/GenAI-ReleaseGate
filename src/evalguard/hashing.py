"""Stable, content-based hashing utilities.

Reproducibility (docs/ARCHITECTURE.md §12) depends on these being deterministic across
processes and Python versions: same content in, same hash out, every time. That rules out
anything keyed on object identity, dict insertion order, or float repr quirks — hence the
canonical-JSON approach below.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    """Hash a string as UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(obj: Any) -> str:
    """Serialize to JSON with sorted keys and no incidental whitespace.

    `sort_keys=True` makes the hash independent of dict/field insertion order.
    `default=str` handles values json.dumps can't natively serialize (e.g. Enum, datetime,
    Path) by falling back to their string form, so callers can hash Pydantic `model_dump()`
    output directly without pre-processing it.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_json(obj: Any) -> str:
    """Hash any JSON-serializable object (dict, list, Pydantic `model_dump()` output, ...)."""
    return sha256_text(_canonical_json(obj))


def hash_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file's raw bytes without loading the whole file into memory at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_hashes(*hashes: str) -> str:
    """Combine several independent hashes into one (e.g. a run's config_hash).

    Sorting first means the combined hash doesn't depend on the order the caller happened
    to pass the individual hashes in — only on the *set* of inputs.
    """
    joined = "|".join(sorted(hashes))
    return sha256_text(joined)
