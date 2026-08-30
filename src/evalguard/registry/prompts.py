"""The Prompt Registry: create/load/version/hash production and candidate prompts.

Prompts are identified by content hash, not by version string (docs/ARCHITECTURE.md
§12) — a file edited without a version bump is refused, not silently accepted. "Exactly
one production prompt per name" is enforced here, at the registry layer, rather than as
a DB constraint — see Phase 0's simplified schema, which deliberately doesn't carry a
partial unique index for it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from evalguard.db.store import get_prompt, insert_prompt
from evalguard.enums import PromptRole
from evalguard.hashing import sha256_text
from evalguard.models import Prompt

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS_DIR = _REPO_ROOT / "prompts"


class PromptContentDriftError(ValueError):
    """A (name, version) pair already exists with different content.

    This is the reproducibility guard: version strings are for humans, but the content
    hash is the thing that actually gets pinned into a run's manifest. If the file
    changed without a version bump, the two hashes disagree and this fires — the fix
    is to bump the version, not to silence the error.
    """


class ProductionConflictError(ValueError):
    """Registering a second production prompt for a name without explicitly replacing it."""


def compute_prompt_hash(template: str) -> str:
    return sha256_text(template)


def load_prompt_from_file(path: str | Path, *, name: str, version: str, role: PromptRole) -> Prompt:
    """Read a prompt template off disk and build the (not-yet-persisted) contract.

    Pure — does not touch the database. `register()` is the DB-aware half.
    """
    template = Path(path).read_text(encoding="utf-8")
    return Prompt(
        name=name,
        version=version,
        role=role,
        template=template,
        content_hash=compute_prompt_hash(template),
    )


def get_by_name_version(conn: sqlite3.Connection, name: str, version: str) -> Prompt | None:
    row = conn.execute(
        "SELECT id FROM prompts WHERE name = ? AND version = ?", (name, version)
    ).fetchone()
    if row is None:
        return None
    return get_prompt(conn, row["id"])


def get_production(conn: sqlite3.Connection, name: str) -> Prompt | None:
    row = conn.execute(
        "SELECT id FROM prompts WHERE name = ? AND role = ?", (name, PromptRole.PRODUCTION.value)
    ).fetchone()
    if row is None:
        return None
    return get_prompt(conn, row["id"])


def list_versions(conn: sqlite3.Connection, name: str) -> list[Prompt]:
    rows = conn.execute(
        "SELECT id FROM prompts WHERE name = ? ORDER BY version", (name,)
    ).fetchall()
    return [get_prompt(conn, row["id"]) for row in rows]


def register(
    conn: sqlite3.Connection, prompt: Prompt, *, allow_replace_production: bool = False
) -> Prompt:
    """Insert a prompt, enforcing content-hash stability and production uniqueness.

    Behavior:
    - Same (name, version) already registered, same hash -> idempotent, returns the
      existing row (no duplicate insert).
    - Same (name, version), different hash -> `PromptContentDriftError`.
    - New row with role=PRODUCTION while a *different* production prompt already
      exists for `name` -> `ProductionConflictError`, unless
      `allow_replace_production=True`, in which case the old one is demoted to
      ARCHIVED first.
    """
    existing = get_by_name_version(conn, prompt.name, prompt.version)
    if existing is not None:
        if existing.content_hash == prompt.content_hash:
            return existing
        raise PromptContentDriftError(
            f"{prompt.name}@{prompt.version} already registered with a different hash "
            f"({existing.content_hash[:12]} != {prompt.content_hash[:12]}) — "
            "bump the version instead of editing an existing one in place."
        )

    if prompt.role is PromptRole.PRODUCTION:
        current_production = get_production(conn, prompt.name)
        if current_production is not None and current_production.version != prompt.version:
            if not allow_replace_production:
                raise ProductionConflictError(
                    f"{prompt.name}@{current_production.version} is already production; "
                    f"pass allow_replace_production=True to promote {prompt.version} over it."
                )
            _archive(conn, current_production)

    return insert_prompt(conn, prompt)


def _archive(conn: sqlite3.Connection, prompt: Prompt) -> None:
    conn.execute("UPDATE prompts SET role = ? WHERE id = ?", (PromptRole.ARCHIVED.value, prompt.id))
    conn.commit()


def register_from_file(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    name: str,
    version: str,
    role: PromptRole,
    allow_replace_production: bool = False,
) -> Prompt:
    """Convenience wrapper: load from disk, then register in one call."""
    prompt = load_prompt_from_file(path, name=name, version=version, role=role)
    return register(conn, prompt, allow_replace_production=allow_replace_production)
