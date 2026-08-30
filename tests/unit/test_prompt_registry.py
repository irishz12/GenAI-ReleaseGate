"""Prompt Registry tests: create/load/version/hash for production and candidate prompts.

Uses the real template files under prompts/production and prompts/candidate — not
fixtures — since those files are themselves a Phase 1 deliverable.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from evalguard.enums import PromptRole
from evalguard.models import Prompt
from evalguard.registry.prompts import (
    DEFAULT_PROMPTS_DIR,
    ProductionConflictError,
    PromptContentDriftError,
    compute_prompt_hash,
    get_by_name_version,
    get_production,
    list_versions,
    load_prompt_from_file,
    register,
    register_from_file,
)

PRODUCTION_FILE = DEFAULT_PROMPTS_DIR / "production" / "support_agent.v1.md"
CANDIDATE_FILE = DEFAULT_PROMPTS_DIR / "candidate" / "support_agent.v2.md"
CANDIDATE_V3_FILE = DEFAULT_PROMPTS_DIR / "candidate" / "support_agent.v3.md"
CANDIDATE_V3_1_FILE = DEFAULT_PROMPTS_DIR / "candidate" / "support_agent.v3.1.md"
CANDIDATE_V3_2_FILE = DEFAULT_PROMPTS_DIR / "candidate" / "support_agent.v3.2.md"

# Phase 8.3: V1/V2 must never change while V3 is designed — pinning their exact
# content hashes turns "do not modify V1/V2" into an enforced invariant, not just an
# intention. If either ever changes (even whitespace), this fails immediately rather
# than silently invalidating every V1-vs-V2 comparison already on record.
V1_FROZEN_HASH = "ff3b009f288663eb52510fdf34c81e584f16825e772798ae5618eb30f2a37287"
V2_FROZEN_HASH = "97e18d73abb468dc900109c54992ae1cda38625bbe4fea9866f1dbf5b13755eb"
# Phase 9: V3 must not change either while V3.1 (its cost-only fix) is designed.
V3_FROZEN_HASH = "e8ad113c575904c6182df8767661e32d7c9ddcc5142b59f6564680a255a78184"
# Phase 9: V3.1 must not change while V3.2 (a looser-bound variant) is designed.
V3_1_FROZEN_HASH = "d0b60381ccc119833e996ad7c4842480a5e1c87c99d91aeeafc9bd8c439a6d46"


def test_prompt_template_files_exist() -> None:
    assert PRODUCTION_FILE.exists()
    assert CANDIDATE_FILE.exists()


def test_v3_prompt_file_exists() -> None:
    assert CANDIDATE_V3_FILE.exists()


def test_v1_and_v2_content_hashes_are_unchanged_by_v3_design(conn: sqlite3.Connection) -> None:
    v1 = load_prompt_from_file(
        PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )
    v2 = load_prompt_from_file(
        CANDIDATE_FILE, name="support_agent", version="v2", role=PromptRole.CANDIDATE
    )
    assert v1.content_hash == V1_FROZEN_HASH
    assert v2.content_hash == V2_FROZEN_HASH


def test_v3_is_genuinely_different_content_from_v1_and_v2() -> None:
    v3 = load_prompt_from_file(
        CANDIDATE_V3_FILE, name="support_agent", version="v3", role=PromptRole.CANDIDATE
    )
    assert v3.content_hash != V1_FROZEN_HASH
    assert v3.content_hash != V2_FROZEN_HASH


def test_v3_registers_as_a_candidate_prompt(conn: sqlite3.Connection) -> None:
    v3 = register_from_file(
        conn, CANDIDATE_V3_FILE, name="support_agent", version="v3", role=PromptRole.CANDIDATE
    )
    assert v3.id is not None
    assert v3.role is PromptRole.CANDIDATE
    assert v3.version == "v3"


def test_v3_1_prompt_file_exists() -> None:
    assert CANDIDATE_V3_1_FILE.exists()


def test_v3_content_hash_is_unchanged_by_v3_1_design(conn: sqlite3.Connection) -> None:
    v3 = load_prompt_from_file(
        CANDIDATE_V3_FILE, name="support_agent", version="v3", role=PromptRole.CANDIDATE
    )
    assert v3.content_hash == V3_FROZEN_HASH


def test_v3_1_is_genuinely_different_content_from_v1_v2_and_v3() -> None:
    v3_1 = load_prompt_from_file(
        CANDIDATE_V3_1_FILE, name="support_agent", version="v3.1", role=PromptRole.CANDIDATE
    )
    assert v3_1.content_hash not in (V1_FROZEN_HASH, V2_FROZEN_HASH, V3_FROZEN_HASH)


def test_v3_1_registers_as_a_candidate_prompt(conn: sqlite3.Connection) -> None:
    v3_1 = register_from_file(
        conn, CANDIDATE_V3_1_FILE, name="support_agent", version="v3.1", role=PromptRole.CANDIDATE
    )
    assert v3_1.id is not None
    assert v3_1.role is PromptRole.CANDIDATE
    assert v3_1.version == "v3.1"

    fetched = get_by_name_version(conn, "support_agent", "v3.1")
    assert fetched is not None
    assert fetched.content_hash == v3_1.content_hash


def test_v3_2_prompt_file_exists() -> None:
    assert CANDIDATE_V3_2_FILE.exists()


def test_v3_1_content_hash_is_unchanged_by_v3_2_design(conn: sqlite3.Connection) -> None:
    v3_1 = load_prompt_from_file(
        CANDIDATE_V3_1_FILE, name="support_agent", version="v3.1", role=PromptRole.CANDIDATE
    )
    assert v3_1.content_hash == V3_1_FROZEN_HASH


def test_v3_2_is_genuinely_different_content_from_earlier_versions() -> None:
    v3_2 = load_prompt_from_file(
        CANDIDATE_V3_2_FILE, name="support_agent", version="v3.2", role=PromptRole.CANDIDATE
    )
    assert v3_2.content_hash not in (
        V1_FROZEN_HASH,
        V2_FROZEN_HASH,
        V3_FROZEN_HASH,
        V3_1_FROZEN_HASH,
    )


def test_v3_2_registers_as_a_candidate_prompt(conn: sqlite3.Connection) -> None:
    v3_2 = register_from_file(
        conn, CANDIDATE_V3_2_FILE, name="support_agent", version="v3.2", role=PromptRole.CANDIDATE
    )
    assert v3_2.id is not None
    assert v3_2.role is PromptRole.CANDIDATE
    assert v3_2.version == "v3.2"

    fetched = get_by_name_version(conn, "support_agent", "v3.2")
    assert fetched is not None
    assert fetched.content_hash == v3_2.content_hash


def test_compute_prompt_hash_is_stable_and_content_sensitive() -> None:
    assert compute_prompt_hash("hello") == compute_prompt_hash("hello")
    assert compute_prompt_hash("hello") != compute_prompt_hash("hello!")


def test_load_prompt_from_file_computes_matching_hash() -> None:
    prompt = load_prompt_from_file(
        PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )
    assert prompt.content_hash == compute_prompt_hash(prompt.template)
    assert prompt.id is None  # not yet persisted


def test_register_from_file_persists_production_and_candidate(conn: sqlite3.Connection) -> None:
    production = register_from_file(
        conn, PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )
    candidate = register_from_file(
        conn, CANDIDATE_FILE, name="support_agent", version="v2", role=PromptRole.CANDIDATE
    )

    assert production.id is not None
    assert candidate.id is not None
    assert production.role is PromptRole.PRODUCTION
    assert candidate.role is PromptRole.CANDIDATE
    # v1 and v2 are genuinely different prompt text, so different hashes.
    assert production.content_hash != candidate.content_hash


def test_get_by_name_version_round_trips(conn: sqlite3.Connection) -> None:
    register_from_file(
        conn, PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )
    fetched = get_by_name_version(conn, "support_agent", "v1")
    assert fetched is not None
    assert fetched.role is PromptRole.PRODUCTION


def test_get_by_name_version_returns_none_when_missing(conn: sqlite3.Connection) -> None:
    assert get_by_name_version(conn, "nonexistent", "v1") is None


def test_registering_same_name_version_same_content_is_idempotent(
    conn: sqlite3.Connection,
) -> None:
    first = register_from_file(
        conn, PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )
    second = register_from_file(
        conn, PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )
    assert first.id == second.id  # no duplicate row


def test_registering_same_name_version_different_content_raises_drift_error(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    register_from_file(
        conn, PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )

    edited = tmp_path / "edited.md"
    edited.write_text(PRODUCTION_FILE.read_text() + "\nExtra unreviewed line.")

    with pytest.raises(PromptContentDriftError):
        register_from_file(
            conn, edited, name="support_agent", version="v1", role=PromptRole.PRODUCTION
        )


def test_registering_second_production_without_replace_flag_raises(
    conn: sqlite3.Connection,
) -> None:
    register_from_file(
        conn, PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )
    with pytest.raises(ProductionConflictError):
        register_from_file(
            conn,
            CANDIDATE_FILE,
            name="support_agent",
            version="v2",
            role=PromptRole.PRODUCTION,
        )


def test_promoting_candidate_to_production_archives_the_old_one(
    conn: sqlite3.Connection,
) -> None:
    v1 = register_from_file(
        conn, PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )
    v2 = register_from_file(
        conn,
        CANDIDATE_FILE,
        name="support_agent",
        version="v2",
        role=PromptRole.PRODUCTION,
        allow_replace_production=True,
    )

    assert get_production(conn, "support_agent").version == "v2"
    archived = get_by_name_version(conn, "support_agent", v1.version)
    assert archived.role is PromptRole.ARCHIVED
    assert v2.role is PromptRole.PRODUCTION


def test_list_versions_returns_all_registered_versions_for_a_name(
    conn: sqlite3.Connection,
) -> None:
    register_from_file(
        conn, PRODUCTION_FILE, name="support_agent", version="v1", role=PromptRole.PRODUCTION
    )
    register_from_file(
        conn, CANDIDATE_FILE, name="support_agent", version="v2", role=PromptRole.CANDIDATE
    )
    versions = {p.version for p in list_versions(conn, "support_agent")}
    assert versions == {"v1", "v2"}


def test_register_rejects_a_raw_prompt_model_directly(conn: sqlite3.Connection) -> None:
    """register() also works on a plain Prompt, not just via register_from_file."""
    prompt = Prompt(
        name="judge_correctness",
        version="v1",
        role=PromptRole.JUDGE,
        template="Grade this answer.",
        content_hash=compute_prompt_hash("Grade this answer."),
    )
    saved = register(conn, prompt)
    assert saved.id is not None
    assert get_by_name_version(conn, "judge_correctness", "v1").role is PromptRole.JUDGE
