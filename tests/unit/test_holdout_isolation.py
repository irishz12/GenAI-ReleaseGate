"""Holdout isolation tests: `data/holdout/` content must stay local-only and never be
committed; only its count, seed, hash, and integrity metadata may be — and that
metadata lives entirely in `data/manifest.json`, which IS committed.

This file tests the *policy*, not the data: it checks .gitignore actually excludes
holdout content while leaving dev untouched, and that the manifest's schema can never
smuggle case content in alongside the metadata. There's no git repository to run
`git check-ignore` against yet (this project predates `git init`), so the gitignore
check is a small, deliberately narrow pattern match against the two rules this project
actually relies on — not a general .gitignore parser.
"""

from __future__ import annotations

from pathlib import Path

from evalguard.registry.datasets import DEFAULT_DATA_DIR, holdout_available, load_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GITIGNORE = _REPO_ROOT / ".gitignore"

# Fields a case-content leak would show up under, if the manifest schema ever grew one.
_CONTENT_BEARING_KEYS = {
    "input",
    "context",
    "question",
    "reference_answer",
    "canary",
    "cases",
    "constraints",
}


def _is_directory_ignored(rel_dir: str) -> bool:
    """Minimal check: does any non-comment .gitignore line equal `rel_dir` with or
    without a trailing slash? Sufficient for this project's own two rules — not a
    general-purpose gitignore matcher.
    """
    rel_dir = rel_dir.rstrip("/")
    lines = [
        line.strip()
        for line in _GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return any(line.rstrip("/") == rel_dir for line in lines)


def test_gitignore_file_exists() -> None:
    assert _GITIGNORE.exists()


def test_gitignore_excludes_holdout_content_directory() -> None:
    assert _is_directory_ignored("data/holdout/"), (
        "data/holdout/ must be gitignored — its case content is local-only, never committed."
    )


def test_gitignore_does_not_exclude_dev_directory() -> None:
    assert not _is_directory_ignored("data/dev/"), (
        "data/dev/ must remain committable — development cases are usable/shippable."
    )


def test_gitignore_does_not_exclude_manifest() -> None:
    """The manifest (counts/seed/hashes only, no content) must stay committed even
    though holdout's content doesn't."""
    lines = [
        line.strip()
        for line in _GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("manifest.json" in line for line in lines)


# ─────────────────────────────────────────────────────────────────────────────
# manifest carries metadata only — never case content
# ─────────────────────────────────────────────────────────────────────────────


def _flatten_keys(obj, keys: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            _flatten_keys(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            _flatten_keys(item, keys)


def test_manifest_json_has_no_content_bearing_keys() -> None:
    """Structural safeguard: if a future change to the manifest schema ever started
    embedding case text (e.g. a 'sample_cases' field for debugging), this test would
    catch it before it got committed alongside a gitignored holdout directory."""
    import json

    raw = json.loads((DEFAULT_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    all_keys: set[str] = set()
    _flatten_keys(raw, all_keys)
    leaked = all_keys & _CONTENT_BEARING_KEYS
    assert not leaked, f"manifest.json contains content-bearing keys: {leaked}"


def test_manifest_only_contains_counts_seed_and_hashes() -> None:
    """Whitelist the manifest's top-level shape directly — count/seed/hash/provenance
    fields only, per the hardening requirement ("commit only its count, seed, hash,
    and integrity metadata")."""
    manifest = load_manifest()
    assert manifest.seed == 42
    assert isinstance(manifest.total_cases, int)
    assert isinstance(manifest.corpus_hash, str) and len(manifest.corpus_hash) == 64
    assert set(manifest.split_counts) == {"dev", "holdout"}
    assert all(isinstance(v, int) for v in manifest.split_counts.values())
    assert all(len(h) == 64 for h in manifest.case_file_hashes.values())
    assert all(len(h) == 64 for h in manifest.source_file_hashes.values())


# ─────────────────────────────────────────────────────────────────────────────
# holdout_available() reflects real on-disk state, not an assumption
# ─────────────────────────────────────────────────────────────────────────────


def test_holdout_available_matches_actual_file_presence() -> None:
    from evalguard.enums import CaseCategory, CaseSplit

    all_present = all(
        (DEFAULT_DATA_DIR / CaseSplit.HOLDOUT.value / f"{c.value}.jsonl").exists()
        for c in CaseCategory
    )
    assert holdout_available() == all_present


def test_holdout_available_is_false_for_a_directory_with_no_holdout(tmp_path: Path) -> None:
    """A data_dir with dev + manifest but no holdout/ must report unavailable, not error."""
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "grounded_qa.jsonl").write_text("", encoding="utf-8")
    assert holdout_available(tmp_path) is False


def test_holdout_available_requires_every_category_present(tmp_path: Path) -> None:
    """Partial holdout (some category files present, others missing) must not read as
    available — that would be a worse failure mode than "clearly absent"."""
    from evalguard.enums import CaseCategory

    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()
    (holdout_dir / f"{CaseCategory.GROUNDED_QA.value}.jsonl").write_text("", encoding="utf-8")
    assert holdout_available(tmp_path) is False
