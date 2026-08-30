"""The Dataset Registry: loads and validates the generated evaluation-case files.

This module never touches Doc2Dial, HuggingFace, or the network — it only reads the
JSONL files and manifest that `scripts/build_doc2dial_dataset.py` already produced
under `data/`. That split matters: the corpus-derivation pipeline (network, one-time,
~2800 candidates down to 160 cases) is a separate concern from "load this eval suite
and tell me it's structurally sound", which must stay fast, offline, and side-effect
free so it can run in CI on every commit.

Holdout discipline: `load_cases(split=HOLDOUT)` is not gated in Phase 1 — there's no
promotion workflow yet to gate against (that's Phase 7). The convention starting now
is that holdout is read only for structural integrity checks (this module's own tests)
and, later, the final promotion decision — never to iterate on prompts or tune
anything. Nothing in this module inspects holdout *content* for that purpose.

Holdout is also local-only: `data/holdout/` is gitignored (see .gitignore), so it may
simply not exist on a given checkout — that is the intended state, not corruption.
`data/manifest.json` still commits its count, seed, corpus hash, and per-file hashes,
which is enough to (a) regenerate it byte-identically from the same cached raw corpus
and seed, or (b) verify it structurally if it does happen to be present locally,
without ever shipping the case content itself. `holdout_available()` and the
`holdout_available` field on `SplitIntegrityReport` make that distinction explicit
rather than letting an absent holdout silently read as "0 cases, no overlap, all good".
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evalguard.db.store import get_case_by_external_id, insert_case
from evalguard.enums import CaseCategory, CaseSplit
from evalguard.hashing import sha256_text
from evalguard.models import EvaluationCase

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = _REPO_ROOT / "data"


class DatasetManifest(BaseModel):
    """Provenance record for one generated evaluation suite (docs/ARCHITECTURE.md §6)."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    source_version: str
    source_url: str
    source_file_hashes: dict[str, str]
    corpus_hash: str
    seed: int
    generated_at: datetime
    category_counts: dict[str, int]
    split_counts: dict[str, int]
    category_split_counts: dict[str, dict[str, int]]
    case_file_hashes: dict[str, str]
    total_cases: int


class SplitIntegrityReport(BaseModel):
    """Result of `verify_split_integrity` — what was checked, not just pass/fail.

    `holdout_available` distinguishes "checked 40 holdout cases, found no overlap" from
    "holdout wasn't present on disk, so of course there was no overlap" — those are very
    different levels of assurance and collapsing them would be misleading.
    """

    model_config = ConfigDict(extra="forbid")

    dev_count: int
    holdout_count: int
    total_count: int
    holdout_available: bool
    overlapping_external_ids: list[str] = Field(default_factory=list)
    category_counts: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.overlapping_external_ids


class SplitIntegrityError(ValueError):
    """Raised when dev/holdout overlap or the on-disk data is otherwise inconsistent."""


def _case_file_path(data_dir: Path, split: CaseSplit, category: CaseCategory) -> Path:
    return data_dir / split.value / f"{category.value}.jsonl"


def load_case_file(path: str | Path) -> list[EvaluationCase]:
    """Parse one JSONL case file. Missing file -> empty list, not an error."""
    path = Path(path)
    if not path.exists():
        return []
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(EvaluationCase.model_validate_json(line))
    return cases


def load_cases(
    split: CaseSplit,
    *,
    category: CaseCategory | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[EvaluationCase]:
    """Load every case for a split, or just one category within it."""
    categories = [category] if category is not None else list(CaseCategory)
    cases: list[EvaluationCase] = []
    for cat in categories:
        cases.extend(load_case_file(_case_file_path(data_dir, split, cat)))
    return cases


def load_manifest(data_dir: Path = DEFAULT_DATA_DIR) -> DatasetManifest:
    manifest_path = data_dir / "manifest.json"
    return DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def holdout_available(data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """True only if every holdout category file is actually present on disk.

    Local-only by design (§ module docstring) — this is how a test or caller checks
    that *before* asserting on holdout counts/content, instead of getting a
    quietly-wrong "0 cases, no overlap" result from a missing directory.
    """
    return all(
        _case_file_path(data_dir, CaseSplit.HOLDOUT, category).exists() for category in CaseCategory
    )


def verify_split_integrity(data_dir: Path = DEFAULT_DATA_DIR) -> SplitIntegrityReport:
    """Structural checks only: no cross-split overlap, sane counts. Never reads holdout
    case *content* for anything beyond this — see module docstring.
    """
    dev = load_cases(CaseSplit.DEV, data_dir=data_dir)
    holdout = load_cases(CaseSplit.HOLDOUT, data_dir=data_dir)

    dev_ids = {c.external_id for c in dev}
    holdout_ids = {c.external_id for c in holdout}
    overlap = sorted(dev_ids & holdout_ids)

    category_counts: dict[str, int] = {}
    for case in dev + holdout:
        category_counts[case.category.value] = category_counts.get(case.category.value, 0) + 1

    report = SplitIntegrityReport(
        dev_count=len(dev),
        holdout_count=len(holdout),
        total_count=len(dev) + len(holdout),
        holdout_available=holdout_available(data_dir),
        overlapping_external_ids=overlap,
        category_counts=category_counts,
    )
    if not report.ok:
        raise SplitIntegrityError(
            f"{len(overlap)} external_id(s) appear in both dev and holdout: {overlap[:10]}"
        )
    return report


def compute_case_file_hash(path: str | Path) -> str:
    """Hash a case file's exact on-disk bytes — used to detect drift against the manifest."""
    return sha256_text(Path(path).read_text(encoding="utf-8"))


class ManifestVerificationReport(BaseModel):
    """Split into `missing` vs `hash_mismatches` on purpose: a missing holdout file is
    the *expected* state on a fresh checkout (gitignored, local-only). A hash mismatch on
    a file that IS present is real drift/corruption and must never happen for either
    split. Only `hash_mismatches` should ever fail a check; `missing` is informational.
    """

    model_config = ConfigDict(extra="forbid")

    missing: list[str] = Field(default_factory=list)
    hash_mismatches: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.hash_mismatches


def verify_manifest_hashes(data_dir: Path = DEFAULT_DATA_DIR) -> ManifestVerificationReport:
    """Check every case file the manifest declares against its recorded hash.

    A dev file should never be `missing` (dev is committed) and never `hash_mismatches`
    (that would mean it was hand-edited after generation). A holdout file being
    `missing` is fine and expected; a holdout file being in `hash_mismatches` is not —
    if it's present at all, its content must be exactly what the manifest attests to.
    """
    manifest = load_manifest(data_dir)
    missing: list[str] = []
    hash_mismatches: list[str] = []
    for rel_path, expected_hash in manifest.case_file_hashes.items():
        actual_path = data_dir / rel_path
        if not actual_path.exists():
            missing.append(rel_path)
            continue
        if compute_case_file_hash(actual_path) != expected_hash:
            hash_mismatches.append(rel_path)
    return ManifestVerificationReport(missing=missing, hash_mismatches=hash_mismatches)


def ensure_cases_registered(
    conn: sqlite3.Connection, cases: list[EvaluationCase]
) -> list[EvaluationCase]:
    """Idempotently sync JSONL-loaded cases into the `evaluation_cases` table.

    Cases from `load_cases()` always have `id=None` — the runner's `responses` table
    has a foreign key to `evaluation_cases(id)`, so a case must be registered (get a
    real DB id) before a run can reference it. Registering is by `external_id`: a case
    already present is fetched and returned as-is (its existing DB row, unmodified —
    `evaluation_cases` rows are never updated in place), never re-inserted or
    duplicated; a new one is inserted. Calling this twice with the same cases is a
    no-op the second time.
    """
    registered = []
    for case in cases:
        existing = get_case_by_external_id(conn, case.external_id)
        registered.append(existing if existing is not None else insert_case(conn, case))
    return registered
