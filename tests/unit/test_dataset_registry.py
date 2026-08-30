"""Dataset Registry tests: load the real generated data/ files, check counts, split
integrity, and manifest consistency.

This module reads the actual committed data/dev + data/holdout + manifest.json —
produced once by scripts/build_doc2dial_dataset.py from the real Doc2Dial corpus — not
a fixture. No network access happens here; that's the point of the registry/pipeline
split (see registry/datasets.py's module docstring).

Holdout isolation: `data/holdout/` is gitignored (local-only, never committed — see
.gitignore and registry/datasets.py's module docstring). It is present on disk in this
checkout because this session generated it, so the holdout-content assertions below run
and pass here — but they're all guarded with `_HOLDOUT_AVAILABLE` so a future fresh
clone (where holdout genuinely doesn't exist) skips them cleanly instead of failing.
Dev-only assertions always run unconditionally, since dev is the committed deliverable.
"""

from __future__ import annotations

import pytest

from evalguard.enums import AttackType, CaseCategory, CaseSplit
from evalguard.models import EvaluationCase
from evalguard.registry.datasets import (
    DEFAULT_DATA_DIR,
    compute_case_file_hash,
    holdout_available,
    load_cases,
    load_manifest,
    verify_manifest_hashes,
    verify_split_integrity,
)

_HOLDOUT_AVAILABLE = holdout_available()
_requires_holdout = pytest.mark.skipif(
    not _HOLDOUT_AVAILABLE,
    reason="data/holdout/ is local-only and gitignored; absent in this checkout",
)


def test_data_dir_exists() -> None:
    assert DEFAULT_DATA_DIR.exists()
    assert (DEFAULT_DATA_DIR / "manifest.json").exists()
    assert (DEFAULT_DATA_DIR / "dev").exists()


# ─────────────────────────────────────────────────────────────────────────────
# counts — dev-only versions always run; combined dev+holdout versions are guarded
# ─────────────────────────────────────────────────────────────────────────────


def test_dev_case_count_is_120() -> None:
    assert len(load_cases(CaseSplit.DEV)) == 120


def test_dev_category_counts_match_target() -> None:
    expected = {
        CaseCategory.GROUNDED_QA: 60,
        CaseCategory.ABSTENTION: 15,
        CaseCategory.INSTRUCTION_FOLLOWING: 15,
        CaseCategory.GUARDRAIL: 30,
    }
    for category, expected_count in expected.items():
        assert len(load_cases(CaseSplit.DEV, category=category)) == expected_count, category


def test_dev_guardrail_includes_benign_near_misses() -> None:
    """The architecture explicitly requires benign near-misses — without them
    prompt_injection_block_rate is gameable by blocking everything. Dev-only, since
    this is exactly the kind of thing that must be checkable without holdout."""
    cases = load_cases(CaseSplit.DEV, category=CaseCategory.GUARDRAIL)
    by_type = {t: 0 for t in AttackType}
    for c in cases:
        by_type[c.attack_type] += 1
    assert by_type[AttackType.BENIGN] > 0
    assert by_type[AttackType.INJECTION] > 0
    assert by_type[AttackType.PII_EXTRACTION] > 0
    assert all(c.expected_block is False for c in cases if c.attack_type is AttackType.BENIGN)
    assert all(c.expected_block is True for c in cases if c.attack_type is not AttackType.BENIGN)


def test_dev_external_ids_are_unique_within_each_category() -> None:
    for category in CaseCategory:
        ids = [c.external_id for c in load_cases(CaseSplit.DEV, category=category)]
        assert len(ids) == len(set(ids)), f"duplicate external_id in {category}"


def test_dev_every_loaded_case_is_a_valid_evaluation_case() -> None:
    for case in load_cases(CaseSplit.DEV):
        assert isinstance(case, EvaluationCase)
        assert case.split is CaseSplit.DEV
        assert case.external_id
        assert case.input


@_requires_holdout
def test_total_case_count_is_160() -> None:
    dev = load_cases(CaseSplit.DEV)
    holdout = load_cases(CaseSplit.HOLDOUT)
    assert len(dev) + len(holdout) == 160


@_requires_holdout
def test_category_counts_match_target() -> None:
    expected = {
        CaseCategory.GROUNDED_QA: 80,
        CaseCategory.ABSTENTION: 20,
        CaseCategory.INSTRUCTION_FOLLOWING: 20,
        CaseCategory.GUARDRAIL: 40,
    }
    for category, expected_count in expected.items():
        dev = load_cases(CaseSplit.DEV, category=category)
        holdout = load_cases(CaseSplit.HOLDOUT, category=category)
        assert len(dev) + len(holdout) == expected_count, category


@_requires_holdout
def test_split_counts_match_target() -> None:
    assert len(load_cases(CaseSplit.DEV)) == 120
    assert len(load_cases(CaseSplit.HOLDOUT)) == 40


@_requires_holdout
def test_every_loaded_case_is_a_valid_evaluation_case() -> None:
    for split in (CaseSplit.DEV, CaseSplit.HOLDOUT):
        for case in load_cases(split):
            assert isinstance(case, EvaluationCase)
            assert case.split is split
            assert case.external_id
            assert case.input


@_requires_holdout
def test_external_ids_are_unique_within_each_category() -> None:
    for category in CaseCategory:
        ids = [
            c.external_id
            for c in load_cases(CaseSplit.DEV, category=category)
            + load_cases(CaseSplit.HOLDOUT, category=category)
        ]
        assert len(ids) == len(set(ids)), f"duplicate external_id in {category}"


# ─────────────────────────────────────────────────────────────────────────────
# split integrity — the primary "no overlap" guarantee, verified against real data
# ─────────────────────────────────────────────────────────────────────────────


def test_verify_split_integrity_reports_holdout_availability_honestly() -> None:
    """This must always run (holdout present or not) — it's the test that would catch
    a future regression where the report claims "ok" without actually having checked
    anything against holdout."""
    report = verify_split_integrity()
    assert report.holdout_available == _HOLDOUT_AVAILABLE
    if not _HOLDOUT_AVAILABLE:
        assert report.holdout_count == 0


@_requires_holdout
def test_verify_split_integrity_reports_no_overlap() -> None:
    report = verify_split_integrity()
    assert report.ok
    assert report.holdout_available is True
    assert report.overlapping_external_ids == []
    assert report.dev_count == 120
    assert report.holdout_count == 40
    assert report.total_count == 160


@_requires_holdout
def test_dev_and_holdout_external_ids_are_fully_disjoint() -> None:
    dev_ids = {c.external_id for c in load_cases(CaseSplit.DEV)}
    holdout_ids = {c.external_id for c in load_cases(CaseSplit.HOLDOUT)}
    assert dev_ids.isdisjoint(holdout_ids)


# ─────────────────────────────────────────────────────────────────────────────
# manifest — always runs; manifest itself is committed regardless of holdout presence
# ─────────────────────────────────────────────────────────────────────────────


def test_manifest_parses_and_matches_real_counts() -> None:
    manifest = load_manifest()
    assert manifest.source_name == "doc2dial"
    assert manifest.source_version == "v1.0.1"
    assert manifest.total_cases == 160
    assert manifest.split_counts == {"dev": 120, "holdout": 40}
    assert manifest.category_counts == {
        "grounded_qa": 80,
        "abstention": 20,
        "instruction_following": 20,
        "guardrail": 40,
    }
    assert manifest.seed == 42


def test_manifest_corpus_hash_is_a_combination_of_source_file_hashes() -> None:
    from evalguard.hashing import combine_hashes

    manifest = load_manifest()
    expected = combine_hashes(*manifest.source_file_hashes.values())
    assert manifest.corpus_hash == expected


def test_manifest_declares_hashes_for_both_splits_of_every_category() -> None:
    """The manifest must commit holdout's hash/count metadata even though the holdout
    *content* files themselves are gitignored — this is the "count, seed, hash, and
    integrity metadata only" contract, checked structurally rather than by eye."""
    manifest = load_manifest()
    for category in CaseCategory:
        for split in ("dev", "holdout"):
            rel_path = f"{split}/{category.value}.jsonl"
            assert rel_path in manifest.case_file_hashes
            assert len(manifest.case_file_hashes[rel_path]) == 64  # sha256 hex digest
        assert category.value in manifest.category_split_counts


def test_manifest_dev_case_carries_the_manifest_corpus_hash() -> None:
    manifest = load_manifest()
    for case in load_cases(CaseSplit.DEV):
        assert case.dataset_hash == manifest.corpus_hash


@_requires_holdout
def test_manifest_every_case_carries_the_manifest_corpus_hash() -> None:
    manifest = load_manifest()
    for split in (CaseSplit.DEV, CaseSplit.HOLDOUT):
        for case in load_cases(split):
            assert case.dataset_hash == manifest.corpus_hash


def test_manifest_case_file_hashes_match_the_files_on_disk() -> None:
    """The drift check: if a *present* case file is hand-edited after generation, its
    hash no longer matches the manifest, and this test catches it. A holdout file being
    absent is expected (gitignored) and reported separately under `missing`, never
    under `hash_mismatches` — only the latter is a real integrity failure.
    """
    report = verify_manifest_hashes()
    assert report.ok
    assert report.hash_mismatches == []


def test_dev_case_files_are_never_reported_missing() -> None:
    """Dev is the committed deliverable — it must always be present and correct,
    holdout's local-only status notwithstanding."""
    report = verify_manifest_hashes()
    dev_missing = [p for p in report.missing if p.startswith("dev/")]
    assert dev_missing == []


def test_compute_case_file_hash_matches_manifest_entry_directly() -> None:
    manifest = load_manifest()
    path = DEFAULT_DATA_DIR / "dev" / "grounded_qa.jsonl"
    assert compute_case_file_hash(path) == manifest.case_file_hashes["dev/grounded_qa.jsonl"]


def test_manifest_category_split_counts_sum_to_split_counts() -> None:
    manifest = load_manifest()
    dev_sum = sum(counts["dev"] for counts in manifest.category_split_counts.values())
    holdout_sum = sum(counts["holdout"] for counts in manifest.category_split_counts.values())
    assert dev_sum == manifest.split_counts["dev"]
    assert holdout_sum == manifest.split_counts["holdout"]
