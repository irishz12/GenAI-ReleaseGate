#!/usr/bin/env python3
"""One-time corpus-derivation pipeline: real Doc2Dial -> data/dev + data/holdout + manifest.

This script is NOT part of the installed `evalguard` package and is not imported by any
runtime code — it's the thing that produced the committed case files under `data/`, and
it's re-runnable if the corpus source or case-building logic ever changes. Everything
downstream (the Dataset Registry, tests, later phases) only ever reads the JSONL/manifest
output of this script — never Doc2Dial directly.

Usage:
    python scripts/build_doc2dial_dataset.py --download   # fetch + cache the raw corpus
    python scripts/build_doc2dial_dataset.py               # build from the cache

Determinism: SEED is fixed below. Same cached raw files + same SEED -> byte-identical
output, every time (see evalguard.registry.case_builder for the pure functions that make
this true).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.enums import CaseCategory  # noqa: E402
from evalguard.hashing import combine_hashes, hash_file, sha256_text  # noqa: E402
from evalguard.registry.case_builder import (  # noqa: E402
    assert_no_overlap,
    assign_splits,
    build_abstention_cases,
    build_grounded_qa_cases,
    build_guardrail_cases,
    build_instruction_following_cases,
    partition_candidates,
)
from evalguard.registry.datasets import DEFAULT_DATA_DIR  # noqa: E402
from evalguard.registry.doc2dial_source import (  # noqa: E402
    extract_grounded_candidates,
    load_dial_json,
    load_doc_json,
)

SOURCE_URL = "https://doc2dial.github.io/file/doc2dial_v1.0.1.zip"
SOURCE_VERSION = "v1.0.1"
CACHE_DIR = REPO_ROOT / ".cache" / "doc2dial" / "raw"
DOC_JSON = CACHE_DIR / "doc2dial_doc.json"
DIAL_VALIDATION_JSON = CACHE_DIR / "doc2dial_dial_validation.json"

SEED = 42

N_GROUNDED_QA = 80
N_INSTRUCTION_FOLLOWING = 20
N_ABSTENTION = 20
N_GUARDRAIL = 40
ABSTENTION_CONTEXT_BUFFER = 40
DEV_RATIO = 0.75


def download_corpus() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "doc2dial_v1.0.1.zip"
    print(f"Downloading {SOURCE_URL} ...")
    urllib.request.urlretrieve(SOURCE_URL, zip_path)  # noqa: S310 (fixed, known-good URL)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CACHE_DIR)
    print(f"Cached raw corpus under {CACHE_DIR}")


def build() -> None:
    if not DOC_JSON.exists() or not DIAL_VALIDATION_JSON.exists():
        raise SystemExit(f"Raw corpus not found under {CACHE_DIR}. Run with --download first.")

    print("Parsing raw Doc2Dial files...")
    docs = load_doc_json(DOC_JSON)
    dialogues = load_dial_json(DIAL_VALIDATION_JSON)
    candidates = extract_grounded_candidates(docs, dialogues)
    print(f"Extracted {len(candidates)} eligible grounded-QA candidates.")

    corpus_hash = combine_hashes(hash_file(DOC_JSON), hash_file(DIAL_VALIDATION_JSON))

    pools = partition_candidates(
        candidates,
        SEED,
        n_grounded_qa=N_GROUNDED_QA,
        n_instruction_following=N_INSTRUCTION_FOLLOWING,
        n_abstention=N_ABSTENTION,
        abstention_context_buffer=ABSTENTION_CONTEXT_BUFFER,
        n_guardrail=N_GUARDRAIL,
    )
    assert_no_overlap(pools)

    cases_by_category = {
        CaseCategory.GROUNDED_QA: build_grounded_qa_cases(
            pools.grounded_qa, dataset_hash=corpus_hash
        ),
        CaseCategory.INSTRUCTION_FOLLOWING: build_instruction_following_cases(
            pools.instruction_following, SEED, dataset_hash=corpus_hash
        ),
        CaseCategory.ABSTENTION: build_abstention_cases(
            pools.abstention_questions, pools.abstention_contexts, dataset_hash=corpus_hash
        ),
        CaseCategory.GUARDRAIL: build_guardrail_cases(
            pools.guardrail_contexts, SEED, dataset_hash=corpus_hash
        ),
    }

    all_cases = assign_splits(cases_by_category, SEED, dev_ratio=DEV_RATIO)

    # ── write case files ────────────────────────────────────────────────────
    data_dir = DEFAULT_DATA_DIR
    for split in ("dev", "holdout"):
        (data_dir / split).mkdir(parents=True, exist_ok=True)

    case_file_hashes: dict[str, str] = {}
    category_split_counts: dict[str, dict[str, int]] = {}
    split_counts = {"dev": 0, "holdout": 0}
    category_counts: dict[str, int] = {}

    for category in CaseCategory:
        by_split: dict[str, list] = {"dev": [], "holdout": []}
        for case in all_cases:
            if case.category is category:
                by_split[case.split.value].append(case)

        category_split_counts[category.value] = {
            "dev": len(by_split["dev"]),
            "holdout": len(by_split["holdout"]),
        }
        category_counts[category.value] = len(by_split["dev"]) + len(by_split["holdout"])

        for split_name, cases in by_split.items():
            rel_path = f"{split_name}/{category.value}.jsonl"
            out_path = data_dir / rel_path
            lines = [json.dumps(c.model_dump(mode="json"), sort_keys=True) for c in cases]
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            case_file_hashes[rel_path] = sha256_text(out_path.read_text(encoding="utf-8"))
            split_counts[split_name] += len(cases)

    manifest = {
        "source_name": "doc2dial",
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
        "source_file_hashes": {
            "doc2dial_doc.json": hash_file(DOC_JSON),
            "doc2dial_dial_validation.json": hash_file(DIAL_VALIDATION_JSON),
        },
        "corpus_hash": corpus_hash,
        "seed": SEED,
        "generated_at": datetime.now(UTC).isoformat(),
        "category_counts": category_counts,
        "split_counts": split_counts,
        "category_split_counts": category_split_counts,
        "case_file_hashes": case_file_hashes,
        "total_cases": sum(category_counts.values()),
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote {manifest['total_cases']} cases to {data_dir}")
    print(f"  category_counts: {category_counts}")
    print(f"  split_counts: {split_counts}")
    print(f"  category_split_counts: {category_split_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true", help="fetch and cache the raw corpus first"
    )
    args = parser.parse_args()

    if args.download:
        download_corpus()
    build()


if __name__ == "__main__":
    main()
