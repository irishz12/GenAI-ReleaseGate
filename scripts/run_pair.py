#!/usr/bin/env python3
"""Phase 2 demo: run Prompt V1 and Prompt V2 over the same dev cases with FakeClient.

Not part of the installed package — a runnable demonstration of the Phase 2 pipeline,
in the same spirit as scripts/build_doc2dial_dataset.py. Uses only the dev split, only
FakeClient, and does not compute any quality/judge/guardrail scores — those are later
phases. Safe to re-run: execute_run resumes rather than duplicating work.

Usage:
    python scripts/run_pair.py [--db path/to/eval.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import load_models_config  # noqa: E402
from evalguard.db.store import get_connection, init_db  # noqa: E402
from evalguard.enums import CaseSplit, PromptRole  # noqa: E402
from evalguard.hashing import hash_file  # noqa: E402
from evalguard.providers.fake import FakeClient  # noqa: E402
from evalguard.registry.datasets import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ensure_cases_registered,
    load_cases,
    load_manifest,
)
from evalguard.registry.prompts import DEFAULT_PROMPTS_DIR, register_from_file  # noqa: E402
from evalguard.runner.manifest import build_run_manifest, compute_code_hash  # noqa: E402
from evalguard.runner.runner import execute_run  # noqa: E402

FAKE_SEED = 42
GUARDRAIL_CONFIG_PATH = REPO_ROOT / "config" / "guardrails.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "artifacts" / "eval.db"))
    args = parser.parse_args()

    conn = get_connection(args.db)
    init_db(conn)

    models_config = load_models_config()
    dataset_manifest = load_manifest()
    code_hash = compute_code_hash()
    guardrail_config_hash = hash_file(GUARDRAIL_CONFIG_PATH)

    # Dev split only — no holdout access in Phase 2.
    cases = load_cases(CaseSplit.DEV, data_dir=DEFAULT_DATA_DIR)
    cases = ensure_cases_registered(conn, cases)
    print(f"Loaded and registered {len(cases)} dev cases.")

    prompt_v1 = register_from_file(
        conn,
        DEFAULT_PROMPTS_DIR / "production" / "support_agent.v1.md",
        name="support_agent",
        version="v1",
        role=PromptRole.PRODUCTION,
    )
    prompt_v2 = register_from_file(
        conn,
        DEFAULT_PROMPTS_DIR / "candidate" / "support_agent.v2.md",
        name="support_agent",
        version="v2",
        role=PromptRole.CANDIDATE,
    )

    client = FakeClient(seed=FAKE_SEED)

    for prompt in (prompt_v1, prompt_v2):
        manifest = build_run_manifest(
            prompt=prompt,
            cases=cases,
            dataset_split=CaseSplit.DEV,
            dataset_corpus_hash=dataset_manifest.corpus_hash,
            model_id=models_config.generator.model_id,
            model_params=models_config.generator.params.model_dump(),
            fake_seed=FAKE_SEED,
            judge_model_id=models_config.judge.model_id,
            guardrail_config_hash=guardrail_config_hash,
            code_hash=code_hash,
        )
        run = execute_run(
            conn,
            manifest,
            prompt,
            cases,
            client,
            price_per_1k_input=models_config.generator.price_per_1k_input_tokens,
            price_per_1k_output=models_config.generator.price_per_1k_output_tokens,
        )
        print(
            f"{prompt.name}@{prompt.version}: run_id={run.id} status={run.status.value} "
            f"cases={run.case_count} failures={run.failure_count} cost=${run.total_cost_usd:.4f}"
        )

    conn.close()


if __name__ == "__main__":
    main()
