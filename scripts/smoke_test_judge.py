#!/usr/bin/env python3
"""Phase 5 smoke test: judge up to 4 real grounded_qa dev cases.

Generates (or reuses) up to 4 real Qwen responses via the existing runner, then judges
each with the frozen judge (openai.gpt-oss-120b) for answer_correctness and
faithfulness, deriving hallucination_rate from the faithfulness call. No holdout
access, no full dev sweep, no judge evaluation beyond these 4 cases.

Requires the same .env (MANTLE_BASE_URL/MANTLE_API_KEY) and config/models.yaml
`bedrock_mantle` section as Phase 3/4.

Usage:
    python scripts/smoke_test_judge.py [--max-cases 4] [--db path/to/smoke.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import (  # noqa: E402
    BedrockMantleModelsNotConfiguredError,
    load_bedrock_mantle_profile,
)
from evalguard.db.store import get_connection, get_scores_for_response, init_db  # noqa: E402
from evalguard.enums import CaseSplit, PromptRole  # noqa: E402
from evalguard.hashing import hash_file  # noqa: E402
from evalguard.providers.base import CompletionParams  # noqa: E402
from evalguard.providers.bedrock_mantle import BedrockMantleClient, MantleConfigError  # noqa: E402
from evalguard.quality.judge import score_grounded_qa_case_with_judge  # noqa: E402
from evalguard.registry.datasets import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ensure_cases_registered,
    load_cases,
    load_manifest,
)
from evalguard.registry.prompts import DEFAULT_PROMPTS_DIR, register_from_file  # noqa: E402
from evalguard.runner.manifest import build_run_manifest, compute_code_hash  # noqa: E402
from evalguard.runner.runner import execute_run  # noqa: E402

GUARDRAIL_CONFIG_PATH = REPO_ROOT / "config" / "guardrails.yaml"
JUDGE_PROMPTS_DIR = REPO_ROOT / "prompts" / "judge"


def _format_cost(value) -> str:
    return f"${value:.6f}" if value is not None else "UNVERIFIED (no pricing configured)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--db", default=str(REPO_ROOT / "artifacts" / "smoke_mantle.db"))
    args = parser.parse_args()

    if args.max_cases > 4:
        print("Refusing to judge more than 4 dev cases in the Phase 5 smoke test.")
        return 1

    try:
        profile = load_bedrock_mantle_profile()
    except BedrockMantleModelsNotConfiguredError as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        generator_client = BedrockMantleClient(model_id=profile.generator.model_id)
        judge_client = BedrockMantleClient(model_id=profile.judge.model_id)
    except MantleConfigError as exc:
        print(f"Credentials error: {exc}")
        return 1

    print(f"Generator: {profile.generator.model_id}")
    print(f"Judge:     {profile.judge.model_id}  (temperature=0, frozen)")

    conn = get_connection(args.db)
    init_db(conn)

    dataset_manifest = load_manifest()
    dev_cases = load_cases(CaseSplit.DEV, data_dir=DEFAULT_DATA_DIR)  # never holdout
    grounded_qa_cases = [c for c in dev_cases if c.category.value == "grounded_qa"]
    cases = ensure_cases_registered(conn, grounded_qa_cases[: args.max_cases])

    prompt = register_from_file(
        conn,
        DEFAULT_PROMPTS_DIR / "production" / "support_agent.v1.md",
        name="support_agent",
        version="v1",
        role=PromptRole.PRODUCTION,
    )

    generator_manifest = build_run_manifest(
        prompt=prompt,
        cases=cases,
        dataset_split=CaseSplit.DEV,
        dataset_corpus_hash=dataset_manifest.corpus_hash,
        model_id=profile.generator.model_id,
        model_params=profile.generator.params.model_dump(),
        judge_model_id=profile.judge.model_id,
        guardrail_config_hash=hash_file(GUARDRAIL_CONFIG_PATH),
        code_hash=compute_code_hash(),
    )

    print(f"\n[1/2] Ensuring {len(cases)} real generator response(s) exist...")
    run = execute_run(
        conn,
        generator_manifest,
        prompt,
        cases,
        generator_client,
        price_per_1k_input=profile.generator.price_per_1k_input_tokens,
        price_per_1k_output=profile.generator.price_per_1k_output_tokens,
    )
    print(f"  run_id={run.id} status={run.status.value} cases={run.case_count}")

    correctness_template = (JUDGE_PROMPTS_DIR / "correctness.v1.md").read_text()
    faithfulness_template = (JUDGE_PROMPTS_DIR / "faithfulness.v1.md").read_text()
    # Phase 8.2: correctness and faithfulness are frozen at different budgets — see
    # config/models.yaml's judge/judge_faithfulness_max_tokens comments.
    correctness_params = CompletionParams(**profile.judge.params.model_dump())
    faithfulness_params = correctness_params.model_copy(
        update={"max_tokens": profile.judge_faithfulness_max_tokens}
    )

    print(f"\n[2/2] Judging {len(cases)} response(s) (blind to prompt/run identity)...")
    from evalguard.db.store import get_response_by_run_and_case

    for case in cases:
        response = get_response_by_run_and_case(conn, run.id, case.id)
        if response.output_text is None:
            print(f"  case={case.external_id}: generator failed, skipping judge")
            continue

        scores = score_grounded_qa_case_with_judge(
            conn,
            response,
            case,
            judge_client,
            correctness_template=correctness_template,
            faithfulness_template=faithfulness_template,
            correctness_params=correctness_params,
            faithfulness_params=faithfulness_params,
            price_per_1k_input=profile.judge.price_per_1k_input_tokens,
            price_per_1k_output=profile.judge.price_per_1k_output_tokens,
        )
        by_metric = {s.metric.value: s for s in get_scores_for_response(conn, response.id)}
        newly = {s.metric.value for s in scores}

        print(
            f"\n  case={case.external_id} response_id={response.id} "
            f"({len(newly)} newly scored, {len(by_metric) - len(newly)} already scored)"
        )
        for metric in ("answer_correctness", "faithfulness", "hallucination"):
            score = by_metric.get(metric)
            if score is None:
                continue
            cost = _format_cost(score.judge_cost_usd)
            print(
                f"    {metric:22s} value={score.value:.2f} "
                f"tokens(in/out)={score.judge_input_tokens}/{score.judge_output_tokens} "
                f"latency={score.judge_latency_ms}ms cost={cost}"
            )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
