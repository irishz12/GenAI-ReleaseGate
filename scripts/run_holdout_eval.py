#!/usr/bin/env python3
"""THE holdout acceptance test: Production V1 vs Candidate V3, full 40-case HOLDOUT
evaluation. Frozen setup, unchanged: generator qwen.qwen3-next-80b-a3b-instruct,
judge openai.gpt-oss-120b, correctness max_tokens=512, faithfulness max_tokens=1024,
guardrail version 2, policy/metrics unchanged (all Phase 8-9 decisions).

This is the ONE sealed look at holdout this project has ever taken. Deliberately
narrow and inflexible on purpose:
  - CaseSplit.HOLDOUT is hardcoded — there is no flag to point this at dev by
    mistake, unlike scripts/run_dev_eval.py.
  - No --max-cases debug option — no partial/exploratory look at holdout data.
  - No reuse/replay logic — neither V1 nor V3 has ever been run against holdout
    before, so there is nothing valid to reuse; this is a genuine, full, fresh run
    for both prompts (same pipeline already proven correct across every dev run).
  - Writes to its own dedicated DB (artifacts/holdout_eval.db), never dev_eval.db.

Usage:
    python scripts/run_holdout_eval.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import (  # noqa: E402
    load_bedrock_guardrail_config,
    load_bedrock_mantle_profile,
    load_policy_config,
)
from evalguard.db.store import (  # noqa: E402
    get_connection,
    get_guardrail_results_for_response,
    get_response_by_run_and_case,
    get_responses_for_run,
    get_scores_for_response,
    init_db,
    upsert_comparison,
)
from evalguard.enums import (  # noqa: E402
    AttackType,
    CaseCategory,
    CaseSplit,
    GuardrailOutcome,
    MetricName,
    PromptRole,
)
from evalguard.hashing import hash_file  # noqa: E402
from evalguard.models import Run  # noqa: E402
from evalguard.providers.base import CompletionParams  # noqa: E402
from evalguard.providers.bedrock_guardrail import BedrockGuardrailClient  # noqa: E402
from evalguard.providers.bedrock_mantle import BedrockMantleClient  # noqa: E402
from evalguard.quality.aggregate import (  # noqa: E402
    compute_cost_per_query,
    compute_failure_rate,
    compute_guardrail_rate,
    compute_latency_percentiles,
    compute_pass_rate,
)
from evalguard.quality.judge import (  # noqa: E402
    JudgeRetryExhaustedError,
    MalformedJudgeOutputError,
    score_grounded_qa_case_with_judge,
)
from evalguard.quality.scoring import score_run  # noqa: E402
from evalguard.registry.datasets import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ensure_cases_registered,
    holdout_available,
    load_cases,
    load_manifest,
)
from evalguard.registry.prompts import DEFAULT_PROMPTS_DIR, register_from_file  # noqa: E402
from evalguard.regression.compare import compare_runs  # noqa: E402
from evalguard.runner.manifest import build_run_manifest, compute_code_hash  # noqa: E402
from evalguard.runner.runner import execute_run  # noqa: E402

GUARDRAIL_CONFIG_PATH = REPO_ROOT / "config" / "guardrails.yaml"
JUDGE_PROMPTS_DIR = REPO_ROOT / "prompts" / "judge"


def _cost(value: float | None) -> str:
    return f"${value:.6f}" if value is not None else "UNVERIFIED"


def _check_aws_credentials(region: str) -> bool:
    import boto3

    try:
        boto3.client("sts", region_name=region).get_caller_identity()
        return True
    except Exception as exc:  # noqa: BLE001 — any failure here means "can't proceed"
        print(f"AWS credentials check failed: {exc}")
        print("Run `aws login` to refresh your session, then retry.")
        return False


def _judge_grounded_qa(conn, run: Run, cases, judge_client, profile) -> dict[str, int]:
    correctness_template = (JUDGE_PROMPTS_DIR / "correctness.v1.md").read_text()
    faithfulness_template = (JUDGE_PROMPTS_DIR / "faithfulness.v1.md").read_text()
    correctness_params = CompletionParams(**profile.judge.params.model_dump())
    faithfulness_params = correctness_params.model_copy(
        update={"max_tokens": profile.judge_faithfulness_max_tokens}
    )

    stats = {"judged": 0, "skipped_blocked": 0, "failed": 0}
    for case in cases:
        if case.category is not CaseCategory.GROUNDED_QA:
            continue
        response = get_response_by_run_and_case(conn, run.id, case.id)
        if response is None or response.output_text is None:
            continue
        if response.finish_reason.value in ("blocked_input", "blocked_output"):
            stats["skipped_blocked"] += 1
            continue
        try:
            score_grounded_qa_case_with_judge(
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
                allow_holdout=True,  # the one sanctioned exception — see module docstring
            )
            stats["judged"] += 1
        except (MalformedJudgeOutputError, JudgeRetryExhaustedError) as exc:
            stats["failed"] += 1
            print(f"    judge FAILED for case={case.external_id}: {exc}")
    return stats


def _run_costs(conn, run: Run) -> dict[str, float | None]:
    responses = get_responses_for_run(conn, run.id)
    successful = [r for r in responses if r.error_class is None]
    generator_cost = (
        sum(r.cost_usd for r in successful)
        if successful and all(r.cost_usd is not None for r in successful)
        else None
    )
    guardrail_results = [
        gr for r in responses for gr in get_guardrail_results_for_response(conn, r.id)
    ]
    priced_gr = [gr.cost_usd for gr in guardrail_results if gr.cost_usd is not None]
    guardrail_cost = sum(priced_gr) if priced_gr else None
    scores = [s for r in responses for s in get_scores_for_response(conn, r.id)]
    priced_judge = [s.judge_cost_usd for s in scores if s.judge_cost_usd is not None]
    judge_cost = sum(priced_judge) if priced_judge else None
    return {"generator": generator_cost, "guardrail": guardrail_cost, "judge": judge_cost}


def _category_summary(conn, run: Run, cases_by_category, category) -> dict:
    cases = cases_by_category[category]
    responses = [get_response_by_run_and_case(conn, run.id, c.id) for c in cases]
    responses = [r for r in responses if r is not None]
    scores = [s for r in responses for s in get_scores_for_response(conn, r.id)]
    guardrail_results = [
        gr for r in responses for gr in get_guardrail_results_for_response(conn, r.id)
    ]

    summary: dict = {
        "n": len(responses),
        "failure_rate": compute_failure_rate(responses),
        "latency": compute_latency_percentiles(responses),
        "cost_per_query": compute_cost_per_query(responses),
    }

    if category is CaseCategory.GROUNDED_QA:
        by_metric = defaultdict(list)
        for s in scores:
            by_metric[s.metric].append(s)
        summary["answer_correctness"] = compute_pass_rate(by_metric[MetricName.ANSWER_CORRECTNESS])
        summary["faithfulness"] = compute_pass_rate(by_metric[MetricName.FAITHFULNESS])
        summary["hallucination"] = compute_pass_rate(by_metric[MetricName.HALLUCINATION])
    elif category is CaseCategory.ABSTENTION:
        summary["abstention_accuracy"] = compute_pass_rate(
            [s for s in scores if s.metric is MetricName.ABSTENTION_ACCURACY]
        )
    elif category is CaseCategory.INSTRUCTION_FOLLOWING:
        summary["instruction_following"] = compute_pass_rate(
            [s for s in scores if s.metric is MetricName.INSTRUCTION_FOLLOWING]
        )
    elif category is CaseCategory.GUARDRAIL:
        first_result_by_response = {}
        for gr in guardrail_results:
            first_result_by_response.setdefault(gr.response_id, gr)
        case_by_id = {c.id: c for c in cases}
        by_attack = defaultdict(list)
        for response in responses:
            gr = first_result_by_response.get(response.id)
            case = case_by_id.get(response.case_id)
            if gr is not None and case is not None and case.attack_type is not None:
                by_attack[case.attack_type].append(gr)
        summary["prompt_injection_block_rate"] = compute_guardrail_rate(
            by_attack[AttackType.INJECTION], numerator_outcome=GuardrailOutcome.TP
        )
        summary["sensitive_information_protection"] = compute_guardrail_rate(
            by_attack[AttackType.PII_EXTRACTION], numerator_outcome=GuardrailOutcome.TP
        )
        summary["benign_false_positive_rate"] = compute_guardrail_rate(
            by_attack[AttackType.BENIGN], numerator_outcome=GuardrailOutcome.FP
        )

    return summary


def _print_summary(summary: dict) -> None:
    print(f"    n={summary['n']} failure_rate={summary['failure_rate']:.2%}")
    if summary["latency"] is not None:
        p50 = summary["latency"]["p50"]
        p95 = summary["latency"]["p95"]
        print(f"    latency p50={p50:.0f}ms p95={p95:.0f}ms")
    print(f"    generator cost/query={_cost(summary['cost_per_query'])}")
    for key, value in summary.items():
        if key in ("n", "failure_rate", "latency", "cost_per_query"):
            continue
        shown = f"{value:.2%}" if value is not None else "N/A (no data)"
        print(f"    {key}={shown}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "artifacts" / "holdout_eval.db"))
    args = parser.parse_args()

    if not holdout_available():
        print("Refusing: data/holdout/ is not fully present.")
        return 1

    profile = load_bedrock_mantle_profile()
    guardrail_config = load_bedrock_guardrail_config()
    if not _check_aws_credentials(guardrail_config.region):
        return 1
    if guardrail_config.guardrail_version != "2":
        actual = guardrail_config.guardrail_version
        print(f"Refusing: expected guardrail version 2, config has {actual!r}")
        return 1

    generator_client = BedrockMantleClient(model_id=profile.generator.model_id)
    judge_client = BedrockMantleClient(model_id=profile.judge.model_id)
    guardrail_client = BedrockGuardrailClient(
        guardrail_config.guardrail_id,
        guardrail_config.guardrail_version,
        guardrail_config.region,
        pricing=guardrail_config.pricing,
    )

    print(f"Generator: {profile.generator.model_id}")
    print(f"Judge:     {profile.judge.model_id}")
    print(f"Guardrail: {guardrail_config.guardrail_id} v{guardrail_config.guardrail_version}")
    print("\n*** THIS IS THE HOLDOUT SPLIT — the one sealed acceptance look. ***\n")

    conn = get_connection(args.db)
    init_db(conn)

    dataset_manifest = load_manifest()
    holdout_cases = load_cases(CaseSplit.HOLDOUT, data_dir=DEFAULT_DATA_DIR)
    cases = ensure_cases_registered(conn, holdout_cases)
    cases_by_category = defaultdict(list)
    for c in cases:
        cases_by_category[c.category].append(c)
    print(
        f"Loaded {len(cases)} HOLDOUT cases: "
        + ", ".join(f"{cat.value}={len(cs)}" for cat, cs in cases_by_category.items())
    )
    if len(cases) != 40:
        print(f"Refusing: expected exactly 40 holdout cases, got {len(cases)}")
        return 1

    prompt_v1 = register_from_file(
        conn,
        DEFAULT_PROMPTS_DIR / "production" / "support_agent.v1.md",
        name="support_agent",
        version="v1",
        role=PromptRole.PRODUCTION,
    )
    prompt_v3 = register_from_file(
        conn,
        DEFAULT_PROMPTS_DIR / "candidate" / "support_agent.v3.md",
        name="support_agent",
        version="v3",
        role=PromptRole.CANDIDATE,
    )

    code_hash = compute_code_hash()
    guardrail_config_hash = hash_file(GUARDRAIL_CONFIG_PATH)

    runs: dict[str, Run] = {}
    for label, prompt in (("v1", prompt_v1), ("v3", prompt_v3)):
        print(f"\n=== Prompt {label} ({prompt.name}@{prompt.version}) ===")
        manifest = build_run_manifest(
            prompt=prompt,
            cases=cases,
            dataset_split=CaseSplit.HOLDOUT,
            dataset_corpus_hash=dataset_manifest.corpus_hash,
            model_id=profile.generator.model_id,
            model_params=profile.generator.params.model_dump(),
            judge_model_id=profile.judge.model_id,
            guardrail_config_hash=guardrail_config_hash,
            code_hash=code_hash,
        )
        print("[1/3] Generate + Guardrail v2...")
        run = execute_run(
            conn,
            manifest,
            prompt,
            cases,
            generator_client,
            price_per_1k_input=profile.generator.price_per_1k_input_tokens,
            price_per_1k_output=profile.generator.price_per_1k_output_tokens,
            guardrail_client=guardrail_client,
        )
        print(f"  run_id={run.id} status={run.status.value} cases={run.case_count}")

        print("[2/3] Deterministic metrics...")
        scoring_summary = score_run(conn, run.id, allow_holdout=True)  # see module docstring
        print(
            f"  newly_scored={scoring_summary.newly_scored} "
            f"already_scored={scoring_summary.already_scored}"
        )

        print("[3/3] Judge metrics (grounded_qa only)...")
        stats = _judge_grounded_qa(conn, run, cases, judge_client, profile)
        print(f"  {stats}")

        runs[label] = run

    print("\n=== Cost (tracked separately per service) ===")
    for label, run in runs.items():
        costs = _run_costs(conn, run)
        print(
            f"  {label}: generator={_cost(costs['generator'])} "
            f"guardrail={_cost(costs['guardrail'])} judge={_cost(costs['judge'])}"
        )

    print("\n=== Per-suite metrics ===")
    for label, run in runs.items():
        print(f"\n  -- {label} --")
        for category in CaseCategory:
            if not cases_by_category[category]:
                continue
            print(f"  [{category.value}]")
            summary = _category_summary(conn, run, cases_by_category, category)
            _print_summary(summary)

    print("\n=== HOLDOUT Regression comparison (V1 baseline vs v3 candidate) ===")
    policy = load_policy_config()
    comparison = compare_runs(conn, runs["v1"].id, runs["v3"].id, policy)
    comparison = upsert_comparison(conn, comparison)
    print(f"n_paired={comparison.n_paired}")
    print(f"decision={comparison.release.status.value}")
    print(f"summary={comparison.release.summary}")
    for gate in comparison.release.gates:
        print(f"  gate={gate.gate_name:32s} status={gate.status.value:8s} reason={gate.reason}")
    print("\ndeltas:")
    for delta in comparison.deltas:
        ci = f" CI=[{delta.ci_low:+.4f}, {delta.ci_high:+.4f}]" if delta.ci_low is not None else ""
        print(
            f"  {delta.metric.value:32s} baseline={delta.baseline_value:.4f} "
            f"candidate={delta.candidate_value:.4f} delta={delta.delta:+.4f}{ci}"
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
