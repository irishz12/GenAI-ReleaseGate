#!/usr/bin/env python3
"""Phase 8: full DEVELOPMENT evaluation — Prompt V1 vs Prompt V2 over all 120 dev
cases, frozen setup: real generator (qwen.qwen3-next-80b-a3b-instruct), real judge
(openai.gpt-oss-120b), the same real Bedrock Guardrail for both prompts, same model
params/dataset/config. Dev split only — this script never imports or loads holdout
data.

Pipeline per case: Generate -> Guardrail -> deterministic metrics -> judge metrics.
After both prompt runs are complete: regression comparison -> GO/REVIEW/HOLD/INVALID
(config/policy.yaml, unmodified — this run does not tune anything).

Resumable/checkpointed by construction, not by any special-case logic in this script:
- `execute_run` skips any case that already has a response for this exact run.
- `score_run` skips any response that's already scored.
- `score_grounded_qa_case_with_judge` skips any metric a response already has.
A crash, ^C, or rate-limit abort loses at most the one in-flight call — re-running
this script picks up exactly where it left off, at no extra generator/judge/guardrail
cost for work already paid for and persisted.

Usage:
    python scripts/run_dev_eval.py [--db path/to/dev_eval.db] [--max-cases N]

--max-cases is a debug aid (wire-up smoke test on a handful of cases before spending
on the full 120) — omit it for the real Phase 8 run.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import (  # noqa: E402
    BedrockGuardrailNotConfiguredError,
    BedrockMantleModelsNotConfiguredError,
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
from evalguard.models import EvaluationCase, Run  # noqa: E402
from evalguard.providers.base import CompletionParams  # noqa: E402
from evalguard.providers.bedrock_guardrail import BedrockGuardrailClient  # noqa: E402
from evalguard.providers.bedrock_mantle import BedrockMantleClient, MantleConfigError  # noqa: E402
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


def _judge_grounded_qa(conn, run: Run, cases: list[EvaluationCase], judge_client, profile) -> int:
    """Judge every not-yet-judged grounded_qa response in `run`. A single case whose
    judge output fails to parse after retry is recorded here and skipped — it must
    never abort scoring for the rest of the run (same per-case isolation principle as
    runner.execute_run)."""
    correctness_template = (JUDGE_PROMPTS_DIR / "correctness.v1.md").read_text()
    faithfulness_template = (JUDGE_PROMPTS_DIR / "faithfulness.v1.md").read_text()
    # Phase 8.2: correctness and faithfulness are frozen at different budgets — see
    # config/models.yaml's judge/judge_faithfulness_max_tokens comments.
    correctness_params = CompletionParams(**profile.judge.params.model_dump())
    faithfulness_params = correctness_params.model_copy(
        update={"max_tokens": profile.judge_faithfulness_max_tokens}
    )

    judged = 0
    for case in cases:
        if case.category is not CaseCategory.GROUNDED_QA:
            continue
        response = get_response_by_run_and_case(conn, run.id, case.id)
        if response is None or response.output_text is None:
            continue
        try:
            new_scores = score_grounded_qa_case_with_judge(
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
            judged += len(new_scores)
        except (MalformedJudgeOutputError, JudgeRetryExhaustedError) as exc:
            print(f"    judge FAILED for case={case.external_id}: {exc}")
    return judged


def _run_costs(conn, run: Run) -> dict[str, float | None]:
    """Generator/guardrail/judge cost tracked separately — never summed into one
    number, since they come from three different billed services."""
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


def _category_summary(
    conn,
    run: Run,
    cases_by_category: dict[CaseCategory, list[EvaluationCase]],
    category: CaseCategory,
) -> dict:
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
        # One outcome per response — the same contract regression.compare._guardrail_value
        # uses (first guardrail_results row, i.e. the input-stage check; these three
        # rate metrics are about whether the input itself was correctly gated, not the
        # separate output-stage leak check). Pooling both stages here would silently
        # disagree with the number that actually drives the release decision.
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


def _print_summary(label: str, summary: dict) -> None:
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


def _find_regressions(conn, run_v1: Run, run_v2: Run, cases: list[EvaluationCase]) -> list[dict]:
    """Case-level V1-vs-V2 diffs where V2 is strictly worse than V1 — the detail
    behind the aggregate deltas, for root-causing a HOLD/REVIEW decision. Script-local
    (not part of the installed package): a one-off analysis over already-persisted
    rows, not a new pipeline stage."""
    regressions: list[dict] = []

    for case in cases:
        r1 = get_response_by_run_and_case(conn, run_v1.id, case.id)
        r2 = get_response_by_run_and_case(conn, run_v2.id, case.id)
        if r1 is None or r2 is None:
            continue

        if r1.error_class is None and r2.error_class is not None:
            regressions.append(
                {
                    "case": case.external_id,
                    "category": case.category.value,
                    "metric": "failure",
                    "detail": f"V1 succeeded, V2 errored ({r2.error_class.value})",
                }
            )
            continue  # no scores to compare for an errored response

        if case.category is CaseCategory.GROUNDED_QA:
            s1 = {s.metric: s.value for s in get_scores_for_response(conn, r1.id)}
            s2 = {s.metric: s.value for s in get_scores_for_response(conn, r2.id)}
            for metric, worse_if in (
                (MetricName.ANSWER_CORRECTNESS, lambda a, b: b < a),
                (MetricName.FAITHFULNESS, lambda a, b: b < a),
                (MetricName.HALLUCINATION, lambda a, b: b > a),
            ):
                if metric in s1 and metric in s2 and worse_if(s1[metric], s2[metric]):
                    regressions.append(
                        {
                            "case": case.external_id,
                            "category": case.category.value,
                            "metric": metric.value,
                            "detail": f"V1={s1[metric]:.2f} -> V2={s2[metric]:.2f}",
                        }
                    )

        elif case.category in (CaseCategory.ABSTENTION, CaseCategory.INSTRUCTION_FOLLOWING):
            metric = (
                MetricName.ABSTENTION_ACCURACY
                if case.category is CaseCategory.ABSTENTION
                else MetricName.INSTRUCTION_FOLLOWING
            )
            s1 = next(
                (s.value for s in get_scores_for_response(conn, r1.id) if s.metric is metric), None
            )
            s2 = next(
                (s.value for s in get_scores_for_response(conn, r2.id) if s.metric is metric), None
            )
            if s1 == 1.0 and s2 == 0.0:
                regressions.append(
                    {
                        "case": case.external_id,
                        "category": case.category.value,
                        "metric": metric.value,
                        "detail": "V1 passed, V2 failed",
                    }
                )

        elif case.category is CaseCategory.GUARDRAIL:
            g1 = get_guardrail_results_for_response(conn, r1.id)
            g2 = get_guardrail_results_for_response(conn, r2.id)
            o1 = g1[0].outcome if g1 else None
            o2 = g2[0].outcome if g2 else None
            if case.attack_type in (AttackType.INJECTION, AttackType.PII_EXTRACTION):
                if o1 is GuardrailOutcome.TP and o2 is GuardrailOutcome.FN:
                    regressions.append(
                        {
                            "case": case.external_id,
                            "category": case.category.value,
                            "metric": "guardrail_outcome",
                            "detail": "V1 blocked (TP), V2 let it through (FN) — safety regression",
                        }
                    )
            elif (
                case.attack_type is AttackType.BENIGN
                and o1 is GuardrailOutcome.TN
                and o2 is GuardrailOutcome.FP
            ):
                regressions.append(
                    {
                        "case": case.external_id,
                        "category": case.category.value,
                        "metric": "guardrail_outcome",
                        "detail": "V1 correctly passed (TN), V2 incorrectly blocked (FP)",
                    }
                )

    return regressions


def _find_unexpected_guardrail_interventions(
    conn, run: Run, cases: list[EvaluationCase]
) -> list[dict]:
    """Non-guardrail-category cases (attack_type is None, expected_block=False) that
    the real Bedrock Guardrail blocked anyway. These never show up in
    prompt_injection_block_rate/sensitive_information_protection/
    benign_false_positive_rate — those three are only ever computed over
    guardrail-category cases (see quality.scoring._summarize) — so a false positive
    here is otherwise invisible to every metric this pipeline reports."""
    findings = []
    for case in cases:
        if case.category is CaseCategory.GUARDRAIL:
            continue
        response = get_response_by_run_and_case(conn, run.id, case.id)
        if response is None:
            continue
        for gr in get_guardrail_results_for_response(conn, response.id):
            if gr.triggered:
                findings.append(
                    {
                        "case": case.external_id,
                        "category": case.category.value,
                        "stage": gr.stage.value,
                        "detail": gr.detail or "(no detail)",
                    }
                )
    return findings


def _check_aws_credentials(region: str) -> bool:
    """Fast pre-flight check: fail immediately with a clear message if the ambient
    AWS session is expired/invalid, instead of surfacing a raw botocore traceback
    partway through a real (paid) run. Returns True if credentials are valid."""
    import boto3

    try:
        boto3.client("sts", region_name=region).get_caller_identity()
        return True
    except Exception as exc:  # noqa: BLE001 — any failure here means "can't proceed"
        print(f"AWS credentials check failed: {exc}")
        print("Run `aws login` to refresh your session, then retry.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "artifacts" / "dev_eval.db"))
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    try:
        profile = load_bedrock_mantle_profile()
        guardrail_config = load_bedrock_guardrail_config()
    except (BedrockMantleModelsNotConfiguredError, BedrockGuardrailNotConfiguredError) as exc:
        print(f"Config error: {exc}")
        return 1

    if not _check_aws_credentials(guardrail_config.region):
        return 1

    try:
        generator_client = BedrockMantleClient(model_id=profile.generator.model_id)
        judge_client = BedrockMantleClient(model_id=profile.judge.model_id)
    except MantleConfigError as exc:
        print(f"Credentials error: {exc}")
        return 1

    guardrail_client = BedrockGuardrailClient(
        guardrail_config.guardrail_id,
        guardrail_config.guardrail_version,
        guardrail_config.region,
        pricing=guardrail_config.pricing,
    )

    print(f"Generator: {profile.generator.model_id}")
    print(f"Judge:     {profile.judge.model_id}")
    print(f"Guardrail: {guardrail_config.guardrail_id} v{guardrail_config.guardrail_version}")

    conn = get_connection(args.db)
    init_db(conn)

    dataset_manifest = load_manifest()
    dev_cases = load_cases(CaseSplit.DEV, data_dir=DEFAULT_DATA_DIR)  # dev split only
    if args.max_cases is not None:
        # Per-category cap, not a flat head-slice — a smoke run should still exercise
        # every pipeline branch (judge, guardrail, deterministic), not just whichever
        # category happens to load first.
        capped_by_category: dict[CaseCategory, int] = defaultdict(int)
        capped_cases: list[EvaluationCase] = []
        for case in dev_cases:
            if capped_by_category[case.category] < args.max_cases:
                capped_cases.append(case)
                capped_by_category[case.category] += 1
        dev_cases = capped_cases
    cases = ensure_cases_registered(conn, dev_cases)
    cases_by_category = defaultdict(list)
    for c in cases:
        cases_by_category[c.category].append(c)
    print(
        f"\nLoaded and registered {len(cases)} dev cases: "
        + ", ".join(f"{cat.value}={len(cs)}" for cat, cs in cases_by_category.items())
    )

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

    code_hash = compute_code_hash()
    guardrail_config_hash = hash_file(GUARDRAIL_CONFIG_PATH)

    runs: dict[str, Run] = {}
    for label, prompt in (("v1", prompt_v1), ("v2", prompt_v2)):
        print(f"\n=== Prompt {label} ({prompt.name}@{prompt.version}) ===")
        manifest = build_run_manifest(
            prompt=prompt,
            cases=cases,
            dataset_split=CaseSplit.DEV,
            dataset_corpus_hash=dataset_manifest.corpus_hash,
            model_id=profile.generator.model_id,
            model_params=profile.generator.params.model_dump(),
            judge_model_id=profile.judge.model_id,
            guardrail_config_hash=guardrail_config_hash,
            code_hash=code_hash,
        )

        print(f"[1/3] Generate + Guardrail ({len(cases)} cases)...")
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
        print(
            f"  run_id={run.id} status={run.status.value} cases={run.case_count} "
            f"failures={run.failure_count}"
        )

        print("[2/3] Deterministic metrics...")
        scoring_summary = score_run(conn, run.id)
        print(
            f"  newly_scored={scoring_summary.newly_scored} "
            f"already_scored={scoring_summary.already_scored} "
            f"skipped_failed={scoring_summary.skipped_failed}"
        )

        print("[3/3] Judge metrics (grounded_qa only)...")
        judged = _judge_grounded_qa(conn, run, cases, judge_client, profile)
        print(f"  newly judged score rows={judged}")

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
            _print_summary(label, summary)

    print("\n=== Regression comparison ===")
    policy = load_policy_config()
    # Always recompute fresh, then upsert: judge retries across resumed invocations
    # keep filling in scores after the first successful run, and a persisted
    # Comparison from an earlier, less-complete resume must not silently go stale
    # (Phase 8.1 finding, fixed properly in Phase 8.2 — see db.store.upsert_comparison).
    comparison = compare_runs(conn, runs["v1"].id, runs["v2"].id, policy)
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

    print("\n=== Failure analysis: cases where V2 is worse than V1 ===")
    regressions = _find_regressions(conn, runs["v1"], runs["v2"], cases)
    if not regressions:
        print("  none found")
    else:
        for reg in regressions:
            print(
                f"  [{reg['category']}] case={reg['case']} metric={reg['metric']}: {reg['detail']}"
            )

    print("\n=== Unexpected guardrail interventions (non-guardrail-category cases) ===")
    for label, run in runs.items():
        unexpected = _find_unexpected_guardrail_interventions(conn, run, cases)
        if not unexpected:
            print(f"  {label}: none found")
            continue
        for finding in unexpected:
            print(
                f"  {label} [{finding['category']}] case={finding['case']} "
                f"stage={finding['stage']}: {finding['detail']}"
            )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
