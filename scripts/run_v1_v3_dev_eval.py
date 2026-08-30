#!/usr/bin/env python3
"""Phase 8.4: Production V1 vs Candidate V3, full 120-case DEVELOPMENT evaluation.

Frozen setup (unchanged this phase): generator qwen.qwen3-next-80b-a3b-instruct,
judge openai.gpt-oss-120b, correctness max_tokens=512, faithfulness max_tokens=1024
(Phase 8.2), guardrail version "2" (Phase 8.2), policy/metrics unchanged.

Reuses V1's existing Phase 8 generation + judge work wherever the Phase 8.2
guardrail-version change (v1 -> v2) doesn't affect it, instead of paying for it
again:
  - Every V1 response that succeeded for real (finish_reason=stop) under the old
    guardrail is REPLAYED verbatim (same text/tokens/latency; cost is recomputed
    from those tokens, same as any response) rather than re-generated — the
    generator/prompt/params never changed, only the guardrail did.
  - V1's grounded_qa judge scores (already frozen at correctness=512/
    faithfulness=1024 since Phase 8.2) are copied onto a replayed response's new row
    rather than re-judged, whenever the replayed text is byte-identical to what was
    already judged — the judge's input is unchanged, so re-judging would only spend
    money to (at best) reproduce the same verdict.
  - The guardrail check itself is ALWAYS re-run fresh, under version 2, for every
    case — that's the one thing that changed. A case blocked under v1's stricter
    PROMPT_ATTACK setting may no longer be blocked under v2; when that happens there
    is no real completion to replay (the model was never actually called for it
    before), so it gets one genuine, unavoidable new generation.
  - Deterministic scores (instruction_following/abstention_accuracy) are free local
    checks — always recomputed fresh, no reuse bookkeeping needed.

Also fixes a real bug found while building this: a response the guardrail blocked at
input has no real candidate answer at all (just the guardrail's canned message) —
judging it as if it were a real answer previously contaminated grounded_qa quality
metrics with guardrail noise. This run skips judging any blocked response.

V3 is entirely new (no prior data exists for it): full fresh generate + guardrail +
deterministic + judge, same as Phase 8's original run.

No holdout access. No prompt/model/guardrail/policy changes made by this script.

Usage:
    python scripts/run_v1_v3_dev_eval.py [--db path/to/dev_eval.db]
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
    get_run,
    get_scores_for_response,
    init_db,
    insert_score,
    upsert_comparison,
)
from evalguard.enums import (  # noqa: E402
    AttackType,
    CaseCategory,
    CaseSplit,
    FinishReason,
    GuardrailOutcome,
    MetricName,
    PromptRole,
)
from evalguard.hashing import hash_file  # noqa: E402
from evalguard.models import EvaluationCase, Run, Score  # noqa: E402
from evalguard.providers.base import Completion, CompletionParams, Usage  # noqa: E402
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
    load_cases,
    load_manifest,
)
from evalguard.registry.prompts import DEFAULT_PROMPTS_DIR, register_from_file  # noqa: E402
from evalguard.regression.compare import compare_runs  # noqa: E402
from evalguard.runner.manifest import build_run_manifest, compute_code_hash  # noqa: E402
from evalguard.runner.rendering import render_prompt  # noqa: E402
from evalguard.runner.runner import execute_run  # noqa: E402

GUARDRAIL_CONFIG_PATH = REPO_ROOT / "config" / "guardrails.yaml"
JUDGE_PROMPTS_DIR = REPO_ROOT / "prompts" / "judge"
OLD_V1_RUN_ID = 1  # Phase 8's original V1 run in this same db (guardrail v1 era)


def _cost(value: float | None) -> str:
    return f"${value:.6f}" if value is not None else "UNVERIFIED"


class ReplayOrGenerateClient:
    """Serves a cached Completion for any prompt seen in a prior run's real
    generation; falls back to a real client only for prompts with no prior real
    completion (blocked last time, no longer blocked now). Tracks how many calls
    were free replays vs. genuinely new paid generations."""

    def __init__(self, real_client, cache_by_prompt: dict[str, Completion]) -> None:
        self._real_client = real_client
        self._cache = cache_by_prompt
        self.replayed = 0
        self.generated = 0

    def complete(self, prompt: str, params: CompletionParams) -> Completion:
        cached = self._cache.get(prompt)
        if cached is not None:
            self.replayed += 1
            return cached
        self.generated += 1
        return self._real_client.complete(prompt, params)


def _build_v1_replay_cache(
    conn, v1_template: str, cases: list[EvaluationCase]
) -> dict[str, Completion]:
    """One cached Completion per case that had a real (finish_reason=stop) response
    in OLD_V1_RUN_ID — never for a blocked-input case, since no real completion
    exists for those to replay."""
    cache: dict[str, Completion] = {}
    for case in cases:
        old = get_response_by_run_and_case(conn, OLD_V1_RUN_ID, case.id)
        if old is None or old.finish_reason is not FinishReason.STOP:
            continue
        prompt = render_prompt(v1_template, case)
        cache[prompt] = Completion(
            text=old.output_text,
            finish_reason=old.finish_reason,
            usage=Usage(input_tokens=old.input_tokens, output_tokens=old.output_tokens),
            latency_ms=old.latency_ms,
        )
    return cache


def _build_v1_judge_cache(conn, cases: list[EvaluationCase]) -> dict[int, tuple[str, dict]]:
    """case_id -> (old_answer_text, {metric: Score}) from OLD_V1_RUN_ID, grounded_qa
    only — the judge scores already frozen at correctness=512/faithfulness=1024 since
    Phase 8.2. The answer text travels with its scores so the caller can confirm a
    replayed response's text still matches before reusing them."""
    cache: dict[int, tuple[str, dict]] = {}
    for case in cases:
        if case.category is not CaseCategory.GROUNDED_QA:
            continue
        old = get_response_by_run_and_case(conn, OLD_V1_RUN_ID, case.id)
        if old is None or old.output_text is None:
            continue
        scores = {s.metric: s for s in get_scores_for_response(conn, old.id)}
        if scores:
            cache[case.id] = (old.output_text, scores)
    return cache


def _judge_grounded_qa(
    conn,
    run: Run,
    cases: list[EvaluationCase],
    judge_client,
    profile,
    *,
    old_judge_cache: dict[int, tuple[str, dict]] | None = None,
) -> dict[str, int]:
    """Judge every not-yet-judged grounded_qa response in `run`. Skips any response
    the guardrail blocked (no real candidate answer to judge — see module docstring).
    Reuses old_judge_cache's scores when the response text is byte-identical to what
    was already judged (only meaningful for the V1 replay run; None for V3)."""
    correctness_template = (JUDGE_PROMPTS_DIR / "correctness.v1.md").read_text()
    faithfulness_template = (JUDGE_PROMPTS_DIR / "faithfulness.v1.md").read_text()
    correctness_params = CompletionParams(**profile.judge.params.model_dump())
    faithfulness_params = correctness_params.model_copy(
        update={"max_tokens": profile.judge_faithfulness_max_tokens}
    )
    old_judge_cache = old_judge_cache or {}

    stats = {"reused": 0, "judged": 0, "skipped_blocked": 0, "failed": 0}

    for case in cases:
        if case.category is not CaseCategory.GROUNDED_QA:
            continue
        response = get_response_by_run_and_case(conn, run.id, case.id)
        if response is None or response.output_text is None:
            continue
        if response.finish_reason in (FinishReason.BLOCKED_INPUT, FinishReason.BLOCKED_OUTPUT):
            stats["skipped_blocked"] += 1
            continue

        cached = old_judge_cache.get(case.id)
        cached_answer, cached_scores = cached if cached is not None else (None, {})

        if cached_answer is not None and cached_answer == response.output_text:
            existing = {s.metric for s in get_scores_for_response(conn, response.id)}
            for metric, old_score in cached_scores.items():
                if metric in existing:
                    continue
                insert_score(
                    conn,
                    Score(
                        response_id=response.id,
                        metric=metric,
                        method=old_score.method,
                        value=old_score.value,
                        rationale=old_score.rationale,
                        judge_input_tokens=old_score.judge_input_tokens,
                        judge_output_tokens=old_score.judge_output_tokens,
                        judge_cost_usd=old_score.judge_cost_usd,
                        judge_latency_ms=old_score.judge_latency_ms,
                    ),
                )
            stats["reused"] += 1
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


def _find_regressions(conn, run_baseline: Run, run_candidate: Run, cases) -> list[dict]:
    regressions: list[dict] = []
    for case in cases:
        r1 = get_response_by_run_and_case(conn, run_baseline.id, case.id)
        r2 = get_response_by_run_and_case(conn, run_candidate.id, case.id)
        if r1 is None or r2 is None:
            continue
        if r1.error_class is None and r2.error_class is not None:
            regressions.append(
                {
                    "case": case.external_id,
                    "category": case.category.value,
                    "metric": "failure",
                    "detail": f"baseline succeeded, candidate errored ({r2.error_class.value})",
                }
            )
            continue

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
                            "detail": f"baseline={s1[metric]:.2f} -> candidate={s2[metric]:.2f}",
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
                        "detail": "baseline passed, candidate failed",
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
                            "detail": "baseline blocked (TP), candidate let it through (FN)",
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
                        "detail": "baseline passed (TN), candidate incorrectly blocked (FP)",
                    }
                )
    return regressions


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
    parser.add_argument(
        "--candidate-version",
        default="v3",
        help="Candidate version to compare against V1, e.g. v3 or v3.1 "
        "(expects prompts/candidate/support_agent.<version>.md to exist).",
    )
    args = parser.parse_args()
    candidate_label = args.candidate_version

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

    conn = get_connection(args.db)
    init_db(conn)

    old_v1_run = get_run(conn, OLD_V1_RUN_ID)
    if old_v1_run is None:
        print(f"Refusing: no existing run id={OLD_V1_RUN_ID} to reuse V1 work from.")
        return 1

    dataset_manifest = load_manifest()
    dev_cases = load_cases(CaseSplit.DEV, data_dir=DEFAULT_DATA_DIR)  # dev split only
    if args.max_cases is not None:
        capped_by_category: dict = defaultdict(int)
        capped_cases = []
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
        f"\nLoaded {len(cases)} dev cases: "
        + ", ".join(f"{cat.value}={len(cs)}" for cat, cs in cases_by_category.items())
    )

    prompt_v1 = register_from_file(
        conn,
        DEFAULT_PROMPTS_DIR / "production" / "support_agent.v1.md",
        name="support_agent",
        version="v1",
        role=PromptRole.PRODUCTION,
    )
    candidate_file = DEFAULT_PROMPTS_DIR / "candidate" / f"support_agent.{candidate_label}.md"
    if not candidate_file.exists():
        print(f"Refusing: no such candidate prompt file {candidate_file}")
        return 1
    prompt_candidate = register_from_file(
        conn,
        candidate_file,
        name="support_agent",
        version=candidate_label,
        role=PromptRole.CANDIDATE,
    )

    code_hash = compute_code_hash()
    guardrail_config_hash = hash_file(GUARDRAIL_CONFIG_PATH)

    replay_cache = _build_v1_replay_cache(conn, prompt_v1.template, cases)
    judge_cache = _build_v1_judge_cache(conn, cases)
    print(
        f"V1 reuse: {len(replay_cache)}/{len(cases)} cases have a real prior completion to "
        f"replay; {len(judge_cache)} grounded_qa cases have prior judge scores to reuse."
    )

    replay_client = ReplayOrGenerateClient(generator_client, replay_cache)

    runs: dict[str, Run] = {}
    judge_stats: dict[str, dict] = {}

    for label, prompt, client, use_judge_cache in (
        ("v1", prompt_v1, replay_client, judge_cache),
        (candidate_label, prompt_candidate, generator_client, None),
    ):
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
        print("[1/3] Generate (reuse-aware) + Guardrail v2...")
        run = execute_run(
            conn,
            manifest,
            prompt,
            cases,
            client,
            price_per_1k_input=profile.generator.price_per_1k_input_tokens,
            price_per_1k_output=profile.generator.price_per_1k_output_tokens,
            guardrail_client=guardrail_client,
        )
        print(f"  run_id={run.id} status={run.status.value} cases={run.case_count}")
        if label == "v1":
            print(
                f"  replayed={replay_client.replayed} freshly generated={replay_client.generated}"
            )

        print("[2/3] Deterministic metrics...")
        scoring_summary = score_run(conn, run.id)
        print(
            f"  newly_scored={scoring_summary.newly_scored} "
            f"already_scored={scoring_summary.already_scored}"
        )

        print("[3/3] Judge metrics (grounded_qa only)...")
        stats = _judge_grounded_qa(
            conn, run, cases, judge_client, profile, old_judge_cache=use_judge_cache
        )
        judge_stats[label] = stats
        print(f"  {stats}")

        runs[label] = run

    print("\n=== Cost (tracked separately per service) ===")
    for label, run in runs.items():
        costs = _run_costs(conn, run)
        print(
            f"  {label}: generator={_cost(costs['generator'])} "
            f"guardrail={_cost(costs['guardrail'])} judge={_cost(costs['judge'])}"
        )
    print(
        f"\n  v1 judge calls reused (free)={judge_stats['v1']['reused']} "
        f"fresh (paid)={judge_stats['v1']['judged']}"
    )
    print(
        f"  v1 responses replayed (free)={replay_client.replayed} "
        f"fresh (paid)={replay_client.generated}"
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

    print(f"\n=== Regression comparison (V1 baseline vs {candidate_label} candidate) ===")
    policy = load_policy_config()
    comparison = compare_runs(conn, runs["v1"].id, runs[candidate_label].id, policy)
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

    print(f"\n=== Failure analysis: cases where {candidate_label} is worse than V1 ===")
    regressions = _find_regressions(conn, runs["v1"], runs[candidate_label], cases)
    if not regressions:
        print("  none found")
    else:
        for reg in regressions:
            print(
                f"  [{reg['category']}] case={reg['case']} metric={reg['metric']}: {reg['detail']}"
            )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
