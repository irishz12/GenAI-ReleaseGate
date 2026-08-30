#!/usr/bin/env python3
"""Phase 8.2: re-score every grounded_qa response's faithfulness (and derived
hallucination) at the newly frozen budget (max_tokens=1024 — see
config/models.yaml's judge_faithfulness_max_tokens, evidence in Phase 8.1), so every
faithfulness score in artifacts/dev_eval.db comes from the same budget instead of a
mix of "most at 512, a few retried at 1024/etc."

Uses ONLY already-stored responses — no generator calls, no correctness re-scoring
(correctness stays frozen at 512 and already succeeded for every case). Deletes each
grounded_qa response's existing faithfulness/hallucination scores first (whatever
budget they came from), then re-runs the same judge orchestration
(quality.judge.score_grounded_qa_case_with_judge) so it re-derives them fresh and
uniformly. Finishes by recomputing and upserting the V1/V2 comparison so it reflects
the fully re-scored, budget-consistent data (Phase 8.2 comparison-persistence fix).

Usage:
    python scripts/rescore_faithfulness.py [--db path/to/dev_eval.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import load_bedrock_mantle_profile, load_policy_config  # noqa: E402
from evalguard.db.store import (  # noqa: E402
    delete_score,
    get_case,
    get_connection,
    get_responses_for_run,
    get_scores_for_response,
    upsert_comparison,
)
from evalguard.enums import CaseCategory, MetricName  # noqa: E402
from evalguard.providers.base import CompletionParams  # noqa: E402
from evalguard.providers.bedrock_mantle import BedrockMantleClient  # noqa: E402
from evalguard.quality.judge import (  # noqa: E402
    JudgeRetryExhaustedError,
    MalformedJudgeOutputError,
    score_grounded_qa_case_with_judge,
)
from evalguard.regression.compare import compare_runs  # noqa: E402

JUDGE_PROMPTS_DIR = REPO_ROOT / "prompts" / "judge"
RUN_IDS = {"v1": 1, "v2": 2}


def _cost(value: float | None) -> str:
    return f"${value:.6f}" if value is not None else "UNVERIFIED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "artifacts" / "dev_eval.db"))
    args = parser.parse_args()

    conn = get_connection(args.db)
    profile = load_bedrock_mantle_profile()
    judge_client = BedrockMantleClient(model_id=profile.judge.model_id)

    correctness_template = (JUDGE_PROMPTS_DIR / "correctness.v1.md").read_text()
    faithfulness_template = (JUDGE_PROMPTS_DIR / "faithfulness.v1.md").read_text()
    correctness_params = CompletionParams(**profile.judge.params.model_dump())
    faithfulness_params = correctness_params.model_copy(
        update={"max_tokens": profile.judge_faithfulness_max_tokens}
    )
    print(f"Judge: {profile.judge.model_id}")
    print(f"correctness max_tokens={correctness_params.max_tokens} (unchanged, not re-scored)")
    print(f"faithfulness max_tokens={faithfulness_params.max_tokens} (re-scoring all grounded_qa)")
    print()

    total_cost_by_run: dict[str, float | None] = {}

    for label, run_id in RUN_IDS.items():
        print(f"=== {label} (run_id={run_id}) ===")
        responses = get_responses_for_run(conn, run_id)
        cleared = 0
        rescored = 0
        failed = 0

        for response in responses:
            case = get_case(conn, response.case_id)
            if case is None or case.category is not CaseCategory.GROUNDED_QA:
                continue
            if response.output_text is None:
                continue

            existing = {s.metric for s in get_scores_for_response(conn, response.id)}
            if MetricName.FAITHFULNESS in existing:
                delete_score(conn, response.id, MetricName.FAITHFULNESS)
                cleared += 1
            if MetricName.HALLUCINATION in existing:
                delete_score(conn, response.id, MetricName.HALLUCINATION)

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
                if any(s.metric is MetricName.FAITHFULNESS for s in new_scores):
                    rescored += 1
            except (MalformedJudgeOutputError, JudgeRetryExhaustedError) as exc:
                failed += 1
                print(f"    judge FAILED for case={case.external_id}: {exc}")

        print(f"  cleared (old faithfulness scores)={cleared}")
        print(f"  rescored at max_tokens=1024={rescored}")
        print(f"  still failed={failed}")

        judge_costs = [
            s.judge_cost_usd
            for r in get_responses_for_run(conn, run_id)
            for s in get_scores_for_response(conn, r.id)
            if s.judge_cost_usd is not None
        ]
        total_cost_by_run[label] = sum(judge_costs) if judge_costs else None
        print(f"  total judge cost (all metrics, this run): {_cost(total_cost_by_run[label])}\n")

    print("=== Final grounded_qa faithfulness coverage ===")
    for label, run_id in RUN_IDS.items():
        responses = get_responses_for_run(conn, run_id)
        n_gqa = 0
        n_faith = 0
        for r in responses:
            case = get_case(conn, r.case_id)
            if case is None or case.category is not CaseCategory.GROUNDED_QA:
                continue
            if r.output_text is None:
                continue
            n_gqa += 1
            metrics = {s.metric for s in get_scores_for_response(conn, r.id)}
            if MetricName.FAITHFULNESS in metrics:
                n_faith += 1
        print(f"  {label}: {n_faith}/{n_gqa} grounded_qa responses have a faithfulness score")

    print("\n=== Refreshed V1 vs V2 comparison ===")
    policy = load_policy_config()
    comparison = compare_runs(conn, RUN_IDS["v1"], RUN_IDS["v2"], policy)
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
