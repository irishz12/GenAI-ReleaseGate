#!/usr/bin/env python3
"""Phase 8.1 investigation: why did the frozen judge (openai.gpt-oss-120b,
max_tokens=512) never produce valid faithfulness JSON for 6 grounded_qa responses
across the Phase 8 dev run, even after 3 full resume/retry passes?

Uses ONLY already-stored responses from artifacts/dev_eval.db — no generator call of
any kind. The one live call this makes is the judge itself (openai.gpt-oss-120b,
temperature=0, same faithfulness prompt template), varying only max_tokens, to find
the smallest budget that reliably lets a real reasoning-model judge call finish its
JSON instead of running out of budget mid "thought."

This does NOT persist any Score rows and does NOT change config/models.yaml's frozen
judge max_tokens=512 — it's a recommendation, not an applied tune. Findings only.

Usage:
    python scripts/investigate_judge_budget.py [--db path/to/dev_eval.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import load_bedrock_mantle_profile  # noqa: E402
from evalguard.db.store import (  # noqa: E402
    get_case_by_external_id,
    get_connection,
    get_response_by_run_and_case,
    get_scores_for_response,
)
from evalguard.enums import CaseCategory, MetricName  # noqa: E402
from evalguard.models import EvaluationCase, Response  # noqa: E402
from evalguard.providers.base import CompletionParams  # noqa: E402
from evalguard.providers.bedrock_mantle import BedrockMantleClient  # noqa: E402
from evalguard.quality.judge import (  # noqa: E402
    MalformedJudgeOutputError,
    build_blind_judge_case,
    parse_judge_json,
    render_judge_prompt,
)
from evalguard.quality.judge_schemas import FaithfulnessVerdict  # noqa: E402
from evalguard.quality.scoring import require_dev_case  # noqa: E402

JUDGE_PROMPTS_DIR = REPO_ROOT / "prompts" / "judge"
FROZEN_MAX_TOKENS = 512
CANDIDATE_BUDGETS = [FROZEN_MAX_TOKENS, 768, 1024, 1280]

# (run_id, external_id) for every grounded_qa response that never got a faithfulness
# score after 3 full Phase 8 resume/retry passes — found by directly querying
# artifacts/dev_eval.db (see Phase 8.1 report).
FAILING_CASES = [
    (1, "gqa-0068"),
    (1, "gqa-0002"),
    (1, "gqa-0045"),
    (1, "gqa-0053"),
    (1, "gqa-0037"),
    (2, "gqa-0076"),
]


def _confirmed_failing_cases(conn, db_path: str) -> list[tuple[str, EvaluationCase, Response]]:
    confirmed = []
    for run_id, external_id in FAILING_CASES:
        case = get_case_by_external_id(conn, external_id)
        if case is None:
            print(f"  SKIP {external_id}: case not found in {db_path}")
            continue
        if case.category is not CaseCategory.GROUNDED_QA:
            print(f"  SKIP {external_id}: not grounded_qa")
            continue
        require_dev_case(case)
        response = get_response_by_run_and_case(conn, run_id, case.id)
        if response is None or response.output_text is None:
            print(f"  SKIP {external_id}: no stored response for run {run_id}")
            continue
        already_has = any(
            s.metric is MetricName.FAITHFULNESS for s in get_scores_for_response(conn, response.id)
        )
        if already_has:
            print(f"  SKIP {external_id}: already has a faithfulness score (not a failing case)")
            continue
        confirmed.append((external_id, case, response))
    return confirmed


def _run_pass(
    cases_and_responses: list[tuple[str, EvaluationCase, Response]],
    judge_client: BedrockMantleClient,
    template: str,
    params: CompletionParams,
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for external_id, case, response in cases_and_responses:
        blind_case = build_blind_judge_case(case, response)
        prompt = render_judge_prompt(template, blind_case)
        completion = judge_client.complete(prompt, params)
        try:
            parse_judge_json(completion.text, FaithfulnessVerdict)
            results[external_id] = True
        except MalformedJudgeOutputError as exc:
            results[external_id] = False
            print(f"    {external_id}: FAILED ({exc})")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "artifacts" / "dev_eval.db"))
    args = parser.parse_args()

    conn = get_connection(args.db)
    profile = load_bedrock_mantle_profile()
    judge_client = BedrockMantleClient(model_id=profile.judge.model_id)
    faithfulness_template = (JUDGE_PROMPTS_DIR / "faithfulness.v1.md").read_text()

    print(f"Judge: {profile.judge.model_id} (temperature=0, top_p=1.0, frozen)")
    print(f"Candidate budgets: {CANDIDATE_BUDGETS}\n")

    cases_and_responses = _confirmed_failing_cases(conn, args.db)
    print(f"\nConfirmed {len(cases_and_responses)}/{len(FAILING_CASES)} still-failing cases.\n")

    smallest_reliable_budget = None
    for budget in CANDIDATE_BUDGETS:
        print(f"=== budget={budget} ===")
        params = CompletionParams(temperature=0.0, max_tokens=budget, top_p=1.0)
        results = _run_pass(cases_and_responses, judge_client, faithfulness_template, params)
        n_ok = sum(1 for ok in results.values() if ok)
        print(f"  pass 1: {n_ok}/{len(cases_and_responses)} parsed successfully")
        if n_ok < len(cases_and_responses):
            continue

        # Confirm reliability with a second pass at the same budget before declaring it.
        results2 = _run_pass(cases_and_responses, judge_client, faithfulness_template, params)
        n_ok2 = sum(1 for ok in results2.values() if ok)
        total = len(cases_and_responses)
        print(f"  pass 2 (reliability check): {n_ok2}/{total} parsed successfully")
        if n_ok2 == len(cases_and_responses):
            smallest_reliable_budget = budget
            break
        print("  -> not reliable across 2 passes, escalating")

    print()
    if smallest_reliable_budget is not None:
        print(
            f"RESULT: max_tokens={smallest_reliable_budget} reliably parses all "
            f"{len(cases_and_responses)} previously-stuck cases (2/2 passes)."
        )
    else:
        print(f"RESULT: none of {CANDIDATE_BUDGETS} reliably cleared all cases in 2 passes.")
    print("No Score rows were written; config/models.yaml's frozen judge max_tokens is unchanged.")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
