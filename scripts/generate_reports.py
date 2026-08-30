#!/usr/bin/env python3
"""GenAI ReleaseGate — Phase 3: generates the structured JSON reports for every
real, already-persisted comparison in artifacts/dev_eval.db and
artifacts/holdout_eval.db. Pure reshaping of existing rows via
evalguard.reporting.build_experiment_report — no new statistics, no new AWS calls,
nothing fabricated. Output lands under results/reports/<experiment_id>.json.

Usage:
    python scripts/generate_reports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import load_policy_config  # noqa: E402
from evalguard.db.store import get_comparison, get_connection, get_prompt, get_run  # noqa: E402
from evalguard.reporting import build_experiment_report, write_report  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "reports"
DEV_DB = REPO_ROOT / "artifacts" / "dev_eval.db"
HOLDOUT_DB = REPO_ROOT / "artifacts" / "holdout_eval.db"


def _report_one(conn, comparison_id: int, dataset: str, policy) -> str:
    comparison = get_comparison(conn, comparison_id)
    baseline_prompt = get_prompt(conn, get_run(conn, comparison.baseline_run_id).prompt_id)
    candidate_prompt = get_prompt(conn, get_run(conn, comparison.candidate_run_id).prompt_id)
    experiment_id = f"{baseline_prompt.version}_vs_{candidate_prompt.version}_{dataset}"

    report = build_experiment_report(
        conn,
        comparison,
        policy,
        experiment_id=experiment_id,
        baseline_version=baseline_prompt.version,
        candidate_version=candidate_prompt.version,
        dataset=dataset,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_report(report, OUT_DIR / f"{experiment_id}.json")
    print(f"wrote {experiment_id}.json  decision={report.decision}")
    return experiment_id


def main() -> int:
    policy = load_policy_config()

    dev_conn = get_connection(DEV_DB)
    # comparison ids from db/store.py's UNIQUE(baseline_run_id, candidate_run_id):
    # 1=V1vV2, 2=V1vV3, 3=V1vV3.1, 4=V1vV3.2, 5=V2vV3 (Phase 1),
    # 6=V1vV3.3, 7=V1vV3.4 (the real GO).
    for cid in (1, 2, 3, 4, 5, 6, 7):
        _report_one(dev_conn, cid, dataset="dev", policy=policy)

    holdout_conn = get_connection(HOLDOUT_DB)
    # 1=V1vV3 (the original headline REVIEW), 2=V1vV3.4 (the real holdout GO).
    for cid in (1, 2):
        _report_one(holdout_conn, cid, dataset="holdout", policy=policy)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
