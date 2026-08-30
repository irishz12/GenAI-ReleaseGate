#!/usr/bin/env python3
"""One-time maintenance: the four real comparisons (v1v2/v1v3/v2v3 dev,
v1v3 holdout) were computed and persisted before Phase 2 added median_delta/
effect_size/p_value/holm_significant to MetricDelta — their stored rows
predate those fields and deserialize with them as null. This recomputes each
comparison with the current (already-tested) compare_runs, using only
already-persisted response/score/guardrail data — no new AWS calls, no
methodology change, no altered thresholds. It then asserts the release
decision is unchanged before persisting, exactly like
test_holm_correction_never_changes_the_release_decision.

Usage:
    python scripts/recompute_comparisons_with_phase2_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import load_policy_config  # noqa: E402
from evalguard.db.store import get_comparison, get_connection, upsert_comparison  # noqa: E402
from evalguard.regression.compare import compare_runs  # noqa: E402

DEV_DB = REPO_ROOT / "artifacts" / "dev_eval.db"
HOLDOUT_DB = REPO_ROOT / "artifacts" / "holdout_eval.db"


def _recompute(conn, comparison_id: int, policy) -> None:
    before = get_comparison(conn, comparison_id)
    recomputed = compare_runs(conn, before.baseline_run_id, before.candidate_run_id, policy)
    if recomputed.release.status != before.release.status:
        raise RuntimeError(
            f"comparison {comparison_id}: decision changed from "
            f"{before.release.status} to {recomputed.release.status} — refusing to persist"
        )
    after = upsert_comparison(conn, recomputed)
    sample_metric = after.deltas[0]
    print(
        f"comparison {comparison_id}: decision unchanged ({after.release.status.value}); "
        f"e.g. {sample_metric.metric.value} effect_size={sample_metric.effect_size}"
    )


def main() -> int:
    policy = load_policy_config()

    dev_conn = get_connection(DEV_DB)
    for cid in (1, 2, 5):
        _recompute(dev_conn, cid, policy)

    holdout_conn = get_connection(HOLDOUT_DB)
    row = holdout_conn.execute("SELECT id FROM comparisons").fetchone()
    _recompute(holdout_conn, row["id"], policy)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
