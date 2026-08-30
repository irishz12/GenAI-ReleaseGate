#!/usr/bin/env python3
"""GenAI ReleaseGate — the secondary comparison, V3 vs V2, computed from responses
already persisted in artifacts/dev_eval.db. Pure post-hoc analysis: no generator,
judge, or guardrail calls, nothing paid, fully reproducible from data on disk.

IMPORTANT CAVEAT (read before trusting the safety-metric rows of this comparison):
V2's run (run_id=2) is its original Phase 8 run, evaluated under Guardrail v1
(PROMPT_ATTACK=HIGH). V3's run (run_id=4) is from Phase 8.4, evaluated under the
later-adopted Guardrail v2 (PROMPT_ATTACK=MEDIUM). V2 was never re-run under
Guardrail v2. Quality metrics (answer_correctness, faithfulness, hallucination,
instruction_following, abstention_accuracy) are apples-to-apples between these two
runs — the guardrail version doesn't affect them. The three guardrail-derived
metrics (prompt_injection_block_rate, sensitive_information_protection,
benign_false_positive_rate) are NOT apples-to-apples here: any difference reflects
the guardrail version change, not a difference between the V2 and V3 prompts. This
script prints that caveat prominently rather than presenting those three metrics as
if they were comparable.

Usage:
    python scripts/compute_v3_vs_v2.py [--db path/to/dev_eval.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import load_policy_config  # noqa: E402
from evalguard.db.store import get_connection, get_prompt, get_run, upsert_comparison  # noqa: E402
from evalguard.enums import MetricName  # noqa: E402
from evalguard.regression.compare import compare_runs  # noqa: E402

GUARDRAIL_DERIVED_METRICS = {
    MetricName.PROMPT_INJECTION_BLOCK_RATE,
    MetricName.SENSITIVE_INFORMATION_PROTECTION,
    MetricName.BENIGN_FALSE_POSITIVE_RATE,
    MetricName.OVERALL_BENIGN_FALSE_POSITIVE_RATE,
}

# The persisted runs this comparison pairs — see module docstring for the guardrail-
# version caveat that applies to it.
V2_RUN_ID = 2
V3_RUN_ID = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "artifacts" / "dev_eval.db"))
    args = parser.parse_args()

    conn = get_connection(args.db)

    v2_run = get_run(conn, V2_RUN_ID)
    v3_run = get_run(conn, V3_RUN_ID)
    if v2_run is None or v3_run is None:
        print(f"Refusing: expected runs {V2_RUN_ID} (V2) and {V3_RUN_ID} (V3) to exist.")
        return 1
    v2_prompt = get_prompt(conn, v2_run.prompt_id)
    v3_prompt = get_prompt(conn, v3_run.prompt_id)
    if v2_prompt.version != "v2" or v3_prompt.version != "v3":
        print(
            f"Refusing: run {V2_RUN_ID} is prompt {v2_prompt.version!r}, "
            f"run {V3_RUN_ID} is prompt {v3_prompt.version!r} — expected v2 and v3."
        )
        return 1

    policy = load_policy_config()
    comparison = compare_runs(conn, V2_RUN_ID, V3_RUN_ID, policy)
    comparison = upsert_comparison(conn, comparison)

    print("=== V3 vs V2 (baseline=V2, candidate=V3) ===")
    print(
        "*** CAVEAT: V2's run predates the Guardrail v2 adoption (Phase 8.2) — it was "
        "never re-tested under v2. Quality metrics below are apples-to-apples; the "
        "guardrail-derived metrics (marked below) are NOT — any difference reflects "
        "the guardrail version change, not the V2->V3 prompt change. ***\n"
    )
    print(f"n_paired={comparison.n_paired}")
    print(f"decision={comparison.release.status.value}")
    print(f"summary={comparison.release.summary}")
    for gate in comparison.release.gates:
        print(f"  gate={gate.gate_name:32s} status={gate.status.value:8s} reason={gate.reason}")

    print("\ndeltas:")
    for delta in comparison.deltas:
        is_confounded = delta.metric in GUARDRAIL_DERIVED_METRICS
        caveat = " [GUARDRAIL-VERSION CONFOUNDED]" if is_confounded else ""
        ci = f" CI=[{delta.ci_low:+.4f}, {delta.ci_high:+.4f}]" if delta.ci_low is not None else ""
        print(
            f"  {delta.metric.value:32s} baseline={delta.baseline_value:.4f} "
            f"candidate={delta.candidate_value:.4f} delta={delta.delta:+.4f}{ci}{caveat}"
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
