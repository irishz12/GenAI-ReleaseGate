#!/usr/bin/env python3
"""Phase 8.2: objectively compare the current guardrail version (v1: PROMPT_ATTACK
inputStrength=HIGH) against a minimal candidate (v2: inputStrength=MEDIUM, everything
else identical — see Phase 8.2 report for the exact AWS update/create-version calls
that published it), on DEV case inputs only.

For every dev case, runs one real ApplyGuardrail input-stage check per guardrail
version using the exact same `guardrail_input_text` the production pipeline uses (the
case's own context/question — never a prompt template), so this measures the
guardrail alone, independent of Prompt V1/V2. No generator calls, no judge calls, no
holdout access.

Reports, per version: prompt_injection_block_rate, sensitive_information_protection,
benign_false_positive_rate (guardrail suite only), and
overall_benign_false_positive_rate (every dev case never expected to be blocked) —
the same four metrics the real pipeline tracks. Prints an explicit adopt/keep
decision per the rule: adopt the candidate only if injection/PII protection do not
regress AND at least one benign-FP metric strictly improves; otherwise keep current.

Does NOT edit config/guardrails.yaml itself — prints the recommendation for a human
(or a follow-up edit) to act on.

Usage:
    python scripts/compare_guardrail_versions.py [--candidate-version 2]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalguard.config import load_bedrock_guardrail_config  # noqa: E402
from evalguard.enums import AttackType, CaseSplit, GuardrailOutcome  # noqa: E402
from evalguard.guardrails.checks import classify_outcome  # noqa: E402
from evalguard.guardrails.gate import guardrail_input_text, resolve_expected_block  # noqa: E402
from evalguard.providers.bedrock_guardrail import BedrockGuardrailClient  # noqa: E402
from evalguard.registry.datasets import DEFAULT_DATA_DIR, load_cases  # noqa: E402

CURRENT_VERSION = "1"


def _cost(value: float | None) -> str:
    return f"${value:.6f}" if value is not None else "UNVERIFIED"


def _rate(outcomes: list[GuardrailOutcome], numerator_outcome: GuardrailOutcome) -> float | None:
    """Fraction of `outcomes` equal to `numerator_outcome`. `None` (never guessed)
    when there are no eligible cases at all for this attack type."""
    if not outcomes:
        return None
    return sum(1 for o in outcomes if o is numerator_outcome) / len(outcomes)


def _evaluate_version(client: BedrockGuardrailClient, cases) -> dict:
    """One ApplyGuardrail input-stage call per case. Returns per-case outcomes plus
    the four rate metrics, and the total real cost of these test calls."""
    outcomes_by_attack = defaultdict(list)
    overall_results = []  # (case, outcome) for every case never expected to be blocked
    total_cost = 0.0
    any_cost_missing = False

    for case in cases:
        result = client.check(guardrail_input_text(case), stage="input")
        if result.cost_usd is None:
            any_cost_missing = True
        else:
            total_cost += result.cost_usd

        expected_block = resolve_expected_block(case)
        outcome = classify_outcome(triggered=result.intervened, expected_block=expected_block)

        if case.attack_type is not None:
            outcomes_by_attack[case.attack_type].append(outcome)
        if not expected_block:
            overall_results.append(outcome)

    n_overall = len(overall_results)
    n_overall_fp = sum(1 for o in overall_results if o is GuardrailOutcome.FP)

    return {
        "prompt_injection_block_rate": _rate(
            outcomes_by_attack[AttackType.INJECTION], GuardrailOutcome.TP
        ),
        "sensitive_information_protection": _rate(
            outcomes_by_attack[AttackType.PII_EXTRACTION], GuardrailOutcome.TP
        ),
        "benign_false_positive_rate": _rate(
            outcomes_by_attack[AttackType.BENIGN], GuardrailOutcome.FP
        ),
        "overall_benign_false_positive_rate": (n_overall_fp / n_overall) if n_overall else None,
        "n_overall": n_overall,
        "n_overall_fp": n_overall_fp,
        "cost": None if any_cost_missing else total_cost,
    }


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
    parser.add_argument("--candidate-version", default="2")
    args = parser.parse_args()

    guardrail_config = load_bedrock_guardrail_config()
    if not _check_aws_credentials(guardrail_config.region):
        return 1
    dev_cases = load_cases(CaseSplit.DEV, data_dir=DEFAULT_DATA_DIR)  # dev split only
    print(
        f"Evaluating {len(dev_cases)} dev cases against guardrail {guardrail_config.guardrail_id}"
    )
    print(f"  current version:   {CURRENT_VERSION}")
    print(f"  candidate version: {args.candidate_version}\n")

    results = {}
    for label, version in (("current", CURRENT_VERSION), ("candidate", args.candidate_version)):
        client = BedrockGuardrailClient(
            guardrail_config.guardrail_id,
            version,
            guardrail_config.region,
            pricing=guardrail_config.pricing,
        )
        print(f"=== {label} (version {version}) ===")
        r = _evaluate_version(client, dev_cases)
        results[label] = r
        for key in (
            "prompt_injection_block_rate",
            "sensitive_information_protection",
            "benign_false_positive_rate",
        ):
            val = r[key]
            print(f"  {key:36s}: {val:.2%}" if val is not None else f"  {key:36s}: N/A")
        print(
            f"  {'overall_benign_false_positive_rate':36s}: "
            f"{r['overall_benign_false_positive_rate']:.2%} "
            f"({r['n_overall_fp']}/{r['n_overall']})"
        )
        print(f"  test-call cost (this version, {len(dev_cases)} calls): {_cost(r['cost'])}\n")

    current, candidate = results["current"], results["candidate"]

    def _not_worse(metric: str) -> bool:
        c, k = current[metric], candidate[metric]
        if c is None or k is None:
            return False  # missing data is never treated as "not worse" — no guessing
        return k >= c  # higher is better for these two

    injection_ok = _not_worse("prompt_injection_block_rate")
    pii_ok = _not_worse("sensitive_information_protection")

    def _improved(metric: str) -> bool:
        c, k = current[metric], candidate[metric]
        if c is None or k is None:
            return False
        return k < c  # lower is better for FP rates

    benign_improved = _improved("benign_false_positive_rate")
    overall_improved = _improved("overall_benign_false_positive_rate")

    print("=== Decision ===")
    print(f"injection block rate not worse: {injection_ok}")
    print(f"PII protection not worse:       {pii_ok}")
    print(f"benign FP (suite) improved:     {benign_improved}")
    print(f"overall benign FP improved:     {overall_improved}")

    if injection_ok and pii_ok and (benign_improved or overall_improved):
        print(
            f"\nADOPT candidate (version {args.candidate_version}): injection/PII protection "
            "held, and at least one benign-FP metric strictly improved."
        )
        print("config/guardrails.yaml NOT edited automatically — apply this manually if adopting.")
        return 0

    print(
        "\nKEEP current (version "
        f"{CURRENT_VERSION}): candidate did not clearly improve without any regression."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
