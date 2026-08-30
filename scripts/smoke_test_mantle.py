#!/usr/bin/env python3
"""Phase 3 tiny real smoke test: confirm the generator via /models, then 1 connectivity
call + up to 2 real dev cases against Bedrock Mantle.

Deliberately tiny, dev-only, and availability-gated:
- 1 GET /models call FIRST — the frozen generator id must appear in the response
  before any inference is attempted. If it doesn't, this script stops and reports;
  it never substitutes a different model.
- exactly 1 chat-completion call after that (BedrockMantleClient.check_connectivity)
- at most 2 cases from data/dev/ (never data/holdout/)
- no full evaluation sweep, no judge scoring, no guardrail scoring
- persists to a *separate* smoke-test DB by default so it never mixes into
  artifacts/eval.db's real run history

Requires:
- .env with MANTLE_BASE_URL + MANTLE_API_KEY (see .env.example)
- config/models.yaml's `bedrock_mantle.generator` section — frozen model ids, not
  guessed; do not edit this to substitute a model without explicit instruction.

Security: never prints/logs the API key. Only reports token counts, latency, and cost.

Usage:
    python scripts/smoke_test_mantle.py [--max-cases 2] [--db path/to/smoke.db]
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
from evalguard.db.store import get_connection, get_responses_for_run, init_db  # noqa: E402
from evalguard.enums import CaseSplit, PromptRole  # noqa: E402
from evalguard.hashing import hash_file  # noqa: E402
from evalguard.providers.bedrock_mantle import BedrockMantleClient, MantleConfigError  # noqa: E402
from evalguard.registry.datasets import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ensure_cases_registered,
    load_cases,
    load_manifest,
)
from evalguard.registry.prompts import DEFAULT_PROMPTS_DIR, register_from_file  # noqa: E402
from evalguard.runner.manifest import build_run_manifest, compute_code_hash  # noqa: E402
from evalguard.runner.runner import compute_cost_usd, execute_run  # noqa: E402

GUARDRAIL_CONFIG_PATH = REPO_ROOT / "config" / "guardrails.yaml"


def _format_cost(value) -> str:
    return f"${value:.6f}" if value is not None else "UNVERIFIED (no pricing configured)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--db", default=str(REPO_ROOT / "artifacts" / "smoke_mantle.db"))
    args = parser.parse_args()

    if args.max_cases > 2:
        print("Refusing to run more than 2 dev cases in the Phase 3 smoke test.")
        return 1

    # ── resolve config (never prints/logs the resolved key itself) ─────────────
    try:
        profile = load_bedrock_mantle_profile()
    except BedrockMantleModelsNotConfiguredError as exc:
        print(f"Config error: {exc}")
        return 1

    try:
        client = BedrockMantleClient(model_id=profile.generator.model_id)
    except MantleConfigError as exc:
        print(f"Credentials error: {exc}")
        return 1

    print(f"Frozen generator model: {profile.generator.model_id}")
    print(f"Frozen judge model (not called this phase): {profile.judge.model_id}")
    print(f"Endpoint: {client.base_url}")  # base URL only — never the key

    # ── 0. confirm the frozen generator is actually available (1 real call) ────
    print("\n[0/2] Checking /models for the frozen generator id...")
    try:
        available_models = client.list_models()
    except Exception as exc:  # noqa: BLE001 — top-level smoke test, report and exit
        print(f"/models call FAILED: {exc}")
        return 1

    print(f"  {len(available_models)} model(s) returned by /models")
    if profile.generator.model_id not in available_models:
        print(
            f"\nGenerator '{profile.generator.model_id}' is NOT in the /models list. "
            "Stopping — not substituting another model."
        )
        if available_models:
            print("  Available model ids:")
            for model_id in available_models:
                print(f"    - {model_id}")
        else:
            print("  (no model ids parsed from the /models response)")
        return 1

    print(f"  Confirmed: '{profile.generator.model_id}' is available.")

    # ── 1. connectivity call ────────────────────────────────────────────────────
    print("\n[1/2] Connectivity check (1 real call)...")
    try:
        completion = client.check_connectivity()
    except Exception as exc:  # noqa: BLE001 — top-level smoke test, report and exit
        print(f"Connectivity check FAILED: {exc}")
        return 1

    print(f"  response text: {completion.text!r}")
    print(
        f"  tokens: input={completion.usage.input_tokens} output={completion.usage.output_tokens}"
    )
    print(f"  latency: {completion.latency_ms} ms")
    connectivity_cost = compute_cost_usd(
        completion.usage.input_tokens,
        completion.usage.output_tokens,
        price_per_1k_input=profile.generator.price_per_1k_input_tokens,
        price_per_1k_output=profile.generator.price_per_1k_output_tokens,
    )
    print(f"  cost: {_format_cost(connectivity_cost)}")

    # ── 2. up to 2 real DEV cases through the actual runner pipeline ───────────
    print(f"\n[2/2] Running {args.max_cases} real dev case(s) through execute_run...")

    conn = get_connection(args.db)
    init_db(conn)

    dataset_manifest = load_manifest()
    dev_cases = load_cases(CaseSplit.DEV, data_dir=DEFAULT_DATA_DIR)  # never holdout
    cases = ensure_cases_registered(conn, dev_cases[: args.max_cases])

    prompt = register_from_file(
        conn,
        DEFAULT_PROMPTS_DIR / "production" / "support_agent.v1.md",
        name="support_agent",
        version="v1",
        role=PromptRole.PRODUCTION,
    )

    manifest = build_run_manifest(
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

    run = execute_run(
        conn,
        manifest,
        prompt,
        cases,
        client,
        price_per_1k_input=profile.generator.price_per_1k_input_tokens,
        price_per_1k_output=profile.generator.price_per_1k_output_tokens,
    )

    responses = get_responses_for_run(conn, run.id)
    print(
        f"\nrun_id={run.id} status={run.status.value} cases={run.case_count} "
        f"failures={run.failure_count}"
    )
    for r in responses:
        print(
            f"  case_id={r.case_id} finish={r.finish_reason.value} "
            f"tokens(in/out)={r.input_tokens}/{r.output_tokens} "
            f"latency={r.latency_ms}ms cost={_format_cost(r.cost_usd)}"
        )
    print(f"\nrun total cost: {_format_cost(run.total_cost_usd)}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
