"""Stage A: GENERATE — executes one prompt version over a frozen case set, persisting
one immutable response per case (docs/ARCHITECTURE.md §1, §3, §11).

Deliberately synchronous and single-attempt-per-case: `FakeClient` has no real I/O
latency to hide behind concurrency, so bounded-concurrency/async is deferred to Phase 3,
when a real network-bound provider (Bedrock Mantle) exists to make that complexity pay
for itself. Likewise, retry/backoff on transient errors is a Phase 3 concern — Phase 2
classifies and records a failure (see evalguard.errors) but does not itself retry.

Resumable by construction: `execute_run` doesn't create a fresh row every call. A run is
identified by its manifest's `config_hash` — call it again with the same manifest and it
finds the existing `runs` row and only processes cases that don't already have a
response, whatever the reason execution stopped last time (crash, kill, ^C). A case that
previously errored is left alone by default; pass `retry_failed=True` to explicitly
re-attempt those specific cases without disturbing the ones that already succeeded.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from evalguard.db.store import (
    delete_response,
    get_responses_for_run,
    get_run_by_config_hash,
    insert_guardrail_result,
    insert_response,
    insert_run,
    update_response_cost,
    update_run_status,
)
from evalguard.enums import FinishReason, GuardrailStage, RunStatus
from evalguard.errors import classify_error
from evalguard.guardrails.gate import (
    build_guardrail_result,
    guardrail_input_text,
    resolve_expected_block,
)
from evalguard.models import EvaluationCase, Prompt, Response, Run
from evalguard.providers.base import CompletionParams, GuardrailClient, LLMClient
from evalguard.runner.manifest import RunManifest, manifest_to_run
from evalguard.runner.rendering import render_prompt


def compute_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    price_per_1k_input: float | None = None,
    price_per_1k_output: float | None = None,
) -> float | None:
    """Token counts × pricing table (docs/ARCHITECTURE.md §5) — or `None` if pricing
    hasn't been verified and configured. Never guesses: an unset price returns `None`,
    not a silently-wrong `0.0` that would understate real spend. `FakeClient` runs get
    a real `0.0` here only because config/models.yaml's fake entries explicitly set
    both prices to `0.0` (a verified fact, not a default) and the caller passes those
    through explicitly."""
    if price_per_1k_input is None or price_per_1k_output is None:
        return None
    return (input_tokens / 1000) * price_per_1k_input + (output_tokens / 1000) * price_per_1k_output


def backfill_response_costs(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    price_per_1k_input: float | None,
    price_per_1k_output: float | None,
) -> int:
    """Recompute `cost_usd` for every successful response in a run from its already-
    stored token counts, using pricing that may have been verified *after* the run
    executed (e.g. Phase 3's smoke test ran before Phase 4 verified real pricing).

    No generation happens here — only `update_response_cost` (a pure recompute over
    stored tokens) is called, and only for responses whose `cost_usd` doesn't already
    match what the current pricing would produce. Returns the number of responses
    actually updated.
    """
    updated = 0
    for response in get_responses_for_run(conn, run_id):
        if response.error_class is not None or response.input_tokens is None:
            continue  # failed or otherwise unpriceable — nothing to recompute
        new_cost = compute_cost_usd(
            response.input_tokens,
            response.output_tokens,
            price_per_1k_input=price_per_1k_input,
            price_per_1k_output=price_per_1k_output,
        )
        if new_cost != response.cost_usd:
            update_response_cost(conn, response.id, new_cost)
            updated += 1
    return updated


def _get_or_create_run(conn: sqlite3.Connection, manifest: RunManifest, prompt_id: int) -> Run:
    existing = get_run_by_config_hash(conn, manifest.config_hash)
    if existing is not None:
        return existing
    return insert_run(conn, manifest_to_run(manifest, prompt_id))


def execute_run(
    conn: sqlite3.Connection,
    manifest: RunManifest,
    prompt: Prompt,
    cases: list[EvaluationCase],
    client: LLMClient,
    *,
    price_per_1k_input: float | None = None,
    price_per_1k_output: float | None = None,
    retry_failed: bool = False,
    guardrail_client: GuardrailClient | None = None,
) -> Run:
    """Execute (or resume) one run. See module docstring for resume semantics.

    Both `prompt` and every case in `cases` must already be registered (have a real DB
    `id`) — `registry.prompts.register(_from_file)` and
    `registry.datasets.ensure_cases_registered` do that. This function only writes
    `responses`; it never registers a prompt or a case itself, to keep "what identity
    does this belong to" and "what did the generator produce" as separate concerns.

    `guardrail_client` is optional and `None` by default — omitting it reproduces
    pre-Phase-6 behavior exactly (no guardrail check, no `guardrail_results` rows).
    When present, the same client/config is used for every case regardless of prompt
    version, so V1 and V2 are always compared under identical guardrail conditions
    (docs/ARCHITECTURE.md Guardrail Layer). Every case's input is checked first; a
    blocked input short-circuits generation entirely (`FinishReason.BLOCKED_INPUT`,
    `client.complete` never called, no cost/tokens recorded). Otherwise generation
    proceeds and, on success, its output is checked too; a blocked output keeps the
    generation's real cost/tokens/latency but replaces `output_text` with the
    guardrail's own message and sets `FinishReason.BLOCKED_OUTPUT`.
    """
    if prompt.id is None:
        raise ValueError("prompt must be registered (have a DB id) before executing a run")
    if any(case.id is None for case in cases):
        raise ValueError("all cases must be registered (have a DB id) before executing a run")

    run = _get_or_create_run(conn, manifest, prompt.id)

    existing_by_case_id = {r.case_id: r for r in get_responses_for_run(conn, run.id)}

    if retry_failed:
        for response in existing_by_case_id.values():
            if response.error_class is not None:
                delete_response(conn, response.id)
        existing_by_case_id = {r.case_id: r for r in get_responses_for_run(conn, run.id)}

    pending = [case for case in cases if case.id not in existing_by_case_id]

    params = CompletionParams(**manifest.model_params)

    for case in pending:
        prompt_text = render_prompt(prompt.template, case)
        expected_block = resolve_expected_block(case)

        input_check = (
            guardrail_client.check(guardrail_input_text(case), stage="input")
            if guardrail_client is not None
            else None
        )
        output_check = None

        if input_check is not None and input_check.intervened:
            response = Response(
                run_id=run.id,
                case_id=case.id,
                output_text=input_check.output_text,
                finish_reason=FinishReason.BLOCKED_INPUT,
                error_class=None,
            )
        else:
            try:
                completion = client.complete(prompt_text, params)
                output_check = (
                    guardrail_client.check(completion.text, stage="output")
                    if guardrail_client is not None
                    else None
                )
                blocked_output = output_check is not None and output_check.intervened
                response = Response(
                    run_id=run.id,
                    case_id=case.id,
                    output_text=output_check.output_text if blocked_output else completion.text,
                    finish_reason=(
                        FinishReason.BLOCKED_OUTPUT if blocked_output else completion.finish_reason
                    ),
                    input_tokens=completion.usage.input_tokens,
                    output_tokens=completion.usage.output_tokens,
                    cost_usd=compute_cost_usd(
                        completion.usage.input_tokens,
                        completion.usage.output_tokens,
                        price_per_1k_input=price_per_1k_input,
                        price_per_1k_output=price_per_1k_output,
                    ),
                    latency_ms=completion.latency_ms,
                    error_class=None,
                )
            except Exception as exc:  # noqa: BLE001 — deliberately broad: one bad case
                # must never abort the run (docs/ARCHITECTURE.md §11 per-case isolation).
                response = Response(
                    run_id=run.id,
                    case_id=case.id,
                    output_text=None,
                    finish_reason=FinishReason.ERROR,
                    error_class=classify_error(exc),
                )

        response = insert_response(conn, response)

        if input_check is not None:
            insert_guardrail_result(
                conn,
                build_guardrail_result(
                    input_check,
                    response_id=response.id,
                    stage=GuardrailStage.INPUT,
                    expected_block=expected_block,
                ),
            )
        if output_check is not None:
            insert_guardrail_result(
                conn,
                build_guardrail_result(
                    output_check,
                    response_id=response.id,
                    stage=GuardrailStage.OUTPUT,
                    expected_block=expected_block,
                ),
            )

    all_responses = get_responses_for_run(conn, run.id)
    is_complete = len(all_responses) >= len(cases)

    priced_responses = [r for r in all_responses if r.cost_usd is not None]
    total_cost_usd = sum(r.cost_usd for r in priced_responses) if priced_responses else None

    return update_run_status(
        conn,
        run.id,
        status=RunStatus.COMPLETED if is_complete else RunStatus.RUNNING,
        case_count=len(cases),
        failure_count=sum(1 for r in all_responses if r.error_class is not None),
        total_cost_usd=total_cost_usd,
        finished_at=datetime.now(UTC) if is_complete else None,
    )
