"""The LLM judge: renders a blind prompt, calls the judge model, validates its output
against a strict schema with bounded retry, and derives faithfulness/hallucination
from claim labels rather than trusting the judge's own arithmetic.

Blindness is structural, not a convention to remember: `build_blind_judge_case` only
accepts `(case, response)` — there is no parameter a run's model_id or a prompt's
version/role could flow through, even by accident. The rendered prompt only ever
contains `BlindJudgeCase`'s four fields (see judge_schemas.py).
"""

from __future__ import annotations

import json
import sqlite3

from pydantic import BaseModel, ValidationError

from evalguard.db.store import get_scores_for_response, insert_score
from evalguard.enums import CaseCategory, MetricName, ScoreMethod
from evalguard.models import EvaluationCase, Response, Score
from evalguard.providers.base import CompletionParams, LLMClient
from evalguard.quality.judge_schemas import (
    BlindJudgeCase,
    ClaimVerdict,
    CorrectnessVerdict,
    FaithfulnessVerdict,
)
from evalguard.quality.scoring import require_dev_case
from evalguard.runner.runner import compute_cost_usd

_DEFAULT_MAX_ATTEMPTS = 2
_RETRY_INSTRUCTION = (
    "\n\nYour previous response was not valid JSON matching the required schema. "
    "Respond with ONLY the JSON object — no prose, no code fences."
)


class MalformedJudgeOutputError(ValueError):
    """Judge output couldn't be parsed as JSON or didn't match the expected schema."""


class JudgeRetryExhaustedError(RuntimeError):
    """Every attempt (bounded — see `call_judge`'s `max_attempts`) produced malformed
    output. This is a real result to report, not something to paper over with a
    default score — a judgment that never arrived is not a passing judgment."""


def render_judge_prompt(template: str, case: BlindJudgeCase) -> str:
    """Substitute the four `BlindJudgeCase` fields into a rubric template. Missing
    context/reference_answer render as empty strings, never the literal `None`."""
    return (
        template.replace("{question}", case.question)
        .replace("{context}", case.context or "")
        .replace("{reference_answer}", case.reference_answer or "")
        .replace("{candidate_answer}", case.candidate_answer)
    )


def _extract_json_object(text: str) -> str:
    """Find the first balanced `{...}` in `text` — tolerates a judge that wraps its
    JSON in prose or a code fence instead of returning it bare."""
    start = text.find("{")
    if start == -1:
        raise MalformedJudgeOutputError("no JSON object found in judge output")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise MalformedJudgeOutputError("unbalanced braces in judge output")


def parse_judge_json[T: BaseModel](text: str, schema: type[T]) -> T:
    """Extract, parse, and validate one judge response against `schema`."""
    json_text = _extract_json_object(text)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise MalformedJudgeOutputError(f"invalid JSON: {exc}") from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise MalformedJudgeOutputError(f"schema validation failed: {exc}") from exc


def call_judge[T: BaseModel](
    client: LLMClient,
    prompt: str,
    params: CompletionParams,
    schema: type[T],
    *,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> tuple[T, int, int, int]:
    """Call the judge and validate its output, retrying up to `max_attempts` times on
    malformed output. Returns (parsed_result, total_input_tokens, total_output_tokens,
    total_latency_ms) — tokens/latency are summed across ALL attempts, including failed
    ones, since real tokens were billed for each.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0
    last_error: MalformedJudgeOutputError | None = None
    current_prompt = prompt

    for attempt in range(1, max_attempts + 1):
        completion = client.complete(current_prompt, params)
        total_input_tokens += completion.usage.input_tokens
        total_output_tokens += completion.usage.output_tokens
        total_latency_ms += completion.latency_ms

        try:
            result = parse_judge_json(completion.text, schema)
            return result, total_input_tokens, total_output_tokens, total_latency_ms
        except MalformedJudgeOutputError as exc:
            last_error = exc
            if attempt < max_attempts:
                current_prompt = prompt + _RETRY_INSTRUCTION

    raise JudgeRetryExhaustedError(
        f"judge output failed to parse after {max_attempts} attempt(s): {last_error}"
    )


def compute_faithfulness_and_hallucination(claims: list[ClaimVerdict]) -> tuple[float, bool]:
    """faithfulness = fraction SUPPORTED; hallucinated = any claim CONTRADICTED.

    An empty claims list is vacuously faithful (nothing unsupported was claimed) —
    this only arises from a candidate answer with no extractable factual content.
    """
    if not claims:
        return 1.0, False
    supported = sum(1 for c in claims if c.label == "SUPPORTED")
    hallucinated = any(c.label == "CONTRADICTED" for c in claims)
    return supported / len(claims), hallucinated


def build_blind_judge_case(case: EvaluationCase, response: Response) -> BlindJudgeCase:
    """The blindness boundary: only `case`/`response` are accepted, so a run's
    model_id or a prompt's version/role has no path into the judge's input."""
    return BlindJudgeCase(
        question=case.input,
        context=case.context,
        reference_answer=case.reference_answer,
        candidate_answer=response.output_text,
    )


def score_grounded_qa_case_with_judge(
    conn: sqlite3.Connection,
    response: Response,
    case: EvaluationCase,
    judge_client: LLMClient,
    *,
    correctness_template: str,
    faithfulness_template: str,
    correctness_params: CompletionParams,
    faithfulness_params: CompletionParams,
    price_per_1k_input: float | None,
    price_per_1k_output: float | None,
    allow_holdout: bool = False,
) -> list[Score]:
    """Judge one grounded_qa response for answer_correctness and faithfulness,
    deriving hallucination from the same faithfulness call. Persists Score rows and
    returns the ones newly created (empty list if this response was already fully
    scored — safe to call again after a partial failure or a fresh case).

    Separate params per call (Phase 8.2): correctness is frozen at max_tokens=512,
    faithfulness at 1024 — Phase 8.1 found the reasoning judge reliably needs the
    larger budget to finish faithfulness's claim-extraction JSON before running out
    of room mid-reasoning. One shared budget silently starved faithfulness/
    hallucination for a real fraction of grounded_qa cases.

    `allow_holdout` defaults to False — see `quality.scoring.require_dev_case`.
    """
    require_dev_case(case, allow_holdout=allow_holdout)
    if case.category is not CaseCategory.GROUNDED_QA:
        raise ValueError(
            f"judge scoring only applies to grounded_qa cases, got {case.category.value}"
        )
    if response.output_text is None:
        raise ValueError(f"response {response.id} has no output_text to judge")

    existing_metrics = {s.metric for s in get_scores_for_response(conn, response.id)}
    blind_case = build_blind_judge_case(case, response)
    new_scores: list[Score] = []

    if MetricName.ANSWER_CORRECTNESS not in existing_metrics:
        prompt = render_judge_prompt(correctness_template, blind_case)
        verdict, in_tok, out_tok, latency_ms = call_judge(
            judge_client, prompt, correctness_params, CorrectnessVerdict
        )
        cost = compute_cost_usd(
            in_tok,
            out_tok,
            price_per_1k_input=price_per_1k_input,
            price_per_1k_output=price_per_1k_output,
        )
        new_scores.append(
            insert_score(
                conn,
                Score(
                    response_id=response.id,
                    metric=MetricName.ANSWER_CORRECTNESS,
                    method=ScoreMethod.JUDGE,
                    value=verdict.score,
                    rationale=verdict.rationale,
                    judge_input_tokens=in_tok,
                    judge_output_tokens=out_tok,
                    judge_cost_usd=cost,
                    judge_latency_ms=latency_ms,
                ),
            )
        )

    if MetricName.FAITHFULNESS not in existing_metrics:
        prompt = render_judge_prompt(faithfulness_template, blind_case)
        verdict, in_tok, out_tok, latency_ms = call_judge(
            judge_client, prompt, faithfulness_params, FaithfulnessVerdict
        )
        faithfulness, hallucinated = compute_faithfulness_and_hallucination(verdict.claims)
        cost = compute_cost_usd(
            in_tok,
            out_tok,
            price_per_1k_input=price_per_1k_input,
            price_per_1k_output=price_per_1k_output,
        )
        claim_summary = (
            "; ".join(f"{c.label}: {c.claim}" for c in verdict.claims) or "no claims extracted"
        )
        new_scores.append(
            insert_score(
                conn,
                Score(
                    response_id=response.id,
                    metric=MetricName.FAITHFULNESS,
                    method=ScoreMethod.JUDGE,
                    value=faithfulness,
                    rationale=claim_summary,
                    judge_input_tokens=in_tok,
                    judge_output_tokens=out_tok,
                    judge_cost_usd=cost,
                    judge_latency_ms=latency_ms,
                ),
            )
        )

        if MetricName.HALLUCINATION not in existing_metrics:
            new_scores.append(
                insert_score(
                    conn,
                    Score(
                        response_id=response.id,
                        metric=MetricName.HALLUCINATION,
                        method=ScoreMethod.JUDGE,
                        value=1.0 if hallucinated else 0.0,
                        rationale=(
                            f"derived from the faithfulness judge call on response "
                            f"{response.id}; not a separate API call"
                        ),
                    ),
                )
            )

    return new_scores
