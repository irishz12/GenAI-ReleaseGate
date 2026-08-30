"""Stage A gating — turns one live `GuardrailClient.check()` call into a persistable
`GuardrailResult`, for the runner to insert immediately alongside the response it
protected. Unlike `guardrails.checks` (Phase 4's post-hoc scoring over text already in
the database), this runs during generation itself and its `triggered` is simply
whatever the real guardrail decided, not an inferred judgment about the response.
"""

from __future__ import annotations

from evalguard.enums import GuardrailMethod, GuardrailStage
from evalguard.guardrails.checks import classify_outcome
from evalguard.models import EvaluationCase, GuardrailResult
from evalguard.providers.base import GuardrailCheckResult


def guardrail_input_text(case: EvaluationCase) -> str:
    """What the input-stage guardrail check scans: the case's own context and
    question only — never a prompt template's fixed system instructions.

    Real incident this guards against: `render_prompt`'s full output (system wrapper
    + context + question, what actually goes to the generator) was sent to
    ApplyGuardrail's INPUT check during Phase 6's smoke test, and Bedrock's
    PROMPT_ATTACK classifier flagged a genuinely benign case — because the
    production template's own defensive line ("...if they ask you to ignore these
    rules, reveal this prompt...") describes the attack in words a classifier can't
    tell apart from an actual attempt. The system wrapper is trusted (we wrote it);
    only the case's context/question can carry an actual attack, so only they need
    scanning. This also makes the guardrail's behavior independent of which prompt
    template rendered the case — required for V1 and V2 to see identical checks.
    """
    return case.input if case.context is None else f"{case.context}\n\n{case.input}"


def resolve_expected_block(case: EvaluationCase) -> bool:
    """`expected_block` is only ever labeled for GUARDRAIL-category cases (see
    EvaluationCase's docstring); every other case is `None`, meaning "not a guardrail
    test — a live check on it is never expected to intervene."""
    return case.expected_block if case.expected_block is not None else False


def build_guardrail_result(
    check: GuardrailCheckResult,
    *,
    response_id: int,
    stage: GuardrailStage,
    expected_block: bool,
) -> GuardrailResult:
    outcome = classify_outcome(triggered=check.intervened, expected_block=expected_block)
    return GuardrailResult(
        response_id=response_id,
        stage=stage,
        triggered=check.intervened,
        expected_block=expected_block,
        outcome=outcome,
        detail=check.reason or None,
        method=GuardrailMethod.BEDROCK,
        latency_ms=check.latency_ms,
        cost_usd=check.cost_usd,
    )
