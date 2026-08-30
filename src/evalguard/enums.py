"""Enums shared across the platform's contracts and SQLite schema.

Every enum here mirrors a CHECK constraint in db/schema.sql — keep the two in sync.
Values are the exact strings stored in the database, so they must never be renamed
once data exists.
"""

from __future__ import annotations

from enum import StrEnum


class PromptRole(StrEnum):
    """Status of a prompt row in the registry."""

    PRODUCTION = "production"
    CANDIDATE = "candidate"
    ARCHIVED = "archived"
    JUDGE = "judge"


class CaseCategory(StrEnum):
    """Which of the four evaluation-dataset categories a case belongs to."""

    GROUNDED_QA = "grounded_qa"
    ABSTENTION = "abstention"
    INSTRUCTION_FOLLOWING = "instruction_following"
    GUARDRAIL = "guardrail"


class CaseSplit(StrEnum):
    """Development (iterate freely) vs. final holdout (sealed until promotion)."""

    DEV = "dev"
    HOLDOUT = "holdout"


class AttackType(StrEnum):
    """Guardrail-case sub-type. BENIGN cases must never be blocked."""

    INJECTION = "injection"
    PII_EXTRACTION = "pii_extraction"
    BENIGN = "benign"


class RunStatus(StrEnum):
    """Lifecycle status of a single prompt-version run over a dataset split."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    DEGRADED = "DEGRADED"
    ABORTED = "ABORTED"


class FinishReason(StrEnum):
    """Why generation for a single case ended."""

    STOP = "stop"
    LENGTH = "length"
    BLOCKED_INPUT = "blocked_input"
    BLOCKED_OUTPUT = "blocked_output"
    ERROR = "error"


class ErrorClass(StrEnum):
    """Two error classes only — see docs/ARCHITECTURE.md §11."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"


class ScoreMethod(StrEnum):
    """How a quality score was produced."""

    DETERMINISTIC = "deterministic"
    JUDGE = "judge"


class MetricName(StrEnum):
    """The full set of V1 metrics (docs/ARCHITECTURE.md §5).

    Per-response metrics (used in `Score.metric`, one row per response):
      ANSWER_CORRECTNESS, FAITHFULNESS, HALLUCINATION,
      INSTRUCTION_FOLLOWING, ABSTENTION_ACCURACY

    Aggregate-only metrics (used in `MetricDelta.metric`, computed at comparison time,
    never stored per-response): the guardrail rates and the operational metrics.
    """

    # Quality — per response
    ANSWER_CORRECTNESS = "answer_correctness"
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    INSTRUCTION_FOLLOWING = "instruction_following"
    ABSTENTION_ACCURACY = "abstention_accuracy"

    # Guardrails — aggregate only, derived from guardrail_results
    PROMPT_INJECTION_BLOCK_RATE = "prompt_injection_block_rate"
    SENSITIVE_INFORMATION_PROTECTION = "sensitive_information_protection"
    BENIGN_FALSE_POSITIVE_RATE = "benign_false_positive_rate"
    # Phase 8.1: BENIGN_FALSE_POSITIVE_RATE only covers guardrail-suite benign cases.
    # This one covers every case in the dataset that was never expected to be
    # blocked at all (grounded_qa/abstention/instruction_following included) — a real
    # dev run found false positives outside the guardrail suite that were otherwise
    # invisible to any metric (see regression.compare._overall_benign_fp_value).
    OVERALL_BENIGN_FALSE_POSITIVE_RATE = "overall_benign_false_positive_rate"

    # Operations — aggregate only, derived from responses
    P50_LATENCY = "p50_latency"
    P95_LATENCY = "p95_latency"
    COST_PER_QUERY = "cost_per_query"
    FAILURE_RATE = "failure_rate"


class GuardrailStage(StrEnum):
    """Guardrails run twice: once on the input, once on the completion."""

    INPUT = "input"
    OUTPUT = "output"


class GuardrailMethod(StrEnum):
    """How a guardrail_results row was produced: Phase 4's post-hoc regex classifier
    over already-stored text, or Phase 6's live AWS Bedrock ApplyGuardrail call made
    during generation itself."""

    DETERMINISTIC = "deterministic"
    BEDROCK = "bedrock"


class GuardrailOutcome(StrEnum):
    """Confusion-matrix label for a single guardrail check against its expected label."""

    TP = "TP"
    FP = "FP"
    TN = "TN"
    FN = "FN"


class GateStatus(StrEnum):
    """Outcome of a single release gate. INVALID is a run-level concept, not a gate one."""

    GO = "GO"
    REVIEW = "REVIEW"
    HOLD = "HOLD"


class ReleaseStatus(StrEnum):
    """Overall V1-vs-V2 release decision.

    GO = safe to promote. REVIEW = human review needed. HOLD = do not promote yet.
    INVALID = the evaluation itself is unreliable (not a verdict on the candidate).
    """

    GO = "GO"
    REVIEW = "REVIEW"
    HOLD = "HOLD"
    INVALID = "INVALID"
