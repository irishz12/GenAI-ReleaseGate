"""Blind judge input contract and strict output schemas.

`BlindJudgeCase` is the entire blindness safeguard on the input side: it's the only
shape the judge prompt is ever rendered from, and `extra="forbid"` means a
prompt_version/role/model_id field accidentally passed in raises immediately rather
than silently reaching the judge.

The output schemas are deliberately narrow. `FaithfulnessVerdict` asks the judge for
claim labels only — faithfulness and hallucination are computed from those labels in
Python (see judge.py), not trusted as the judge's own arithmetic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BlindJudgeCase(BaseModel):
    """Everything the judge is allowed to see. Nothing else."""

    model_config = ConfigDict(extra="forbid")

    question: str
    context: str | None
    reference_answer: str | None
    candidate_answer: str


class CorrectnessVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["correct", "partially_correct", "incorrect"]
    rationale: str


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    label: Literal["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]


class FaithfulnessVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimVerdict]
