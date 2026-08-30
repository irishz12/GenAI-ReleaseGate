"""Renders a prompt template against one evaluation case — the one substitution
mechanism used identically for both V1 and V2, so a metric delta between them is
attributable to the prompt text, never to how the case was plugged in.

Uses plain string replacement rather than `str.format()`: the production/candidate
templates only ever contain the two placeholders below, but `.format()` would raise on
any other literal `{`/`}` a future prompt might contain (e.g. a JSON formatting
instruction) — replacement doesn't have that failure mode.
"""

from __future__ import annotations

from evalguard.models import EvaluationCase

_CONTEXT_PLACEHOLDER = "{context}"
_QUESTION_PLACEHOLDER = "{question}"


def render_prompt(template: str, case: EvaluationCase) -> str:
    """Substitute `{context}` and `{question}` in `template` with this case's data.

    `case.context` may be `None` (not every category has one) — rendered as an empty
    string rather than the literal text "None".
    """
    return template.replace(_CONTEXT_PLACEHOLDER, case.context or "").replace(
        _QUESTION_PLACEHOLDER, case.input
    )
