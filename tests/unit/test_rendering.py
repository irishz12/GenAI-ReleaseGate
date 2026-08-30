"""render_prompt must substitute context/question consistently — the one place V1 and
V2 share, so a metric delta between them is attributable to prompt text alone.
"""

from __future__ import annotations

from evalguard.enums import CaseCategory, CaseSplit
from evalguard.models import EvaluationCase
from evalguard.runner.rendering import render_prompt


def _case(**overrides) -> EvaluationCase:
    defaults = dict(
        external_id="gqa-0001",
        category=CaseCategory.GROUNDED_QA,
        split=CaseSplit.DEV,
        input="What is the refund window?",
        context="Refunds are accepted within 30 days.",
        dataset_hash="hash",
    )
    defaults.update(overrides)
    return EvaluationCase(**defaults)


def test_render_prompt_substitutes_both_placeholders() -> None:
    template = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    rendered = render_prompt(template, _case())
    assert "Refunds are accepted within 30 days." in rendered
    assert "What is the refund window?" in rendered
    assert "{context}" not in rendered
    assert "{question}" not in rendered


def test_render_prompt_handles_missing_context_as_empty_string() -> None:
    template = "Context:\n{context}\n\nQuestion:\n{question}"
    rendered = render_prompt(template, _case(context=None))
    assert "Context:\n\n\nQuestion:" in rendered
    assert "None" not in rendered


def test_render_prompt_is_consistent_across_two_different_templates() -> None:
    """V1 and V2 differ in instructions, but the *mechanism* plugging in the case data
    must be identical — same case data ends up in both, just wrapped differently."""
    case = _case()
    template_v1 = "SYSTEM A\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    template_v2 = (
        "SYSTEM B (different rules)\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    rendered_v1 = render_prompt(template_v1, case)
    rendered_v2 = render_prompt(template_v2, case)

    assert case.context in rendered_v1
    assert case.context in rendered_v2
    assert case.input in rendered_v1
    assert case.input in rendered_v2
    assert "SYSTEM A" in rendered_v1 and "SYSTEM A" not in rendered_v2
    assert "SYSTEM B" in rendered_v2 and "SYSTEM B" not in rendered_v1


def test_render_prompt_does_not_mutate_the_template_string() -> None:
    template = "Context:\n{context}\n\nQuestion:\n{question}"
    render_prompt(template, _case())
    assert template == "Context:\n{context}\n\nQuestion:\n{question}"


def test_render_prompt_is_deterministic() -> None:
    template = "Context:\n{context}\n\nQuestion:\n{question}"
    case = _case()
    assert render_prompt(template, case) == render_prompt(template, case)


def test_render_prompt_against_the_real_production_and_candidate_templates() -> None:
    from evalguard.registry.prompts import DEFAULT_PROMPTS_DIR

    v1_template = (DEFAULT_PROMPTS_DIR / "production" / "support_agent.v1.md").read_text()
    v2_template = (DEFAULT_PROMPTS_DIR / "candidate" / "support_agent.v2.md").read_text()
    case = _case()

    rendered_v1 = render_prompt(v1_template, case)
    rendered_v2 = render_prompt(v2_template, case)

    assert case.context in rendered_v1
    assert case.input in rendered_v1
    assert case.context in rendered_v2
    assert case.input in rendered_v2
    assert "{context}" not in rendered_v1
    assert "{question}" not in rendered_v1
    assert "{context}" not in rendered_v2
    assert "{question}" not in rendered_v2
