"""Tests for the text-normalization and candidate-quality filters added during the
Phase 1 hardening pass (evalguard.registry.doc2dial_source).

Each filter here exists because a manual audit of the generated dev sample found a
concrete bad case by name — these tests pin down the exact inputs that were found bad
and confirm the fix, rather than testing the filters in the abstract.
"""

from __future__ import annotations

from evalguard.registry.doc2dial_source import (
    _is_filler_answer,
    _is_fragment_question,
    _is_vague_question,
    _normalize_text,
)

# ─────────────────────────────────────────────────────────────────────────────
# _normalize_text — collapses tokenizer-artifact spacing
# ─────────────────────────────────────────────────────────────────────────────


def test_normalize_text_collapses_apostrophe_spacing() -> None:
    # The exact artifact found in gqa-0079 / gqa-0053 during the dev-sample audit.
    raw = "Here 's some additional information that may help you decide what 's right for you :"
    assert _normalize_text(raw) == (
        "Here's some additional information that may help you decide what's right for you:"
    )


def test_normalize_text_collapses_contraction_spacing() -> None:
    raw = "We 're not recommending that you start at age 62 , your full retirement age ."
    assert _normalize_text(raw) == (
        "We're not recommending that you start at age 62, your full retirement age."
    )


def test_normalize_text_collapses_repeated_whitespace_and_newlines() -> None:
    raw = (
        "Prescription Drug Coverage helps pay for medications doctors prescribe for "
        "\n treatment .   For more"
    )
    normalized = _normalize_text(raw)
    assert "\n" not in normalized
    assert "  " not in normalized
    assert normalized.endswith("treatment. For more")


def test_normalize_text_is_a_no_op_on_already_clean_text() -> None:
    clean = "What is the retirement age for full benefits?"
    assert _normalize_text(clean) == clean


def test_normalize_text_is_idempotent() -> None:
    raw = "Here 's some information , and what 's next ."
    once = _normalize_text(raw)
    twice = _normalize_text(once)
    assert once == twice


# ─────────────────────────────────────────────────────────────────────────────
# _is_fragment_question — rejects multi-turn continuations taken out of context
# ─────────────────────────────────────────────────────────────────────────────


def test_fragment_question_detects_conjunction_starters() -> None:
    # Exact cases found in the dev-sample audit (abst-0015, abst-0002, gqa-0065).
    assert _is_fragment_question("and how can i pay it?")
    assert _is_fragment_question("ok and that is the only condition?")
    assert _is_fragment_question("and how they determine if I am eligible for benefits")


def test_fragment_question_detects_bare_affirmation_starters() -> None:
    # gqa-0073 / gqa-0004 from the second audit pass: no question at all, just a reply
    # to an unstated prior yes/no question.
    assert _is_fragment_question("Yes, that's what I just said.")
    assert _is_fragment_question("No, I already told you that.")


def test_fragment_question_allows_self_contained_questions() -> None:
    assert not _is_fragment_question("What is the refund window for this benefit?")
    assert not _is_fragment_question("Is there early or late retirement?")
    assert not _is_fragment_question("How much will I pay for these benefits?")


def test_fragment_question_is_case_insensitive() -> None:
    assert _is_fragment_question("AND how can i pay it?")
    assert _is_fragment_question("Ok, and that is the only condition?")


# ─────────────────────────────────────────────────────────────────────────────
# _is_filler_answer — rejects generic non-answers
# ─────────────────────────────────────────────────────────────────────────────


def test_filler_answer_detects_transition_phrases() -> None:
    # Exact cases found in the dev-sample audit (gqa-0079, gqa-0010).
    assert _is_filler_answer(
        "Here's some additional information that may help you decide what's right for you:"
    )
    assert _is_filler_answer(
        "I suggest you check with national and interstate legislation for this."
    )


def test_filler_answer_allows_real_content_answers() -> None:
    assert not _is_filler_answer("You can dial 1-844-651-0077 for help with your application.")
    assert not _is_filler_answer("Yes, they must show evidence that we ask for.")


# ─────────────────────────────────────────────────────────────────────────────
# _is_vague_question — rejects short objectless questions
# ─────────────────────────────────────────────────────────────────────────────


def test_vague_question_detects_short_objectless_questions() -> None:
    # Exact case found in the dev-sample audit (abst-0003).
    assert _is_vague_question("How can I request?")


def test_vague_question_allows_longer_questions_ending_in_the_same_verb() -> None:
    # A longer question ending in the same trailing verb is fine — only short,
    # genuinely objectless ones should be flagged.
    assert not _is_vague_question(
        "Can someone from the veterans crisis line call me back today for help?"
    )


def test_vague_question_allows_short_questions_with_a_clear_object() -> None:
    assert not _is_vague_question("Can I qualify for VA benefits?")
    assert not _is_vague_question("What are the fees for that?")
