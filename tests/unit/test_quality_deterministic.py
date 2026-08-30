"""Tests for evalguard.quality.deterministic — instruction_following and abstention
checkers, plus the shared detect_refusal helper.
"""

from __future__ import annotations

import pytest

from evalguard.quality.deterministic import (
    UnknownConstraintTypeError,
    check_abstention,
    check_instruction_following,
    detect_refusal,
)

# ─────────────────────────────────────────────────────────────────────────────
# detect_refusal
# ─────────────────────────────────────────────────────────────────────────────


def test_detect_refusal_true_for_known_phrases() -> None:
    assert detect_refusal("I don't have enough information to answer that.")
    assert detect_refusal("Sorry, I cannot provide that.")
    assert detect_refusal("I DON'T KNOW.")  # case-insensitive


def test_detect_refusal_false_for_substantive_answer() -> None:
    assert not detect_refusal("You can dial 1-844-651-0077 for help with your application.")


def test_detect_refusal_false_for_empty_string() -> None:
    assert not detect_refusal("")


# ─────────────────────────────────────────────────────────────────────────────
# check_instruction_following — one test per constraint type, plus edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_max_words_passes_within_limit() -> None:
    assert check_instruction_following("one two three", {"type": "max_words", "max_words": 5})


def test_max_words_fails_over_limit() -> None:
    assert not check_instruction_following(
        "one two three four five six", {"type": "max_words", "max_words": 5}
    )


def test_max_words_boundary_is_inclusive() -> None:
    assert check_instruction_following(
        "one two three four five", {"type": "max_words", "max_words": 5}
    )


def test_exact_sentences_passes_with_correct_count() -> None:
    assert check_instruction_following(
        "This is one sentence.", {"type": "exact_sentences", "sentence_count": 1}
    )


def test_exact_sentences_fails_with_wrong_count() -> None:
    assert not check_instruction_following(
        "First sentence. Second sentence.", {"type": "exact_sentences", "sentence_count": 1}
    )


def test_exact_sentences_handles_question_and_exclamation_marks() -> None:
    assert check_instruction_following(
        "Is this correct?", {"type": "exact_sentences", "sentence_count": 1}
    )
    assert check_instruction_following(
        "Great job!", {"type": "exact_sentences", "sentence_count": 1}
    )


def test_prefix_passes_when_present() -> None:
    assert check_instruction_following("Answer: 30 days", {"type": "prefix", "value": "Answer:"})


def test_prefix_fails_when_absent() -> None:
    assert not check_instruction_following("30 days", {"type": "prefix", "value": "Answer:"})


def test_prefix_ignores_leading_whitespace() -> None:
    assert check_instruction_following("   Answer: 30 days", {"type": "prefix", "value": "Answer:"})


def test_json_object_passes_with_required_keys() -> None:
    assert check_instruction_following(
        '{"answer": "30 days"}', {"type": "json_object", "required_keys": ["answer"]}
    )


def test_json_object_fails_when_key_missing() -> None:
    assert not check_instruction_following(
        '{"result": "30 days"}', {"type": "json_object", "required_keys": ["answer"]}
    )


def test_json_object_fails_on_invalid_json() -> None:
    assert not check_instruction_following(
        "not json at all", {"type": "json_object", "required_keys": ["answer"]}
    )


def test_json_object_fails_when_not_a_dict() -> None:
    assert not check_instruction_following(
        '["answer", "30 days"]', {"type": "json_object", "required_keys": ["answer"]}
    )


def test_no_terminal_punctuation_passes_without_punctuation() -> None:
    assert check_instruction_following("30 days", {"type": "no_terminal_punctuation"})


def test_no_terminal_punctuation_fails_with_period() -> None:
    assert not check_instruction_following("30 days.", {"type": "no_terminal_punctuation"})


def test_no_terminal_punctuation_fails_on_empty_string() -> None:
    assert not check_instruction_following("", {"type": "no_terminal_punctuation"})


def test_unknown_constraint_type_raises() -> None:
    with pytest.raises(UnknownConstraintTypeError):
        check_instruction_following("text", {"type": "not_a_real_type"})


def test_missing_constraint_type_raises() -> None:
    with pytest.raises(UnknownConstraintTypeError):
        check_instruction_following("text", {})


# ─────────────────────────────────────────────────────────────────────────────
# check_abstention
# ─────────────────────────────────────────────────────────────────────────────


def test_check_abstention_passes_when_expected_and_refused() -> None:
    assert check_abstention(
        "I don't have enough information to answer that.", expected_abstention=True
    )


def test_check_abstention_fails_when_expected_but_answered() -> None:
    assert not check_abstention("The answer is 30 days.", expected_abstention=True)


def test_check_abstention_passes_when_not_expected_and_answered() -> None:
    assert check_abstention("The answer is 30 days.", expected_abstention=False)


def test_check_abstention_fails_when_not_expected_but_refused() -> None:
    assert not check_abstention(
        "I don't have enough information to answer that.", expected_abstention=False
    )
