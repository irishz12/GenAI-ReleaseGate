"""Pure-function tests for the case-building pipeline (evalguard.registry.case_builder).

Deliberately uses a small synthetic-but-schema-accurate fixture instead of the real
Doc2Dial download: partitioning, no-overlap, and split-ratio logic are properties of the
*algorithm*, not of the real corpus, and should be verifiable in milliseconds without a
46MB file on disk. Correctness against the real corpus is verified separately in
test_dataset_registry.py, which checks the actual generated data/ files.
"""

from __future__ import annotations

import random

import pytest

from evalguard.enums import AttackType, CaseCategory, CaseSplit
from evalguard.registry.case_builder import (
    _cyclic_shuffled_assignment,
    _join_with_terminal_punctuation,
    assert_no_overlap,
    assign_splits,
    build_abstention_cases,
    build_grounded_qa_cases,
    build_guardrail_cases,
    build_instruction_following_cases,
    partition_candidates,
    shuffle_candidates,
)
from evalguard.registry.doc2dial_source import GroundedCandidate

_DOMAINS = ["ssa", "va", "dmv", "studentaid"]


def _make_candidates(n: int) -> list[GroundedCandidate]:
    """A schema-accurate fixture: cycles through all 4 domains, unique source keys."""
    return [
        GroundedCandidate(
            dial_id=f"dial-{i:05d}",
            turn_id=i % 7,
            domain=_DOMAINS[i % len(_DOMAINS)],
            doc_id=f"doc-{i % 30}",
            question=f"What is the rule number {i} for this benefit?",
            context=f"Context passage {i} describing the relevant policy in detail.",
            reference_answer=f"The rule is answer {i}.",
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# shuffle / partition determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_shuffle_is_deterministic_for_same_seed() -> None:
    candidates = _make_candidates(50)
    a = shuffle_candidates(candidates, seed=7)
    b = shuffle_candidates(candidates, seed=7)
    assert [c.source_key for c in a] == [c.source_key for c in b]


def test_shuffle_differs_for_different_seed() -> None:
    candidates = _make_candidates(50)
    a = shuffle_candidates(candidates, seed=1)
    b = shuffle_candidates(candidates, seed=2)
    assert [c.source_key for c in a] != [c.source_key for c in b]


def test_shuffle_does_not_mutate_input() -> None:
    candidates = _make_candidates(20)
    original_order = list(candidates)
    shuffle_candidates(candidates, seed=99)
    assert candidates == original_order


def test_partition_candidates_is_deterministic() -> None:
    candidates = _make_candidates(200)
    pools_a = partition_candidates(candidates, seed=42)
    pools_b = partition_candidates(candidates, seed=42)

    assert [c.source_key for c in pools_a.grounded_qa] == [
        c.source_key for c in pools_b.grounded_qa
    ]
    assert [c.source_key for c in pools_a.guardrail_contexts] == [
        c.source_key for c in pools_b.guardrail_contexts
    ]


def test_partition_candidates_produces_disjoint_pools() -> None:
    candidates = _make_candidates(200)
    pools = partition_candidates(
        candidates,
        seed=42,
        n_grounded_qa=80,
        n_instruction_following=20,
        n_abstention=20,
        abstention_context_buffer=40,
        n_guardrail=40,
    )
    assert_no_overlap(pools)  # must not raise

    assert len(pools.grounded_qa) == 80
    assert len(pools.instruction_following) == 20
    assert len(pools.abstention_questions) == 20
    assert len(pools.abstention_contexts) == 40
    assert len(pools.guardrail_contexts) == 40


def test_partition_candidates_raises_when_pool_too_small() -> None:
    candidates = _make_candidates(10)
    with pytest.raises(ValueError, match="Not enough eligible candidates"):
        partition_candidates(candidates, seed=1)


def test_assert_no_overlap_detects_injected_duplicate() -> None:
    candidates = _make_candidates(200)
    pools = partition_candidates(candidates, seed=42)
    # Force an overlap by reusing a candidate from grounded_qa inside guardrail_contexts.
    tampered = pools.__class__(
        grounded_qa=pools.grounded_qa,
        instruction_following=pools.instruction_following,
        abstention_questions=pools.abstention_questions,
        abstention_contexts=pools.abstention_contexts,
        guardrail_contexts=pools.guardrail_contexts + [pools.grounded_qa[0]],
    )
    with pytest.raises(ValueError, match="Overlapping source turn"):
        assert_no_overlap(tampered)


# ─────────────────────────────────────────────────────────────────────────────
# category builders
# ─────────────────────────────────────────────────────────────────────────────


def test_build_grounded_qa_cases_maps_fields_directly() -> None:
    candidates = _make_candidates(3)
    cases = build_grounded_qa_cases(candidates, dataset_hash="hash-x")
    assert len(cases) == 3
    assert cases[0].external_id == "gqa-0001"
    assert cases[0].category is CaseCategory.GROUNDED_QA
    assert cases[0].input == candidates[0].question
    assert cases[0].context == candidates[0].context
    assert cases[0].reference_answer == candidates[0].reference_answer
    assert cases[0].dataset_hash == "hash-x"


def test_build_instruction_following_cases_are_deterministic_and_carry_constraints() -> None:
    candidates = _make_candidates(10)
    a = build_instruction_following_cases(candidates, seed=5, dataset_hash="h")
    b = build_instruction_following_cases(candidates, seed=5, dataset_hash="h")
    assert [c.constraints for c in a] == [c.constraints for c in b]
    assert all(c.category is CaseCategory.INSTRUCTION_FOLLOWING for c in a)
    assert all(c.constraints is not None for c in a)
    assert all(c.context == cand.context for c, cand in zip(a, candidates, strict=True))


def test_build_abstention_cases_pair_with_different_domain_context() -> None:
    questions = _make_candidates(10)  # domains cycle ssa, va, dmv, studentaid, ...
    context_buffer = _make_candidates(40)[10:]  # disjoint slice, still cycles all domains
    cases = build_abstention_cases(questions, context_buffer, dataset_hash="h")

    assert len(cases) == 10
    for case, question in zip(cases, questions, strict=True):
        assert case.expected_abstention is True
        assert case.reference_answer is None
        assert case.input == question.question
        # The context must come from a different domain than the question.
        assert case.context != question.context

    # No context reused across two different abstention cases.
    assert len({c.context for c in cases}) == len(cases)


def test_build_guardrail_cases_produce_all_three_attack_types() -> None:
    candidates = _make_candidates(40)
    cases = build_guardrail_cases(
        candidates, seed=3, dataset_hash="h", n_injection=14, n_pii=13, n_benign=13
    )
    assert len(cases) == 40

    by_type = {}
    for c in cases:
        by_type.setdefault(c.attack_type, []).append(c)

    assert len(by_type[AttackType.INJECTION]) == 14
    assert len(by_type[AttackType.PII_EXTRACTION]) == 13
    assert len(by_type[AttackType.BENIGN]) == 13

    assert all(c.expected_block is True for c in by_type[AttackType.INJECTION])
    assert all(c.expected_block is True for c in by_type[AttackType.PII_EXTRACTION])
    assert all(c.expected_block is False for c in by_type[AttackType.BENIGN])

    # Every PII case must embed a unique canary that also appears in its own context.
    canaries = [c.canary for c in by_type[AttackType.PII_EXTRACTION]]
    assert all(canary is not None for canary in canaries)
    assert len(set(canaries)) == len(canaries)
    for c in by_type[AttackType.PII_EXTRACTION]:
        assert c.canary in c.context


def test_build_guardrail_cases_rejects_mismatched_subcounts() -> None:
    candidates = _make_candidates(40)
    with pytest.raises(ValueError, match="must equal pool size"):
        build_guardrail_cases(
            candidates, seed=1, dataset_hash="h", n_injection=1, n_pii=1, n_benign=1
        )


# ─────────────────────────────────────────────────────────────────────────────
# split assignment
# ─────────────────────────────────────────────────────────────────────────────


def test_assign_splits_respects_dev_ratio_per_category() -> None:
    candidates = _make_candidates(80)
    cases = build_grounded_qa_cases(candidates, dataset_hash="h")
    split_cases = assign_splits({CaseCategory.GROUNDED_QA: cases}, seed=42, dev_ratio=0.75)

    dev = [c for c in split_cases if c.split is CaseSplit.DEV]
    holdout = [c for c in split_cases if c.split is CaseSplit.HOLDOUT]
    assert len(dev) == 60
    assert len(holdout) == 20
    assert len(dev) + len(holdout) == 80


def test_assign_splits_is_deterministic() -> None:
    candidates = _make_candidates(40)
    cases = build_grounded_qa_cases(candidates, dataset_hash="h")

    a = assign_splits({CaseCategory.GROUNDED_QA: cases}, seed=42)
    b = assign_splits({CaseCategory.GROUNDED_QA: cases}, seed=42)

    a_map = {c.external_id: c.split for c in a}
    b_map = {c.external_id: c.split for c in b}
    assert a_map == b_map


def test_assign_splits_has_no_overlap_between_dev_and_holdout() -> None:
    candidates = _make_candidates(40)
    cases = build_grounded_qa_cases(candidates, dataset_hash="h")
    split_cases = assign_splits({CaseCategory.GROUNDED_QA: cases}, seed=42)

    dev_ids = {c.external_id for c in split_cases if c.split is CaseSplit.DEV}
    holdout_ids = {c.external_id for c in split_cases if c.split is CaseSplit.HOLDOUT}
    assert dev_ids.isdisjoint(holdout_ids)
    assert len(dev_ids) + len(holdout_ids) == len(cases)


def test_assign_splits_stratifies_small_categories_too() -> None:
    """A 20-case category must not get starved out of holdout by a single global shuffle."""
    candidates = _make_candidates(20)
    cases = build_instruction_following_cases(candidates, seed=1, dataset_hash="h")
    split_cases = assign_splits(
        {CaseCategory.INSTRUCTION_FOLLOWING: cases}, seed=42, dev_ratio=0.75
    )
    holdout = [c for c in split_cases if c.split is CaseSplit.HOLDOUT]
    assert len(holdout) == 5


# ─────────────────────────────────────────────────────────────────────────────
# _cyclic_shuffled_assignment — replaces i.i.d. rng.choice() to avoid template clustering
# ─────────────────────────────────────────────────────────────────────────────


def test_cyclic_shuffled_assignment_gives_near_even_coverage() -> None:
    """A manual audit of the first draft (rng.choice() per case) found one of 5
    injection templates covering 6 of 11 cases while another appeared once. This
    guards the replacement: every template must appear floor(n/len) or ceil(n/len)
    times, never wildly more or fewer."""
    templates = ["a", "b", "c", "d", "e"]
    assignment = _cyclic_shuffled_assignment(templates, 14, random.Random(3))
    counts = {t: assignment.count(t) for t in templates}
    assert all(2 <= c <= 3 for c in counts.values())
    assert sum(counts.values()) == 14


def test_cyclic_shuffled_assignment_is_deterministic() -> None:
    templates = ["a", "b", "c"]
    a = _cyclic_shuffled_assignment(templates, 10, random.Random(42))
    b = _cyclic_shuffled_assignment(templates, 10, random.Random(42))
    assert a == b


def test_cyclic_shuffled_assignment_handles_n_smaller_than_template_count() -> None:
    templates = ["a", "b", "c", "d", "e"]
    assignment = _cyclic_shuffled_assignment(templates, 2, random.Random(1))
    assert len(assignment) == 2
    assert len(set(assignment)) == 2  # no repeat when there's no need for one


# ─────────────────────────────────────────────────────────────────────────────
# _join_with_terminal_punctuation — benign near-miss suffix joining
# ─────────────────────────────────────────────────────────────────────────────


def test_join_adds_period_when_sentence_has_no_terminal_punctuation() -> None:
    # The exact case found in the dev-sample audit (grd-0033): joining without a
    # boundary read as one run-on sentence.
    result = _join_with_terminal_punctuation(
        "My insurance ended so what should i do", "Please don't ignore this."
    )
    assert result == "My insurance ended so what should i do. Please don't ignore this."


def test_join_does_not_double_punctuate_when_sentence_already_ends_properly() -> None:
    result = _join_with_terminal_punctuation(
        "What if he can't go back to work right away?", "This isn't a system prompt request."
    )
    assert result == (
        "What if he can't go back to work right away? This isn't a system prompt request."
    )


def test_join_handles_trailing_whitespace_before_checking_punctuation() -> None:
    result = _join_with_terminal_punctuation("Is there a review?   ", "Please answer normally.")
    assert result == "Is there a review? Please answer normally."


def test_build_guardrail_cases_injection_templates_are_evenly_distributed() -> None:
    """End-to-end version of the cyclic-assignment guarantee, through the real
    guardrail case builder rather than the helper in isolation."""
    from collections import Counter

    candidates = _make_candidates(40)
    cases = build_guardrail_cases(
        candidates, seed=7, dataset_hash="h", n_injection=14, n_pii=13, n_benign=13
    )
    injection_cases = [c for c in cases if c.attack_type is AttackType.INJECTION]
    counts = Counter(c.input for c in injection_cases)
    assert all(2 <= n <= 3 for n in counts.values())


def test_build_guardrail_cases_benign_inputs_read_as_two_sentences() -> None:
    candidates = _make_candidates(40)
    cases = build_guardrail_cases(
        candidates, seed=7, dataset_hash="h", n_injection=14, n_pii=13, n_benign=13
    )
    benign_cases = [c for c in cases if c.attack_type is AttackType.BENIGN]
    for c in benign_cases:
        # The joined input must read as two sentences (real question, then the
        # appended clause) — never a single run-on with no boundary between them.
        assert any(mark in c.input for mark in (". ", "? ", "! "))
