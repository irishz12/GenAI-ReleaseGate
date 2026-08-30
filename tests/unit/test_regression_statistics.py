"""Phase 2 statistical enhancements: median paired difference, paired effect size
(Cohen's d), a bootstrap-derived two-sided p-value, and Holm-Bonferroni multiple-
comparison correction. These sit *alongside* the existing mean delta + paired
bootstrap 95% CI (regression/compare.py) — they add descriptive/inferential
evidence, they never feed back into or alter the release-policy decision itself.
"""

from __future__ import annotations

from evalguard.regression.statistics import (
    bootstrap_two_sided_p_value,
    holm_bonferroni,
    paired_cohens_d,
    paired_median_delta,
)

# ─────────────────────────────────────────────────────────────────────────────
# paired_median_delta
# ─────────────────────────────────────────────────────────────────────────────


def test_median_delta_is_none_when_empty() -> None:
    assert paired_median_delta([]) is None


def test_median_delta_of_odd_length() -> None:
    assert paired_median_delta([0.1, 0.3, 0.2]) == 0.2


def test_median_delta_of_even_length_averages_middle_two() -> None:
    assert paired_median_delta([0.1, 0.2, 0.3, 0.4]) == 0.25


def test_median_delta_is_robust_to_a_single_outlier_unlike_the_mean() -> None:
    diffs = [0.01, 0.02, 0.01, 0.02, 10.0]  # one wild outlier
    median = paired_median_delta(diffs)
    mean = sum(diffs) / len(diffs)
    assert median < 0.1
    assert mean > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# paired_cohens_d
# ─────────────────────────────────────────────────────────────────────────────


def test_cohens_d_is_none_below_two_samples() -> None:
    assert paired_cohens_d([]) is None
    assert paired_cohens_d([0.1]) is None


def test_cohens_d_is_none_when_diffs_have_zero_variance() -> None:
    assert paired_cohens_d([0.1, 0.1, 0.1]) is None


def test_cohens_d_is_positive_for_a_consistent_positive_shift() -> None:
    d = paired_cohens_d([1.0, 1.1, 0.9, 1.2, 0.8])
    assert d is not None
    assert d > 0


def test_cohens_d_scales_with_effect_magnitude() -> None:
    small = paired_cohens_d([0.05, -0.03, 0.04, -0.02, 0.06, -0.01])
    large = paired_cohens_d([1.05, 0.97, 1.04, 0.98, 1.06, 0.99])
    assert abs(large) > abs(small)


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_two_sided_p_value
# ─────────────────────────────────────────────────────────────────────────────


def test_p_value_is_none_below_two_samples() -> None:
    assert bootstrap_two_sided_p_value([]) is None
    assert bootstrap_two_sided_p_value([0.1]) is None


def test_p_value_is_deterministic_across_calls() -> None:
    diffs = [0.05, -0.02, 0.1, 0.0, -0.05, 0.08, 0.03]
    assert bootstrap_two_sided_p_value(diffs) == bootstrap_two_sided_p_value(diffs)


def test_p_value_is_small_for_a_clearly_nonzero_consistent_diff() -> None:
    diffs = [0.5] * 20
    p = bootstrap_two_sided_p_value(diffs)
    assert p is not None
    assert p < 0.05


def test_p_value_is_large_when_diffs_straddle_zero_symmetrically() -> None:
    diffs = [0.5, -0.5, 0.4, -0.4, 0.6, -0.6, 0.3, -0.3]
    p = bootstrap_two_sided_p_value(diffs)
    assert p is not None
    assert p > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# holm_bonferroni
# ─────────────────────────────────────────────────────────────────────────────


def test_holm_bonferroni_empty_input() -> None:
    assert holm_bonferroni({}) == {}


def test_holm_bonferroni_single_hypothesis_matches_uncorrected_alpha() -> None:
    result = holm_bonferroni({"a": 0.03}, alpha=0.05)
    assert result == {"a": True}


def test_holm_bonferroni_rejects_all_when_every_p_value_is_tiny() -> None:
    result = holm_bonferroni({"a": 0.001, "b": 0.002, "c": 0.003}, alpha=0.05)
    assert result == {"a": True, "b": True, "c": True}


def test_holm_bonferroni_step_down_stops_rejecting_once_one_fails() -> None:
    """Classic Holm worked example: sorted p-values 0.01, 0.02, 0.20, 0.50 against
    alpha=0.05 and m=4. Thresholds are alpha/(m-i+1) for rank i=1..4: 0.0125, 0.0167,
    0.025, 0.05. p=0.01 < 0.0125 -> reject. p=0.02 > 0.0167 -> fail, and everything
    ranked below a failure is also not rejected (monotone step-down), even though
    0.20 or 0.50 might individually clear a later, laxer threshold."""
    result = holm_bonferroni(
        {"a": 0.01, "b": 0.02, "c": 0.20, "d": 0.50},
        alpha=0.05,
    )
    assert result == {"a": True, "b": False, "c": False, "d": False}


def test_holm_bonferroni_is_stricter_than_uncorrected_alpha() -> None:
    """p=0.04 alone would pass an uncorrected alpha=0.05 test, but as the 2nd of 2
    hypotheses its Holm threshold is alpha/(m-i+1) = 0.05/1 = 0.05 for rank 2 — wait,
    reason about rank order: p-values sorted [0.03, 0.04], m=2. Rank 1 threshold
    0.05/2=0.025, rank 2 threshold 0.05/1=0.05. p=0.03 > 0.025 -> already fails at
    rank 1, so nothing downstream is rejected either."""
    result = holm_bonferroni({"a": 0.03, "b": 0.04}, alpha=0.05)
    assert result == {"a": False, "b": False}
