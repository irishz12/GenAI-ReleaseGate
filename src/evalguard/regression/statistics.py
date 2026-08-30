"""Phase 2 statistical enhancements: median paired difference, paired effect size
(Cohen's d), a bootstrap-derived two-sided p-value, and Holm-Bonferroni multiple-
comparison correction.

These sit *alongside* the existing mean delta + paired bootstrap 95% CI in
regression/compare.py — they add descriptive statistics and multiple-comparison-
adjusted evidence on top of what was already there. None of them feed back into or
alter the release-policy decision (policy/engine.py): `decide_release` still looks
only at each gate's raw delta against config/policy.yaml's thresholds, exactly as
before. A comparison's real-world release decision (GO/REVIEW/HOLD/INVALID) is not
recomputed or reinterpreted because of anything in this module.
"""

from __future__ import annotations

import random
import statistics as _stdlib_statistics

# Same deterministic bootstrap parameters as paired_bootstrap_ci_95 in compare.py —
# the p-value is derived from an independently-seeded resample of the same diffs, so
# it's consistent with (though not literally reusing) that CI's resampling.
_P_VALUE_SEED = 42
_P_VALUE_RESAMPLES = 2000


def paired_median_delta(diffs: list[float]) -> float | None:
    """Median of per-case paired differences (candidate - baseline). `None` when
    empty — mirrors paired_bootstrap_ci_95's "nothing to summarize" convention.

    Reported alongside the mean delta because it's robust to the occasional wild
    outlier a single mis-scored case can produce, where the mean is not.
    """
    if not diffs:
        return None
    return _stdlib_statistics.median(diffs)


def paired_cohens_d(diffs: list[float]) -> float | None:
    """Effect size for paired differences: mean(diffs) / sample_stdev(diffs).

    `None` below 2 samples (no variance to estimate) or when the diffs have exactly
    zero variance (division by zero would otherwise produce +/-inf, which is not a
    meaningful effect size — a constant shift's *significance* is better read off the
    CI/p-value, not an undefined effect size).
    """
    if len(diffs) < 2:
        return None
    stdev = _stdlib_statistics.stdev(diffs)
    if stdev == 0:
        return None
    mean = _stdlib_statistics.mean(diffs)
    return mean / stdev


def bootstrap_two_sided_p_value(
    diffs: list[float],
    *,
    n_resamples: int = _P_VALUE_RESAMPLES,
    seed: int = _P_VALUE_SEED,
) -> float | None:
    """Two-sided bootstrap p-value for the null hypothesis that the true mean paired
    difference is zero, estimated from the same paired-bootstrap resampling scheme
    as paired_bootstrap_ci_95. `None` below 2 samples.

    p = 2 * min(P(resample mean <= 0), P(resample mean >= 0)), capped at 1.0 — the
    standard way to turn a bootstrap distribution symmetric-tail-count into a
    two-sided p-value without assuming normality.
    """
    n = len(diffs)
    if n < 2:
        return None
    rng = random.Random(seed)
    resample_means = [
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    ]
    p_le_zero = sum(1 for m in resample_means if m <= 0) / n_resamples
    p_ge_zero = sum(1 for m in resample_means if m >= 0) / n_resamples
    return min(1.0, 2 * min(p_le_zero, p_ge_zero))


def holm_bonferroni(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down multiple-comparison correction.

    Sorts the m hypotheses by ascending p-value. Walking in that order, hypothesis
    at rank i (1-indexed) is rejected only while p(i) <= alpha / (m - i + 1) AND
    every hypothesis ranked before it was also rejected — the first failure makes
    every subsequent (laxer-threshold) hypothesis fail too, which is what keeps the
    family-wise error rate at alpha regardless of how many metrics are tested at
    once in a single comparison.

    Returns a dict from the same keys back to a bool: True = still significant
    after correction, False = not. Never used to alter a release decision — this is
    a descriptive answer to "if we were testing many metrics for significance at
    once, which would still hold up," reported alongside, not instead of, the gate
    outcome.
    """
    if not p_values:
        return {}
    ranked = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ranked)
    result: dict[str, bool] = {}
    still_rejecting = True
    for i, (key, p) in enumerate(ranked, start=1):
        threshold = alpha / (m - i + 1)
        if still_rejecting and p <= threshold:
            result[key] = True
        else:
            still_rejecting = False
            result[key] = False
    return result
