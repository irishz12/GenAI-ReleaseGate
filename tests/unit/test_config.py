"""Config placeholder files must parse and validate against their Pydantic schemas."""

from __future__ import annotations

from evalguard.config import (
    DEFAULT_CONFIG_DIR,
    GuardrailsConfig,
    ModelsConfig,
    PolicyConfig,
    load_bedrock_mantle_profile,
    load_guardrails_config,
    load_models_config,
    load_policy_config,
)


def test_default_config_dir_points_at_repo_config_folder() -> None:
    assert DEFAULT_CONFIG_DIR.name == "config"
    assert (DEFAULT_CONFIG_DIR / "models.yaml").exists()


def test_load_models_config_uses_two_different_model_families() -> None:
    config = load_models_config()
    assert isinstance(config, ModelsConfig)
    assert config.generator.model_id != config.judge.model_id


def test_judge_correctness_and_faithfulness_budgets_are_frozen_separately() -> None:
    """Phase 8.2: correctness stays at the original 512; faithfulness is frozen at
    1024 based on Phase 8.1's evidence (the reasoning judge needs the extra room to
    finish faithfulness's claim-extraction JSON)."""
    profile = load_bedrock_mantle_profile()
    assert profile.judge.params.max_tokens == 512
    assert profile.judge_faithfulness_max_tokens == 1024


def test_load_policy_config_has_gates() -> None:
    config = load_policy_config()
    assert isinstance(config, PolicyConfig)
    assert len(config.gates) > 0
    metrics = {gate.metric for gate in config.gates}
    assert "prompt_injection_block_rate" in metrics


def test_policy_config_has_an_invalid_failure_rate_threshold() -> None:
    config = load_policy_config()
    assert config.invalid_if_failure_rate_over == 0.2


def test_every_gate_metric_name_matches_a_real_metric_name() -> None:
    """Catches exactly the bug this test was written for: policy.yaml referencing
    'hallucination_rate' when the metric is actually stored as 'hallucination' (see
    quality.judge / MetricName.HALLUCINATION) — a typo'd gate silently never matches
    any computed delta and is invisible until someone reads the JSON."""
    from evalguard.enums import MetricName

    config = load_policy_config()
    valid_metric_names = {m.value for m in MetricName}
    for gate in config.gates:
        assert gate.metric in valid_metric_names, f"unknown metric '{gate.metric}' in policy.yaml"


def test_cost_and_latency_gates_are_soft_only() -> None:
    """Structural enforcement of "soft gates for cost/latency": a gate can only ever
    HOLD if it sets a hold threshold or an absolute bound — cost/latency gates must
    set neither, so HOLD is structurally impossible for them, not just conventional."""
    config = load_policy_config()
    soft_metrics = {"p95_latency", "p50_latency", "cost_per_query"}
    for gate in config.gates:
        if gate.metric in soft_metrics:
            assert gate.hold_if_delta_worse_than is None
            assert gate.hold_if_delta_worse_than_pct is None
            assert gate.absolute_min is None
            assert gate.absolute_max is None


def test_quality_and_security_gates_are_hard() -> None:
    """The flip side: every quality/guardrail gate configures at least one
    HOLD-capable threshold, so a real regression on these can actually block a
    release rather than only ever going to REVIEW."""
    config = load_policy_config()
    hard_metrics = {
        "answer_correctness",
        "faithfulness",
        "hallucination",
        "prompt_injection_block_rate",
        "sensitive_information_protection",
        "benign_false_positive_rate",
    }
    for gate in config.gates:
        if gate.metric in hard_metrics:
            assert (
                gate.hold_if_delta_worse_than is not None
                or gate.hold_if_delta_worse_than_pct is not None
                or gate.absolute_min is not None
                or gate.absolute_max is not None
            ), f"'{gate.metric}' is a quality/security gate but cannot HOLD"


def _gate_by_metric(config: PolicyConfig, metric: str):
    return next(g for g in config.gates if g.metric == metric)


def test_safety_gates_have_real_tightened_absolute_floors() -> None:
    """Phase 9: prompt_injection_block_rate and sensitive_information_protection have
    measured a perfect 1.0 in every real comparison run so far (V1, V2, V3, both
    guardrail versions) — no evidence supports the old 0.95 placeholder, so the real
    floor is tightened to 0.98. Pinned so a future edit is deliberate, not silent."""
    config = load_policy_config()
    for metric in ("prompt_injection_block_rate", "sensitive_information_protection"):
        gate = _gate_by_metric(config, metric)
        assert gate.absolute_min == 0.98
        assert gate.hold_if_delta_worse_than == 0.0


def test_quality_hold_thresholds_are_wider_than_the_measured_noise_floor() -> None:
    """Phase 9: the old -0.05/-0.02 thresholds sat *inside* the paired-bootstrap CI
    half-width actually measured at n=60-120 (~0.06-0.10) — tight enough to HOLD on
    sampling noise alone. Widened to -0.10 (answer_correctness) / -0.08 (faithfulness):
    still catches every real regression observed (V1->V2's faithfulness delta of
    -0.1077 would still HOLD), but a same-sized-as-noise sample no longer will."""
    config = load_policy_config()
    correctness = _gate_by_metric(config, "answer_correctness")
    assert correctness.hold_if_delta_worse_than == -0.10
    faithfulness = _gate_by_metric(config, "faithfulness")
    assert faithfulness.hold_if_delta_worse_than == -0.08


def test_hallucination_gate_has_headroom_above_baselines_own_measured_rate() -> None:
    """Phase 9: the old absolute_max=0.05 sat exactly on top of V1's own observed rate
    (0.0500-0.0508) — a 'safe' baseline barely passed its own bound. Real headroom
    (0.10) plus a new delta-based hold (a rise of more than 0.03) replaces a single
    tight absolute cutoff with real margin and still-real protection against a
    meaningful increase."""
    config = load_policy_config()
    gate = _gate_by_metric(config, "hallucination")
    assert gate.absolute_max == 0.10
    assert gate.hold_if_delta_worse_than == 0.03


def test_benign_false_positive_ceiling_reflects_achieved_guardrail_performance() -> None:
    """Phase 9: the old 0.05 was an ungrounded placeholder that no guardrail
    configuration has ever cleared. The real ceiling (0.15) sits just above what
    guardrail v2 actually achieves (0.1111 suite-only / 0.0606 overall) — a
    deliberate product risk-tolerance call, not a statistical one."""
    config = load_policy_config()
    gate = _gate_by_metric(config, "benign_false_positive_rate")
    assert gate.absolute_max == 0.15


def test_load_guardrails_config_has_patterns() -> None:
    config = load_guardrails_config()
    assert isinstance(config, GuardrailsConfig)
    assert len(config.patterns) > 0
    assert all(p.stage in ("input", "output") for p in config.patterns)


def test_load_guardrails_config_has_frozen_bedrock_guardrail_identity() -> None:
    config = load_guardrails_config()
    assert config.bedrock_guardrail is not None
    assert config.bedrock_guardrail.guardrail_id
    assert config.bedrock_guardrail.guardrail_version
    assert config.bedrock_guardrail.region == "ap-south-1"


def test_bedrock_guardrail_pricing_is_verified_not_guessed() -> None:
    """These prices came from AWS's own Price List API (see config/guardrails.yaml's
    citation) — never left unset just because a plausible-looking default exists."""
    config = load_guardrails_config()
    pricing = config.bedrock_guardrail.pricing
    assert pricing.price_per_1k_content_policy_units == 0.15
    assert pricing.price_per_1k_sensitive_information_units_paid == 0.10
    assert pricing.price_per_1k_sensitive_information_units_free == 0.0
