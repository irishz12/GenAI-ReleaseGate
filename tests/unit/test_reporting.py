"""Phase 3 structured reporting: turns a real, already-persisted Comparison into a
JSON-serializable ExperimentReport with the exact field set the portfolio spec
requires. Built only from already-computed Comparison/PolicyConfig data — no
fabricated values, no new statistics computed here.
"""

from __future__ import annotations

import json

from evalguard.config import GateConfig, PolicyConfig
from evalguard.enums import GateStatus, MetricName, ReleaseStatus
from evalguard.models import Comparison, GateResult, MetricDelta, ReleaseDecision
from evalguard.reporting import ExperimentReport, build_experiment_report, write_report


def _policy() -> PolicyConfig:
    return PolicyConfig(
        gates=[
            GateConfig(
                metric="answer_correctness",
                direction="higher_is_better",
                review_if_delta_worse_than=0.0,
                hold_if_delta_worse_than=-0.10,
            ),
            GateConfig(
                metric="cost_per_query",
                direction="lower_is_better",
                review_if_delta_worse_than_pct=15.0,
            ),
        ],
        invalid_if_failure_rate_over=0.2,
    )


def _review_comparison() -> Comparison:
    correctness_delta = MetricDelta(
        metric=MetricName.ANSWER_CORRECTNESS,
        baseline_value=0.75,
        candidate_value=0.75,
        delta=0.0,
        ci_low=-0.05,
        ci_high=0.05,
        median_delta=0.0,
        effect_size=None,
        p_value=0.9,
        holm_significant=False,
    )
    cost_delta = MetricDelta(
        metric=MetricName.COST_PER_QUERY,
        baseline_value=0.0001,
        candidate_value=0.00014,
        delta=0.00004,
    )
    gates = [
        GateResult(
            gate_name="answer_correctness",
            status=GateStatus.GO,
            threshold=None,
            observed=0.0,
            reason="within configured thresholds (delta=+0.0000)",
        ),
        GateResult(
            gate_name="cost_per_query",
            status=GateStatus.REVIEW,
            threshold=15.0,
            observed=40.0,
            reason="delta +40.00% crossed the review threshold +15.00%",
        ),
    ]
    release = ReleaseDecision(
        status=ReleaseStatus.REVIEW,
        gates=gates,
        summary="REVIEW — review: cost_per_query",
    )
    return Comparison(
        id=99,
        baseline_run_id=2,
        candidate_run_id=4,
        n_paired=120,
        deltas=[correctness_delta, cost_delta],
        release=release,
    )


def _go_comparison() -> Comparison:
    delta = MetricDelta(
        metric=MetricName.ANSWER_CORRECTNESS,
        baseline_value=0.70,
        candidate_value=0.80,
        delta=0.10,
    )
    gate = GateResult(
        gate_name="answer_correctness",
        status=GateStatus.GO,
        threshold=None,
        observed=0.10,
        reason="within configured thresholds (delta=+0.1000)",
    )
    release = ReleaseDecision(
        status=ReleaseStatus.GO, gates=[gate], summary="GO — all 1 gate(s) passed"
    )
    return Comparison(
        id=100, baseline_run_id=1, candidate_run_id=2, n_paired=40, deltas=[delta], release=release
    )


# ─────────────────────────────────────────────────────────────────────────────
# build_experiment_report
# ─────────────────────────────────────────────────────────────────────────────


def test_report_carries_the_exact_required_fields() -> None:
    report = build_experiment_report(
        _review_comparison(),
        _policy(),
        experiment_id="v2_vs_v3_dev",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    dumped = report.model_dump()
    expected_fields = {
        "experiment_id",
        "baseline_version",
        "candidate_version",
        "dataset",
        "valid_cases",
        "metrics",
        "baseline_values",
        "candidate_values",
        "delta",
        "confidence_interval",
        "effect_size",
        "threshold",
        "gate_results",
        "decision",
        "decision_reasons",
        "timestamp",
    }
    assert expected_fields <= set(dumped.keys())


def test_report_identity_fields_are_exactly_what_was_passed_in() -> None:
    report = build_experiment_report(
        _review_comparison(),
        _policy(),
        experiment_id="v2_vs_v3_dev",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    assert report.experiment_id == "v2_vs_v3_dev"
    assert report.baseline_version == "v2"
    assert report.candidate_version == "v3"
    assert report.dataset == "dev"
    assert report.valid_cases == 120


def test_report_metrics_and_values_come_straight_from_the_comparisons_deltas() -> None:
    report = build_experiment_report(
        _review_comparison(),
        _policy(),
        experiment_id="x",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    assert set(report.metrics) == {"answer_correctness", "cost_per_query"}
    assert report.baseline_values["answer_correctness"] == 0.75
    assert report.candidate_values["cost_per_query"] == 0.00014
    assert report.delta["cost_per_query"] == 0.00004
    assert report.confidence_interval["answer_correctness"] == [-0.05, 0.05]
    assert report.confidence_interval["cost_per_query"] is None
    assert report.effect_size["answer_correctness"] is None
    assert report.effect_size["cost_per_query"] is None


def test_report_threshold_is_sourced_from_policy_config_not_fabricated() -> None:
    report = build_experiment_report(
        _review_comparison(),
        _policy(),
        experiment_id="x",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    assert report.threshold["answer_correctness"] == {
        "review_if_delta_worse_than": 0.0,
        "hold_if_delta_worse_than": -0.10,
    }
    assert report.threshold["cost_per_query"] == {"review_if_delta_worse_than_pct": 15.0}


def test_report_gate_results_are_the_real_gate_results_unmodified() -> None:
    comparison = _review_comparison()
    report = build_experiment_report(
        comparison,
        _policy(),
        experiment_id="x",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    assert report.gate_results == comparison.release.gates


def test_report_decision_reasons_lists_every_non_go_gate() -> None:
    report = build_experiment_report(
        _review_comparison(),
        _policy(),
        experiment_id="x",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    assert report.decision == "REVIEW"
    assert len(report.decision_reasons) == 1
    assert "cost_per_query" in report.decision_reasons[0]
    assert "answer_correctness" not in " ".join(report.decision_reasons)


def test_report_decision_reasons_falls_back_to_summary_when_every_gate_passes() -> None:
    report = build_experiment_report(
        _go_comparison(),
        _policy(),
        experiment_id="x",
        baseline_version="v1",
        candidate_version="v2",
        dataset="dev",
    )
    assert report.decision == "GO"
    assert report.decision_reasons == ["GO — all 1 gate(s) passed"]


def test_report_timestamp_is_the_comparisons_persisted_created_at_not_generation_time() -> None:
    comparison = _go_comparison()
    report = build_experiment_report(
        comparison,
        _policy(),
        experiment_id="x",
        baseline_version="v1",
        candidate_version="v2",
        dataset="dev",
    )
    assert report.timestamp == comparison.created_at


def test_report_round_trips_through_json() -> None:
    report = build_experiment_report(
        _review_comparison(),
        _policy(),
        experiment_id="x",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    restored = ExperimentReport.model_validate_json(report.model_dump_json())
    assert restored == report


def test_write_report_writes_valid_json_to_disk(tmp_path) -> None:
    report = build_experiment_report(
        _go_comparison(),
        _policy(),
        experiment_id="x",
        baseline_version="v1",
        candidate_version="v2",
        dataset="dev",
    )
    out_path = tmp_path / "x.json"
    write_report(report, out_path)
    on_disk = json.loads(out_path.read_text())
    assert on_disk["experiment_id"] == "x"
    assert on_disk["decision"] == "GO"
