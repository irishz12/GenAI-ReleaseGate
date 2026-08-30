"""Phase 3 structured reporting: turns a real, already-persisted Comparison into a
JSON-serializable ExperimentReport with the exact field set the portfolio spec
requires. Built only from already-computed Comparison/PolicyConfig data — no
fabricated values, no new statistics computed here.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from evalguard.config import GateConfig, PolicyConfig
from evalguard.db.store import (
    insert_case,
    insert_guardrail_result,
    insert_prompt,
    insert_response,
    insert_run,
    insert_score,
)
from evalguard.enums import (
    CaseCategory,
    CaseSplit,
    FinishReason,
    GateStatus,
    GuardrailMethod,
    GuardrailOutcome,
    GuardrailStage,
    MetricName,
    PromptRole,
    ReleaseStatus,
    ScoreMethod,
)
from evalguard.models import (
    Comparison,
    EvaluationCase,
    GateResult,
    GuardrailResult,
    MetricDelta,
    Prompt,
    ReleaseDecision,
    Response,
    Run,
    Score,
)
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


def _make_run_with_costs(
    conn: sqlite3.Connection,
    *,
    version: str,
    generator_cost: float | None,
    guardrail_cost: float | None,
    judge_cost: float | None,
) -> Run:
    """Registers a real prompt + run + one response/guardrail-result/score with the
    given (real, not fabricated at test time) per-service costs, so the reporting
    layer's cost breakdown has genuine backing data to compute from — mirrors
    exactly how scripts/generate_reports.py sees real dev/holdout data."""
    prompt = insert_prompt(
        conn,
        Prompt(
            name="support_agent",
            version=version,
            role=PromptRole.CANDIDATE,
            template="Answer using only the context.",
            content_hash=f"hash-{version}",
        ),
    )
    run = insert_run(
        conn,
        Run(
            prompt_id=prompt.id,
            model_id="fake-generator-v1",
            judge_model_id="fake-judge-v1",
            guardrail_config_hash="ghash",
            code_sha="deadbeef",
            config_hash=f"chash-{version}",
        ),
    )
    case = insert_case(
        conn,
        EvaluationCase(
            external_id=f"case-{version}",
            category=CaseCategory.GROUNDED_QA,
            split=CaseSplit.DEV,
            input="What is the refund window?",
            context="Refunds within 30 days.",
            reference_answer="30 days",
            dataset_hash="dataset-hash-1",
        ),
    )
    response = insert_response(
        conn,
        Response(
            run_id=run.id,
            case_id=case.id,
            output_text="30 days",
            finish_reason=FinishReason.STOP,
            input_tokens=10,
            output_tokens=3,
            cost_usd=generator_cost,
            latency_ms=100,
        ),
    )
    if guardrail_cost is not None:
        insert_guardrail_result(
            conn,
            GuardrailResult(
                response_id=response.id,
                stage=GuardrailStage.OUTPUT,
                triggered=False,
                expected_block=False,
                outcome=GuardrailOutcome.TN,
                method=GuardrailMethod.BEDROCK,
                cost_usd=guardrail_cost,
            ),
        )
    if judge_cost is not None:
        insert_score(
            conn,
            Score(
                response_id=response.id,
                metric=MetricName.ANSWER_CORRECTNESS,
                method=ScoreMethod.JUDGE,
                value=1.0,
                judge_cost_usd=judge_cost,
            ),
        )
    return run


def _review_comparison(conn: sqlite3.Connection) -> Comparison:
    baseline_run = _make_run_with_costs(
        conn, version="v2", generator_cost=0.001, guardrail_cost=0.0002, judge_cost=0.0005
    )
    candidate_run = _make_run_with_costs(
        conn, version="v3", generator_cost=0.0014, guardrail_cost=0.0002, judge_cost=0.0005
    )

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
        baseline_run_id=baseline_run.id,
        candidate_run_id=candidate_run.id,
        n_paired=120,
        deltas=[correctness_delta, cost_delta],
        release=release,
    )


def _go_comparison(conn: sqlite3.Connection) -> Comparison:
    baseline_run = _make_run_with_costs(
        conn, version="v1", generator_cost=0.001, guardrail_cost=None, judge_cost=0.0004
    )
    candidate_run = _make_run_with_costs(
        conn, version="v2b", generator_cost=0.0009, guardrail_cost=None, judge_cost=0.0004
    )

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
        id=100,
        baseline_run_id=baseline_run.id,
        candidate_run_id=candidate_run.id,
        n_paired=40,
        deltas=[delta],
        release=release,
    )


# ─────────────────────────────────────────────────────────────────────────────
# build_experiment_report
# ─────────────────────────────────────────────────────────────────────────────


def test_report_carries_the_exact_required_fields(conn: sqlite3.Connection) -> None:
    report = build_experiment_report(
        conn,
        _review_comparison(conn),
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
        "baseline_cost_breakdown",
        "candidate_cost_breakdown",
    }
    assert expected_fields <= set(dumped.keys())


def test_report_identity_fields_are_exactly_what_was_passed_in(conn: sqlite3.Connection) -> None:
    report = build_experiment_report(
        conn,
        _review_comparison(conn),
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


def test_report_metrics_and_values_come_straight_from_the_comparisons_deltas(
    conn: sqlite3.Connection,
) -> None:
    report = build_experiment_report(
        conn,
        _review_comparison(conn),
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


def test_report_threshold_is_sourced_from_policy_config_not_fabricated(
    conn: sqlite3.Connection,
) -> None:
    report = build_experiment_report(
        conn,
        _review_comparison(conn),
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


def test_report_gate_results_are_the_real_gate_results_unmodified(
    conn: sqlite3.Connection,
) -> None:
    comparison = _review_comparison(conn)
    report = build_experiment_report(
        conn,
        comparison,
        _policy(),
        experiment_id="x",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    assert report.gate_results == comparison.release.gates


def test_report_decision_reasons_lists_every_non_go_gate(conn: sqlite3.Connection) -> None:
    report = build_experiment_report(
        conn,
        _review_comparison(conn),
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


def test_report_decision_reasons_falls_back_to_summary_when_every_gate_passes(
    conn: sqlite3.Connection,
) -> None:
    report = build_experiment_report(
        conn,
        _go_comparison(conn),
        _policy(),
        experiment_id="x",
        baseline_version="v1",
        candidate_version="v2",
        dataset="dev",
    )
    assert report.decision == "GO"
    assert report.decision_reasons == ["GO — all 1 gate(s) passed"]


def test_report_timestamp_is_the_comparisons_persisted_created_at_not_generation_time(
    conn: sqlite3.Connection,
) -> None:
    comparison = _go_comparison(conn)
    report = build_experiment_report(
        conn,
        comparison,
        _policy(),
        experiment_id="x",
        baseline_version="v1",
        candidate_version="v2",
        dataset="dev",
    )
    assert report.timestamp == comparison.created_at


def test_report_round_trips_through_json(conn: sqlite3.Connection) -> None:
    report = build_experiment_report(
        conn,
        _review_comparison(conn),
        _policy(),
        experiment_id="x",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    restored = ExperimentReport.model_validate_json(report.model_dump_json())
    assert restored == report


def test_write_report_writes_valid_json_to_disk(conn: sqlite3.Connection, tmp_path) -> None:
    report = build_experiment_report(
        conn,
        _go_comparison(conn),
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


# ─────────────────────────────────────────────────────────────────────────────
# cost breakdown — Phase 3: generator/guardrail/judge cost, split by service,
# computed from already-persisted Response.cost_usd / GuardrailResult.cost_usd /
# Score.judge_cost_usd. Purely additive: cost_per_query (the release-gate metric)
# is untouched — these are extra visibility fields, not a new gate input.
# ─────────────────────────────────────────────────────────────────────────────


def test_cost_breakdown_sums_all_three_real_services(conn: sqlite3.Connection) -> None:
    report = build_experiment_report(
        conn,
        _review_comparison(conn),
        _policy(),
        experiment_id="x",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    baseline = report.baseline_cost_breakdown
    assert baseline.generator_cost == 0.001
    assert baseline.guardrail_cost == 0.0002
    assert baseline.judge_cost == 0.0005
    assert baseline.total_evaluation_cost == pytest.approx(0.001 + 0.0002 + 0.0005)

    candidate = report.candidate_cost_breakdown
    assert candidate.generator_cost == 0.0014
    assert candidate.guardrail_cost == 0.0002
    assert candidate.judge_cost == 0.0005
    assert candidate.total_evaluation_cost == pytest.approx(0.0014 + 0.0002 + 0.0005)


def test_cost_breakdown_leaves_missing_components_as_none_not_zero(
    conn: sqlite3.Connection,
) -> None:
    """_go_comparison's runs have no guardrail cost at all (DETERMINISTIC-only
    guardrail checks have no cost_usd) — this must read as None, never a fabricated
    0.0, matching compute_cost_per_query's existing 'unverified, not zero' rule."""
    report = build_experiment_report(
        conn,
        _go_comparison(conn),
        _policy(),
        experiment_id="x",
        baseline_version="v1",
        candidate_version="v2",
        dataset="dev",
    )
    baseline = report.baseline_cost_breakdown
    assert baseline.generator_cost == 0.001
    assert baseline.guardrail_cost is None
    assert baseline.judge_cost == 0.0004


def test_cost_breakdown_total_is_none_if_any_component_is_missing(
    conn: sqlite3.Connection,
) -> None:
    """total_evaluation_cost must never be a silently-partial sum (the same
    discipline compute_cost_per_query already enforces for its own aggregate) — a
    'total' that quietly drops a missing component would misrepresent itself as
    complete. _go_comparison's runs have no guardrail cost data at all, so the total
    must be None even though generator_cost and judge_cost are both real numbers."""
    report = build_experiment_report(
        conn,
        _go_comparison(conn),
        _policy(),
        experiment_id="x",
        baseline_version="v1",
        candidate_version="v2",
        dataset="dev",
    )
    assert report.baseline_cost_breakdown.total_evaluation_cost is None
    assert report.candidate_cost_breakdown.total_evaluation_cost is None


def test_cost_breakdown_never_alters_the_release_gate_metric(conn: sqlite3.Connection) -> None:
    """The cost breakdown is purely additional evidence — cost_per_query (the actual
    release-gate metric, generator-only, computed by regression/compare.py) must be
    byte-identical whether or not the cost breakdown is present."""
    comparison = _review_comparison(conn)
    report = build_experiment_report(
        conn,
        comparison,
        _policy(),
        experiment_id="x",
        baseline_version="v2",
        candidate_version="v3",
        dataset="dev",
    )
    original_cost_delta = next(d for d in comparison.deltas if d.metric.value == "cost_per_query")
    assert report.delta["cost_per_query"] == original_cost_delta.delta
    assert report.baseline_values["cost_per_query"] == original_cost_delta.baseline_value
    assert report.candidate_values["cost_per_query"] == original_cost_delta.candidate_value
