import type { Decision } from "./types";

export interface DecisionMeta {
  status: Decision;
  label: string;
  explanation: string;
}

export const DECISION_META: Record<Decision, DecisionMeta> = {
  GO: {
    status: "GO",
    label: "GO",
    explanation: "The candidate satisfies every release requirement — safe to ship.",
  },
  REVIEW: {
    status: "REVIEW",
    label: "REVIEW",
    explanation:
      "Evidence is inconclusive or a trade-off requires a human decision before shipping.",
  },
  HOLD: {
    status: "HOLD",
    label: "HOLD",
    explanation: "A hard release criterion was violated — automatic release block.",
  },
  INVALID: {
    status: "INVALID",
    label: "INVALID",
    explanation:
      "The evaluation itself is not trustworthy enough to decide (e.g. too many failed responses to trust the comparison).",
  },
};

/**
 * As of this writing, no real comparison in results/reports/*.json has ever
 * produced GO (v1_vs_v2_dev=HOLD, v1_vs_v3_dev=REVIEW, v2_vs_v3_dev=REVIEW,
 * v1_vs_v3_holdout=REVIEW). That's not a gap in the product — it's what
 * actually happened in this candidate's release history.
 *
 * This fixture exists ONLY to show that the engine supports and correctly
 * reaches GO. It is the same shape of deterministic, hand-built delta set
 * used by the backend's "Decision Engine Validation Scenarios"
 * (tests/unit/test_policy_engine.py::test_scenario_go_deterministic_fixture_every_gate_within_threshold),
 * evaluated against the real config/policy.yaml thresholds. It is a
 * VALIDATION FIXTURE, never a real V1/V2/V3 result — labeled as such
 * everywhere it's rendered.
 */
export const GO_VALIDATION_FIXTURE = {
  source:
    "tests/unit/test_policy_engine.py — Decision Engine Validation Scenarios (deterministic synthetic fixture, not a real experiment)",
  gates: [
    { gate_name: "answer_correctness", status: "GO" as const, delta: 0.05 },
    { gate_name: "faithfulness", status: "GO" as const, delta: 0.05 },
    { gate_name: "hallucination", status: "GO" as const, delta: -0.01 },
    { gate_name: "prompt_injection_block_rate", status: "GO" as const, delta: 0.0 },
    { gate_name: "sensitive_information_protection", status: "GO" as const, delta: 0.0 },
    { gate_name: "benign_false_positive_rate", status: "GO" as const, delta: 0.0 },
    { gate_name: "p95_latency", status: "GO" as const, delta: 0.0 },
    { gate_name: "cost_per_query", status: "GO" as const, delta: 0.0 },
  ],
};
