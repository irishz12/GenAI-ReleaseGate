import { getAllReports } from "./reports";

/**
 * Guardrail v1-vs-v2 head-to-head numbers. These four figures are NOT in
 * results/reports/*.json — they come from a direct, real
 * `ApplyGuardrail`-call comparison (scripts/compare_guardrail_versions.py,
 * both guardrail configs tested against the same 120 dev cases) that was
 * never persisted as a `Comparison` row, only printed and recorded. The
 * exact same four numbers, with the exact same citation, are already
 * hardcoded in scripts/generate_result_charts.py (the source of
 * results/charts/guardrail_improvement.png) — this is not a new number,
 * it's the same real measurement carried into the frontend.
 */
export const GUARDRAIL_COMPARISON = {
  source: "scripts/compare_guardrail_versions.py (Phase 8.2), 120 dev cases",
  v1: {
    label: "Guardrail v1 (PROMPT_ATTACK=HIGH)",
    suiteBenignFalsePositiveRate: 0.5556,
    overallBenignFalsePositiveRate: 0.1212,
  },
  v2: {
    label: "Guardrail v2 (PROMPT_ATTACK=MEDIUM)",
    suiteBenignFalsePositiveRate: 0.1111,
    overallBenignFalsePositiveRate: 0.0606,
  },
} as const;

/** Attack-blocking / PII-protection rates ARE in the real reports — every
 * comparison measured a perfect 1.0 on both, under both guardrail
 * versions, so any one report's baseline/candidate values already answer
 * this. Pulled here rather than re-typed. */
export function getAttackBlockingRates() {
  const reports = getAllReports();
  const anyReport = reports.v1_vs_v2_dev;
  return {
    promptInjectionBlockRate: anyReport.baseline_values.prompt_injection_block_rate,
    sensitiveInformationProtection:
      anyReport.baseline_values.sensitive_information_protection,
  };
}
