import { getReport } from "./reports";
import type { Decision } from "./types";

export type PromptVersion = "v1" | "v2" | "v3" | "v3.1" | "v3.2" | "v3.3" | "v3.4";

export interface PromptStory {
  version: PromptVersion;
  role: "Production baseline" | "Candidate" | "Refined candidate";
  title: string;
  summary: string;
  /** null for v1: it is the incumbent, never itself "decided" against a
   * baseline. Sourced from the real comparison each version was actually
   * decided by (see `decidedBy` below) — never hand-typed. */
  decision: Decision | null;
  /** Which real, persisted comparison produced `decision`. */
  decidedBy: string | null;
}

// Narrative framing only (role labels, one-line summaries) — every decision
// value below is read out of the real reports, not asserted independently.
// This is the full, real release history: V2 was HELD, V3/V3.1/V3.2/V3.3 were
// each REVIEWed for a real reason, and V3.4 is the first (and so far only)
// candidate to pass every gate — on both dev and the sealed holdout.
export const PROMPT_STORY: PromptStory[] = [
  {
    version: "v1",
    role: "Production baseline",
    title: "V1 — Production",
    summary:
      "Answer from context only; plain refusal when the context lacks enough information.",
    decision: null,
    decidedBy: null,
  },
  {
    version: "v2",
    role: "Candidate",
    title: "V2 — Candidate",
    summary:
      "Added an emphatic insufficient-information rule to fix under-abstention.",
    decision: getReport("v1_vs_v2_dev").decision,
    decidedBy: "v1_vs_v2_dev",
  },
  {
    version: "v3",
    role: "Refined candidate",
    title: "V3 — Refined candidate",
    summary:
      "Root-caused V2's faithfulness regression and fixed it with two targeted rule changes.",
    decision: getReport("v1_vs_v3_dev").decision,
    decidedBy: "v1_vs_v3_dev",
  },
  {
    version: "v3.1",
    role: "Candidate",
    title: "V3.1",
    summary: "Added a rigid 2-sentence output cap to cut cost — reintroduced a faithfulness regression.",
    decision: getReport("v1_vs_v3.1_dev").decision,
    decidedBy: "v1_vs_v3.1_dev",
  },
  {
    version: "v3.2",
    role: "Candidate",
    title: "V3.2",
    summary: "Loosened the cap to 3 sentences — still over the cost threshold, faithfulness still flagged.",
    decision: getReport("v1_vs_v3.2_dev").decision,
    decidedBy: "v1_vs_v3.2_dev",
  },
  {
    version: "v3.3",
    role: "Candidate",
    title: "V3.3",
    summary:
      "Replaced the sentence cap with a qualitative anti-preamble instruction — closed most of the cost gap, missed the 15% threshold by 1.3pp.",
    decision: getReport("v1_vs_v3.3_dev").decision,
    decidedBy: "v1_vs_v3.3_dev",
  },
  {
    version: "v3.4",
    role: "Candidate",
    title: "V3.4",
    summary:
      "Trimmed the instruction text itself (the actual dominant cost driver) — passed all 8 release gates on dev and on the sealed holdout.",
    decision: getReport("v1_vs_v3.4_dev").decision,
    decidedBy: "v1_vs_v3.4_dev",
  },
];

/**
 * V1 -> V2 -> V3 dev-set values for one metric, for the three-way
 * comparison table/chart. V1's value comes from v1_vs_v2_dev (the run V2
 * was actually compared against); V3's value comes from v1_vs_v3_dev
 * (compared against a separate but equivalent V1 replay run — same
 * prompt, evaluated under the later guardrail version, which doesn't
 * affect quality metrics). Both V1 figures are real and close but not
 * always identical, because pairing only includes a case when both sides
 * of THAT SPECIFIC comparison have a valid score — see README.md's
 * Limitations section. Never averaged or reconciled into one synthetic V1
 * number; each is exactly what its own comparison measured.
 */
export function getThreeWayDevValue(metric: string) {
  const v1v2 = getReport("v1_vs_v2_dev");
  const v1v3 = getReport("v1_vs_v3_dev");
  return {
    v1: v1v2.baseline_values[metric],
    v2: v1v2.candidate_values[metric],
    v3: v1v3.candidate_values[metric],
    v1FromV3Comparison: v1v3.baseline_values[metric],
  };
}

function pctChange(delta: number, baseline: number): number {
  return (delta / baseline) * 100;
}

/**
 * The headline result: V3.4 is the first (and, as of this writing, only)
 * candidate to pass every real release gate — on the 120-case dev set AND
 * on the 40-case sealed holdout. Every number here is read directly from
 * the two real, persisted comparisons (v1_vs_v3.4_dev, v1_vs_v3.4_holdout)
 * — nothing computed independently of what the policy engine actually
 * decided. This is a REAL result, not the decision-engine validation
 * fixture (see data/decisions.ts) — the fixture existed only to prove GO
 * was reachable before any real candidate achieved it; it played no part
 * in this outcome.
 */
export function getV34Headline() {
  const dev = getReport("v1_vs_v3.4_dev");
  const holdout = getReport("v1_vs_v3.4_holdout");

  return {
    dev,
    holdout,
    devCostPct: pctChange(dev.delta.cost_per_query, dev.baseline_values.cost_per_query),
    holdoutCostPct: pctChange(
      holdout.delta.cost_per_query,
      holdout.baseline_values.cost_per_query,
    ),
    devCorrectnessDeltaPp: dev.delta.answer_correctness * 100,
    devFaithfulnessDeltaPp: dev.delta.faithfulness * 100,
    devHallucinationPct: dev.candidate_values.hallucination * 100,
    devP95DeltaPct: pctChange(dev.delta.p95_latency, dev.baseline_values.p95_latency),
    devPromptInjectionBlockRate: dev.candidate_values.prompt_injection_block_rate * 100,
    devSensitiveInfoProtection: dev.candidate_values.sensitive_information_protection * 100,
  };
}

/** The full rejected/reviewed candidate history before V3.4 — every entry
 * is a real, persisted comparison's own decision and reason, read from the
 * report, not hand-typed. */
export function getFailedCandidates() {
  const v2 = getReport("v1_vs_v2_dev");
  const v3 = getReport("v1_vs_v3_dev");
  const v31 = getReport("v1_vs_v3.1_dev");
  const v32 = getReport("v1_vs_v3.2_dev");
  const v33 = getReport("v1_vs_v3.3_dev");

  const costPct = (r: typeof v3) => pctChange(r.delta.cost_per_query, r.baseline_values.cost_per_query);

  return [
    { version: "V2", decision: v2.decision, reasons: v2.decision_reasons },
    { version: "V3", decision: v3.decision, reasons: v3.decision_reasons, costPct: costPct(v3) },
    { version: "V3.1", decision: v31.decision, reasons: v31.decision_reasons, costPct: costPct(v31) },
    { version: "V3.2", decision: v32.decision, reasons: v32.decision_reasons, costPct: costPct(v32) },
    { version: "V3.3", decision: v33.decision, reasons: v33.decision_reasons, costPct: costPct(v33) },
  ];
}

/** Cost delta (%) vs. release decision for every real candidate, V2 through V3.4 —
 * the cost/quality-tradeoff visualization: every candidate is plotted, whether or
 * not cost was the reason it was reviewed/held, so the chart shows the real trend
 * (cost overrun shrinking release after release) rather than only the cases where
 * cost happened to be the triggering gate. V2's cost delta is real (from
 * v1_vs_v2_dev) even though its HOLD was driven by faithfulness/safety, not cost. */
export function getCostVsDecisionData() {
  const reports: Array<[string, ReturnType<typeof getReport>]> = [
    ["V2", getReport("v1_vs_v2_dev")],
    ["V3", getReport("v1_vs_v3_dev")],
    ["V3.1", getReport("v1_vs_v3.1_dev")],
    ["V3.2", getReport("v1_vs_v3.2_dev")],
    ["V3.3", getReport("v1_vs_v3.3_dev")],
    ["V3.4", getReport("v1_vs_v3.4_dev")],
  ];
  return reports.map(([version, r]) => ({
    version,
    decision: r.decision,
    costPct: pctChange(r.delta.cost_per_query, r.baseline_values.cost_per_query),
  }));
}

/** The headline holdout finding for V3 (the original, still-real REVIEW
 * story) — the one result this project surfaced before V3.4 existed. Every
 * number here is read directly from the real, sealed-holdout comparison;
 * the two "human decision" fields are the one piece of narrative fact that
 * isn't a JSON field (there is no "human_decision" column in the schema) —
 * it's project history, recorded in README.md and git history, not
 * something the policy engine computed. Kept visually and structurally
 * separate from the automated decision for exactly that reason. */
export function getHoldoutHeadline() {
  const report = getReport("v1_vs_v3_holdout");
  return {
    report,
    correctnessDeltaPp: report.delta.answer_correctness * 100,
    ci: report.confidence_interval.answer_correctness,
    automatedDecision: report.decision,
    humanDecision: "V3 NOT PROMOTED" as const,
    humanDecisionNote:
      "A human reviewer, given this REVIEW-with-a-confirmed-regression evidence, declined to promote V3. V1 remained the production baseline — a human judgment call, not an automated HOLD.",
  };
}
