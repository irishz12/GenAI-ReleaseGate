import { getReport } from "./reports";
import type { Decision } from "./types";

export type PromptVersion = "v1" | "v2" | "v3";

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

/** The headline holdout finding — the one result this whole product exists
 * to surface. Every number here is read directly from the real,
 * sealed-holdout comparison; the two "human decision" fields are the one
 * piece of narrative fact that isn't a JSON field (there is no
 * "human_decision" column in the schema) — it's project history, recorded
 * in README.md §2 and git history, not something the policy engine
 * computed. It is kept visually and structurally separate from the
 * automated decision for exactly that reason. */
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
