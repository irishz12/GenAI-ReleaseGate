// Display metadata for every metric name that appears in results/reports/*.json.
// Purely presentational (label, unit, formatting, which direction is "better") —
// no numbers live here. Every metric key must match evalguard.enums.MetricName's
// `.value` string exactly, since that's what keys every report's metric dicts.

export type MetricDirection = "higher_is_better" | "lower_is_better";

export type FormatKind = "pct" | "pct-signed" | "ms" | "usd";

export interface MetricMeta {
  key: string;
  label: string;
  group: "quality" | "safety" | "cost_latency";
  direction: MetricDirection;
  formatKind: FormatKind;
  format: (value: number) => string;
}

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const ms = (v: number) => `${v.toFixed(0)} ms`;
const usd = (v: number) => `$${(v * 1000).toFixed(4)} / 1k queries`;

export const METRICS: Record<string, MetricMeta> = {
  answer_correctness: {
    key: "answer_correctness",
    label: "Answer correctness",
    group: "quality",
    direction: "higher_is_better",
    formatKind: "pct",
    format: pct,
  },
  faithfulness: {
    key: "faithfulness",
    label: "Faithfulness",
    group: "quality",
    direction: "higher_is_better",
    formatKind: "pct",
    format: pct,
  },
  hallucination: {
    key: "hallucination",
    label: "Hallucination rate",
    group: "quality",
    direction: "lower_is_better",
    formatKind: "pct",
    format: pct,
  },
  instruction_following: {
    key: "instruction_following",
    label: "Instruction following",
    group: "quality",
    direction: "higher_is_better",
    formatKind: "pct",
    format: pct,
  },
  abstention_accuracy: {
    key: "abstention_accuracy",
    label: "Abstention accuracy",
    group: "quality",
    direction: "higher_is_better",
    formatKind: "pct",
    format: pct,
  },
  prompt_injection_block_rate: {
    key: "prompt_injection_block_rate",
    label: "Prompt injection block rate",
    group: "safety",
    direction: "higher_is_better",
    formatKind: "pct",
    format: pct,
  },
  sensitive_information_protection: {
    key: "sensitive_information_protection",
    label: "Sensitive-information protection",
    group: "safety",
    direction: "higher_is_better",
    formatKind: "pct",
    format: pct,
  },
  benign_false_positive_rate: {
    key: "benign_false_positive_rate",
    label: "Benign false positives (guardrail suite)",
    group: "safety",
    direction: "lower_is_better",
    formatKind: "pct",
    format: pct,
  },
  overall_benign_false_positive_rate: {
    key: "overall_benign_false_positive_rate",
    label: "Benign false positives (all non-attack cases)",
    group: "safety",
    direction: "lower_is_better",
    formatKind: "pct",
    format: pct,
  },
  p50_latency: {
    key: "p50_latency",
    label: "p50 latency",
    group: "cost_latency",
    direction: "lower_is_better",
    formatKind: "ms",
    format: ms,
  },
  p95_latency: {
    key: "p95_latency",
    label: "p95 latency",
    group: "cost_latency",
    direction: "lower_is_better",
    formatKind: "ms",
    format: ms,
  },
  cost_per_query: {
    key: "cost_per_query",
    label: "Cost per query",
    group: "cost_latency",
    direction: "lower_is_better",
    formatKind: "usd",
    format: usd,
  },
  failure_rate: {
    key: "failure_rate",
    label: "Failure rate",
    group: "cost_latency",
    direction: "lower_is_better",
    formatKind: "pct",
    format: pct,
  },
};

export function metricLabel(key: string): string {
  return METRICS[key]?.label ?? key;
}

export function formatMetric(key: string, value: number): string {
  const meta = METRICS[key];
  return meta ? meta.format(value) : value.toFixed(4);
}

/** Signed percentage-point delta display for ratio-style (0-1) metrics — used
 * for correctness/faithfulness/etc. deltas, which are always reported in the
 * report's raw 0-1 units, never already-percent. */
export function formatDeltaPct(value: number): string {
  const pp = value * 100;
  const sign = pp >= 0 ? "+" : "";
  return `${sign}${pp.toFixed(2)} pp`;
}

export function formatCI(ci: [number, number] | null): string {
  if (!ci) return "—";
  return `[${(ci[0] * 100).toFixed(1)}, ${(ci[1] * 100).toFixed(1)}] pp`;
}
