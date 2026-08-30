// Mirrors src/evalguard/reporting/models.py's ExperimentReport field-for-field.
// Every field name here matches the JSON under results/reports/*.json verbatim —
// there is no rekeying, no renaming, nothing translated for display convenience.

export type Decision = "GO" | "REVIEW" | "HOLD" | "INVALID";
export type GateStatus = "GO" | "REVIEW" | "HOLD";

export interface GateResult {
  gate_name: string;
  status: GateStatus;
  threshold: number | null;
  observed: number | null;
  reason: string;
}

export interface ExperimentReport {
  experiment_id: string;
  baseline_version: string;
  candidate_version: string;
  dataset: "dev" | "holdout";
  valid_cases: number;
  metrics: string[];
  baseline_values: Record<string, number>;
  candidate_values: Record<string, number>;
  delta: Record<string, number>;
  confidence_interval: Record<string, [number, number] | null>;
  effect_size: Record<string, number | null>;
  threshold: Record<string, Record<string, number>>;
  gate_results: GateResult[];
  decision: Decision;
  decision_reasons: string[];
  timestamp: string;
}
