import "server-only";

import fs from "node:fs";
import path from "node:path";

import type { ExperimentReport } from "./types";

// READ-ONLY access to this project's real, already-computed comparisons.
// results/reports/*.json is produced entirely by the Python backend
// (scripts/generate_reports.py, src/evalguard/reporting/) — never by this
// frontend. Nothing here calls a network API, calls Bedrock, or writes to
// disk; it only parses JSON that already exists on disk at build/render
// time. This is the ONLY place in the frontend that touches the
// filesystem — every page and component gets its data by importing from
// this module or from data/experiments.ts, data/metrics.ts, etc., which
// are themselves built on top of it.
const REPORTS_DIR = path.join(process.cwd(), "..", "results", "reports");

const REPORT_IDS = [
  "v1_vs_v2_dev",
  "v1_vs_v3_dev",
  "v1_vs_v3.1_dev",
  "v1_vs_v3.2_dev",
  "v2_vs_v3_dev",
  "v1_vs_v3.3_dev",
  "v1_vs_v3.4_dev",
  "v1_vs_v3_holdout",
  "v1_vs_v3.4_holdout",
] as const;

export type ReportId = (typeof REPORT_IDS)[number];

function readReport(id: ReportId): ExperimentReport {
  const filePath = path.join(REPORTS_DIR, `${id}.json`);
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as ExperimentReport;
}

let cache: Record<ReportId, ExperimentReport> | null = null;

/** Every real, persisted comparison — the full V1→V2→V3→V3.1→V3.2→V3.3→V3.4
 * release history, on both dev and holdout. Cached per server process — the
 * files are immutable build artifacts, not live data, so re-reading them
 * on every request would be pure overhead, not a freshness concern. */
export function getAllReports(): Record<ReportId, ExperimentReport> {
  if (!cache) {
    cache = Object.fromEntries(
      REPORT_IDS.map((id) => [id, readReport(id)]),
    ) as Record<ReportId, ExperimentReport>;
  }
  return cache;
}

export function getReport(id: ReportId): ExperimentReport {
  return getAllReports()[id];
}
