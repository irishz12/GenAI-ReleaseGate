import "server-only";

import fs from "node:fs";
import path from "node:path";

// Reads the real, committed prompt template files verbatim — never
// paraphrased or reconstructed. Same server-only, fs-at-build-time pattern
// as data/reports.ts.
const PROMPTS_DIR = path.join(process.cwd(), "..", "prompts");

const FILES: Record<"v1" | "v2" | "v3", string> = {
  v1: "production/support_agent.v1.md",
  v2: "candidate/support_agent.v2.md",
  v3: "candidate/support_agent.v3.md",
};

export function getPromptTemplate(version: "v1" | "v2" | "v3"): string {
  return fs.readFileSync(path.join(PROMPTS_DIR, FILES[version]), "utf-8").trim();
}
