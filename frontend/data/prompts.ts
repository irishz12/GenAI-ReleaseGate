import "server-only";

import fs from "node:fs";
import path from "node:path";

import type { PromptVersion } from "./experiments";

// Reads the real, committed prompt template files verbatim — never
// paraphrased or reconstructed. Same server-only, fs-at-build-time pattern
// as data/reports.ts.
const PROMPTS_DIR = path.join(process.cwd(), "..", "prompts");

const FILES: Record<PromptVersion, string> = {
  v1: "production/support_agent.v1.md",
  v2: "candidate/support_agent.v2.md",
  v3: "candidate/support_agent.v3.md",
  "v3.1": "candidate/support_agent.v3.1.md",
  "v3.2": "candidate/support_agent.v3.2.md",
  "v3.3": "candidate/support_agent.v3.3.md",
  "v3.4": "candidate/support_agent.v3.4.md",
};

export function getPromptTemplate(version: PromptVersion): string {
  return fs.readFileSync(path.join(PROMPTS_DIR, FILES[version]), "utf-8").trim();
}
