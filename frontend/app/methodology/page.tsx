import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const SECTIONS = [
  {
    title: "Dev vs. sealed holdout",
    body: "120 development cases + 40 sealed holdout cases, sourced from the doc2dial dialogue corpus (v1.0.1), seeded (seed=42) and content-hashed so the split is reproducible without committing holdout content. data/holdout/ is gitignored; only its count, seed, and integrity hashes are version-controlled. The holdout is consumed exactly once, by design.",
  },
  {
    title: "Paired evaluation",
    body: "V1 and a candidate run separately over the same frozen dataset and the same pinned model config, then join per case_id. Pairing means the comparison is driven by prompt behavior, not by which run happened to get easier cases.",
  },
  {
    title: "Blind LLM judge",
    body: "Correctness and faithfulness are scored by a judge model at temperature 0 that never sees which prompt version or run produced an answer — hallucination is derived from the same faithfulness call at no extra cost. Deterministic checks (instruction-following, abstention detection) are used wherever code alone can check them.",
  },
  {
    title: "AWS Bedrock Guardrail",
    body: "A real Amazon Bedrock Guardrail (native ApplyGuardrail, prompt-injection + PII policies) checks every input and output. Guardrail calibration (see Safety) was a real, direct head-to-head test of two configurations against the same 120 dev cases.",
  },
  {
    title: "Bootstrap confidence intervals",
    body: "A paired bootstrap 95% CI (2,000 resamples, fixed seed) accompanies every mean delta — deterministic: identical DB state always produces an identical interval. Median paired difference, effect size (Cohen's d), and Holm-Bonferroni multiple-comparison correction are computed alongside, never in place of, the mean and CI.",
  },
  {
    title: "Release policy",
    body: "A declarative config/policy.yaml defines gates per metric. Quality and safety gates are hard (can HOLD); cost and latency gates are soft (can only REVIEW — structurally impossible to HOLD, by construction). The overall decision is worst-gate-wins: any HOLD -> HOLD, else any REVIEW -> REVIEW, else GO. A comparison is INVALID, independent of any gate, if too many paired cases failed on either side.",
  },
  {
    title: "Reproducibility",
    body: "Every prompt is identified by a SHA-256 content hash; a run's identity is a content hash over its prompt, case set, model params, and code. Charts and JSON reports are regenerable from artifacts/*.db with no new AWS calls.",
  },
];

export default function MethodologyPage() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Methodology</h1>
        <p className="mt-2 max-w-2xl text-ink-muted">
          How every number on this site was actually produced.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {SECTIONS.map((s) => (
          <Card key={s.title}>
            <CardHeader>
              <CardTitle>{s.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-ink-muted">{s.body}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
