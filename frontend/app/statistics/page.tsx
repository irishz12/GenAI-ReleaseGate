import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { DecisionBadge } from "@/components/decision-badge";
import { FixtureBadge } from "@/components/fixture-badge";
import { DeltaCIChart } from "@/components/charts/delta-ci-chart";
import { getAllReports } from "@/data/reports";

const COMPARISON_LABELS: Record<string, string> = {
  v1_vs_v2_dev: "V2 vs V1 (dev)",
  v1_vs_v3_dev: "V3 vs V1 (dev)",
  v2_vs_v3_dev: "V3 vs V2 (dev)",
  v1_vs_v3_holdout: "V3 vs V1 (holdout, sealed)",
};

export default function StatisticsPage() {
  const reports = getAllReports();

  const correctnessRows = Object.entries(reports).map(([id, report]) => ({
    label: COMPARISON_LABELS[id],
    delta: report.delta.answer_correctness * 100,
    ciLow: report.confidence_interval.answer_correctness?.[0] ?? null,
    ciHigh: report.confidence_interval.answer_correctness?.[1] ?? null,
    ciLowPp:
      report.confidence_interval.answer_correctness != null
        ? report.confidence_interval.answer_correctness[0] * 100
        : null,
    ciHighPp:
      report.confidence_interval.answer_correctness != null
        ? report.confidence_interval.answer_correctness[1] * 100
        : null,
    effectSize: report.effect_size.answer_correctness,
    decision: report.decision,
  }));

  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Statistics</h1>
        <p className="mt-2 max-w-2xl text-ink-muted">
          Every comparison reports a paired mean delta and a paired bootstrap 95% confidence
          interval (2,000 resamples, fixed seed — identical DB state always produces an identical
          interval), plus a paired effect size (Cohen&apos;s d). A Holm-Bonferroni correction is
          applied across the metrics tested within each comparison, computed the same way, but
          none of it feeds back into the release decision — the gates below still read only the
          raw delta and CI.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Answer correctness — paired delta with 95% CI, all four comparisons</CardTitle>
          <CardDescription>
            A bar whose interval crosses zero is statistically indistinguishable from no change.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <DeltaCIChart data={correctnessRows} height={320} />
        </CardContent>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2">
        {correctnessRows.map((r) => (
          <Card key={r.label}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">{r.label}</CardTitle>
                <DecisionBadge decision={r.decision} />
              </div>
            </CardHeader>
            <CardContent className="flex flex-col gap-1 text-sm">
              <p>
                delta:{" "}
                <span className="font-mono">
                  {r.delta >= 0 ? "+" : ""}
                  {r.delta.toFixed(2)} pp
                </span>
              </p>
              <p>
                95% CI:{" "}
                <span className="font-mono">
                  {r.ciLowPp !== null
                    ? `[${r.ciLowPp.toFixed(1)}, ${r.ciHighPp!.toFixed(1)}] pp`
                    : "—"}
                </span>
              </p>
              <p>
                effect size (Cohen&apos;s d):{" "}
                <span className="font-mono">
                  {r.effectSize !== null ? r.effectSize.toFixed(3) : "n/a (zero variance)"}
                </span>
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Dev vs. holdout: the generalization gap</CardTitle>
          <CardDescription>V3 vs V1, answer correctness</CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-ink-muted">
          <p>
            On the 120-case dev set, V3&apos;s correctness delta was{" "}
            <span className="font-mono text-ink">
              {(reports.v1_vs_v3_dev.delta.answer_correctness * 100).toFixed(2)} pp
            </span>{" "}
            with a CI that straddles zero. On the 40-case sealed holdout — cases neither prompt
            had touched before that one run — the same comparison measured{" "}
            <span className="font-mono text-hold">
              {(reports.v1_vs_v3_holdout.delta.answer_correctness * 100).toFixed(2)} pp
            </span>{" "}
            with a CI that excludes zero. This reversal, not any single number, is the headline
            result this project exists to surface.
          </p>
        </CardContent>
      </Card>

      <Card className="border-dashed border-accent bg-surface-muted">
        <CardHeader>
          <div className="flex flex-wrap items-center gap-3">
            <CardTitle>Real results vs. validation fixtures</CardTitle>
            <FixtureBadge />
          </div>
        </CardHeader>
        <CardContent className="text-sm text-ink-muted">
          <p>
            Every number on this page comes from{" "}
            <code className="font-mono text-xs">results/reports/*.json</code> — real,
            already-computed comparisons over actual generator/judge/guardrail responses. As of
            this writing, no real comparison has ever produced GO (see the{" "}
            <a href="/release-gate" className="underline underline-offset-2">
              Release Gate
            </a>{" "}
            page for why, and for the clearly-labeled synthetic fixture that proves GO is
            reachable). No historical decision shown here was recomputed or reinterpreted to
            produce a different answer.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
