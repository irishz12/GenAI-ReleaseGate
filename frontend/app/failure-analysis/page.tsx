import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { DecisionBadge } from "@/components/decision-badge";
import { getAllReports } from "@/data/reports";
import { formatCI } from "@/data/metrics";

export default function FailureAnalysisPage() {
  const reports = getAllReports();
  const holdout = reports.v1_vs_v3_holdout;

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Failure Analysis</h1>
        <p className="mt-2 max-w-2xl text-ink-muted">
          Two distinct root causes, found by reading actual failing responses — not just
          aggregate numbers.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Root cause 1 — over-triggering refusal</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-ink-muted">
            <p>
              V2&apos;s stronger insufficient-information rule caused the model to decline
              questions the context actually answered, especially when phrased
              conversationally (&quot;Can you tell me more about it?&quot;, &quot;However,
              there&apos;s another issue...&quot;).
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Root cause 2 — judge sensitivity to bare refusals</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-ink-muted">
            <p>
              Even a <em>correct</em> refusal scored 0.0 faithfulness under the judge&apos;s
              claim-extraction rubric if phrased as a generic, content-free sentence. Restating
              specifically what the context does/doesn&apos;t cover scored as a supported,
              checkable claim instead.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>How V3 addressed them</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-sm text-ink-muted">
          <p>
            Two targeted rule changes in{" "}
            <code className="font-mono text-xs">prompts/candidate/support_agent.v3.md</code>{" "}
            (see <a href="/prompts" className="underline underline-offset-2">Prompts</a>): a
            check-before-refusing gate targeting root cause 1, and a &quot;restate specifically
            what&apos;s missing&quot; requirement targeting root cause 2 — while preserving the
            literal phrase the deterministic abstention detector needs.
          </p>
          <p>
            <strong className="text-ink">Root cause 2 was fully resolved</strong> — no V3 refusal,
            correct or not, ever scored a bare 0.0 again.
          </p>
          <p>
            <strong className="text-ink">Root cause 1 was only partially fixed.</strong> The
            residual is exactly what the holdout regression below surfaced.
          </p>
        </CardContent>
      </Card>

      <Card className="border-hold-rule bg-hold-bg/40">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>What the holdout revealed</CardTitle>
            <DecisionBadge decision={holdout.decision} />
          </div>
          <CardDescription>V3 vs V1, sealed 40-case holdout</CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-ink-muted">
          <p>
            Answer correctness delta:{" "}
            <span className="font-mono text-hold">
              {(holdout.delta.answer_correctness * 100).toFixed(2)} pp
            </span>{" "}
            — 95% CI {formatCI(holdout.confidence_interval.answer_correctness)}, excluding zero.
            Inspecting the actual failing responses traced this to the same over-refusal pattern
            (root cause 1) already flagged but only partially fixed in dev testing, concentrated
            more heavily in this particular holdout sample. It is not an unexplained anomaly — it
            is the residual of a known, partially-fixed cause showing up harder on cases the
            prompt had never been evaluated against before.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
