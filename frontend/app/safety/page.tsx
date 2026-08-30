import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { MetricBarChart } from "@/components/charts/metric-bar-chart";
import { GUARDRAIL_COMPARISON, getAttackBlockingRates } from "@/data/safety";

export default function SafetyPage() {
  const attack = getAttackBlockingRates();
  const fpData = [
    {
      label: "Guardrail v1",
      value: GUARDRAIL_COMPARISON.v1.suiteBenignFalsePositiveRate,
      fill: "#6b7280",
    },
    {
      label: "Guardrail v2",
      value: GUARDRAIL_COMPARISON.v2.suiteBenignFalsePositiveRate,
      fill: "#15803d",
    },
  ];
  const overallFpData = [
    {
      label: "Guardrail v1",
      value: GUARDRAIL_COMPARISON.v1.overallBenignFalsePositiveRate,
      fill: "#6b7280",
    },
    {
      label: "Guardrail v2",
      value: GUARDRAIL_COMPARISON.v2.overallBenignFalsePositiveRate,
      fill: "#15803d",
    },
  ];

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Safety</h1>
        <p className="mt-2 max-w-2xl text-ink-muted">
          A real AWS Bedrock Guardrail (native <code className="font-mono text-xs">ApplyGuardrail</code>,
          prompt-injection + PII policies) sits in front of every generated response. The
          objective is <strong>effective safety with minimal disruption</strong> — a guardrail that
          blocks everything is trivially safe and trivially useless.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Prompt injection block rate</CardTitle>
            <CardDescription>Both guardrail versions</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="font-mono text-4xl font-semibold text-go">
              {(attack.promptInjectionBlockRate * 100).toFixed(0)}%
            </p>
            <p className="mt-2 text-sm text-ink-muted">Unchanged across v1 → v2.</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Sensitive-information protection</CardTitle>
            <CardDescription>Both guardrail versions</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="font-mono text-4xl font-semibold text-go">
              {(attack.sensitiveInformationProtection * 100).toFixed(0)}%
            </p>
            <p className="mt-2 text-sm text-ink-muted">Unchanged across v1 → v2.</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Benign false positives — guardrail suite</CardTitle>
            <CardDescription>9 benign near-miss cases</CardDescription>
          </CardHeader>
          <CardContent>
            <MetricBarChart data={fpData} formatKind="pct" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Benign false positives — all non-attack cases</CardTitle>
            <CardDescription>99 of 120 dev cases</CardDescription>
          </CardHeader>
          <CardContent>
            <MetricBarChart data={overallFpData} formatKind="pct" />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>What changed</CardTitle>
          <CardDescription>Source: {GUARDRAIL_COMPARISON.source}</CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-ink-muted">
          <p>
            A single-parameter change — attack-detection strength{" "}
            <code className="font-mono text-xs">PROMPT_ATTACK</code> from{" "}
            <code className="font-mono text-xs">HIGH</code> to{" "}
            <code className="font-mono text-xs">MEDIUM</code>, PII policy untouched — cut benign
            false positives from{" "}
            <strong className="text-ink">
              {(GUARDRAIL_COMPARISON.v1.suiteBenignFalsePositiveRate * 100).toFixed(2)}%
            </strong>{" "}
            to{" "}
            <strong className="text-ink">
              {(GUARDRAIL_COMPARISON.v2.suiteBenignFalsePositiveRate * 100).toFixed(2)}%
            </strong>{" "}
            on the guardrail suite, with zero cost to attack-blocking or PII protection.
            <strong className="text-ink"> Adopted.</strong>
          </p>
          <p className="mt-3">
            V2&apos;s HOLD decision (see{" "}
            <a href="/release-gate" className="underline underline-offset-2">
              Release Gate
            </a>
            ) includes a benign-false-positive-rate violation measured at 55.56% — that number is
            from Guardrail v1, the version active when V2 was tested. V2 was never re-evaluated
            under Guardrail v2, so it&apos;s unknown whether V2 would clear the safety cap under
            the calibrated guardrail. This is a guardrail-version effect, not evidence that V2&apos;s
            prompt is worse than V3&apos;s on this axis.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
