import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { DecisionBadge } from "@/components/decision-badge";
import { FixtureBadge } from "@/components/fixture-badge";
import { getAllReports } from "@/data/reports";
import { DECISION_META, GO_VALIDATION_FIXTURE } from "@/data/decisions";
import { metricLabel } from "@/data/metrics";
import type { GateStatus } from "@/data/types";

const COMPARISON_LABELS: Record<string, string> = {
  v1_vs_v2_dev: "V2 vs V1 — dev",
  v1_vs_v3_dev: "V3 vs V1 — dev",
  "v1_vs_v3.1_dev": "V3.1 vs V1 — dev",
  "v1_vs_v3.2_dev": "V3.2 vs V1 — dev",
  v2_vs_v3_dev: "V3 vs V2 — dev",
  "v1_vs_v3.3_dev": "V3.3 vs V1 — dev",
  "v1_vs_v3.4_dev": "V3.4 vs V1 — dev",
  v1_vs_v3_holdout: "V3 vs V1 — holdout (sealed)",
  "v1_vs_v3.4_holdout": "V3.4 vs V1 — holdout (sealed)",
};

const GATE_BADGE: Record<GateStatus, "go" | "review" | "hold"> = {
  GO: "go",
  REVIEW: "review",
  HOLD: "hold",
};

export default function ReleaseGatePage() {
  const reports = getAllReports();

  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Release Gate</h1>
        <p className="mt-2 max-w-2xl text-ink-muted">
          One declarative policy (<code className="font-mono text-xs">config/policy.yaml</code>),
          worst-gate-wins: every configured gate is evaluated independently, and the overall
          decision is the single worst outcome among them. This is the centerpiece of GenAI
          ReleaseGate — every result below is real, computed evidence.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {(["GO", "REVIEW", "HOLD", "INVALID"] as const).map((status) => (
          <Card key={status}>
            <CardHeader>
              <div className="flex items-center gap-2">
                <DecisionBadge decision={status} />
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-ink-muted">{DECISION_META[status].explanation}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div>
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-2xl font-semibold tracking-tight">Real experiment results</h2>
          <Badge variant="real">REAL DATA</Badge>
        </div>
        <p className="mt-1 text-ink-muted">
          Every real, persisted comparison — the full release history. V2 through V3.3 were each
          HELD or REVIEWed for a real reason; <strong className="text-ink">V3.4 is the first
          candidate to pass all 8 gates</strong>, on both the dev set and the sealed holdout. The
          decision engine evaluates every gate and applies the configured policy —{" "}
          <strong className="text-ink">no manual override was used</strong> to produce any result
          below, including V3.4&apos;s GO.
        </p>
      </div>

      <div className="flex flex-col gap-6">
        {Object.entries(reports).map(([id, report]) => (
          <Card key={id}>
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <CardTitle>{COMPARISON_LABELS[id]}</CardTitle>
                <DecisionBadge decision={report.decision} />
              </div>
              <CardDescription>
                {report.valid_cases} paired cases · {report.dataset} split
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <ul className="flex flex-col gap-1 text-sm text-ink-muted">
                {report.decision_reasons.map((reason) => (
                  <li key={reason}>— {reason}</li>
                ))}
              </ul>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Gate</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Reason</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {report.gate_results.map((gate) => (
                    <TableRow key={gate.gate_name}>
                      <TableCell className="font-medium">
                        {metricLabel(gate.gate_name)}
                      </TableCell>
                      <TableCell>
                        <Badge variant={GATE_BADGE[gate.status]}>{gate.status}</Badge>
                      </TableCell>
                      <TableCell className="text-ink-muted">{gate.reason}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        ))}
      </div>

      <div>
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-2xl font-semibold tracking-tight">Decision-engine validation fixture</h2>
          <FixtureBadge />
        </div>
        <p className="mt-1 text-ink-muted">
          Before V3.4&apos;s real GO existed, this clearly-labeled synthetic fixture — NOT a real
          experiment — was used to prove the engine could reach GO at all. Same shape as the
          backend&apos;s own{" "}
          <code className="font-mono text-xs">
            test_scenario_go_deterministic_fixture_every_gate_within_threshold
          </code>
          , evaluated against the real <code className="font-mono text-xs">config/policy.yaml</code>{" "}
          thresholds. It played no part in V3.4&apos;s result and is kept only for engine
          validation.
        </p>
      </div>
      <Card className="border-dashed border-accent bg-surface-muted">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle>What GO looks like (synthetic, not a real result)</CardTitle>
            <div className="flex items-center gap-2">
              <FixtureBadge />
              <DecisionBadge decision="GO" />
            </div>
          </div>
          <CardDescription>Source: {GO_VALIDATION_FIXTURE.source}</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Gate</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Synthetic delta</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {GO_VALIDATION_FIXTURE.gates.map((g) => (
                <TableRow key={g.gate_name}>
                  <TableCell className="font-medium">{metricLabel(g.gate_name)}</TableCell>
                  <TableCell>
                    <Badge variant="go">{g.status}</Badge>
                  </TableCell>
                  <TableCell className="font-mono text-ink-muted">
                    {g.delta >= 0 ? "+" : ""}
                    {g.delta.toFixed(2)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
