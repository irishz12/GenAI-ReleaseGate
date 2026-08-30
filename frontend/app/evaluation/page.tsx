import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
import { MetricBarChart } from "@/components/charts/metric-bar-chart";
import { getThreeWayDevValue } from "@/data/experiments";
import { METRICS, formatMetric } from "@/data/metrics";

const HEADLINE_METRICS = [
  "answer_correctness",
  "faithfulness",
  "hallucination",
  "abstention_accuracy",
  "cost_per_query",
  "p95_latency",
];

const COLORS = { v1: "#6b7280", v2: "#1d4ed8", v3: "#15803d" };

export default function EvaluationPage() {
  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Evaluation</h1>
        <p className="mt-2 max-w-2xl text-ink-muted">
          V1 → V2 → V3 on the 120-case development set, across quality, cost, and latency. Every
          value below is read from the real, persisted comparisons — nothing is hand-typed.
        </p>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        {HEADLINE_METRICS.map((key) => {
          const meta = METRICS[key];
          const values = getThreeWayDevValue(key);
          const data = [
            { label: "V1", value: values.v1, fill: COLORS.v1 },
            { label: "V2", value: values.v2, fill: COLORS.v2 },
            { label: "V3", value: values.v3, fill: COLORS.v3 },
          ];
          return (
            <Card key={key}>
              <CardHeader>
                <CardTitle>{meta.label}</CardTitle>
                <CardDescription>
                  {meta.direction === "higher_is_better" ? "Higher is better" : "Lower is better"}
                  {key === "cost_per_query" && " — generator (model inference) cost only"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <MetricBarChart data={data} formatKind={meta.formatKind} />
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Full metric table (dev, 120 paired cases)</CardTitle>
          <CardDescription>
            V1/V2 from the real v1_vs_v2_dev comparison; V3 from the real v1_vs_v3_dev comparison.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                <TableHead>V1</TableHead>
                <TableHead>V2</TableHead>
                <TableHead>V3</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.keys(METRICS).map((key) => {
                const values = getThreeWayDevValue(key);
                if (values.v1 === undefined) return null;
                return (
                  <TableRow key={key}>
                    <TableCell className="font-medium">{METRICS[key].label}</TableCell>
                    <TableCell className="font-mono">{formatMetric(key, values.v1)}</TableCell>
                    <TableCell className="font-mono">{formatMetric(key, values.v2)}</TableCell>
                    <TableCell className="font-mono">{formatMetric(key, values.v3)}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
