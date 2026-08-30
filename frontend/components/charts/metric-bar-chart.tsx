"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FormatKind } from "@/data/metrics";

export interface MetricBarDatum {
  label: string;
  value: number;
  fill: string;
}

// Formatting is done here, from a serializable `formatKind` string, rather
// than accepting a formatter function prop — a Server Component can't pass
// a closure across the client-component boundary, and every page that
// renders this chart is a Server Component reading data/metrics.ts's static
// format functions.
function formatValue(kind: FormatKind, v: number): string {
  switch (kind) {
    case "pct":
      return `${(v * 100).toFixed(2)}%`;
    case "ms":
      return `${v.toFixed(0)} ms`;
    case "usd":
      return `$${(v * 1000).toFixed(4)}/1k`;
  }
}

export function MetricBarChart({
  data,
  formatKind,
  height = 260,
}: {
  data: MetricBarDatum[];
  formatKind: FormatKind;
  height?: number;
}) {
  const formatter = (v: number) => formatValue(formatKind, v);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--rule)" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: "var(--ink-muted)", fontSize: 12 }}
          axisLine={{ stroke: "var(--rule)" }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: "var(--ink-muted)", fontSize: 12 }}
          tickFormatter={formatter}
          axisLine={false}
          tickLine={false}
          width={64}
        />
        <Tooltip
          formatter={(value: number) => formatter(value)}
          contentStyle={{
            background: "var(--surface)",
            border: "1px solid var(--rule)",
            borderRadius: 8,
            fontSize: 12,
            color: "var(--ink)",
          }}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.label} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
