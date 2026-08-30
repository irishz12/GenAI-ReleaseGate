"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface DeltaCIDatum {
  label: string;
  /** delta in percentage points */
  delta: number;
  /** [low, high] in percentage points, or null when no CI was computed */
  ciLow: number | null;
  ciHigh: number | null;
}

function TooltipContent({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: DeltaCIDatum }[];
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-rule bg-surface px-3 py-2 text-xs shadow-md">
      <p className="font-semibold text-ink">{d.label}</p>
      <p className="text-ink-muted">delta: {d.delta >= 0 ? "+" : ""}{d.delta.toFixed(2)} pp</p>
      {d.ciLow !== null && d.ciHigh !== null && (
        <p className="text-ink-muted">
          95% CI: [{d.ciLow.toFixed(1)}, {d.ciHigh.toFixed(1)}] pp
        </p>
      )}
    </div>
  );
}

/** Paired-delta chart with bootstrap 95% CI error bars. A bar whose CI
 * crosses zero visually straddles the reference line — that's the point:
 * it makes "statistically indistinguishable from no change" visible, not
 * just a number in a table. */
export function DeltaCIChart({ data, height = 280 }: { data: DeltaCIDatum[]; height?: number }) {
  const chartData = data.map((d) => ({
    ...d,
    errorBar:
      d.ciLow !== null && d.ciHigh !== null
        ? ([d.delta - d.ciLow, d.ciHigh - d.delta] as [number, number])
        : undefined,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} margin={{ top: 16, right: 16, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--rule)" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: "var(--ink-muted)", fontSize: 12 }}
          axisLine={{ stroke: "var(--rule)" }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: "var(--ink-muted)", fontSize: 12 }}
          tickFormatter={(v) => `${v}pp`}
          axisLine={false}
          tickLine={false}
          width={48}
        />
        <ReferenceLine y={0} stroke="var(--ink-faint)" />
        <Tooltip content={<TooltipContent />} />
        <Bar dataKey="delta" radius={[4, 4, 4, 4]} barSize={40}>
          {chartData.map((d) => (
            <Cell key={d.label} fill={d.delta >= 0 ? "var(--go)" : "var(--hold)"} />
          ))}
          <ErrorBar dataKey="errorBar" stroke="var(--ink-faint)" width={6} strokeWidth={1.5} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
