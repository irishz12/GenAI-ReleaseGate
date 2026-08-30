import Link from "next/link";
import { ArrowRight, ArrowDown } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { DecisionBadge } from "@/components/decision-badge";
import { PROMPT_STORY, getV34Headline, getFailedCandidates } from "@/data/experiments";
import { DECISION_META } from "@/data/decisions";
import type { Decision } from "@/data/types";

const STATES: Decision[] = ["GO", "REVIEW", "HOLD", "INVALID"];

export default function HomePage() {
  const headline = getV34Headline();
  const failed = getFailedCandidates();

  return (
    <div className="flex flex-col gap-10">
      {/* Hero */}
      <section className="flex flex-col gap-4">
        <p className="font-mono text-xs tracking-widest text-ink-faint uppercase">
          GenAI ReleaseGate
        </p>
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-balance sm:text-5xl">
          Can We Safely Ship a Better LLM?
        </h1>
        <p className="max-w-2xl text-ink-muted">
          <strong className="text-ink">The business problem:</strong> before deploying a new
          prompt, can we prove it&apos;s safe, useful, operationally acceptable, and worth
          shipping — with evidence, not a single benchmark score?
        </p>

        <div className="flex flex-wrap items-center gap-2 pt-1">
          {STATES.map((s) => (
            <div
              key={s}
              className="flex items-center gap-2 rounded-lg border border-rule bg-surface px-3 py-1.5"
            >
              <DecisionBadge decision={s} />
              <span className="text-xs text-ink-muted">{DECISION_META[s].explanation}</span>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap gap-3 pt-1">
          <Button asChild>
            <Link href="/release-gate">
              See the release gate <ArrowRight />
            </Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/methodology">Read the methodology</Link>
          </Button>
        </div>
      </section>

      <Separator />

      {/* The real release journey */}
      <section className="flex flex-col gap-5">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">The real release journey</h2>
          <p className="mt-1 text-ink-muted">
            Six real candidates, each decided by the same automated policy engine. The gate
            actually rejected or reviewed five of them before one passed.
          </p>
        </div>
        <div className="flex flex-col items-stretch gap-1 sm:flex-row sm:flex-wrap sm:items-center sm:gap-2">
          {PROMPT_STORY.map((p, i) => (
            <div key={p.version} className="flex items-center gap-2">
              <div className="flex min-w-[7rem] flex-col items-center gap-1.5 rounded-lg border border-rule bg-surface px-3 py-2 text-center">
                <span className="font-mono text-sm font-semibold">{p.title.split(" ")[0]}</span>
                {p.decision ? (
                  <DecisionBadge decision={p.decision} />
                ) : (
                  <span className="text-xs text-ink-faint">Production baseline</span>
                )}
              </div>
              {i < PROMPT_STORY.length - 1 && (
                <ArrowRight className="hidden size-4 shrink-0 text-ink-faint sm:block" />
              )}
              {i < PROMPT_STORY.length - 1 && (
                <ArrowDown className="size-4 shrink-0 text-ink-faint sm:hidden" />
              )}
            </div>
          ))}
        </div>
      </section>

      <Separator />

      {/* V3.4 headline card — the real GO */}
      <section className="flex flex-col gap-5">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-2xl font-semibold tracking-tight">
            V3.4 — First Candidate to Pass the Release Gate
          </h2>
          <Badge variant="real">REAL DATA</Badge>
        </div>
        <Card className="border-go-rule bg-go-bg/40">
          <CardContent className="grid gap-6 pt-6 sm:grid-cols-2">
            <div className="flex flex-col gap-1">
              <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
                Dev — 120 cases
              </p>
              <div className="flex items-center gap-2">
                <DecisionBadge decision={headline.dev.decision} />
                <span className="font-mono text-sm text-ink-muted">
                  cost {headline.devCostPct >= 0 ? "+" : ""}
                  {headline.devCostPct.toFixed(2)}%
                </span>
              </div>
            </div>
            <div className="flex flex-col gap-1">
              <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
                Sealed holdout — 40 cases
              </p>
              <div className="flex items-center gap-2">
                <DecisionBadge decision={headline.holdout.decision} />
                <span className="font-mono text-sm text-ink-muted">
                  cost {headline.holdoutCostPct >= 0 ? "+" : ""}
                  {headline.holdoutCostPct.toFixed(2)}%
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-4 sm:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Quality (dev)</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-1 text-sm text-ink-muted">
              <p>
                Correctness{" "}
                <span className="font-mono text-ink">
                  +{headline.devCorrectnessDeltaPp.toFixed(2)} pp
                </span>
              </p>
              <p>
                Faithfulness{" "}
                <span className="font-mono text-ink">
                  +{headline.devFaithfulnessDeltaPp.toFixed(2)} pp
                </span>
              </p>
              <p>
                Hallucination{" "}
                <span className="font-mono text-ink">
                  {headline.devHallucinationPct.toFixed(2)}%
                </span>
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Safety</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-1 text-sm text-ink-muted">
              <p>
                Injection blocking{" "}
                <span className="font-mono text-ink">
                  {headline.devPromptInjectionBlockRate.toFixed(0)}%
                </span>
              </p>
              <p>
                Sensitive-info protection{" "}
                <span className="font-mono text-ink">
                  {headline.devSensitiveInfoProtection.toFixed(0)}%
                </span>
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Performance</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-1 text-sm text-ink-muted">
              <p>
                p95 latency vs V1 (dev){" "}
                <span className="font-mono text-go">
                  {headline.devP95DeltaPct.toFixed(1)}%
                </span>
              </p>
            </CardContent>
          </Card>
        </div>
        <p className="text-sm text-ink-faint">
          &ldquo;GO&rdquo; here means <strong>passed the release gate — not the same as
          production deployment</strong>: a promotion candidate with statistically-grounded
          evidence behind it, not a claim that this has been deployed to production. &ldquo;Cost&rdquo;
          above is generator (model inference) cost only — judge and guardrail costs are real,
          separately tracked, and not part of this figure.
        </p>
      </section>

      <Separator />

      {/* Failed / reviewed candidates */}
      <section className="flex flex-col gap-5">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Candidates the gate rejected</h2>
          <p className="mt-1 text-ink-muted">
            V3.4 wasn&apos;t the first attempt — it&apos;s the fifth. The gate reviewed or held
            back every one of the others.
          </p>
        </div>
        <div className="flex flex-col gap-2">
          {failed.map((c) => (
            <div
              key={c.version}
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-rule bg-surface px-4 py-3"
            >
              <div className="flex items-center gap-3">
                <span className="w-14 font-mono text-sm font-semibold">{c.version}</span>
                <DecisionBadge decision={c.decision} />
              </div>
              <p className="text-sm text-ink-muted">
                {c.reasons.join("; ")}
                {c.costPct !== undefined && (
                  <span className="font-mono"> ({c.costPct >= 0 ? "+" : ""}{c.costPct.toFixed(2)}% cost)</span>
                )}
              </p>
            </div>
          ))}
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-go-rule bg-go-bg/40 px-4 py-3">
            <div className="flex items-center gap-3">
              <span className="w-14 font-mono text-sm font-semibold">V3.4</span>
              <DecisionBadge decision="GO" />
            </div>
            <p className="font-mono text-sm text-ink-muted">
              +{headline.devCostPct.toFixed(2)}% dev / +{headline.holdoutCostPct.toFixed(2)}%
              holdout
            </p>
          </div>
        </div>
      </section>

      <Separator />

      {/* Dev -> Holdout */}
      <section className="flex flex-col gap-4">
        <h2 className="text-2xl font-semibold tracking-tight">Dev → sealed holdout</h2>
        <div className="flex flex-col items-center gap-2 rounded-lg border border-rule bg-surface-muted p-6 text-center">
          <p className="font-medium">Development evaluation — 120 cases</p>
          <ArrowDown className="size-4 text-ink-faint" />
          <DecisionBadge decision="GO" />
          <ArrowDown className="size-4 text-ink-faint" />
          <p className="font-medium">Sealed holdout — 40 cases</p>
          <ArrowDown className="size-4 text-ink-faint" />
          <DecisionBadge decision="GO" />
          <ArrowDown className="size-4 text-ink-faint" />
          <p className="font-medium">Promotion evidence</p>
        </div>
        <p className="max-w-2xl text-sm text-ink-muted">
          The candidate was not promoted solely because it passed development evaluation. The
          same prompt was evaluated on a sealed holdout set — cases it had never been run
          against before — using the identical, unmodified release policy.
        </p>
      </section>
    </div>
  );
}
