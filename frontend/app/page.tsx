import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { DecisionBadge } from "@/components/decision-badge";
import { PROMPT_STORY, getHoldoutHeadline } from "@/data/experiments";
import { DECISION_META } from "@/data/decisions";
import { formatCI } from "@/data/metrics";
import type { Decision } from "@/data/types";

const STATES: Decision[] = ["GO", "REVIEW", "HOLD", "INVALID"];

export default function HomePage() {
  const headline = getHoldoutHeadline();

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
          <strong className="text-ink">The business problem:</strong> an enterprise
          customer-support AI team needs to know whether a new prompt version is safe and
          genuinely better before releasing it to customers — not by eyeballing a handful of
          outputs, but with paired statistical evidence and an auditable decision.
        </p>

        {/* The four supported release states — always visible, never buried */}
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
              See the release decisions <ArrowRight />
            </Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/methodology">Read the methodology</Link>
          </Button>
        </div>
      </section>

      <Separator />

      {/* Three prompt versions */}
      <section className="flex flex-col gap-5">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">The three-prompt story</h2>
          <p className="mt-1 text-ink-muted">
            V1 is the production baseline. V2 and V3 are candidates, each evaluated against V1 by
            the same automated pipeline.
          </p>
        </div>
        <div className="grid gap-4 sm:grid-cols-3">
          {PROMPT_STORY.map((p) => (
            <Card key={p.version}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>{p.title}</CardTitle>
                  {p.decision && <DecisionBadge decision={p.decision} />}
                </div>
                <CardDescription>{p.role}</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-ink-muted">{p.summary}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <Separator />

      {/* Headline holdout finding */}
      <section className="flex flex-col gap-5">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">
            The headline finding: a caught regression
          </h2>
          <p className="mt-1 text-ink-muted">
            V3 looked like an improvement in development. The sealed 40-case holdout split told a
            different story.
          </p>
        </div>
        <Card className="border-hold-rule bg-hold-bg/40">
          <CardContent className="grid gap-6 pt-6 sm:grid-cols-3">
            <div>
              <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
                Correctness delta (V3 vs V1, holdout)
              </p>
              <p className="mt-1 font-mono text-3xl font-semibold text-hold">
                {headline.correctnessDeltaPp.toFixed(2)} pp
              </p>
              <p className="mt-1 text-sm text-ink-muted">
                95% CI: {formatCI(headline.ci)} — excludes zero
              </p>
            </div>
            <div>
              <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
                Automated policy decision
              </p>
              <div className="mt-2">
                <DecisionBadge decision={headline.automatedDecision} />
              </div>
              <p className="mt-2 text-sm text-ink-muted">
                A confirmed regression that crosses the review threshold, not the harsher hold
                threshold — REVIEW is the evidence-matched decision here.
              </p>
            </div>
            <div>
              <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
                Human decision
              </p>
              <p className="mt-2 font-mono text-lg font-semibold text-ink">
                {headline.humanDecision}
              </p>
              <p className="mt-2 text-sm text-ink-muted">{headline.humanDecisionNote}</p>
            </div>
          </CardContent>
        </Card>
        <p className="text-sm text-ink-faint">
          The automated decision and the human decision are shown separately on purpose — the
          policy engine surfaces evidence and a REVIEW state, it does not itself decide to keep a
          candidate out of production.
        </p>
      </section>

      <Separator />

      {/* Safety proof point — a third, independent axis (not quality) this
          system decides on */}
      <section className="flex flex-col gap-4">
        <h2 className="text-2xl font-semibold tracking-tight">Safety is evaluated too</h2>
        <div className="grid gap-4 sm:grid-cols-3">
          <Card>
            <CardContent className="pt-6">
              <p className="font-mono text-2xl font-semibold text-go">100%</p>
              <p className="mt-1 text-sm text-ink-muted">Prompt injection blocked (tested)</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <p className="font-mono text-2xl font-semibold text-go">100%</p>
              <p className="mt-1 text-sm text-ink-muted">Sensitive information protected (tested)</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <p className="font-mono text-2xl font-semibold text-ink">
                55.56% <span className="text-ink-faint">→</span> 11.11%
              </p>
              <p className="mt-1 text-sm text-ink-muted">
                Benign false positives, guardrail v1 → v2
              </p>
            </CardContent>
          </Card>
        </div>
        <Link
          href="/safety"
          className="text-sm font-medium text-accent underline underline-offset-2"
        >
          See the full guardrail calibration →
        </Link>
      </section>
    </div>
  );
}
