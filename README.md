# GenAI ReleaseGate

Statistical release gating for production LLM agents — decide whether a new prompt is
safe and genuinely better before it ships, with evidence instead of a single benchmark
score.

**Live Demo:** https://genai-releasegate.vercel.app

---

## The Problem

Most LLM prompt changes are judged by eyeballing a handful of outputs. That misses
exactly the failures that matter most: a change that improves one metric can silently
regress another; a prompt that looks like a clear win on the data you iterated against
can fail to generalize; a safety guardrail tuned to be maximally strict can quietly
block a large fraction of legitimate traffic. None of that shows up in a spot check.

## Business Use Case

An enterprise customer-support AI team needs to answer one question before every
release: **"Can we safely ship this LLM change?"** GenAI ReleaseGate answers it with a
repeatable pipeline — generate, score, statistically compare, and decide — rather than
a person's judgment call on a handful of examples.

## What GenAI ReleaseGate Does

Generates responses under a production prompt and a candidate, scores them for quality
(correctness, faithfulness, hallucination), safety (a real AWS Bedrock Guardrail), cost,
and latency, runs a paired statistical comparison, and evaluates a declarative policy
that answers exactly one of four ways:

| State | Meaning |
|---|---|
| **GO** | The candidate satisfies every release requirement — passed the release gate. |
| **REVIEW** | Evidence is inconclusive, or a trade-off requires a human decision. |
| **HOLD** | A hard release criterion was violated — automatic release block. |
| **INVALID** | The evaluation itself isn't trustworthy enough to decide. |

## Real Prompt Release Experiment

Six real candidates, evaluated against production baseline V1 by the same automated
pipeline. The gate rejected or reviewed five of them before one passed:

```
V1 (Production baseline)
  │
  ▼
V2  ──▶  HOLD      faithfulness regression; benign false-positive issue under the
                    guardrail configuration active for that run
  │
  ▼
V3  ──▶  REVIEW    cost +25.79%
  │
  ▼
V3.1 ──▶ REVIEW    cost +17.02% — AND a reintroduced faithfulness regression
  │
  ▼
V3.2 ──▶ REVIEW    cost +22.54% — faithfulness still flagged
  │
  ▼
V3.3 ──▶ REVIEW    cost +16.30% — missed the 15% threshold by 1.3pp
  │
  ▼
V3.4 ──▶ GO        all 8 gates passed — dev cost +12.17%, holdout cost +10.81%
```

V3.1 and V3.2 tried to cut cost with a rigid 2- and 3-sentence output cap — neither
closed the gap, and both reintroduced the faithfulness regression V3 had fixed (a hard
length limit truncates the specific-restatement content the judge rewards). V3.3
replaced the cap with a qualitative anti-preamble instruction and got much closer
without the faithfulness cost, but still missed the threshold. V3.4's fix came from a
different, data-derived insight: measuring actual response tokens showed the cost
overrun was dominated by *input* tokens (the instruction text itself, sent on every
call) — not output length. Trimming that got it under the line. Full prompt text and
reasoning for every version: [`/prompts`](prompts/), [Failure Analysis](#failure-analysis).

## V3.4 Results

**GO on both the 120-case development set and the 40-case sealed holdout — no manual
override, no threshold changed to produce this result.**

| | Dev (n=120) | Holdout (n=40) |
|---|---|---|
| Decision | **GO** | **GO** |
| Answer correctness | 74.39% → 82.37% (+7.98pp) | 59.47% → 69.58% (+10.11pp) |
| Faithfulness | 84.40% → 88.93% (+4.53pp) | 83.60% → 86.40% (+2.81pp) |
| Hallucination | 5.08% → 1.69% | 5.26% → 0.00% |
| Instruction following | 60.00% → 66.67% | 60.00% → 60.00% |
| Abstention accuracy | 6.67% → 93.33% | 0.00% → 100.00% |
| Prompt injection block rate | 100% → 100% | 100% → 100% |
| Sensitive-info block rate | 100% → 100% | 100% → 100% |
| Cost per query (generator inference only) | +12.17% | +10.81% |
| p95 latency | −30.6% | −16.0% |

All 8 real, persisted comparisons (V2 through V3.4, dev and holdout) are in
[`results/reports/`](results/reports/) as structured JSON — nothing summarized here is
hand-typed independently of that data.

**What "cost per query" measures:** `cost_per_query` — the metric the release gate
reads — is the generator's (model inference) cost only. Judge and guardrail calls are
real, separately billed services with their own cost, tracked independently in the
database and printed separately by the evaluation scripts (`generator=... guardrail=...
judge=...`); they are not included in the `cost_per_query` gate. Nothing on this site
calls it "total system cost."

## Dev vs Sealed Holdout

```
Development evaluation (120 cases)
        │
        ▼
   V3.4 = GO
        │
        ▼
Sealed holdout (40 cases, never touched before this run)
        │
        ▼
   V3.4 = GO
        │
        ▼
   Promotion evidence
```

The candidate was not treated as ready just because it passed development evaluation.
The identical prompt was evaluated again on a sealed holdout split, using the same
unmodified release policy — this is the exact discipline that caught V3's real
regression earlier in this project's history (correctness delta −7.53pp, CI
[−17.7, −1.2] — see [Failure Analysis](#failure-analysis)) and is what makes V3.4's GO
a stronger claim than a single dev-set win. Precisely: **V3.4 passed the same release
policy on unseen holdout data** — not a claim that this "proves" the improvement will
generalize to production traffic in general.

**Statistical honesty:** V3.4's quality improvements were directionally consistent
across Dev and Holdout. The release decision itself replicated: GO on both datasets.
Answer correctness excludes zero on dev (CI [+0.8, +15.8]pp) but straddles zero on the
smaller 40-case holdout (CI [−3.3, +26.7]pp) — directionally consistent, not
independently statistically confirmed there. We do not claim every quality metric was
statistically significant on both splits — only abstention accuracy and (on dev) cost
per query survive Holm-Bonferroni correction across the tested metrics.

## Release Decision Engine

One declarative policy (`config/policy.yaml`), worst-gate-wins: every configured gate
evaluates independently, and the overall decision is the single worst outcome among
them — any HOLD → HOLD, else any REVIEW → REVIEW, else GO. Quality and safety metrics
are hard gates that can HOLD; cost and latency are soft gates that can only ever
REVIEW, structurally impossible to HOLD by construction (no hold threshold or absolute
bound is configured for them). A comparison is INVALID, independent of any gate, if too
many paired cases failed on either side to trust it.

The decision engine evaluates every gate and applies the configured policy — no manual
override was used to produce any result in this project, including V3.4's GO. A
separate, clearly-labeled synthetic **validation fixture** exists in the test suite to
prove the engine can reach GO in principle; it predates V3.4's real result and played no
part in it (`tests/unit/test_policy_engine.py`).

## Statistical Methodology

Every comparison reports a paired mean delta (joined by `case_id`), a paired bootstrap
95% confidence interval (2,000 resamples, fixed seed — identical data always produces
an identical interval), a median paired difference, a paired effect size (Cohen's d),
and a Holm-Bonferroni correction across the metrics tested in that comparison. None of
this feeds back into the release decision — the gates read only the raw delta and CI,
exactly as before these additions existed.

## AWS Bedrock Guardrails

A real Amazon Bedrock Guardrail (native `ApplyGuardrail`, prompt-injection + PII
policies) checks every input and output. Guardrail v1 (`PROMPT_ATTACK=HIGH`) achieved a
100% block rate on every real attack case in the evaluated suite but over-blocked
benign traffic badly — 55.56% of its own test suite. Guardrail v2 (`PROMPT_ATTACK=MEDIUM`,
PII policy untouched) preserved that same 100% block rate on the evaluated
prompt-injection and PII-extraction cases while cutting benign false positives to
11.11% — a 5x reduction at zero cost to the measured block rate. Adopted, and used for
every comparison from V3 onward. (V2's HOLD includes a 55.56% benign-FP reading from
Guardrail v1 — V2 was never re-tested under v2.)

These are block rates measured on this project's own evaluation suite (30 guardrail
cases on dev, 10 on holdout — prompt-injection, PII-extraction, and benign near-miss
cases combined), not a guarantee about production traffic in general — see
[Limitations](#limitations).

## Architecture

```
Prompt (V1 / candidate)
   │
   ▼
Stage A: Generate (real Bedrock Mantle call) + Guardrail check (real ApplyGuardrail)
   │
   ▼
Stage B: Deterministic scoring + blind LLM judge (correctness, faithfulness, hallucination)
   │
   ▼
Stage C: Pair by case_id → statistics (mean, CI, median, effect size, Holm) → policy engine
   │
   ▼
GO / REVIEW / HOLD / INVALID  +  structured JSON report (results/reports/*.json)
   │
   ▼
Next.js dashboard (frontend/) — visualization only, reads committed JSON,
never calls Bedrock. Live at genai-releasegate.vercel.app.
```

## Failure Analysis

Two real root causes, found by reading actual failing responses:

- **Over-triggering refusal** (V2): a stronger insufficient-information rule caused the
  model to decline questions the context actually answered, especially phrased
  conversationally.
- **Judge sensitivity to bare refusals** (V2): even a correct refusal scored 0.0
  faithfulness under the judge's claim-extraction rubric if generic; restating
  specifically what the context does/doesn't cover scored as a supported claim.

V3 fixed both with two targeted rule changes, but a residual over-refusal case remained
— and that residual is what the sealed holdout caught (correctness delta −7.53pp, CI
[−17.7, −1.2], decision REVIEW; a human reviewer declined to promote V3, so V1 stayed
production).

Getting V3 under the cost gate took three more iterations: V3.1/V3.2's sentence caps
cut cost but broke faithfulness again by truncating the same specificity that fixed
root cause #2. V3.3's qualitative anti-preamble instruction avoided that failure mode
and got close (16.30%) without the regression. V3.4 closed the rest of the gap by
measuring where the cost actually came from — input tokens (the instruction template
itself, fixed overhead on every call: exactly 111 extra tokens per case vs V1, zero
variance) dominated over output length — and trimmed the instruction wording itself,
not the response.

## Reproducibility

- Every prompt is identified by a SHA-256 content hash — an edited file without a
  version bump is refused, not silently accepted.
- A run's identity is a content hash over its prompt, case set, model params, and code
  — resumable by construction.
- Bootstrap CI, median, effect size, and Holm correction all use a fixed seed —
  identical DB state always reproduces identical statistics.
- `data/holdout/` is gitignored; only its count, seed, and per-file SHA-256 hashes are
  committed (`data/manifest.json`) — enough to verify integrity without shipping
  sealed content.
- Charts, comparisons, and JSON reports are all regenerable from `artifacts/*.db` with
  no new AWS calls (`scripts/generate_result_charts.py`, `scripts/generate_reports.py`,
  `scripts/recompute_comparisons_with_phase2_stats.py`).

## Limitations

- Holdout is 40 cases — enough for V3's original correctness regression to exclude
  zero, but not enough to independently confirm V3.4's correctness improvement (its CI
  straddles zero there, though directionally consistent with dev).
- Single seed (42) for the dev/holdout split — not re-validated across multiple splits.
- The judge is a single model at temperature 0 — not cross-validated against a second
  judge.
- Guardrail calibration only explored `PROMPT_ATTACK` strength; PII policy was never
  varied.
- A metric's reported baseline can differ slightly across comparisons it appears in,
  since pairing only includes a case when both sides of *that specific* comparison have
  a valid score — never averaged into one synthetic number.
- "GO" here means the candidate passed every configured release gate — a promotion
  candidate backed by dev + holdout evidence, not a claim that it has been deployed to
  production.

## Tech Stack

**Backend:** Python 3.12, `uv`, `pytest`, `ruff`, `boto3` (AWS Bedrock Mantle + Bedrock
Guardrail), SQLite, Pydantic.

**Frontend:** Next.js 16 (App Router), TypeScript, Tailwind CSS v4, hand-authored
shadcn/ui-style components, Recharts, `next-themes` — reads `results/reports/*.json` at
build/render time, no live model calls.

**CI:** GitHub Actions (`.github/workflows/ci.yml`) — lint, unit, statistical, and
reporting tests on every push/PR, entirely offline.
