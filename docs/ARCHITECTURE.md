# GenAI Prompt Evaluation & Guardrails Platform — Architecture (V1)

**Status of everything below §0:** this is the **original Phase 0 design document**,
written 2026-08-29 before implementation began. It is kept, unedited apart from a few
explicit corrections marked inline, as a historical record of the original design
reasoning — not a description of the current system. **For what's actually built and
running today, see §0 immediately below.**

> This revision trims the original design down to what a first working version actually needs.
> Statistical and calibration machinery that doesn't pay for itself at this dataset size has been
> moved to **§14 Future Extensions** rather than built now.

---

## 0. Current Implemented Architecture

This section describes the system as it actually exists today — not the plan in §1
onward. Every stage below is real, built, and exercised by the real V1→V3.4 experiment
history in this repository.

```
Prompt / Dataset   (prompts/*.md, data/dev+holdout/*.jsonl, content-hashed)
        │
        ▼
Generation          (real Bedrock Mantle call — src/evalguard/providers/bedrock_mantle.py)
        │
        ▼
Guardrails           (real Bedrock ApplyGuardrail, input + output — providers/bedrock_guardrail.py)
        │
        ▼
Quality/Safety Evaluation   (deterministic checks + blind LLM judge — quality/, guardrails/)
        │
        ▼
Statistical Comparison   (paired by case_id, mean/median delta, bootstrap CI, effect
                          size, Holm correction — regression/compare.py, regression/statistics.py)
        │
        ▼
Release Policy       (declarative gates, worst-gate-wins — policy/engine.py + config/policy.yaml)
        │
        ▼
Structured Reports    (Pydantic → JSON — reporting/ → results/reports/*.json)
        │
        ▼
Next.js Dashboard     (frontend/ — reads the JSON above at build time)
```

**Explicitly:**

- The dashboard is a **presentation layer**. It reads already-persisted, structured
  results (`results/reports/*.json`) via `frontend/data/reports.ts` at build/render
  time.
- It does **not** make live model, judge, or guardrail calls of any kind, and does not
  invoke the evaluation pipeline.
- It does **not** make release decisions independently — it only renders the
  `decision` and `gate_results` fields that are already present in the JSON it reads.
- **`src/evalguard/policy/engine.py`, evaluated against `config/policy.yaml`, is the
  sole source of every GO/REVIEW/HOLD/INVALID decision** in this project. Nothing in
  the frontend, and nothing in this document, overrides or recomputes that.
- This describes the **evaluation pipeline and its presentation layer** — it is not a
  claim that the candidate LLM prompt itself has been deployed to production. "GO"
  means a candidate passed the release gate on real dev + holdout evidence; see
  README.md's Limitations section for the exact distinction.

The read-only FastAPI layer sketched in the original design (§14/§15 below) was never
built — once the dashboard read JSON directly, it became unnecessary, not merely
deferred.

---

## 1. Recommended Architecture

This is a **batch evaluation pipeline**, not a service. A CLI process runs an evaluation, writes
rows to SQLite, and exits. Everything else reads those rows. No queue, no broker, no orchestration
engine — the workload is sequential batch generation followed by pure-function scoring.

### Three stages, connected only by the database

```
Stage A: GENERATE   (network-bound, non-deterministic)
Stage B: EVALUATE   (deterministic checks + one judge call)
Stage C: DECIDE      (pure functions, no network)
```

Stage A produces **immutable** `responses`. Stage B is a function over stored responses. Stage C
is a function over stored scores. This buys reproducibility (re-score without re-generating),
testability (B and C run on fixture rows, no network), and judge integrity (the judge reads a
response with the variant label stripped, so it can't grade by knowing which prompt it's looking
at).

### Paired comparison

V1 and V2 run separately over the **same frozen dataset** and the **same pinned model
config**, then join per `case_id`. Pairing means the comparison is driven by prompt behavior, not
by which run happened to get easier cases — important at a dataset size of ~160, where unpaired
noise would swamp a real signal.

### Deterministic-first

Whatever can be checked with code is checked with code: abstention detection, instruction-format
compliance, PII/canary matching. The judge is used only where it's actually needed — answer
correctness against a reference, and faithfulness/hallucination against the source context.

---

## 2. Architecture Diagram

```
   prompts/ · data/ · config/          (files, hashed on load)
        │
        ▼
   PROMPT + CASE REGISTRY  (name, version, hash, status/split)
        │
        ▼
   RUN MANIFEST (frozen): prompt_hash · dataset_hash · model+params ·
                          judge_model · guardrail_config_hash · seed
        │
 ══════════ STAGE A: GENERATE (async, bounded concurrency) ══════════
        │
   ┌────┴─────┐                  ┌──────────┐
   │ RUN: V1  │                  │ RUN: V2  │   same cases, same model
   └────┬─────┘                  └────┬─────┘   same guardrail config
        │                             │
        └──────────────┬──────────────┘
                        ▼
              GUARDRAIL — INPUT CHECK
              (block → short-circuit, record)
                        ▼
              LLM PROVIDER ADAPTER
              ┌──────────────────────┐
              │ BedrockMantleClient  │   retry + backoff (transient only)
              │ FakeClient (tests)   │   tokens, latency, cost captured
              └──────────────────────┘
                        ▼
              GUARDRAIL — OUTPUT CHECK
              (PII / canary leakage)
                        ▼
             ╔══════ responses (IMMUTABLE) ══════╗
                        │
 ══════════ STAGE B: EVALUATE (pure fn over stored responses) ══════════
        ┌───────────────┼────────────────┐
        ▼                                ▼
 DETERMINISTIC CHECKS            LLM JUDGE (different model family,
 abstention · format ·           temp=0, blind to variant, structured
 required fields · PII           JSON: correctness, faithfulness)
        └───────────────┬────────────────┘
                        ▼
             ╔══════ scores · guardrail_results ══════╗
                        │
 ══════════ STAGE C: DECIDE (pure, deterministic, no network) ══════════
                        ▼
              REGRESSION: V1 ⋈ V2 on case_id
              per-metric delta + simple paired 95% CI
                        ▼
              POLICY ENGINE (config/policy.yaml)
              GO / REVIEW / HOLD / INVALID + failing-gate reasons
                        ▼
             ╔══════════ comparisons ══════════╗
                        │
              Markdown/JSON report
              (FastAPI + Next.js read this later — Phase 8/9)

        ALL STAGES WRITE ONE SQLITE FILE: artifacts/eval.db
```

---

## 3. End-to-End Data Flow

`evalguard compare --baseline v1 --candidate v2 --split dev`:

1. **Resolve & freeze.** Hash both prompts and the dataset split. Resolve the pinned generator
   model and the pinned judge model from `config/models.yaml`. Record the git SHA. Refuse to run
   on a dataset/prompt whose file content doesn't match its recorded hash.

2. **Generate (Stage A).** For each run, walk the cases with bounded `asyncio` concurrency
   (default 8). Per case: input guardrail check → (if not blocked) generator call with retry on
   transient errors → output guardrail check → one `responses` row with output text, tokens,
   latency, cost, guardrail flags, and success/failure status. A single case failure never aborts
   the run — it's recorded and counted toward `failure_rate`.

3. **Evaluate (Stage B).** Deterministic checkers run over every response first (cheap, no
   network). The judge is called only for `answer_correctness` and `faithfulness`
   (`hallucination_rate` is derived from the same faithfulness call — see §5). Judge input strips
   the run/variant identity and shuffles case order before scoring.

4. **Decide (Stage C).** Inner-join V1 and V2 responses on `case_id` (only cases both variants
   answered are compared). Compute per-metric deltas, a simple paired confidence interval, cost
   change, and latency change. Evaluate the declarative gates in `config/policy.yaml`, produce
   GO / REVIEW / HOLD / INVALID with the specific reason for each failing gate.

5. **Emit.** Write one `comparisons` row (deltas + gate results as a JSON payload, per §9). Render
   a Markdown report. Exit code 0 for GO/REVIEW, 1 for HOLD, 2 for INVALID.

---

## 4. Component Responsibilities

**Prompt Registry** — loads prompt files, computes a content hash, tracks `status`
(`production | candidate | archived`). Promotion from candidate to production requires a GO
comparison against the **holdout** split.

**Evaluation Dataset** — JSONL cases validated against a per-category schema, tagged with a
`split` (`dev` or `holdout`). No separate dataset-version table in V1 — case rows carry the
dataset hash directly (see §9).

**Evaluation Runner** — renders the prompt, calls the generator through the provider adapter,
applies guardrail checks, persists responses. Knows nothing about what "good" means — that's
Stage B's job. Tracks tokens, latency (generator time, separate from guardrail overhead), cost,
and per-case failures.

**Quality Evaluation Engine** — deterministic checkers for abstention accuracy and instruction
following; one judge call per response for correctness + faithfulness/hallucination. Judge prompt
is versioned like any other prompt (hashed, stored in `prompts`).

**Guardrail Layer** — input check (blocks before generation) and output check (PII/canary
detection) against the completion. Every guardrail case carries an `expected_block` label so a
result resolves to TP/FP/TN/FN, rolled up into `prompt_injection_block_rate`,
`sensitive_information_protection`, and `benign_false_positive_rate`.

**Regression Engine** — inner-joins V1/V2 on `case_id`, computes per-metric deltas and a simple
paired bootstrap 95% CI (a single ~20-line function — see §10). No McNemar test, no minimum-
detectable-effect reporting in V1 (§14).

**Release Policy Engine** — declarative YAML gates evaluated against the delta table. Produces
GO/REVIEW/HOLD/INVALID plus a human-readable reason per failing gate. Safety gates (guardrail
metrics) are absolute; quality/cost gates are relative to baseline.

**Results Store** — one SQLite file, 7 tables (§9).

**FastAPI layer** (Phase 8, read-only) — exposes runs/comparisons once the pipeline is stable.

**Next.js dashboard** (Phase 9) — out of scope for V1 implementation.

---

## 5. Core Metrics (V1)

**Quality**
| Metric | Method |
|---|---|
| `answer_correctness` | LLM judge vs. reference answer |
| `faithfulness` | LLM judge, claim-level support against context |
| `hallucination_rate` | derived from the same faithfulness call (fraction of responses with ≥1 unsupported/contradicted claim) |
| `instruction_following` | deterministic (format/length/required-field checks) |
| `abstention_accuracy` | deterministic (expected-abstention label vs. detected refusal) |

**Guardrails**
| Metric | Method |
|---|---|
| `prompt_injection_block_rate` | deterministic, TP rate on injection cases |
| `sensitive_information_protection` | deterministic, canary/PII leak detection |
| `benign_false_positive_rate` | deterministic, FP rate on benign near-miss cases |

**Operations**
| Metric | Method |
|---|---|
| `p50_latency`, `p95_latency` | measured generator latency (ms) |
| `cost_per_query` | tokens × pricing table |
| `failure_rate` | unhandled errors / total cases |

**Derived (Stage C)**
- V1 vs. V2 delta per metric above, with a simple paired 95% CI where applicable
- cost change, latency change
- GO / REVIEW / HOLD / INVALID

Faithfulness and hallucination sharing one judge call is the only "clever" thing left in the
metric design — it halves judge cost and keeps the two numbers mutually consistent by
construction.

---

## 6. Dataset Plan

| Category | Cases | Tests |
|---|---|---|
| `grounded_qa` | 80 | correctness + faithfulness given context |
| `abstention` | 20 | refuses when context is insufficient |
| `instruction_following` | 20 | format/constraint compliance |
| `guardrail` | 40 | injection + PII, **including benign near-misses** |
| **Total** | **~160** | |

Split: **~120 development** (iterate freely) / **~40 final holdout** (touched only for the
promotion decision). The guardrail category must include benign-looking cases that are *not*
attacks — without them `prompt_injection_block_rate` is gameable by blocking everything, and
`benign_false_positive_rate` has nothing to measure.

This is a small corpus by design. It's sized to make Phase 0–7 buildable and demonstrable
directly; growing it is a dataset-authoring task, not an architecture change.

---

## 7. Technology Choices

| Choice | Why |
|---|---|
| Python 3.12 + uv | reproducible lockfile, ecosystem fit |
| SQLite (stdlib `sqlite3`) | single-writer batch workload; zero ops; portable artifact |
| Raw SQL, no ORM | 7 tables, mostly analytical reads — an ORM adds indirection for no benefit here |
| Pydantic v2 | validates dataset rows, config, and judge JSON output |
| asyncio + Semaphore | the workload is pure network I/O; bounded concurrency in ~20 lines |
| Typer CLI | the pipeline *is* a CLI |
| YAML policy config | thresholds need to be editable and diffable without touching code |
| Bedrock Mantle behind an `LLMClient` Protocol | one interface, two implementations (Mantle, Fake); generator and judge are separate client instances pointed at **different model families** |
| pytest + `FakeClient` | whole pipeline runs offline, no credentials, no cost |

---

## 8. Folder Structure

```
genai-eval-guardrails/
├── pyproject.toml
├── config/
│   ├── models.yaml          # generator + judge model ids, params, pricing
│   ├── policy.yaml          # release gates
│   └── guardrails.yaml      # injection patterns, PII rules
├── prompts/
│   ├── production/support_agent.v1.md
│   ├── candidate/support_agent.v2.md
│   └── judge/{correctness,faithfulness}.v1.md
├── data/
│   ├── dev/{grounded_qa,abstention,instruction_following,guardrail}.jsonl
│   └── holdout/{...same four files...}
├── src/evalguard/
│   ├── cli.py
│   ├── config.py
│   ├── hashing.py
│   ├── manifest.py
│   ├── db/
│   │   ├── schema.sql
│   │   └── store.py
│   ├── registry/
│   │   ├── prompts.py
│   │   └── cases.py
│   ├── providers/
│   │   ├── base.py           # LLMClient Protocol
│   │   ├── bedrock_mantle.py
│   │   ├── fake.py
│   │   └── pricing.py
│   ├── runner/
│   │   └── runner.py
│   ├── quality/
│   │   ├── deterministic.py
│   │   └── judge.py
│   ├── guardrails/
│   │   ├── checks.py
│   │   └── scoring.py
│   ├── regression/
│   │   └── compare.py        # join + deltas + simple CI, one file
│   └── policy/
│       └── engine.py
├── tests/
│   └── unit/
└── docs/
    └── ARCHITECTURE.md

# Note: this tree is the original Phase 0 design sketch — some file names below
# evolved during implementation (e.g. registry/cases.py -> registry/datasets.py +
# case_builder.py + doc2dial_source.py; guardrails/scoring.py -> guardrails/gate.py).
# The originally-sketched report/markdown.py was never built as such, but a
# structured reporting package WAS built later (src/evalguard/reporting/,
# Pydantic models -> results/reports/*.json) once a stable, machine-readable
# artifact was needed for the frontend to consume — see README.md's
# "Statistical Methodology" section and results/reports/ itself.
```

Removed relative to the original draft: `datasets.py` (folded into `cases.py`), `rubrics.py` +
calibration files, `regression/stats.py` as a separate module, `api/` (still Phase 8, not
scaffolded yet).

---

## 9. Database / Schema (7 tables)

SQLite, WAL mode, `PRAGMA foreign_keys=ON`.

```sql
CREATE TABLE prompts (
  id            INTEGER PRIMARY KEY,
  name          TEXT NOT NULL,             -- 'support_agent', 'judge_correctness', ...
  version       TEXT NOT NULL,
  role          TEXT NOT NULL CHECK (role IN ('production','candidate','archived','judge')),
  template      TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  UNIQUE (name, version)
);

CREATE TABLE evaluation_cases (
  id                  INTEGER PRIMARY KEY,
  external_id         TEXT NOT NULL UNIQUE,
  category            TEXT NOT NULL CHECK (category IN
                        ('grounded_qa','abstention','instruction_following','guardrail')),
  split               TEXT NOT NULL CHECK (split IN ('dev','holdout')),
  input               TEXT NOT NULL,
  context             TEXT,                 -- grounding passage, if any
  reference_answer    TEXT,
  constraints_json    TEXT,                 -- format/length rules for instruction_following
  expected_abstention INTEGER,              -- 0/1, nullable
  attack_type         TEXT,                 -- injection | pii_extraction | benign | NULL
  canary              TEXT,                 -- secret that must never appear in output
  expected_block      INTEGER,              -- 0/1, guardrail cases only
  dataset_hash        TEXT NOT NULL         -- hash of the file this case was loaded from
);

CREATE TABLE runs (
  id                    INTEGER PRIMARY KEY,
  prompt_id             INTEGER NOT NULL REFERENCES prompts(id),
  model_id              TEXT NOT NULL,
  model_params_json     TEXT NOT NULL,
  judge_model_id        TEXT NOT NULL,
  guardrail_config_hash TEXT NOT NULL,
  code_sha              TEXT NOT NULL,
  config_hash           TEXT NOT NULL,      -- hash of the full manifest
  status                TEXT NOT NULL CHECK (status IN
                          ('RUNNING','COMPLETED','DEGRADED','ABORTED')),
  case_count            INTEGER,
  failure_count         INTEGER NOT NULL DEFAULT 0,
  total_cost_usd        REAL,
  started_at            TEXT NOT NULL,
  finished_at           TEXT
);

CREATE TABLE responses (
  id                    INTEGER PRIMARY KEY,
  run_id                INTEGER NOT NULL REFERENCES runs(id),
  case_id               INTEGER NOT NULL REFERENCES evaluation_cases(id),
  output_text           TEXT,               -- NULL if blocked or errored
  finish_reason         TEXT,               -- stop | length | blocked_input | error
  input_tokens          INTEGER,
  output_tokens         INTEGER,
  cost_usd              REAL,
  latency_ms            INTEGER,            -- generator call only
  error_class           TEXT,               -- NULL on success
  created_at            TEXT NOT NULL,
  UNIQUE (run_id, case_id)
);

CREATE TABLE scores (
  id            INTEGER PRIMARY KEY,
  response_id   INTEGER NOT NULL REFERENCES responses(id),
  metric        TEXT NOT NULL,              -- answer_correctness | faithfulness | ...
  method        TEXT NOT NULL CHECK (method IN ('deterministic','judge')),
  value         REAL NOT NULL,              -- 0..1 or 0/1
  rationale     TEXT,                       -- judge explanation, if applicable
  UNIQUE (response_id, metric)
);

CREATE TABLE guardrail_results (
  id             INTEGER PRIMARY KEY,
  response_id    INTEGER NOT NULL REFERENCES responses(id),
  stage          TEXT NOT NULL CHECK (stage IN ('input','output')),
  triggered      INTEGER NOT NULL,          -- did a check fire
  expected_block INTEGER NOT NULL,          -- from evaluation_cases.expected_block
  outcome        TEXT NOT NULL CHECK (outcome IN ('TP','FP','TN','FN')),
  detail         TEXT                       -- which check, what matched
);

CREATE TABLE comparisons (
  id                INTEGER PRIMARY KEY,
  baseline_run_id   INTEGER NOT NULL REFERENCES runs(id),
  candidate_run_id  INTEGER NOT NULL REFERENCES runs(id),
  n_paired          INTEGER NOT NULL,
  decision          TEXT NOT NULL CHECK (decision IN ('GO','REVIEW','HOLD','INVALID')),
  deltas_json       TEXT NOT NULL,          -- per-metric {baseline, candidate, delta, ci_low, ci_high}
  gates_json        TEXT NOT NULL,          -- per-gate {status, threshold, observed, reason}
  created_at        TEXT NOT NULL,
  UNIQUE (baseline_run_id, candidate_run_id)
);
```

**Why deltas and gates are JSON, not two more tables.** At ~12 metrics and a handful of gates per
comparison, a `metric_deltas` + `gate_results` table pair is normalization with no query benefit
in V1 — nothing needs to filter across comparisons by individual metric yet. One `comparisons` row
per V1-vs-V2 run is easy to read, easy to render into a report, and easy to widen into real tables
later if the API layer ends up needing to query deltas directly (§14).

---

## 10. Synchronous vs Asynchronous

| Work | Mode | Why |
|---|---|---|
| Config load, hashing, CLI orchestration | Sync | trivial, sequential |
| Generator calls within a run | Async, semaphore-bounded (default 8) | pure network I/O |
| Judge calls | Async, same or lower semaphore | same reason |
| Guardrail checks | Sync, in-process | regex/string matching, microseconds |
| DB writes | Sync, single writer | SQLite has one writer; simplest correct option |
| Deterministic scoring, regression, policy | Sync | pure functions, CPU-trivial |
| V1 run vs. V2 run | Sequential | avoids one run's throttling contaminating the other's latency numbers |

---

## 11. Error-Handling Strategy

Two error classes, not three:

- **Transient** (throttling, 5xx, timeout) — retry with backoff, max 3 attempts.
- **Permanent** (validation, content filter, auth) — no retry, recorded as a failed response.

Every case runs in its own try/except — one bad case never aborts a run. A run's `failure_rate`
is `failure_count / case_count`. If `failure_rate > 2%` on either run, or the two runs don't cover
the same set of cases, the eventual decision is **INVALID**, not HOLD — the tool distinguishes
"I couldn't measure this" from "the candidate is worse," because collapsing those two into one
signal is how eval gates get ignored. Judge output that fails JSON parsing gets one repair retry,
then is recorded as a scoring failure and excluded from that metric's aggregate.

---

## 12. Reproducibility Strategy

Every run stores a `config_hash` over prompt hash, dataset hash, model + params, judge model,
guardrail config hash, and code SHA. Prompts and cases are content-hashed, not just version-
labeled — a file edited without a version bump fails the hash check and the run refuses to start.

`responses` is immutable. Re-scoring (fixing a checker, adjusting a judge prompt) reads stored
responses and produces a new `comparisons` row with zero new generator calls — this is the
reproducibility property that matters most day to day, and it costs nothing extra to keep.

`FakeClient` lets the entire pipeline run offline with scripted responses, so tests don't depend
on network access or API cost.

---

## 13. Evaluation Integrity Strategy

Kept simple, but not skipped:

1. **Blind judging.** The judge sees the case and the response, never the run/variant identity or
   the other variant's answer — enforced by the projection Stage B reads from, not by convention.
2. **Judge is a different model family from the generator, temperature 0.** Recorded in
   `runs.judge_model_id`; if it's ever the same family as the generator that's a visible fact in
   the run row, not a hidden assumption.
3. **Deterministic-first.** Every metric that doesn't need judgment doesn't get it — fewer judge
   calls means less variance and less surface for bias.
4. **Guardrail cases include benign near-misses**, so a guardrail that blocks everything shows up
   as a bad `benign_false_positive_rate`, not a perfect score.
5. **Holdout discipline.** Dev is for iteration; holdout is read only for the promotion decision.

Judge self-consistency sampling and formal human-calibration (Cohen's κ) are **not** part of V1 —
see §14. They're the right next investment once the pipeline is running end to end, not a
precondition for a first working version.

---

## 14. Future Extensions (explicitly deferred)

These were part of the original design and remain reasonable ideas — they're deferred because
they don't pay for themselves at V1's dataset size (~160 cases) and would add complexity before
there's a working baseline to justify it.

| Deferred | Why it's not in V1 |
|---|---|
| **3× judge self-consistency sampling** | Adds judge cost/latency for a variance estimate that matters more once the dataset is large enough to detect small effects |
| **Judge calibration set + Cohen's κ gate** | Requires a human-labelled gold set; worth building once the judge's rubrics have stabilized |
| **Statistical power / minimum-detectable-effect reporting, McNemar's test** | Real, but a bigger subsystem than a ~160-case V1 needs; the simple paired CI in §9/§10 covers "is this delta probably real" well enough to start |
| **Extensive golden-report regression testing for the evaluator itself** | Useful once report format has stabilized; premature now |
| **Automatic prompt optimization** | Actively in tension with using a human-authored candidate against an untouched holdout |
| **Multi-agent orchestration** | No task decomposition here that needs it |
| **MLflow / experiment tracking platform** | The `runs` + `comparisons` tables already cover this at V1 scale |
| **Kafka / Redis / worker pools** | No streaming, no multi-tenant concurrency need |
| **Separate `datasets` table, `metric_deltas`/`gate_results` tables, `guardrail_events` detail table** | Normalization with no current query benefit — revisit if the API layer needs to filter/query at that grain |
| **Vector DB / RAG** | Context is supplied per case; retrieval is deliberately held constant so prompt deltas stay attributable |
| **Kubernetes / microservices** | One process, minutes of runtime |

A Next.js dashboard (originally scoped as Phase 9) has since been built and deployed
(`frontend/`, live at the URL in README.md's Live Demo line) — it reads
`results/reports/*.json` directly at build time rather than through a live API, which
made the originally-planned read-only FastAPI layer (Phase 8) unnecessary: there is no
API server in this project, by design, not because it's still pending.

---

## 15. Build Phases

1. **Skeleton** — config, hashing, `schema.sql`, Typer stub.
2. **Registries** — prompt + case loading/validation, dev/holdout split.
3. **Runner against `FakeClient`** — full async runner, retries, persistence, no real cost.
4. **Bedrock Mantle adapter** — real generator + judge clients, pricing.
5. **Deterministic quality checks** — abstention, instruction following. Demonstrable on its own.
6. **Guardrail layer** — input/output checks, TP/FP/TN/FN scoring.
7. **LLM judge** — correctness + faithfulness/hallucination, blind projection.
8. **Regression + policy** — paired deltas, simple CI, GO/REVIEW/HOLD/INVALID, report. **This is
   where the project becomes a release gate** — everything before it is instrumentation.
9. ~~FastAPI read layer~~ — superseded (see §14): the dashboard reads JSON directly.
10. **Next.js dashboard** — built and deployed; see README.md's Live Demo line.

Critical path to a demo: **1 → 3 → 5 → 8** — a working V1-vs-V2 gate with deterministic metrics
and zero judge calls, before the judge or guardrails are even built.
