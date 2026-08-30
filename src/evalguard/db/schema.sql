-- SQLite schema for the GenAI Prompt Evaluation & Guardrails Platform (V1).
-- 7 tables, per docs/ARCHITECTURE.md §9. Applied via db/store.py:init_db().
--
-- CHECK constraints mirror the enums in evalguard.enums — keep both in sync.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS prompts (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    version       TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('production', 'candidate', 'archived', 'judge')),
    template      TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS evaluation_cases (
    id                   INTEGER PRIMARY KEY,
    external_id          TEXT NOT NULL UNIQUE,
    category             TEXT NOT NULL CHECK (category IN
                           ('grounded_qa', 'abstention', 'instruction_following', 'guardrail')),
    split                TEXT NOT NULL CHECK (split IN ('dev', 'holdout')),
    input                TEXT NOT NULL,
    context              TEXT,
    reference_answer     TEXT,
    constraints_json     TEXT,
    expected_abstention  INTEGER CHECK (expected_abstention IN (0, 1)),
    attack_type          TEXT CHECK (attack_type IN ('injection', 'pii_extraction', 'benign')),
    canary               TEXT,
    expected_block       INTEGER CHECK (expected_block IN (0, 1)),
    dataset_hash         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                     INTEGER PRIMARY KEY,
    prompt_id              INTEGER NOT NULL REFERENCES prompts(id),
    model_id               TEXT NOT NULL,
    model_params_json      TEXT NOT NULL,
    judge_model_id         TEXT NOT NULL,
    guardrail_config_hash  TEXT NOT NULL,
    code_sha               TEXT NOT NULL,
    config_hash            TEXT NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN
                             ('RUNNING', 'COMPLETED', 'DEGRADED', 'ABORTED')),
    case_count             INTEGER,
    failure_count          INTEGER NOT NULL DEFAULT 0,
    total_cost_usd         REAL,
    started_at             TEXT NOT NULL,
    finished_at            TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    case_id        INTEGER NOT NULL REFERENCES evaluation_cases(id),
    output_text    TEXT,
    finish_reason  TEXT NOT NULL CHECK (finish_reason IN
                     ('stop', 'length', 'blocked_input', 'blocked_output', 'error')),
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    cost_usd       REAL,
    latency_ms     INTEGER,
    error_class    TEXT CHECK (error_class IN ('transient', 'permanent')),
    created_at     TEXT NOT NULL,
    UNIQUE (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS scores (
    id           INTEGER PRIMARY KEY,
    response_id  INTEGER NOT NULL REFERENCES responses(id),
    metric       TEXT NOT NULL CHECK (metric IN (
                   'answer_correctness', 'faithfulness', 'hallucination',
                   'instruction_following', 'abstention_accuracy',
                   'prompt_injection_block_rate', 'sensitive_information_protection',
                   'benign_false_positive_rate',
                   'p50_latency', 'p95_latency', 'cost_per_query', 'failure_rate'
                 )),
    method       TEXT NOT NULL CHECK (method IN ('deterministic', 'judge')),
    value        REAL NOT NULL,
    rationale    TEXT,
    judge_input_tokens   INTEGER,
    judge_output_tokens  INTEGER,
    judge_cost_usd       REAL,
    judge_latency_ms     INTEGER,
    UNIQUE (response_id, metric)
);

CREATE TABLE IF NOT EXISTS guardrail_results (
    id              INTEGER PRIMARY KEY,
    response_id     INTEGER NOT NULL REFERENCES responses(id),
    stage           TEXT NOT NULL CHECK (stage IN ('input', 'output')),
    triggered       INTEGER NOT NULL CHECK (triggered IN (0, 1)),
    expected_block  INTEGER NOT NULL CHECK (expected_block IN (0, 1)),
    outcome         TEXT NOT NULL CHECK (outcome IN ('TP', 'FP', 'TN', 'FN')),
    detail          TEXT,
    method          TEXT NOT NULL DEFAULT 'deterministic'
                      CHECK (method IN ('deterministic', 'bedrock')),
    latency_ms      INTEGER,
    cost_usd        REAL
);

CREATE TABLE IF NOT EXISTS comparisons (
    id                 INTEGER PRIMARY KEY,
    baseline_run_id    INTEGER NOT NULL REFERENCES runs(id),
    candidate_run_id   INTEGER NOT NULL REFERENCES runs(id),
    n_paired           INTEGER NOT NULL,
    decision           TEXT NOT NULL CHECK (decision IN ('GO', 'REVIEW', 'HOLD', 'INVALID')),
    deltas_json        TEXT NOT NULL,
    gates_json         TEXT NOT NULL,
    summary            TEXT,
    created_at         TEXT NOT NULL,
    UNIQUE (baseline_run_id, candidate_run_id)
);

CREATE INDEX IF NOT EXISTS idx_responses_run ON responses(run_id);
CREATE INDEX IF NOT EXISTS idx_scores_response ON scores(response_id);
CREATE INDEX IF NOT EXISTS idx_guardrail_results_response ON guardrail_results(response_id);
CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_hash);
