"""SQLite connection helpers and typed insert/get functions for the 7-table schema.

Every `insert_*` takes a Pydantic model (with `id=None`) and returns the same model with
`id` set to the new rowid. Every `get_*` reconstructs the Pydantic model from a row,
parsing JSON columns and casting enum strings back to their enum types. This is the only
module that knows the DB is SQLite — everything above it (runner, quality, regression,
etc., in later phases) only ever sees the Pydantic contracts from `evalguard.models`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from evalguard.enums import MetricName, RunStatus
from evalguard.models import (
    Comparison,
    EvaluationCase,
    GateResult,
    GuardrailResult,
    MetricDelta,
    Prompt,
    ReleaseDecision,
    Response,
    Run,
    Score,
)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# ─────────────────────────────────────────────────────────────────────────────
# connection / lifecycle
# ─────────────────────────────────────────────────────────────────────────────


def get_connection(path: str | Path) -> sqlite3.Connection:
    """Open a connection with the pragmas this schema depends on.

    WAL mode is the right call even for a single-writer tool: it lets a report-rendering
    read happen concurrently with an in-progress run without blocking on the writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, sql_type: str) -> None:
    """Add `column` to `table` if an older schema version doesn't have it yet.

    `CREATE TABLE IF NOT EXISTS` only helps for a brand-new database — a table created
    under an earlier schema.sql (e.g. an existing artifacts/*.db from before the
    judge_* columns existed) keeps its original columns forever unless something adds
    the new ones explicitly.
    """
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")


def init_db(conn: sqlite3.Connection) -> None:
    """Apply schema.sql, then backfill any columns a pre-existing database predates."""
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    _ensure_column(conn, "scores", "judge_input_tokens", "INTEGER")
    _ensure_column(conn, "scores", "judge_output_tokens", "INTEGER")
    _ensure_column(conn, "scores", "judge_cost_usd", "REAL")
    _ensure_column(conn, "scores", "judge_latency_ms", "INTEGER")
    _ensure_column(conn, "guardrail_results", "method", "TEXT NOT NULL DEFAULT 'deterministic'")
    _ensure_column(conn, "guardrail_results", "latency_ms", "INTEGER")
    _ensure_column(conn, "guardrail_results", "cost_usd", "REAL")
    conn.commit()


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _bool_to_int(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _int_to_bool(value: int | None) -> bool | None:
    return None if value is None else bool(value)


# ─────────────────────────────────────────────────────────────────────────────
# prompts
# ─────────────────────────────────────────────────────────────────────────────


def insert_prompt(conn: sqlite3.Connection, prompt: Prompt) -> Prompt:
    cur = conn.execute(
        """
        INSERT INTO prompts (name, version, role, template, content_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            prompt.name,
            prompt.version,
            prompt.role.value,
            prompt.template,
            prompt.content_hash,
            _iso(prompt.created_at),
        ),
    )
    conn.commit()
    return prompt.model_copy(update={"id": cur.lastrowid})


def get_prompt(conn: sqlite3.Connection, prompt_id: int) -> Prompt | None:
    row = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    if row is None:
        return None
    return Prompt(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        role=row["role"],
        template=row["template"],
        content_hash=row["content_hash"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# evaluation_cases
# ─────────────────────────────────────────────────────────────────────────────


def insert_case(conn: sqlite3.Connection, case: EvaluationCase) -> EvaluationCase:
    cur = conn.execute(
        """
        INSERT INTO evaluation_cases
            (external_id, category, split, input, context, reference_answer,
             constraints_json, expected_abstention, attack_type, canary,
             expected_block, dataset_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case.external_id,
            case.category.value,
            case.split.value,
            case.input,
            case.context,
            case.reference_answer,
            json.dumps(case.constraints) if case.constraints is not None else None,
            _bool_to_int(case.expected_abstention),
            case.attack_type.value if case.attack_type else None,
            case.canary,
            _bool_to_int(case.expected_block),
            case.dataset_hash,
        ),
    )
    conn.commit()
    return case.model_copy(update={"id": cur.lastrowid})


def get_case(conn: sqlite3.Connection, case_id: int) -> EvaluationCase | None:
    row = conn.execute("SELECT * FROM evaluation_cases WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    return EvaluationCase(
        id=row["id"],
        external_id=row["external_id"],
        category=row["category"],
        split=row["split"],
        input=row["input"],
        context=row["context"],
        reference_answer=row["reference_answer"],
        constraints=json.loads(row["constraints_json"]) if row["constraints_json"] else None,
        expected_abstention=_int_to_bool(row["expected_abstention"]),
        attack_type=row["attack_type"],
        canary=row["canary"],
        expected_block=_int_to_bool(row["expected_block"]),
        dataset_hash=row["dataset_hash"],
    )


def get_case_by_external_id(conn: sqlite3.Connection, external_id: str) -> EvaluationCase | None:
    """Look up a case by its dataset-file identity rather than its DB rowid.

    Cases loaded from data/dev/*.jsonl (registry.datasets.load_cases) always have
    `id=None` — they only get a DB id once registered here. This is how the runner
    resolves "is this case already in the registry, and if so, what's its id?"
    """
    row = conn.execute(
        "SELECT id FROM evaluation_cases WHERE external_id = ?", (external_id,)
    ).fetchone()
    if row is None:
        return None
    return get_case(conn, row["id"])


# ─────────────────────────────────────────────────────────────────────────────
# runs
# ─────────────────────────────────────────────────────────────────────────────


def insert_run(conn: sqlite3.Connection, run: Run) -> Run:
    cur = conn.execute(
        """
        INSERT INTO runs
            (prompt_id, model_id, model_params_json, judge_model_id,
             guardrail_config_hash, code_sha, config_hash, status,
             case_count, failure_count, total_cost_usd, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run.prompt_id,
            run.model_id,
            json.dumps(run.model_params),
            run.judge_model_id,
            run.guardrail_config_hash,
            run.code_sha,
            run.config_hash,
            run.status.value,
            run.case_count,
            run.failure_count,
            run.total_cost_usd,
            _iso(run.started_at),
            _iso(run.finished_at) if run.finished_at else None,
        ),
    )
    conn.commit()
    return run.model_copy(update={"id": cur.lastrowid})


def get_run(conn: sqlite3.Connection, run_id: int) -> Run | None:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return Run(
        id=row["id"],
        prompt_id=row["prompt_id"],
        model_id=row["model_id"],
        model_params=json.loads(row["model_params_json"]),
        judge_model_id=row["judge_model_id"],
        guardrail_config_hash=row["guardrail_config_hash"],
        code_sha=row["code_sha"],
        config_hash=row["config_hash"],
        status=row["status"],
        case_count=row["case_count"],
        failure_count=row["failure_count"],
        total_cost_usd=row["total_cost_usd"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
    )


def get_run_by_config_hash(conn: sqlite3.Connection, config_hash: str) -> Run | None:
    """The resume mechanism: a run is identified by its manifest's config_hash, not by
    a fresh row per invocation. Two calls with the same manifest resolve to the same
    run row here."""
    row = conn.execute("SELECT id FROM runs WHERE config_hash = ?", (config_hash,)).fetchone()
    if row is None:
        return None
    return get_run(conn, row["id"])


def update_run_status(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: RunStatus,
    case_count: int | None = None,
    failure_count: int | None = None,
    total_cost_usd: float | None = None,
    finished_at: datetime | None = None,
) -> Run:
    """Update a run's mutable bookkeeping fields (never its identity fields —
    prompt_id/model_id/config_hash/etc. are set once at creation and never change).

    `case_count`/`failure_count`/`total_cost_usd` are overwritten unconditionally,
    `None` included — the caller (runner.execute_run) always recomputes and passes its
    actual current value, and for `total_cost_usd` specifically, `None` is itself a
    meaningful result ("pricing not configured, cost is genuinely unknown" — see
    runner.compute_cost_usd), not a "leave this alone" sentinel, so COALESCE would be
    wrong here. `finished_at` is the one field that keeps COALESCE, and in the
    direction that protects it: once a run's completion timestamp is set, a later call
    (e.g. an idempotent re-invocation of an already-COMPLETED run) must never bump it
    to a fresher "now" — `COALESCE(finished_at, ?)` keeps the first value set.
    """
    conn.execute(
        """
        UPDATE runs
        SET status = ?,
            case_count = ?,
            failure_count = ?,
            total_cost_usd = ?,
            finished_at = COALESCE(finished_at, ?)
        WHERE id = ?
        """,
        (
            status.value,
            case_count,
            failure_count,
            total_cost_usd,
            _iso(finished_at) if finished_at else None,
            run_id,
        ),
    )
    conn.commit()
    return get_run(conn, run_id)


# ─────────────────────────────────────────────────────────────────────────────
# responses
# ─────────────────────────────────────────────────────────────────────────────


def insert_response(conn: sqlite3.Connection, response: Response) -> Response:
    cur = conn.execute(
        """
        INSERT INTO responses
            (run_id, case_id, output_text, finish_reason, input_tokens, output_tokens,
             cost_usd, latency_ms, error_class, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            response.run_id,
            response.case_id,
            response.output_text,
            response.finish_reason.value,
            response.input_tokens,
            response.output_tokens,
            response.cost_usd,
            response.latency_ms,
            response.error_class.value if response.error_class else None,
            _iso(response.created_at),
        ),
    )
    conn.commit()
    return response.model_copy(update={"id": cur.lastrowid})


def get_response(conn: sqlite3.Connection, response_id: int) -> Response | None:
    row = conn.execute("SELECT * FROM responses WHERE id = ?", (response_id,)).fetchone()
    if row is None:
        return None
    return Response(
        id=row["id"],
        run_id=row["run_id"],
        case_id=row["case_id"],
        output_text=row["output_text"],
        finish_reason=row["finish_reason"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cost_usd=row["cost_usd"],
        latency_ms=row["latency_ms"],
        error_class=row["error_class"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def update_response_cost(
    conn: sqlite3.Connection, response_id: int, cost_usd: float | None
) -> Response:
    """Recompute-and-store `cost_usd` from a response's already-stored token counts and
    a (possibly newly-verified) price — never touches `output_text`/tokens/finish_reason.

    This is the one sanctioned exception to "responses are immutable"
    (docs/ARCHITECTURE.md §12: "Cost is computed from stored token counts, so a price
    change re-derives historical cost without re-running anything" — cost is
    explicitly designed to be re-derivable, unlike the generation record itself).
    """
    conn.execute("UPDATE responses SET cost_usd = ? WHERE id = ?", (cost_usd, response_id))
    conn.commit()
    return get_response(conn, response_id)


def get_response_by_run_and_case(
    conn: sqlite3.Connection, run_id: int, case_id: int
) -> Response | None:
    """The `UNIQUE(run_id, case_id)` lookup — this is what the runner calls to decide
    whether a case has already been attempted for a given run (resume/skip logic)."""
    row = conn.execute(
        "SELECT id FROM responses WHERE run_id = ? AND case_id = ?", (run_id, case_id)
    ).fetchone()
    if row is None:
        return None
    return get_response(conn, row["id"])


def get_responses_for_run(conn: sqlite3.Connection, run_id: int) -> list[Response]:
    rows = conn.execute("SELECT id FROM responses WHERE run_id = ?", (run_id,)).fetchall()
    return [get_response(conn, row["id"]) for row in rows]


def delete_response(conn: sqlite3.Connection, response_id: int) -> None:
    """Used only by the runner's `retry_failed=True` path, to clear a previously-errored
    response before re-attempting that case. Responses are otherwise immutable —
    Stage B (scoring, later phases) depends on that; this is the one deliberate
    exception, and it only ever removes a row that already recorded a failure."""
    conn.execute("DELETE FROM responses WHERE id = ?", (response_id,))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# scores
# ─────────────────────────────────────────────────────────────────────────────


def insert_score(conn: sqlite3.Connection, score: Score) -> Score:
    cur = conn.execute(
        """
        INSERT INTO scores
            (response_id, metric, method, value, rationale,
             judge_input_tokens, judge_output_tokens, judge_cost_usd, judge_latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            score.response_id,
            score.metric.value,
            score.method.value,
            score.value,
            score.rationale,
            score.judge_input_tokens,
            score.judge_output_tokens,
            score.judge_cost_usd,
            score.judge_latency_ms,
        ),
    )
    conn.commit()
    return score.model_copy(update={"id": cur.lastrowid})


def get_scores_for_response(conn: sqlite3.Connection, response_id: int) -> list[Score]:
    rows = conn.execute("SELECT * FROM scores WHERE response_id = ?", (response_id,)).fetchall()
    return [
        Score(
            id=row["id"],
            response_id=row["response_id"],
            metric=row["metric"],
            method=row["method"],
            value=row["value"],
            rationale=row["rationale"],
            judge_input_tokens=row["judge_input_tokens"],
            judge_output_tokens=row["judge_output_tokens"],
            judge_cost_usd=row["judge_cost_usd"],
            judge_latency_ms=row["judge_latency_ms"],
        )
        for row in rows
    ]


def delete_score(conn: sqlite3.Connection, response_id: int, metric: MetricName) -> None:
    """Remove one (response_id, metric) score so it can be re-scored under a changed
    setting (Phase 8.2: re-scoring faithfulness at a new frozen budget) without
    hitting `UNIQUE(response_id, metric)` on the next insert. Scores are otherwise
    immutable — this is a deliberate, narrow exception, same spirit as
    `delete_response`'s `retry_failed` path."""
    conn.execute(
        "DELETE FROM scores WHERE response_id = ? AND metric = ?", (response_id, metric.value)
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# guardrail_results
# ─────────────────────────────────────────────────────────────────────────────


def insert_guardrail_result(conn: sqlite3.Connection, result: GuardrailResult) -> GuardrailResult:
    cur = conn.execute(
        """
        INSERT INTO guardrail_results
            (response_id, stage, triggered, expected_block, outcome, detail, method,
             latency_ms, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.response_id,
            result.stage.value,
            _bool_to_int(result.triggered),
            _bool_to_int(result.expected_block),
            result.outcome.value,
            result.detail,
            result.method.value,
            result.latency_ms,
            result.cost_usd,
        ),
    )
    conn.commit()
    return result.model_copy(update={"id": cur.lastrowid})


def get_guardrail_results_for_response(
    conn: sqlite3.Connection, response_id: int
) -> list[GuardrailResult]:
    rows = conn.execute(
        "SELECT * FROM guardrail_results WHERE response_id = ?", (response_id,)
    ).fetchall()
    return [
        GuardrailResult(
            id=row["id"],
            response_id=row["response_id"],
            stage=row["stage"],
            triggered=_int_to_bool(row["triggered"]),
            expected_block=_int_to_bool(row["expected_block"]),
            outcome=row["outcome"],
            detail=row["detail"],
            method=row["method"],
            latency_ms=row["latency_ms"],
            cost_usd=row["cost_usd"],
        )
        for row in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# comparisons
# ─────────────────────────────────────────────────────────────────────────────


def insert_comparison(conn: sqlite3.Connection, comparison: Comparison) -> Comparison:
    deltas_json = json.dumps([d.model_dump(mode="json") for d in comparison.deltas])
    gates_json = json.dumps([g.model_dump(mode="json") for g in comparison.release.gates])
    cur = conn.execute(
        """
        INSERT INTO comparisons
            (baseline_run_id, candidate_run_id, n_paired, decision,
             deltas_json, gates_json, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            comparison.baseline_run_id,
            comparison.candidate_run_id,
            comparison.n_paired,
            comparison.release.status.value,
            deltas_json,
            gates_json,
            comparison.release.summary,
            _iso(comparison.created_at),
        ),
    )
    conn.commit()
    return comparison.model_copy(update={"id": cur.lastrowid})


def upsert_comparison(conn: sqlite3.Connection, comparison: Comparison) -> Comparison:
    """Insert a comparison, or replace it in place if one already exists for this
    (baseline_run_id, candidate_run_id) pair — recomputing after more judge/guardrail
    scores land (Phase 8.2) must not collide with the UNIQUE constraint, and must not
    leave a stale decision sitting next to newer scores. `created_at` is left as the
    original insert time; only the derived numbers (n_paired, decision, deltas,
    gates, summary) are refreshed. Raw responses/scores are untouched either way —
    they remain the real source of truth this is only ever recomputed from."""
    deltas_json = json.dumps([d.model_dump(mode="json") for d in comparison.deltas])
    gates_json = json.dumps([g.model_dump(mode="json") for g in comparison.release.gates])
    conn.execute(
        """
        INSERT INTO comparisons
            (baseline_run_id, candidate_run_id, n_paired, decision,
             deltas_json, gates_json, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (baseline_run_id, candidate_run_id) DO UPDATE SET
            n_paired = excluded.n_paired,
            decision = excluded.decision,
            deltas_json = excluded.deltas_json,
            gates_json = excluded.gates_json,
            summary = excluded.summary
        """,
        (
            comparison.baseline_run_id,
            comparison.candidate_run_id,
            comparison.n_paired,
            comparison.release.status.value,
            deltas_json,
            gates_json,
            comparison.release.summary,
            _iso(comparison.created_at),
        ),
    )
    conn.commit()
    # lastrowid isn't reliable across the insert vs. ON CONFLICT UPDATE paths in
    # Python's sqlite3 — just look the row back up by its real unique key.
    saved = get_comparison_by_run_pair(
        conn, comparison.baseline_run_id, comparison.candidate_run_id
    )
    return comparison.model_copy(update={"id": saved.id})


def _row_to_comparison(row: sqlite3.Row) -> Comparison:
    deltas = [MetricDelta.model_validate(d) for d in json.loads(row["deltas_json"])]
    gates = [GateResult.model_validate(g) for g in json.loads(row["gates_json"])]
    release = ReleaseDecision(status=row["decision"], gates=gates, summary=row["summary"])
    return Comparison(
        id=row["id"],
        baseline_run_id=row["baseline_run_id"],
        candidate_run_id=row["candidate_run_id"],
        n_paired=row["n_paired"],
        deltas=deltas,
        release=release,
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def get_comparison(conn: sqlite3.Connection, comparison_id: int) -> Comparison | None:
    row = conn.execute("SELECT * FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
    return None if row is None else _row_to_comparison(row)


def get_comparison_by_run_pair(
    conn: sqlite3.Connection, baseline_run_id: int, candidate_run_id: int
) -> Comparison | None:
    """Look up a comparison by its (baseline_run_id, candidate_run_id) pair — the same
    get-before-insert check `runner._get_or_create_run` does for runs via
    `get_run_by_config_hash`, so a caller can resume idempotently instead of hitting
    the `UNIQUE(baseline_run_id, candidate_run_id)` constraint on a second attempt."""
    row = conn.execute(
        "SELECT * FROM comparisons WHERE baseline_run_id = ? AND candidate_run_id = ?",
        (baseline_run_id, candidate_run_id),
    ).fetchone()
    return None if row is None else _row_to_comparison(row)
