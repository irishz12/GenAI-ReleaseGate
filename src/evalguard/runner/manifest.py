"""The frozen run manifest: everything a run needs, hashed reproducibly, before a
single generator call happens (docs/ARCHITECTURE.md §12).

A `RunManifest` is the *plan*; a `Run` (evalguard.models, persisted via db.store) is the
DB record of executing that plan. Two manifests built from identical inputs always
produce the same `config_hash` — that's both the reproducibility guarantee and the
resume mechanism (runner.execute_run looks up an existing `runs` row by config_hash).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evalguard.enums import CaseSplit, RunStatus
from evalguard.hashing import combine_hashes, hash_file, sha256_json
from evalguard.models import EvaluationCase, Prompt, Run

_REPO_ROOT = Path(__file__).resolve().parents[3]


class RunManifest(BaseModel):
    """Everything needed to execute one prompt version over one frozen case set."""

    model_config = ConfigDict(extra="forbid")

    prompt_name: str
    prompt_version: str
    prompt_hash: str
    dataset_split: CaseSplit
    case_external_ids: list[str] = Field(default_factory=list)  # sorted, frozen
    case_set_hash: str
    dataset_corpus_hash: str
    model_id: str
    model_params: dict
    # Only meaningful for FakeClient runs — None for a real-provider run (Phase 3's
    # BedrockMantleClient has no seed concept; its determinism comes from the real
    # model's own temperature/sampling settings, not from evalguard).
    fake_seed: int | None = None
    judge_model_id: str
    guardrail_config_hash: str
    code_hash: str
    config_hash: str


def compute_code_hash(package_dir: Path | None = None) -> str:
    """Content hash of every `.py` file under the evalguard package.

    Stands in for a git commit SHA (docs/ARCHITECTURE.md calls out `code_sha`) — this
    project has no git repository yet. Combining per-file hashes means the result
    changes whenever any package code changes, so two runs against different code never
    collide on the same config_hash and get silently treated as "the same run, just
    resuming". If/when this repo gets a git history, swapping in
    `git rev-parse HEAD` here is a one-line change confined to this function.
    """
    package_dir = package_dir or (_REPO_ROOT / "src" / "evalguard")
    py_files = sorted(p for p in package_dir.rglob("*.py"))
    return combine_hashes(*(hash_file(p) for p in py_files))


def build_run_manifest(
    *,
    prompt: Prompt,
    cases: list[EvaluationCase],
    dataset_split: CaseSplit,
    dataset_corpus_hash: str,
    model_id: str,
    model_params: dict,
    fake_seed: int | None = None,
    judge_model_id: str,
    guardrail_config_hash: str,
    code_hash: str,
) -> RunManifest:
    """Freeze one run's plan and compute its config_hash.

    `config_hash` is a canonical-JSON hash (`sha256_json`) over every field that
    determines run behavior — prompt identity, the exact case set, model
    identity/params, the judge/guardrail config that *would* apply, and the code
    itself. Two manifests are "the same run" iff every one of those matches.
    """
    case_external_ids = sorted(c.external_id for c in cases)
    case_set_hash = sha256_json(case_external_ids)

    payload = {
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_hash": prompt.content_hash,
        "dataset_split": dataset_split.value,
        "case_set_hash": case_set_hash,
        "dataset_corpus_hash": dataset_corpus_hash,
        "model_id": model_id,
        "model_params": model_params,
        "fake_seed": fake_seed,
        "judge_model_id": judge_model_id,
        "guardrail_config_hash": guardrail_config_hash,
        "code_hash": code_hash,
    }
    config_hash = sha256_json(payload)

    return RunManifest(
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        prompt_hash=prompt.content_hash,
        dataset_split=dataset_split,
        case_external_ids=case_external_ids,
        case_set_hash=case_set_hash,
        dataset_corpus_hash=dataset_corpus_hash,
        model_id=model_id,
        model_params=model_params,
        fake_seed=fake_seed,
        judge_model_id=judge_model_id,
        guardrail_config_hash=guardrail_config_hash,
        code_hash=code_hash,
        config_hash=config_hash,
    )


def manifest_to_run(manifest: RunManifest, prompt_id: int) -> Run:
    """Map a frozen manifest onto an insertable `Run` row.

    `fake_seed` is folded into `model_params_json` when present — there's no dedicated
    `seed` column on `runs` (docs/ARCHITECTURE.md §9 deliberately keeps the schema to 7
    tables with no extra columns); it's still part of `config_hash` either way, present
    or not, this is just where it lives at rest. Omitted entirely for a real-provider
    run (`fake_seed is None`) rather than persisting a misleading `"fake_seed": null`.
    """
    model_params = dict(manifest.model_params)
    if manifest.fake_seed is not None:
        model_params["fake_seed"] = manifest.fake_seed

    return Run(
        prompt_id=prompt_id,
        model_id=manifest.model_id,
        model_params=model_params,
        judge_model_id=manifest.judge_model_id,
        guardrail_config_hash=manifest.guardrail_config_hash,
        code_sha=manifest.code_hash,
        config_hash=manifest.config_hash,
        status=RunStatus.RUNNING,
    )
