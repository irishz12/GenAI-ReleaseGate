"""RunManifest tests: the frozen plan must hash reproducibly and change whenever
anything that affects run behavior changes — prompt, case set, model config, judge
model, guardrail config, or code.
"""

from __future__ import annotations

from pathlib import Path

from evalguard.enums import CaseCategory, CaseSplit, PromptRole
from evalguard.models import EvaluationCase, Prompt
from evalguard.runner.manifest import build_run_manifest, compute_code_hash, manifest_to_run


def _prompt(**overrides) -> Prompt:
    defaults = dict(
        name="support_agent",
        version="v1",
        role=PromptRole.PRODUCTION,
        template="Answer: {question} given {context}",
        content_hash="prompt-hash-v1",
    )
    defaults.update(overrides)
    return Prompt(**defaults)


def _case(external_id: str, **overrides) -> EvaluationCase:
    defaults = dict(
        external_id=external_id,
        category=CaseCategory.GROUNDED_QA,
        split=CaseSplit.DEV,
        input="What is the refund window?",
        context="Refunds are accepted within 30 days.",
        dataset_hash="corpus-hash",
    )
    defaults.update(overrides)
    return EvaluationCase(**defaults)


def _manifest_kwargs(**overrides):
    defaults = dict(
        prompt=_prompt(),
        cases=[_case("gqa-0001"), _case("gqa-0002")],
        dataset_split=CaseSplit.DEV,
        dataset_corpus_hash="corpus-hash",
        model_id="fake-generator-v1",
        model_params={"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        fake_seed=42,
        judge_model_id="fake-judge-v1",
        guardrail_config_hash="guardrail-hash",
        code_hash="code-hash",
    )
    defaults.update(overrides)
    return defaults


def test_build_run_manifest_is_deterministic() -> None:
    a = build_run_manifest(**_manifest_kwargs())
    b = build_run_manifest(**_manifest_kwargs())
    assert a.config_hash == b.config_hash
    assert a == b


def test_config_hash_is_independent_of_case_order() -> None:
    a = build_run_manifest(**_manifest_kwargs(cases=[_case("gqa-0001"), _case("gqa-0002")]))
    b = build_run_manifest(**_manifest_kwargs(cases=[_case("gqa-0002"), _case("gqa-0001")]))
    assert a.config_hash == b.config_hash
    assert a.case_external_ids == b.case_external_ids == ["gqa-0001", "gqa-0002"]


def test_config_hash_changes_when_prompt_hash_changes() -> None:
    a = build_run_manifest(**_manifest_kwargs(prompt=_prompt(content_hash="hash-a")))
    b = build_run_manifest(**_manifest_kwargs(prompt=_prompt(content_hash="hash-b")))
    assert a.config_hash != b.config_hash


def test_config_hash_changes_when_case_set_changes() -> None:
    a = build_run_manifest(**_manifest_kwargs(cases=[_case("gqa-0001")]))
    b = build_run_manifest(**_manifest_kwargs(cases=[_case("gqa-0001"), _case("gqa-0002")]))
    assert a.config_hash != b.config_hash
    assert a.case_set_hash != b.case_set_hash


def test_config_hash_changes_when_dataset_corpus_hash_changes() -> None:
    a = build_run_manifest(**_manifest_kwargs(dataset_corpus_hash="corpus-a"))
    b = build_run_manifest(**_manifest_kwargs(dataset_corpus_hash="corpus-b"))
    assert a.config_hash != b.config_hash


def test_config_hash_changes_when_model_params_change() -> None:
    a = build_run_manifest(**_manifest_kwargs(model_params={"temperature": 0.0}))
    b = build_run_manifest(**_manifest_kwargs(model_params={"temperature": 0.7}))
    assert a.config_hash != b.config_hash


def test_config_hash_changes_when_fake_seed_changes() -> None:
    a = build_run_manifest(**_manifest_kwargs(fake_seed=1))
    b = build_run_manifest(**_manifest_kwargs(fake_seed=2))
    assert a.config_hash != b.config_hash


def test_config_hash_changes_when_judge_model_id_changes() -> None:
    a = build_run_manifest(**_manifest_kwargs(judge_model_id="judge-a"))
    b = build_run_manifest(**_manifest_kwargs(judge_model_id="judge-b"))
    assert a.config_hash != b.config_hash


def test_config_hash_changes_when_guardrail_config_hash_changes() -> None:
    a = build_run_manifest(**_manifest_kwargs(guardrail_config_hash="g-a"))
    b = build_run_manifest(**_manifest_kwargs(guardrail_config_hash="g-b"))
    assert a.config_hash != b.config_hash


def test_config_hash_changes_when_code_hash_changes() -> None:
    a = build_run_manifest(**_manifest_kwargs(code_hash="code-a"))
    b = build_run_manifest(**_manifest_kwargs(code_hash="code-b"))
    assert a.config_hash != b.config_hash


def test_v1_and_v2_manifests_differ_only_in_prompt_identity() -> None:
    """Same dataset, same model, same everything except the prompt — exactly the
    "same generator model, same cases" comparison setup docs/ARCHITECTURE.md calls for."""
    shared = _manifest_kwargs()
    manifest_v1 = build_run_manifest(**shared)
    manifest_v2 = build_run_manifest(
        **{**shared, "prompt": _prompt(version="v2", content_hash="prompt-hash-v2")}
    )

    assert manifest_v1.config_hash != manifest_v2.config_hash
    assert manifest_v1.case_set_hash == manifest_v2.case_set_hash
    assert manifest_v1.dataset_corpus_hash == manifest_v2.dataset_corpus_hash
    assert manifest_v1.model_id == manifest_v2.model_id
    assert manifest_v1.model_params == manifest_v2.model_params


def test_manifest_to_run_folds_fake_seed_into_model_params() -> None:
    manifest = build_run_manifest(**_manifest_kwargs(fake_seed=99))
    run = manifest_to_run(manifest, prompt_id=7)
    assert run.model_params["fake_seed"] == 99
    assert run.prompt_id == 7
    assert run.config_hash == manifest.config_hash
    assert run.id is None  # not yet persisted


# ─────────────────────────────────────────────────────────────────────────────
# compute_code_hash
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_code_hash_is_deterministic() -> None:
    from evalguard.runner.manifest import _REPO_ROOT

    package_dir = _REPO_ROOT / "src" / "evalguard"
    assert compute_code_hash(package_dir) == compute_code_hash(package_dir)


def test_compute_code_hash_changes_when_a_file_changes(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "a.py").write_text("x = 1\n")
    before = compute_code_hash(package_dir)

    (package_dir / "a.py").write_text("x = 2\n")
    after = compute_code_hash(package_dir)

    assert before != after


def test_compute_code_hash_is_independent_of_file_visit_order(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    (dir_a / "m1.py").write_text("one")
    (dir_a / "m2.py").write_text("two")

    dir_b = tmp_path / "b"
    dir_b.mkdir()
    (dir_b / "m2.py").write_text("two")
    (dir_b / "m1.py").write_text("one")

    assert compute_code_hash(dir_a) == compute_code_hash(dir_b)
