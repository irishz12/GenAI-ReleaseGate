"""Config schemas and a YAML loader for config/models.yaml, policy.yaml, guardrails.yaml.

These are placeholders for V1: the schemas are real and validated, but nothing in Phase 0
reads AWS credentials or calls a provider. `.env` (see .env.example) is loaded via
`load_dotenv()` so later phases can pick up AWS_* vars through the environment without
another config layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = _REPO_ROOT / "config"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# config/models.yaml
# ─────────────────────────────────────────────────────────────────────────────


class ModelParams(_Base):
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0


class ModelEntry(_Base):
    provider: Literal["fake", "bedrock_mantle"]
    model_id: str
    params: ModelParams = ModelParams()
    # Optional and unset by default (never 0.0 by default): a cost figure must come
    # from a price the user has actually verified and typed in here. `None` means
    # "not yet verified" — evalguard.runner.runner.compute_cost_usd refuses to guess
    # and returns None rather than silently reporting an understated $0.00.
    price_per_1k_input_tokens: float | None = None
    price_per_1k_output_tokens: float | None = None


class ModelProfile(_Base):
    """A generator/judge pair for one provider — the fields Phase 3 needs to run a
    real client, kept separate from the default `fake` pair above so filling this in
    can never accidentally change what the offline pipeline uses."""

    generator: ModelEntry
    judge: ModelEntry
    # Phase 8.2: the faithfulness judge call (claim extraction) needs more room than
    # correctness — `judge.params.max_tokens` (512) stays correctness's frozen budget;
    # this is faithfulness's own frozen budget, set from Phase 8.1's real evidence
    # (512 reliably produced malformed/truncated JSON; 1024 reliably did not).
    judge_faithfulness_max_tokens: int


class ModelsConfig(_Base):
    generator: ModelEntry
    judge: ModelEntry
    # Real-provider model identity for Phase 3 (docs/ARCHITECTURE.md: "generator/judge
    # model IDs from config/models.yaml"). The endpoint/API key are connection secrets
    # and come from .env instead (see providers/bedrock_mantle.py:MantleConfig) — never
    # from this file. Optional: the default `fake` pair above works with nothing here.
    bedrock_mantle: ModelProfile | None = None


# ─────────────────────────────────────────────────────────────────────────────
# config/policy.yaml
# ─────────────────────────────────────────────────────────────────────────────


class GateConfig(_Base):
    metric: str
    direction: Literal["higher_is_better", "lower_is_better"]
    hold_if_delta_worse_than: float | None = None
    review_if_delta_worse_than: float | None = None
    hold_if_delta_worse_than_pct: float | None = None
    review_if_delta_worse_than_pct: float | None = None
    absolute_min: float | None = None
    absolute_max: float | None = None


class PolicyConfig(_Base):
    gates: list[GateConfig]
    # A comparison whose paired cases failed (errored on either side) more often than
    # this is unreliable regardless of what the gates say — see regression.compare's
    # INVALID handling. 0.2 is a DEVELOPMENT DEFAULT (see config/policy.yaml), not a
    # tuned production threshold.
    invalid_if_failure_rate_over: float = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# config/guardrails.yaml
# ─────────────────────────────────────────────────────────────────────────────


class GuardrailPattern(_Base):
    name: str
    stage: Literal["input", "output"]
    pattern: str


class BedrockGuardrailPricing(_Base):
    """Per-1K-text-unit prices — never guessed. See config/guardrails.yaml for the
    AWS Price List API citation each field comes from."""

    price_per_1k_content_policy_units: float | None = None
    price_per_1k_sensitive_information_units_paid: float | None = None
    price_per_1k_sensitive_information_units_free: float | None = None


class BedrockGuardrailConfig(_Base):
    """Identity of the real AWS Bedrock Guardrail (Phase 6). Never a secret — the
    connection credential is the ambient AWS CLI/SSO session boto3 picks up on its
    own, not anything read from this file or from .env."""

    guardrail_id: str
    guardrail_version: str
    region: str
    pricing: BedrockGuardrailPricing = BedrockGuardrailPricing()


class GuardrailsConfig(_Base):
    patterns: list[GuardrailPattern] = []
    bedrock_guardrail: BedrockGuardrailConfig | None = None


# ─────────────────────────────────────────────────────────────────────────────
# loader
# ─────────────────────────────────────────────────────────────────────────────


def load_yaml_config[T: BaseModel](path: str | Path, model: type[T]) -> T:
    """Load a YAML file and validate it against a Pydantic config model."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return model.model_validate(data or {})


def load_models_config(path: str | Path = DEFAULT_CONFIG_DIR / "models.yaml") -> ModelsConfig:
    return load_yaml_config(path, ModelsConfig)


def load_policy_config(path: str | Path = DEFAULT_CONFIG_DIR / "policy.yaml") -> PolicyConfig:
    return load_yaml_config(path, PolicyConfig)


def load_guardrails_config(
    path: str | Path = DEFAULT_CONFIG_DIR / "guardrails.yaml",
) -> GuardrailsConfig:
    return load_yaml_config(path, GuardrailsConfig)


class BedrockMantleModelsNotConfiguredError(RuntimeError):
    """Raised when config/models.yaml has no `bedrock_mantle` section — this is the
    "don't guess a model id" equivalent of MantleConfigError for connection config."""


def load_bedrock_mantle_profile(
    path: str | Path = DEFAULT_CONFIG_DIR / "models.yaml",
) -> ModelProfile:
    """The generator/judge model identity Phase 3 actually calls — never guessed,
    never defaulted; must be explicitly present under `bedrock_mantle:` in
    config/models.yaml."""
    config = load_models_config(path)
    if config.bedrock_mantle is None:
        raise BedrockMantleModelsNotConfiguredError(
            f"{path} has no `bedrock_mantle:` section — add one with real generator/"
            "judge model ids before using BedrockMantleClient."
        )
    return config.bedrock_mantle


class BedrockGuardrailNotConfiguredError(RuntimeError):
    """Raised when config/guardrails.yaml has no `bedrock_guardrail` section."""


def load_bedrock_guardrail_config(
    path: str | Path = DEFAULT_CONFIG_DIR / "guardrails.yaml",
) -> BedrockGuardrailConfig:
    """The frozen guardrail identity Phase 6 actually calls — never guessed; must be
    explicitly present under `bedrock_guardrail:` in config/guardrails.yaml."""
    config = load_guardrails_config(path)
    if config.bedrock_guardrail is None:
        raise BedrockGuardrailNotConfiguredError(
            f"{path} has no `bedrock_guardrail:` section — add one with a real "
            "guardrail_id/guardrail_version before using BedrockGuardrailClient."
        )
    return config.bedrock_guardrail
