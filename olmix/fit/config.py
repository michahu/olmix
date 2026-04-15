"""Pydantic configuration models for olmix fit.

Defines the YAML-driven configuration for `olmix fit --config <yaml>`.
"""

from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, Discriminator, Tag, model_validator

PathType = Union[Path, PathLike[Any], str]


class SwarmDataConfig(BaseModel):
    """Paths to the swarm CSV files (ratios and metrics)."""

    ratios: str
    metrics: str


class PriorsConfig(BaseModel):
    """Token distribution across domains (inline priors)."""

    relative_sizes: dict[str, float]
    token_counts: dict[str, int]
    expanded_relative_sizes: dict[str, float] | None = None

    def to_tuple(self) -> tuple[dict[str, float], dict[str, int]]:
        """Return the (relative_sizes, token_counts) tuple expected by run_fit."""
        return (dict(self.relative_sizes), dict(self.token_counts))


class RegressionConfig(BaseModel):
    """Regression model settings."""

    type: str = "log_linear"
    seed: int = 0
    n_test: int = 0
    train_split: float = 1.0
    aggregate_task_families: bool = False


class ProposerConfig(BaseModel):
    """Mixture proposer settings."""

    type: str = "exact"
    temperature: float | None = None
    kl_reg: float | None = None
    fit_only: bool = False
    make_worst_mix: bool = False
    expanded_kl_source_mixtures: dict[str, dict[str, float]] | None = None


class ConstraintsConfig(BaseModel):
    """Token budget constraints."""

    enabled: bool = False
    target_tokens: int | None = None
    repetition_factor: float = 4.0


class FilteringConfig(BaseModel):
    """Domain/metric filtering."""

    drop_metrics: list[str] = []
    obj_weights: dict[str, float] = {}


class InLoopEvalConfig(BaseModel):
    """Eval config for in-loop (WandB) metrics.

    Tasks are nested by family: {family: {task_id: wandb_metric_name}}.
    Used by ``olmix launch`` (task_ids) and ``olmix fit`` (metric_names).
    """

    type: Literal["inloop"] = "inloop"
    tasks: dict[str, dict[str, str]]

    @property
    def metric_names(self) -> list[str]:
        """Flattened list of WandB metric names (CSV column names)."""
        return [name for family in self.tasks.values() for name in family.values()]

    @property
    def task_ids(self) -> list[str]:
        """Flattened list of olmo-core task IDs."""
        return [tid for family in self.tasks.values() for tid in family.keys()]

    @property
    def task_families(self) -> dict[str, list[str]]:
        """Family → list of metric names."""
        return {family: list(mapping.values()) for family, mapping in self.tasks.items()}


class OfflineEvalConfig(BaseModel):
    """Eval config for offline metrics.

    Tasks are nested by family: {family: [metric_name, ...]}.
    Used by ``olmix fit`` only.
    """

    type: Literal["offline"] = "offline"
    tasks: dict[str, list[str]]

    @property
    def metric_names(self) -> list[str]:
        """Flattened list of metric names (CSV column names)."""
        return [name for names in self.tasks.values() for name in names]

    @property
    def task_families(self) -> dict[str, list[str]]:
        """Family → list of metric names."""
        return dict(self.tasks)


def _eval_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "offline")
    return getattr(v, "type", "offline")


EvalConfig = Annotated[
    Union[
        Annotated[InLoopEvalConfig, Tag("inloop")],
        Annotated[OfflineEvalConfig, Tag("offline")],
    ],
    Discriminator(_eval_discriminator),
]


class FitConfig(BaseModel):
    """Top-level fit configuration, composed of all sub-configs."""

    swarm: SwarmDataConfig
    priors: PriorsConfig
    eval: EvalConfig | None = None
    regression: RegressionConfig = RegressionConfig()
    proposer: ProposerConfig = ProposerConfig()
    constraints: ConstraintsConfig = ConstraintsConfig()
    filtering: FilteringConfig = FilteringConfig()

    @model_validator(mode="after")
    def validate_expanded_kl_config(self) -> "FitConfig":
        expanded_prior = self.priors.expanded_relative_sizes
        source_mixtures = self.proposer.expanded_kl_source_mixtures

        if (expanded_prior is None) != (source_mixtures is None):
            raise ValueError(
                "priors.expanded_relative_sizes and proposer.expanded_kl_source_mixtures must be provided together"
            )

        if expanded_prior is None:
            return self

        collapsed_sources = set(self.priors.relative_sizes)
        expanded_keys = set(expanded_prior)

        if set(source_mixtures) - collapsed_sources:
            extra_sources = sorted(set(source_mixtures) - collapsed_sources)
            raise ValueError(f"expanded_kl_source_mixtures contains unknown collapsed sources: {extra_sources}")

        covered_keys: set[str] = set()
        for source, mixture in source_mixtures.items():
            total = sum(mixture.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"expanded_kl_source_mixtures[{source}] must sum to 1.0, got {total}")
            mixture_keys = set(mixture)
            missing = sorted(mixture_keys - expanded_keys)
            if missing:
                raise ValueError(
                    f"expanded_kl_source_mixtures[{source}] contains keys missing from expanded_relative_sizes: {missing}"
                )
            overlap = sorted(mixture_keys & covered_keys)
            if overlap:
                raise ValueError(f"expanded_kl_source_mixtures leaf keys overlap across sources: {overlap}")
            covered_keys.update(mixture_keys)

        identity_sources = collapsed_sources - set(source_mixtures)
        missing_identity = sorted(source for source in identity_sources if source not in expanded_keys)
        if missing_identity:
            raise ValueError(
                "expanded_relative_sizes must include uncollapsed sources directly when expanded KL is enabled: "
                f"{missing_identity}"
            )

        expected_keys = covered_keys | identity_sources
        if expanded_keys != expected_keys:
            missing = sorted(expected_keys - expanded_keys)
            extra = sorted(expanded_keys - expected_keys)
            details = []
            if missing:
                details.append(f"missing keys {missing}")
            if extra:
                details.append(f"unexpected keys {extra}")
            raise ValueError("expanded_relative_sizes does not match the collapsed-to-leaf expansion: " + ", ".join(details))

        return self

    @classmethod
    def _evaluate_fraction(cls, value: str) -> float:
        """Safely evaluate fraction strings like '7/52.0' to floats."""
        # Try to parse as a simple division expression
        if "/" in value:
            try:
                parts = value.split("/")
                if len(parts) == 2:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    return numerator / denominator
            except (ValueError, ZeroDivisionError):
                pass
        # Fallback to direct float conversion
        return float(value)

    @classmethod
    def from_yaml(cls, path: PathType) -> "FitConfig":
        """Load a FitConfig from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Preprocess fraction strings in filtering.obj_weights
        if "filtering" in data and isinstance(data["filtering"], dict):
            for field_name in ["obj_weights"]:
                if field_name in data["filtering"] and isinstance(data["filtering"][field_name], dict):
                    result = {}
                    for key, value in data["filtering"][field_name].items():
                        if isinstance(value, str):
                            result[key] = cls._evaluate_fraction(value)
                        else:
                            result[key] = float(value)
                    data["filtering"][field_name] = result

        return cls(**data)
