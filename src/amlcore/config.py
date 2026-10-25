"""Load and validate the project configuration (configs/default.yaml).

Why a typed, validated config instead of a raw dict: a typo in a key or a bad value
(say, split fractions that add up to more than 1) fails at load time with a clear
message, rather than as a confusing error halfway through a run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# config.py lives at <root>/src/amlcore/config.py, so the root is two folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEMO_ROOT = PROJECT_ROOT / "data" / "demo"
CLASS_WEIGHTINGS = ("none", "sqrt")


@dataclass(frozen=True)
class PathsConfig:
    """Where data and outputs live, as absolute paths."""

    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class DataConfig:
    """Which AMLworld variant to read, and where its period ends (exclusive; None keeps all)."""

    variant: str
    end: pd.Timestamp | None


@dataclass(frozen=True)
class SplitConfig:
    """Temporal split fractions; test is whatever remains after train and val."""

    train_frac: float
    val_frac: float


@dataclass(frozen=True)
class FeaturesConfig:
    """Look-back windows (strings like "24h", for readable feature names) and the age cap."""

    windows: tuple[str, ...]
    age_cap: str


@dataclass(frozen=True)
class RuleThresholds:
    """Thresholds for the scenario rules in rules.py."""

    structuring_min_usd: float
    structuring_max_usd: float
    velocity_min_count: float
    fan_in_min_count: float
    pass_through_min_ratio: float
    pass_through_max_ratio: float
    new_account_max_age_hours: float
    new_account_min_usd: float


@dataclass(frozen=True)
class RulesConfig:
    """Rule thresholds, and each rule's weight in rule_score (keyed by rule name)."""

    thresholds: RuleThresholds
    weights: dict[str, float]


@dataclass(frozen=True)
class ModelConfig:
    """LightGBM hyperparameters (configs/default.yaml says what each one does)."""

    class_weighting: str  # "none" or "sqrt"
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    early_stopping_rounds: int
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Alert budgets as fractions of transactions (0.005 = the top 0.5%), and the bootstrap."""

    alert_budgets: tuple[float, ...]
    primary_budget: float
    bootstrap_block: str
    bootstrap_resamples: int


@dataclass(frozen=True)
class Config:
    """The full configuration, with shortcuts for the most-used values."""

    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    features: FeaturesConfig
    rules: RulesConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    fx_to_usd: dict[str, float]

    @property
    def variant(self) -> str:
        return self.data.variant

    @property
    def trans_path(self) -> Path:
        return self.paths.raw_dir / f"{self.variant}_Trans.csv"

    @property
    def patterns_path(self) -> Path:
        return self.paths.raw_dir / f"{self.variant}_Patterns.txt"

    @property
    def currencies(self) -> list[str]:
        """The currency vocabulary, sorted so currency codes are the same on every run."""
        return sorted(self.fx_to_usd)


# Our own small parser, because pandas 3 deprecates lowercase "d" in pd.Timedelta("7d").
_WINDOW_UNITS = {"m": "minutes", "h": "hours", "d": "days"}


def parse_window(window: str) -> pd.Timedelta:
    """Parse a window like "30m", "24h" or "7d" into a Timedelta."""
    number, unit = window[:-1], window[-1:]
    if unit not in _WINDOW_UNITS or not number.isdigit() or int(number) == 0:
        raise ValueError(f"cannot parse window {window!r}; use e.g. 30m, 24h, 7d")
    return pd.Timedelta(**{_WINDOW_UNITS[unit]: int(number)})


def load_config(path: str | Path | None = None) -> Config:
    """Read a YAML config file (default: configs/default.yaml) into a validated Config."""
    with open(Path(path) if path is not None else DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return _build_config(raw)


def demo_config(base: Config) -> Config:
    """The base config pointed at the bundled synthetic DEMO data in data/demo.

    data.end is turned off: it trims AMLworld's artificial tail, which the synthetic
    data does not have, and the default cutoff would drop the demo's last days.
    """
    paths = PathsConfig(*(DEMO_ROOT / name for name in ("raw", "processed", "models", "reports")))
    return replace(base, paths=paths, data=DataConfig(variant="DEMO", end=None))


def _build_config(raw: dict[str, Any]) -> Config:
    """Turn the parsed YAML dict into a Config, validating each section."""
    config = Config(
        paths=PathsConfig(**{k: _resolve(v) for k, v in raw["paths"].items()}),
        data=DataConfig(variant=str(raw["data"]["variant"]), end=_parse_end(raw["data"]["end"])),
        split=SplitConfig(float(raw["split"]["train_frac"]), float(raw["split"]["val_frac"])),
        features=FeaturesConfig(
            windows=tuple(str(w) for w in raw["features"]["windows"]),
            age_cap=str(raw["features"]["age_cap"]),
        ),
        rules=_build_rules(raw["rules"]),
        model=_build_model(raw["model"]),
        evaluation=EvaluationConfig(
            alert_budgets=tuple(float(b) for b in raw["evaluation"]["alert_budgets"]),
            primary_budget=float(raw["evaluation"]["primary_budget"]),
            bootstrap_block=str(raw["evaluation"]["bootstrap_block"]),
            bootstrap_resamples=int(raw["evaluation"]["bootstrap_resamples"]),
        ),
        fx_to_usd={str(k): float(v) for k, v in raw["fx_to_usd"].items()},
    )
    _validate(config)
    return config


def _resolve(path: str) -> Path:
    """Make a config path absolute, so commands work from any working directory."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _parse_end(value: Any) -> pd.Timestamp | None:
    """Parse data.end; a typo should fail here, not silently keep or drop data."""
    if value is None:
        return None
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise ValueError(f"cannot parse data.end {value!r}; use e.g. '2022-09-11 00:00'") from exc


def _build_rules(raw: dict[str, Any]) -> RulesConfig:
    """Read the rules section; an unknown or missing threshold name is an error."""
    try:
        thresholds = RuleThresholds(**{k: float(v) for k, v in raw["thresholds"].items()})
    except TypeError as exc:
        raise ValueError(f"bad rules.thresholds: {exc}") from exc
    return RulesConfig(thresholds=thresholds, weights={str(k): float(v) for k, v in raw["weights"].items()})


def _build_model(raw: dict[str, Any]) -> ModelConfig:
    """Read the model section with each value cast to its declared type."""
    types = {"class_weighting": str, "n_estimators": int, "learning_rate": float, "num_leaves": int,
             "min_child_samples": int, "subsample": float, "colsample_bytree": float,
             "reg_lambda": float, "early_stopping_rounds": int, "seed": int}
    missing = set(types) - set(raw)
    if missing:
        raise ValueError(f"model section is missing {sorted(missing)}")
    return ModelConfig(**{name: cast(raw[name]) for name, cast in types.items()})


def _validate(config: Config) -> None:
    """Checks whose failure would otherwise surface as a silent wrong result."""
    s, t, e = config.split, config.rules.thresholds, config.evaluation
    if s.train_frac <= 0 or s.val_frac <= 0 or s.train_frac + s.val_frac >= 1:
        raise ValueError("split fractions must be positive, with train_frac + val_frac < 1")
    for window in (*config.features.windows, config.features.age_cap):
        parse_window(window)
    if "24h" not in config.features.windows:
        raise ValueError("features.windows must include 24h: the pass-through ratio and the rules use it")
    if not t.structuring_min_usd < t.structuring_max_usd:
        raise ValueError("rules: structuring_min_usd must be below structuring_max_usd")
    if not t.pass_through_min_ratio < t.pass_through_max_ratio:
        raise ValueError("rules: pass_through_min_ratio must be below pass_through_max_ratio")
    if any(w < 0 for w in config.rules.weights.values()) or not any(config.rules.weights.values()):
        raise ValueError("rules: weights must be >= 0, with at least one positive")
    if config.model.class_weighting not in CLASS_WEIGHTINGS:
        raise ValueError(f"model.class_weighting must be one of {CLASS_WEIGHTINGS}")
    if not all(0 < b < 1 for b in e.alert_budgets) or e.primary_budget not in e.alert_budgets:
        raise ValueError("alert budgets must be in (0, 1), and primary_budget must be one of them")
    if config.fx_to_usd.get("US Dollar") != 1.0 or any(r <= 0 for r in config.fx_to_usd.values()):
        raise ValueError("fx_to_usd must have 'US Dollar: 1.0' and only positive rates")
