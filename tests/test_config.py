"""Tests for amlcore.config: the default loads, and bad values fail at load time."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from amlcore.config import DEFAULT_CONFIG_PATH, demo_config, load_config, parse_window


def test_default_config_loads() -> None:
    config = load_config()
    assert config.data.end == pd.Timestamp("2022-09-11")
    assert config.evaluation.primary_budget == 0.005
    assert config.currencies == sorted(config.currencies)


def test_demo_config_points_at_the_bundled_data() -> None:
    demo = demo_config(load_config())
    assert demo.trans_path.exists() and demo.data.end is None


@pytest.mark.parametrize("window, expected", [("30m", pd.Timedelta(minutes=30)), ("7d", pd.Timedelta(days=7))])
def test_parse_window(window: str, expected: pd.Timedelta) -> None:
    assert parse_window(window) == expected


@pytest.mark.parametrize("bad", ["24hrs", "h", "0h", "-1h"])
def test_bad_windows_raise(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_window(bad)


@pytest.mark.parametrize("section, key, value", [
    ("split", "train_frac", 0.9),  # leaves no test split
    ("model", "class_weighting", "balanced"),
    ("evaluation", "primary_budget", 0.02),  # not one of the budgets
    ("features", "windows", ["1h", "7d"]),  # pass-through and rules need 24h
])
def test_bad_values_fail_at_load(tmp_path: Path, section: str, key: str, value) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw[section][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)
