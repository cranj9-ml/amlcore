"""Tests for amlcore.rules: each scenario's boundary, and the weighted score."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from amlcore.rules import RULE_NAMES, rule_flags, rule_score

QUIET = {  # a row that fires no rule
    "amount_usd": 500.0, "src_out_count_24h": 1, "dst_in_count_24h": 1,
    "pass_through_ratio_24h": np.nan, "currency_mismatch": 0, "src_age_secs": 172800.0,
}


def row(**overrides: float) -> pd.DataFrame:
    return pd.DataFrame([{**QUIET, **overrides}])


def fired(feats: pd.DataFrame, config) -> list[str]:
    flags = rule_flags(feats, config.rules.thresholds).iloc[0]
    return [name for name in RULE_NAMES if flags[f"flag_{name}"]]


def test_quiet_row_fires_nothing(config) -> None:
    assert fired(row(), config) == []


@pytest.mark.parametrize("overrides, expected", [
    ({"amount_usd": 9000.0}, ["structuring"]),
    ({"amount_usd": 10000.0}, []),  # the band is [9000, 10000)
    ({"src_out_count_24h": 10}, ["velocity_24h"]),
    ({"dst_in_count_24h": 10}, ["fan_in_24h"]),
    ({"pass_through_ratio_24h": 1.2}, ["pass_through"]),
    ({"pass_through_ratio_24h": 1.21}, []),
    ({"currency_mismatch": 1}, ["cross_currency"]),
    ({"src_age_secs": 3600.0, "amount_usd": 10000.0}, ["new_account_large"]),
    ({"src_age_secs": np.nan, "amount_usd": 10000.0}, []),  # age unknown during warm-up
])
def test_each_rule_boundary(config, overrides: dict[str, float], expected: list[str]) -> None:
    assert fired(row(**overrides), config) == expected


def test_weighted_score_by_hand(config) -> None:
    # structuring (2) + cross_currency (1) + pass_through (2)
    feats = row(amount_usd=9500.0, currency_mismatch=1, pass_through_ratio_24h=1.0)
    assert rule_score(feats, config.rules).iloc[0] == 5.0


def test_weights_must_cover_every_rule(config) -> None:
    rules = replace(config.rules, weights={"structuring": 1.0})
    with pytest.raises(ValueError, match="exactly"):
        rule_score(row(), rules)
