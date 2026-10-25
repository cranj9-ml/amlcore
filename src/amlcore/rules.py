"""Rules-based baseline: six classic transaction-monitoring scenarios.

This is what most banks run today, so it is the bar any model has to clear. Each rule
flags transactions matching a known red flag; rule_score is the weighted sum of the
flags, which ranks alerts so the rules are judged at the same alert budget as the model.
Thresholds and weights live in configs/default.yaml and are set a priori from common
practice, never tuned on this data.

Rules read only point-in-time features (features.py), so they obey the same no-leakage
guarantee as the model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from amlcore.config import RulesConfig, RuleThresholds


@dataclass(frozen=True)
class Rule:
    """A named scenario: what it looks for, and how to flag it from the feature table."""

    name: str
    description: str
    flag: Callable[[pd.DataFrame, RuleThresholds], pd.Series]


def _structuring(f: pd.DataFrame, t: RuleThresholds) -> pd.Series:
    """Amount in [min, max): payments sized to stay under the $10k reporting threshold."""
    return (f["amount_usd"] >= t.structuring_min_usd) & (f["amount_usd"] < t.structuring_max_usd)


def _velocity_24h(f: pd.DataFrame, t: RuleThresholds) -> pd.Series:
    """The sender made many payments in the last 24 hours."""
    return f["src_out_count_24h"] >= t.velocity_min_count


def _fan_in_24h(f: pd.DataFrame, t: RuleThresholds) -> pd.Series:
    """The receiver got many payments in the last 24 hours (a collection account).

    This counts incoming payments, not distinct senders, so it is a proxy for fan-in.
    """
    return f["dst_in_count_24h"] >= t.fan_in_min_count


def _pass_through(f: pd.DataFrame, t: RuleThresholds) -> pd.Series:
    """The payment roughly equals what the sender just received: money in, money out.

    NaN (nothing received in 24h) compares as False, so it never fires without inflows.
    """
    ratio = f["pass_through_ratio_24h"]
    return (ratio >= t.pass_through_min_ratio) & (ratio <= t.pass_through_max_ratio)


def _cross_currency(f: pd.DataFrame, t: RuleThresholds) -> pd.Series:
    """Paid in one currency and received in another: a common layering step."""
    return f["currency_mismatch"] == 1


def _new_account_large(f: pd.DataFrame, t: RuleThresholds) -> pd.Series:
    """A large payment from an account first seen very recently.

    Age is missing during the data's first 48h (features.account_age_seconds); missing
    compares as False, so the rule can't fire while every account looks new only because
    the data just began.
    """
    is_new = f["src_age_secs"] < t.new_account_max_age_hours * 3600
    return is_new & (f["amount_usd"] >= t.new_account_min_usd)


RULES = (
    Rule("structuring", "Payment just under the $10k reporting threshold", _structuring),
    Rule("velocity_24h", "Sender made many payments in the last 24h", _velocity_24h),
    Rule("fan_in_24h", "Receiver got many payments in the last 24h", _fan_in_24h),
    Rule("pass_through", "Payment is 80-120% of the sender's 24h inflow", _pass_through),
    Rule("cross_currency", "Paid and received in different currencies", _cross_currency),
    Rule("new_account_large", "Large payment from a newly seen account", _new_account_large),
)
RULE_NAMES = tuple(rule.name for rule in RULES)


def rule_flags(feats: pd.DataFrame, thresholds: RuleThresholds) -> pd.DataFrame:
    """One 0/1 column per rule, named flag_<rule>."""
    return pd.DataFrame(
        {f"flag_{rule.name}": rule.flag(feats, thresholds).astype("int8") for rule in RULES},
        index=feats.index,
    )


def rule_score(feats: pd.DataFrame, config: RulesConfig) -> pd.Series:
    """Weighted sum of the flags. Every rule must have exactly one weight."""
    if set(config.weights) != set(RULE_NAMES):
        raise ValueError(f"rule weights must be given for exactly {list(RULE_NAMES)}, "
                         f"got {sorted(config.weights)}")
    flags = rule_flags(feats, config.thresholds)
    score = sum(config.weights[name] * flags[f"flag_{name}"] for name in RULE_NAMES)
    return pd.Series(score, index=feats.index, name="rule_score", dtype="float64")
