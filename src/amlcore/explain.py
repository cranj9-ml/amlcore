"""Reason codes: why a transaction was alerted, in words an investigator can check.

Why: an alert nobody can explain doesn't get worked, and a model nobody can explain
doesn't get approved. Each alert gets its top three reasons, plus any rule scenarios that
fired, so the investigator starts from evidence rather than a bare score.

How: TreeSHAP, computed by LightGBM itself (pred_contrib=True), splits each row's raw
score (log-odds) exactly into one contribution per feature plus a constant, the model's
average output. A positive contribution pushed the row towards "laundering". The reasons
are the largest positive contributions, each rendered from a template with the actual
value and the typical (training median) value, e.g. "receiver made 42 payments in the
past 7 days (typical: 3)". Exact, not approximate: contributions plus the constant equal
the raw score for every row, and a test checks it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from amlcore.config import RulesConfig
from amlcore.features import PAYMENT_FORMATS
from amlcore.rules import rule_flags

ROLES = {"src": "sender", "dst": "receiver"}
WINDOWS = {"1h": "past hour", "24h": "past 24 hours", "7d": "past 7 days"}
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
ALERT_COLUMNS = ["txn_id", "timestamp", "src_id", "dst_id", "amount_usd"]


@dataclass(frozen=True)
class Reason:
    """One feature's push towards "laundering", with the text shown to investigators."""

    feature: str
    contribution: float  # SHAP value, in log-odds
    text: str


# --- SHAP -------------------------------------------------------------------------------------


def shap_contributions(model: lgb.LGBMClassifier | lgb.Booster, x: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per-row, per-feature contributions to the raw score, and the constant (expected value).

    Takes the fitted classifier or a booster loaded from model.txt. Uses the trees early
    stopping kept, the same ones that produce the scores.
    """
    booster, best = _booster(model)
    raw = booster.predict(x, pred_contrib=True, num_iteration=best)
    contrib = pd.DataFrame(raw[:, :-1], columns=x.columns, index=x.index)
    return contrib, float(raw[0, -1]) if len(raw) else 0.0


def raw_scores(model: lgb.LGBMClassifier | lgb.Booster, x: pd.DataFrame) -> np.ndarray:
    """The model's log-odds, the quantity SHAP decomposes."""
    booster, best = _booster(model)
    return booster.predict(x, raw_score=True, num_iteration=best)


def _booster(model: lgb.LGBMClassifier | lgb.Booster) -> tuple[lgb.Booster, int | None]:
    """The underlying booster and the iteration to predict with. A booster saved with
    save_model already holds only the trees up to the best iteration."""
    if isinstance(model, lgb.Booster):
        return model, None
    return model.booster_, model.best_iteration_ or None


def global_importance(contrib: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature: how much each feature moves scores on average."""
    imp = contrib.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.to_numpy()})


# --- Templates --------------------------------------------------------------------------------


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _count(x: float) -> str:
    return f"{x:,.0f}"


def _duration(seconds: float) -> str:
    """Seconds as the largest sensible unit: 90 -> "2 minutes", 7200 -> "2 hours"."""
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            n = round(seconds / size)
            return f"{n} {unit}{'s' if n != 1 else ''}"
    return "under a minute"


def _with_typical(text: str, fmt, typical: float | None) -> str:
    if typical is None or (isinstance(typical, float) and math.isnan(typical)):
        return text
    return f"{text} (typical: {fmt(typical)})"


def describe(feature: str, value: float, typical: float | None = None,
             currencies: Sequence[str] = (), age_cap: float | None = None) -> str:
    """One feature's value as a sentence; unknown features fall back to "name = value".

    age_cap (seconds) is where account age is capped, so a capped value reads "at least".
    """
    missing = value is None or (isinstance(value, float) and math.isnan(value))
    if m := re.fullmatch(r"(src|dst)_(in|out)_count_(\w+)", feature):
        verb = "received" if m[2] == "in" else "made"
        return _with_typical(f"{ROLES[m[1]]} {verb} {_count(value)} payments in the "
                             f"{WINDOWS.get(m[3], m[3])}", _count, typical)
    if m := re.fullmatch(r"(src|dst)_(in|out)_sum_(\w+)", feature):
        verb = "received" if m[2] == "in" else "paid out"
        return _with_typical(f"{ROLES[m[1]]} {verb} {_money(value)} in the "
                             f"{WINDOWS.get(m[3], m[3])}", _money, typical)
    if m := re.fullmatch(r"(src|dst)_(in|out)_degree", feature):
        text = (f"{ROLES[m[1]]} had paid {_count(value)} distinct accounts before this" if m[2] == "out"
                else f"{ROLES[m[1]]} had been paid by {_count(value)} distinct accounts before this")
        return _with_typical(text, _count, typical)
    if m := re.fullmatch(r"(src|dst)_secs_since_prev", feature):
        if missing:
            return f"first activity seen for the {ROLES[m[1]]}"
        return _with_typical(f"{ROLES[m[1]]}'s previous activity was {_duration(value)} earlier",
                             _duration, typical)
    if m := re.fullmatch(r"(src|dst)_age_secs", feature):
        role = ROLES[m[1]]
        if missing:  # the data's warm-up period: every account would look young
            return f"{role}'s age unknown this early in the data"
        if value == 0:
            return f"{role} first seen with this payment"
        if age_cap is not None and value >= age_cap:
            return f"{role} first seen at least {_duration(age_cap)} earlier"
        return _with_typical(f"{role} first seen {_duration(value)} earlier", _duration, typical)
    if feature == "pass_through_ratio_24h":
        if missing:
            return "sender received nothing in the past 24 hours"
        return f"pays out {value:.0%} of what the sender received in the past 24 hours"
    if m := re.fullmatch(r"src_out_burst_(\w+)_(\w+)", feature):
        if missing:
            return f"sender made no payments in the {WINDOWS.get(m[2], m[2])}"
        return (f"sender's payments in the {WINDOWS.get(m[1], m[1])} are {value:.1f}x "
                f"its usual rate over the {WINDOWS.get(m[2], m[2])}")
    if feature == "amount_usd":
        return _with_typical(f"amount {_money(value)}", _money, typical)
    if feature == "amount_ratio":
        converted = not missing and abs(value - 1) > 0.01
        return "amount converted between currencies" if converted else "amount unchanged by conversion"
    flags = {
        "currency_mismatch": ("paid and received in different currencies", "paid and received in the same currency"),
        "cross_bank": ("payment between two banks", "payment within one bank"),
        "self_transfer": ("sender pays itself", "sender and receiver differ"),
        "round_amount": ("round amount", "not a round amount"),
    }
    if feature in flags:
        return flags[feature][0] if value else flags[feature][1]
    if feature == "hour":
        return f"made at {int(value):02d}:00"
    if feature == "weekday":
        return f"made on a {WEEKDAYS[int(value)]}"
    if feature == "payment_format_code":
        code = int(value)
        return f"paid by {PAYMENT_FORMATS[code]}" if 0 <= code < len(PAYMENT_FORMATS) else "unknown payment format"
    if m := re.fullmatch(r"(payment|receiving)_currency_code", feature):
        code, verb = int(value), "paid" if m[1] == "payment" else "received"
        return f"{verb} in {currencies[code]}" if 0 <= code < len(currencies) else f"{verb} in an unknown currency"
    shown = "missing" if missing else f"{value:.3g}"
    return _with_typical(f"{feature} = {shown}", lambda t: f"{t:.3g}", typical)


# --- Reasons per alert ------------------------------------------------------------------------


def top_reasons(values: pd.Series, contrib: pd.Series, typical: pd.Series, k: int = 3,
                currencies: Sequence[str] = (), age_cap: float | None = None) -> list[Reason]:
    """The k features that pushed this row hardest towards laundering (positive SHAP only).

    Negative contributions are evidence against laundering; they don't explain an alert.
    """
    positive = contrib[contrib > 0].sort_values(ascending=False).head(k)
    return [Reason(f, float(c), describe(f, values[f], typical.get(f), currencies, age_cap))
            for f, c in positive.items()]


def typical_values(train: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Median of each feature over training rows: the "typical" shown next to a value."""
    return train[list(columns)].median(numeric_only=True)


def alert_queue(model: lgb.LGBMClassifier, columns: Sequence[str], part: pd.DataFrame,
                scores: np.ndarray, typical: pd.Series, rules: RulesConfig, budget: float,
                currencies: Sequence[str] = (), age_cap: float | None = None, k: int = 3) -> pd.DataFrame:
    """The top `budget` share of `part` by score, each with its reasons and the rules that fired.

    SHAP runs on the alerted rows only, since it costs milliseconds per row. Labels are
    kept as the last columns so the demo can show hits and misses; a real queue wouldn't
    have them.
    """
    n_alerts = max(1, math.floor(budget * len(part) + 1e-9))
    top = part.assign(score=scores).sort_values("score", ascending=False, kind="stable").head(n_alerts)
    contrib, _ = shap_contributions(model, top[list(columns)])
    reasons = [top_reasons(top.loc[i, list(columns)], contrib.loc[i], typical, k, currencies, age_cap)
               for i in top.index]
    flags = rule_flags(top, rules.thresholds)
    fired = [", ".join(c.removeprefix("flag_") for c in flags.columns if row[c]) for _, row in flags.iterrows()]

    out = top[ALERT_COLUMNS + ["score"]].reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    for i in range(k):
        out[f"reason_{i + 1}"] = [r[i].text if i < len(r) else "" for r in reasons]
    out["rules_fired"] = fired
    out["is_laundering"] = top["is_laundering"].to_numpy()
    out["typology"] = top["typology"].astype(str).to_numpy()
    return out
