"""The dashboard's data logic: everything app/dashboard.py shows, as plain tested functions.

The dashboard reads only a run's saved outputs (pipeline.py writes them), never the raw
data or the full feature table, so it opens in seconds even on 5 million transactions.
Every number is recomputed from scores_test.parquet at whatever budget the viewer picks,
with the same tie-aware metrics as the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from amlcore.config import Config, parse_window
from amlcore.explain import Reason, shap_contributions, top_reasons
from amlcore.features import PAYMENT_FORMATS
from amlcore.metrics import alert_count, at_budget, recall_by_typology, selection_probability
from amlcore.pipeline import MAX_DASHBOARD_BUDGET, MODEL, RULES, report_dir
from amlcore.rules import RULES as RULE_LIST

RUN_FILES = ["metrics.json", "scores_test.parquet", "features_top_test.parquet", "typical_values.csv",
             "daily.csv", "shap_importance.csv"]
SCORE_OF = {RULES: "rule_score", MODEL: "model_score"}


@dataclass(frozen=True)
class Run:
    """One saved run, loaded for the dashboard."""

    variant: str
    metrics: dict[str, Any]
    scores: pd.DataFrame  # one row per test transaction
    features_top: pd.DataFrame  # features of the top-scoring test rows, indexed by txn_id
    typical: pd.Series
    daily: pd.DataFrame
    importance: pd.DataFrame
    booster: lgb.Booster

    @property
    def y(self) -> np.ndarray:
        return self.scores["is_laundering"].to_numpy()


def model_path(config: Config) -> Path:
    return config.paths.models_dir / config.variant / "model.txt"


def load_run(config: Config) -> Run:
    """Read a run's outputs; a missing file raises with the command that writes it."""
    out = report_dir(config)
    missing = [name for name in RUN_FILES if not (out / name).exists()]
    if missing or not model_path(config).exists():
        demo = " --demo" if config.variant == "DEMO" else ""
        raise FileNotFoundError(f"no saved run for {config.variant} in {out} "
                                f"(missing {missing or ['model.txt']}); run `python -m amlcore.cli run{demo}`")
    return Run(
        variant=config.variant,
        metrics=json.loads((out / "metrics.json").read_text(encoding="utf-8")),
        scores=pd.read_parquet(out / "scores_test.parquet"),
        features_top=pd.read_parquet(out / "features_top_test.parquet").set_index("txn_id"),
        typical=pd.read_csv(out / "typical_values.csv", index_col="feature")["typical"],
        daily=pd.read_csv(out / "daily.csv", parse_dates=["day"]),
        importance=pd.read_csv(out / "shap_importance.csv"),
        booster=lgb.Booster(model_file=str(model_path(config))),
    )


# --- Metrics at any budget ---------------------------------------------------------------------


def budget_metrics(run: Run, budget: float) -> dict[str, dict[str, float]]:
    """Alerts, expected hits, precision and recall for rules and model at `budget`."""
    out = {}
    for scorer, column in SCORE_OF.items():
        r = at_budget(run.y, run.scores[column].to_numpy(), budget)
        out[scorer] = {"alerts": r.k, "hits": r.hits, "precision": r.precision, "recall": r.recall}
    return out


def recall_curve(run: Run, budgets: np.ndarray) -> pd.DataFrame:
    """Recall at each budget for both scorers, plus the best any ranking could do."""
    n, n_pos = len(run.y), run.y.sum()
    rows = []
    for budget in budgets:
        k = alert_count(n, budget)
        row = {"budget": float(budget), "perfect": min(1.0, k / n_pos)}
        for scorer, column in SCORE_OF.items():
            row[scorer] = float(selection_probability(run.scores[column].to_numpy(), k) @ run.y) / n_pos
        rows.append(row)
    return pd.DataFrame(rows)


def payment_formats(codes: pd.Series) -> pd.Series:
    """Payment format names from their integer codes."""
    names = dict(enumerate(PAYMENT_FORMATS))
    return codes.map(names).fillna("unknown")


def breakdown(run: Run, budget: float, by: str) -> pd.DataFrame:
    """Recall at `budget` for each typology (by="typology") or payment format (by="payment_format"),
    rules against model, largest groups first."""
    if by == "typology":
        groups = run.scores["typology"].to_numpy()
    elif by == "payment_format":
        groups = payment_formats(run.scores["payment_format_code"]).to_numpy()
    else:
        raise ValueError(f"cannot break down by {by!r}")
    tables = {s: recall_by_typology(run.y, run.scores[c].to_numpy(), groups, budget) for s, c in SCORE_OF.items()}
    out = tables[MODEL][["positives"]].copy()
    for scorer in SCORE_OF:
        out[f"recall_{scorer}"] = tables[scorer]["recall"]
    return out.rename_axis(by).reset_index()


def rule_table(run: Run) -> pd.DataFrame:
    """Each rule on the test period: how often it fires, how often it's right, what it catches."""
    rows = []
    for rule in RULE_LIST:
        fired = run.scores[f"flag_{rule.name}"].to_numpy() == 1
        hits = int(run.y[fired].sum())
        rows.append({
            "rule": rule.name, "description": rule.description,
            "fires_on": fired.mean(),
            "precision_when_fired": hits / fired.sum() if fired.any() else float("nan"),
            "laundering_flagged": hits / run.y.sum(),
        })
    return pd.DataFrame(rows)


# --- The alert queue and one alert explained ------------------------------------------------------


def queue(run: Run, budget: float) -> pd.DataFrame:
    """The top `budget` share of test rows by model score, with the rules that fired."""
    if budget > MAX_DASHBOARD_BUDGET + 1e-12:
        raise ValueError(f"the saved run explains alerts only up to a {MAX_DASHBOARD_BUDGET:.0%} budget")
    k = alert_count(len(run.scores), budget)
    top = run.scores.sort_values("model_score", ascending=False, kind="stable").head(k)
    flags = [f"flag_{rule.name}" for rule in RULE_LIST]
    fired = top[flags].apply(lambda row: ", ".join(f.removeprefix("flag_") for f in flags if row[f]), axis=1)
    out = top[["txn_id", "timestamp", "amount_usd", "model_score", "rule_score", "is_laundering",
               "typology"]].assign(payment_format=payment_formats(top["payment_format_code"]),
                                   rules_fired=fired.to_numpy())
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class AlertExplanation:
    """One alert's SHAP breakdown: contributions (log-odds), the constant, and the reasons."""

    contributions: pd.Series  # per feature, largest |contribution| first
    expected_value: float
    raw_score: float
    reasons: list[Reason]


def explain_alert(run: Run, txn_id: int, config: Config, k: int = 3) -> AlertExplanation:
    """Exact TreeSHAP for one alerted transaction, and its top-k reasons as sentences."""
    if txn_id not in run.features_top.index:
        raise KeyError(f"transaction {txn_id} is not among the saved top-scoring test rows")
    x = run.features_top.loc[[txn_id], run.booster.feature_name()]
    contrib, expected = shap_contributions(run.booster, x)
    row = contrib.iloc[0]
    reasons = top_reasons(x.iloc[0], row, run.typical, k, config.currencies,
                          parse_window(config.features.age_cap).total_seconds())
    return AlertExplanation(
        contributions=row.reindex(row.abs().sort_values(ascending=False).index),
        expected_value=expected,
        raw_score=float(row.sum() + expected),
        reasons=reasons,
    )


def reason_shares(run: Run, budget: float, config: Config, sample: int = 300) -> pd.Series:
    """Share of alerts by first reason, numbers masked, on up to `sample` top alerts.

    A sample keeps this interactive: SHAP costs milliseconds per row with ~2,000 trees, so
    the rows go through it in one batch.
    """
    ids = queue(run, budget)["txn_id"].head(sample).to_numpy()
    x = run.features_top.loc[ids, run.booster.feature_name()]
    contrib, _ = shap_contributions(run.booster, x)
    age_cap = parse_window(config.features.age_cap).total_seconds()
    firsts = []
    for i in x.index:
        reasons = top_reasons(x.loc[i], contrib.loc[i], run.typical, 1, config.currencies, age_cap)
        firsts.append(reasons[0].text if reasons else "(no positive reason)")
    masked = pd.Series(firsts).str.replace(r"(?<!past )\b[0-9][0-9,.]*", "N", regex=True)
    return masked.value_counts(normalize=True)
