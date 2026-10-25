"""The two steps of a run: build features, then train, evaluate, explain and report.

  features:  raw files → transactions → 45 point-in-time features
             → data/processed/<variant>/features.parquet
  train:     LightGBM on train (early stopping on val), the rules baseline, both scored on
             val and test → reports/<variant>/ and models/<variant>/model.txt

Outputs in reports/<variant>/:
  report.md            the human-readable summary, headline first
  metrics.json         every number, for scripts
  typology_recall.csv  recall per laundering pattern, test, both scorers
  alerts_test.csv      the test alert queue: top 0.5% with three reasons each
  shap_importance.csv  mean |SHAP| per feature, on a sample of test
  recall_vs_budget.png, shap_importance.png
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from amlcore.config import Config, parse_window
from amlcore.data import SPLITS, TEST, TRAIN, VAL, load_transactions
from amlcore.explain import alert_queue, global_importance, shap_contributions, typical_values
from amlcore.features import build_features, feature_columns
from amlcore.metrics import alert_count, evaluate, recall_by_typology, recall_interval, selection_probability
from amlcore.model import fit_lightgbm, predict_scores
from amlcore.rules import rule_score

RULES, MODEL = "rules", "lightgbm"
SCORERS = (RULES, MODEL)
EVAL_SPLITS = (VAL, TEST)
SHAP_SAMPLE = 10_000  # rows for the global SHAP summary; mean |SHAP| is stable well before this
CURVE_BUDGETS = np.geomspace(0.0005, 0.05, 41)  # 0.05% to 5% of transactions, for the chart


def features_path(config: Config) -> Path:
    return config.paths.processed_dir / config.variant / "features.parquet"


def report_dir(config: Config) -> Path:
    return config.paths.reports_dir / config.variant


def run_features(config: Config) -> pd.DataFrame:
    """Build the feature table from the raw files and save it as parquet."""
    feats = build_features(load_transactions(config), config)
    path = features_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(path, index=False)
    return feats


@dataclass(frozen=True)
class RunResult:
    """Everything the report is built from."""

    metrics: dict[str, Any]
    typology: pd.DataFrame
    alerts: pd.DataFrame
    importance: pd.DataFrame
    curve: pd.DataFrame
    out_dir: Path


def run_train(config: Config, feats: pd.DataFrame | None = None) -> RunResult:
    """Train, score both scorers on val and test, explain the test alerts, write every output."""
    if feats is None:
        feats = pd.read_parquet(features_path(config))
    columns = feature_columns(feats)
    parts = {name: feats[feats["split"] == name] for name in SPLITS}
    train, val, test = parts[TRAIN], parts[VAL], parts[TEST]
    model = fit_lightgbm(train[columns], train["is_laundering"], val[columns], val["is_laundering"],
                         config.model)

    scores = {split: {RULES: rule_score(parts[split], config.rules).to_numpy(),
                      MODEL: predict_scores(model, parts[split][columns])}
              for split in EVAL_SPLITS}
    ev = config.evaluation
    metrics: dict[str, Any] = {
        "model": {"trees": int(model.best_iteration_ or model.n_estimators), "features": len(columns),
                  "scale_pos_weight": float(model.get_params()["scale_pos_weight"])},
        "data": _data_summary(parts),
        **{split: {s: evaluate(parts[split]["is_laundering"].to_numpy(), scores[split][s],
                               ev.alert_budgets, parts[split]["src_id"].to_numpy())
                   for s in SCORERS} for split in EVAL_SPLITS},
    }
    y_test = test["is_laundering"].to_numpy()
    metrics["test_recall_interval"] = {
        s: recall_interval(y_test, scores[TEST][s], test["timestamp"], ev.primary_budget,
                           ev.bootstrap_block, ev.bootstrap_resamples)
        for s in SCORERS
    }

    age_cap = parse_window(config.features.age_cap).total_seconds()
    alerts = alert_queue(model, columns, test, scores[TEST][MODEL], typical_values(train, columns),
                         config.rules, ev.primary_budget, config.currencies, age_cap)
    sample = test.sample(min(SHAP_SAMPLE, len(test)), random_state=config.model.seed)
    result = RunResult(
        metrics=metrics,
        typology=_typology_table(test, scores[TEST], ev.primary_budget),
        alerts=alerts,
        importance=global_importance(shap_contributions(model, sample[columns])[0]),
        curve=_recall_curve(y_test, scores[TEST]),
        out_dir=report_dir(config),
    )
    model_file = config.paths.models_dir / config.variant / "model.txt"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(model_file)
    write_outputs(result, config)
    return result


def _data_summary(parts: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Period, size and base rate of each split, for the top of the report."""
    return [{"split": name, "start": str(p["timestamp"].min()), "end": str(p["timestamp"].max()),
             "transactions": len(p), "laundering": int(p["is_laundering"].sum()),
             "base_rate": float(p["is_laundering"].mean())}
            for name, p in parts.items()]


def _typology_table(test: pd.DataFrame, scores: dict[str, np.ndarray], budget: float) -> pd.DataFrame:
    """Test recall per typology at the primary budget, one column per scorer."""
    y, typology = test["is_laundering"].to_numpy(), test["typology"].astype(str).to_numpy()
    tables = {s: recall_by_typology(y, scores[s], typology, budget) for s in SCORERS}
    out = tables[MODEL][["positives"]].copy()
    for s in SCORERS:
        out[f"recall_{s}"] = tables[s]["recall"]
    return out.reset_index()


def _recall_curve(y: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Recall at many budgets for each scorer, plus the best any ranking could do."""
    n, n_pos = len(y), y.sum()
    rows = []
    for budget in CURVE_BUDGETS:
        k = alert_count(n, budget)
        row = {"budget": budget, "perfect": min(1.0, k / n_pos)}
        row.update({s: float(selection_probability(scores[s], k) @ y) / n_pos for s in SCORERS})
        rows.append(row)
    return pd.DataFrame(rows)


# --- Writing outputs --------------------------------------------------------------------------


def write_outputs(result: RunResult, config: Config) -> None:
    out = result.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(_jsonable(result.metrics), indent=2), encoding="utf-8")
    result.typology.to_csv(out / "typology_recall.csv", index=False)
    result.alerts.to_csv(out / "alerts_test.csv", index=False)
    result.importance.to_csv(out / "shap_importance.csv", index=False)
    (out / "report.md").write_text(render_report(result, config), encoding="utf-8")
    from amlcore.charts import chart_importance, chart_recall_vs_budget  # matplotlib only when writing

    chart_recall_vs_budget(result.curve, config.evaluation.primary_budget, out / "recall_vs_budget.png")
    chart_importance(result.importance, out / "shap_importance.png")


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, numpy numbers and NaN into plain JSON (NaN becomes null)."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.{digits}%}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def _at(evaluation: dict[str, Any], budget: float) -> Any:
    return next(b for b in evaluation["at_budget"] if b.budget == budget)


def render_report(result: RunResult, config: Config) -> str:
    """The human-readable report: headline first, details after."""
    m, budget = result.metrics, config.evaluation.primary_budget
    data_rows = [[d["split"], f"{d['start'][:16]} to {d['end'][:16]}", f"{d['transactions']:,}",
                  f"{d['laundering']:,}", _pct(d["base_rate"], 3)] for d in m["data"]]

    headline = []
    for s in SCORERS:
        ev, ci = m[TEST][s], m["test_recall_interval"][s]
        at, acc = _at(ev, budget), _at(ev["accounts"], budget)
        headline.append([s, _pct(at.recall), f"{_pct(ci.low)} to {_pct(ci.high)}", _pct(at.precision),
                         f"{ev['pr_auc']:.4f}", _pct(acc.recall), _pct(acc.precision)])

    budget_rows = [[split, s, _pct(at.budget), f"{at.k:,}", _pct(at.precision), _pct(at.recall)]
                   for split in EVAL_SPLITS for s in SCORERS for at in m[split][s]["at_budget"]]
    typology_rows = [[r.typology, f"{r.positives:,}", _pct(r.recall_rules), _pct(r.recall_lightgbm)]
                     for r in result.typology.itertuples()]
    shap_rows = [[r.feature, f"{r.mean_abs_shap:.3f}"] for r in result.importance.head(10).itertuples()]
    reason_share = result.alerts["reason_1"].str.replace(r"[0-9][0-9,.$]*", "N", regex=True)
    reason_rows = [[text, _pct(share)] for text, share in reason_share.value_counts(normalize=True).head(5).items()]

    ci = m["test_recall_interval"][MODEL]
    sections = [
        f"# Results: {config.variant}",
        f"LightGBM ({m['model']['trees']} trees, {m['model']['features']} point-in-time features, "
        f"scale_pos_weight {m['model']['scale_pos_weight']:.1f}) against the six-scenario rules "
        "baseline. Trained on train, early stopping on validation; test was scored once.",
        "## Data\n\n" + _table(["Split", "Period", "Transactions", "Laundering", "Base rate"], data_rows),
        f"## Headline: test, {budget:.1%} alert budget\n\n"
        + _table(["Scorer", "Recall", f"95% CI ({ci.n_blocks} blocks)", "Precision", "PR-AUC",
                  "Account recall", "Account precision"], headline),
        "## Every budget\n\n" + _table(["Split", "Scorer", "Budget", "Alerts", "Precision", "Recall"], budget_rows),
        f"## Recall by typology (test, {budget:.1%})\n\n"
        + _table(["Typology", "Positives", "Rules", "LightGBM"], typology_rows),
        "## What moves the score (mean absolute SHAP, test sample)\n\n" + _table(["Feature", "Mean abs SHAP"], shap_rows),
        "## Most common first reason on test alerts\n\n" + _table(["Reason 1", "Share of alerts"], reason_rows),
        "## Notes\n\n" + "\n".join([
            "- Validation was used for early stopping, so validation results are slightly optimistic.",
            "- Precision and recall at a budget are expected values with ties at the cut-off broken at "
            "random (this matters for the coarse rule score).",
            "- The interval resamples whole time blocks of test, because one scheme's payments cluster in time.",
            "- Account level: a sending account is laundering if any of its payments is, scored by its "
            "highest payment score.",
        ]),
    ]
    return "\n\n".join(sections) + "\n"
