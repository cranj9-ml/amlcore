"""End-to-end test: the bundled synthetic demo through the CLI's `run` command.

This proves the plumbing works; the synthetic schemes are easy, so the scores say
nothing about real-world performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from amlcore.features import META_COLUMNS

OUTPUTS = ["report.md", "metrics.json", "typology_recall.csv", "alerts_test.csv",
           "shap_importance.csv", "recall_vs_budget.png", "shap_importance.png",
           "scores_test.parquet", "features_top_test.parquet", "typical_values.csv", "daily.csv"]


@pytest.fixture(scope="module")
def out(demo_run: Path) -> Path:
    """The folder holding the demo run's processed, models and reports subfolders."""
    return demo_run.parent


def test_every_output_is_written(out: Path) -> None:
    for name in OUTPUTS:
        assert (out / "reports" / "DEMO" / name).exists(), name
    assert (out / "processed" / "DEMO" / "features.parquet").exists()
    assert (out / "models" / "DEMO" / "model.txt").exists()


def test_model_beats_the_rules_on_test(out: Path) -> None:
    metrics = json.loads((out / "reports" / "DEMO" / "metrics.json").read_text())
    assert metrics["test"]["lightgbm"]["pr_auc"] > metrics["test"]["rules"]["pr_auc"]
    assert metrics["test"]["lightgbm"]["pr_auc"] > 5 * metrics["test"]["lightgbm"]["base_rate"]


def test_alert_queue_has_three_reasons(out: Path) -> None:
    alerts = pd.read_csv(out / "reports" / "DEMO" / "alerts_test.csv")
    assert len(alerts) >= 1
    assert alerts["reason_1"].notna().all()


def test_report_leads_with_the_headline(out: Path) -> None:
    text = (out / "reports" / "DEMO" / "report.md").read_text()
    assert text.index("## Headline") < text.index("## Every budget")
    assert "accuracy" not in text.lower()


def test_dashboard_outputs_cover_test_and_the_top_rows(out: Path) -> None:
    reports = out / "reports" / "DEMO"
    feats = pd.read_parquet(out / "processed" / "DEMO" / "features.parquet")
    test = feats[feats["split"] == "test"]
    scores = pd.read_parquet(reports / "scores_test.parquet")
    assert len(scores) == len(test) and scores["is_laundering"].sum() == test["is_laundering"].sum()
    assert {"model_score", "rule_score", "flag_structuring"} <= set(scores.columns)
    top = pd.read_parquet(reports / "features_top_test.parquet")
    assert len(top) == max(1, int(0.05 * len(test)))
    assert set(top["txn_id"]) <= set(scores.nlargest(len(top), "model_score")["txn_id"])
    assert len(top.columns) == 1 + 45 and not set(META_COLUMNS[1:]) & set(top.columns)
    daily = pd.read_csv(reports / "daily.csv")
    assert daily["transactions"].sum() == len(feats)
    typical = pd.read_csv(reports / "typical_values.csv", index_col="feature")
    assert len(typical) == 45
