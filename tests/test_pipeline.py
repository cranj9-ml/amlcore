"""End-to-end test: the bundled synthetic demo through the CLI's `run` command.

This proves the plumbing works; the synthetic schemes are easy, so the scores say
nothing about real-world performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from amlcore.cli import main
from amlcore.config import DEFAULT_CONFIG_PATH, DEMO_ROOT

OUTPUTS = ["report.md", "metrics.json", "typology_recall.csv", "alerts_test.csv",
           "shap_importance.csv", "recall_vs_budget.png", "shap_importance.png"]


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the demo once through a config whose outputs all go to a temp folder."""
    root = tmp_path_factory.mktemp("e2e")
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["paths"] = {"raw_dir": str(DEMO_ROOT / "raw"), "processed_dir": str(root / "processed"),
                    "models_dir": str(root / "models"), "reports_dir": str(root / "reports")}
    raw["data"] = {"variant": "DEMO", "end": None}
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    assert main(["run", "--config", str(config_path)]) == 0
    return root


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
