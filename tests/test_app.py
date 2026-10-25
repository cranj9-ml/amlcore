"""Smoke tests for app/dashboard.py with Streamlit's AppTest: it renders every tab without error."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
TABS = ["Overview", "Alert queue", "What drives it", "Rules baseline", "Data"]


def test_dashboard_renders_a_saved_run(demo_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMLCORE_CONFIG", str(demo_run))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == TABS
    labels = [m.label for m in at.metric]
    assert "LightGBM recall" in labels and "Rules recall" in labels
    assert any("Why it was alerted" in m.value for m in at.markdown)


def test_moving_the_budget_changes_the_alert_count(demo_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMLCORE_CONFIG", str(demo_run))
    at = AppTest.from_file(APP, default_timeout=120).run()

    def alerts() -> str:
        return next(m.value for m in at.metric if m.label == "Alerts at this budget")

    small = alerts()
    at.sidebar.select_slider[0].set_value(0.05).run()
    assert not at.exception
    assert int(alerts().replace(",", "")) > int(small.replace(",", ""))


def test_missing_run_shows_how_to_make_one(tmp_path: Path, demo_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = yaml.safe_load(demo_run.read_text())
    raw["paths"].update(reports_dir=str(tmp_path / "reports"), models_dir=str(tmp_path / "models"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("AMLCORE_CONFIG", str(config_path))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.info and "Train on demo data" in at.info[0].value
