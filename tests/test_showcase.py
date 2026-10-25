"""Tests for amlcore.showcase, the dashboard's data logic, on one demo run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from amlcore import showcase as sc
from amlcore.config import PathsConfig, load_config
from amlcore.explain import raw_scores
from amlcore.metrics import alert_count
from amlcore.pipeline import MODEL, RULES


@pytest.fixture(scope="module")
def config(demo_run: Path):
    return load_config(demo_run)


@pytest.fixture(scope="module")
def run(config) -> sc.Run:
    return sc.load_run(config)


def test_budget_metrics_match_the_saved_report(run: sc.Run, demo_run: Path) -> None:
    metrics = json.loads((demo_run.parent / "reports" / "DEMO" / "metrics.json").read_text())
    live = sc.budget_metrics(run, 0.005)
    for scorer in (RULES, MODEL):
        saved = next(b for b in metrics["test"][scorer]["at_budget"] if b["budget"] == 0.005)
        assert live[scorer]["recall"] == pytest.approx(saved["recall"])
        assert live[scorer]["precision"] == pytest.approx(saved["precision"])
        assert live[scorer]["alerts"] == saved["k"]


def test_recall_curve_never_falls_and_stays_under_perfect(run: sc.Run) -> None:
    curve = sc.recall_curve(run, np.geomspace(0.0005, 0.05, 15))
    for scorer in (RULES, MODEL, "perfect"):
        assert curve[scorer].is_monotonic_increasing
    assert (curve[MODEL] <= curve["perfect"] + 1e-12).all()


@pytest.mark.parametrize("by", ["typology", "payment_format"])
def test_breakdowns_cover_every_positive(run: sc.Run, by: str) -> None:
    table = sc.breakdown(run, 0.05, by)
    assert table["positives"].sum() == run.y.sum()
    assert table[[f"recall_{MODEL}", f"recall_{RULES}"]].stack().between(0, 1).all()


def test_breakdown_rejects_unknown_grouping(run: sc.Run) -> None:
    with pytest.raises(ValueError):
        sc.breakdown(run, 0.05, "bank")


def test_rule_table_matches_the_flags(run: sc.Run) -> None:
    table = sc.rule_table(run).set_index("rule")
    assert len(table) == 6
    flags = run.scores["flag_structuring"].to_numpy() == 1
    assert table.loc["structuring", "fires_on"] == pytest.approx(flags.mean())
    assert table.loc["structuring", "laundering_flagged"] == pytest.approx(run.y[flags].sum() / run.y.sum())


def test_queue_is_the_top_k_by_score(run: sc.Run) -> None:
    q = sc.queue(run, 0.05)
    assert len(q) == alert_count(len(run.scores), 0.05)
    assert q["model_score"].is_monotonic_decreasing
    assert q["model_score"].min() >= run.scores["model_score"].nlargest(len(q)).min()
    with pytest.raises(ValueError, match="up to"):
        sc.queue(run, 0.10)


def test_explain_alert_is_exact_and_readable(run: sc.Run, config) -> None:
    txn_id = int(sc.queue(run, 0.05)["txn_id"].iloc[0])
    ex = sc.explain_alert(run, txn_id, config)
    x = run.features_top.loc[[txn_id], run.booster.feature_name()]
    assert ex.contributions.sum() + ex.expected_value == pytest.approx(raw_scores(run.booster, x)[0], abs=1e-6)
    assert ex.contributions.abs().is_monotonic_decreasing
    assert 1 <= len(ex.reasons) <= 3 and all(r.contribution > 0 for r in ex.reasons)
    with pytest.raises(KeyError):
        sc.explain_alert(run, -1, config)


def test_reason_shares_sum_to_one(run: sc.Run, config) -> None:
    shares = sc.reason_shares(run, 0.05, config)
    assert shares.sum() == pytest.approx(1.0)
    assert not shares.index.str.contains(r"\$[0-9]").any()  # amounts are masked


def test_load_run_names_the_command_when_files_are_missing(tmp_path: Path, demo_run: Path) -> None:
    empty = replace(load_config(demo_run), paths=PathsConfig(tmp_path, tmp_path, tmp_path, tmp_path))
    with pytest.raises(FileNotFoundError, match="amlcore.cli run"):
        sc.load_run(empty)
