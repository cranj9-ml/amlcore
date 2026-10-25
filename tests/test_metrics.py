"""Tests for amlcore.metrics: tie-aware budgets by hand, and the block-bootstrap interval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amlcore.metrics import (
    account_level,
    alert_count,
    at_budget,
    evaluate,
    pr_auc,
    recall_by_typology,
    recall_interval,
    selection_probability,
)


def test_pr_auc_by_hand() -> None:
    # Ranking: pos, neg, pos. Precision at each positive: 1/1 and 2/3; mean = 5/6.
    assert pr_auc(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])) == pytest.approx(5 / 6)
    assert np.isnan(pr_auc(np.array([0, 0]), np.array([0.1, 0.2])))


@pytest.mark.parametrize("n, budget, k", [(1000, 0.005, 5), (999, 0.005, 4), (10, 0.001, 1)])
def test_alert_count_rounds_down_but_is_at_least_one(n: int, budget: float, k: int) -> None:
    assert alert_count(n, budget) == k


def test_selection_probability_shares_tied_slots() -> None:
    # k = 2: the 0.9 row is in; three rows tie at 0.5 for the one remaining slot.
    prob = selection_probability(np.array([0.9, 0.5, 0.5, 0.5, 0.1]), 2)
    np.testing.assert_allclose(prob, [1, 1 / 3, 1 / 3, 1 / 3, 0])


def test_at_budget_with_ties_by_hand() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    score = np.array([3, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 20% budget -> k = 2
    result = at_budget(y, score, 0.2)
    assert result.k == 2
    assert result.hits == pytest.approx(1 + 1 / 4)  # the 3, plus a 1-in-4 draw among the 1s
    assert result.recall == pytest.approx(1.25 / 3)
    assert result.precision == pytest.approx(1.25 / 2)


def test_recall_by_typology_by_hand() -> None:
    y = np.array([1, 1, 1, 0])
    table = recall_by_typology(y, np.array([4, 3, 1, 2]), np.array(["cycle", "cycle", "stack", "none"]), 0.5)
    assert table.loc["cycle", "recall"] == 1.0
    assert table.loc["stack", "recall"] == 0.0


def test_account_level_takes_any_label_and_max_score() -> None:
    acc = account_level(np.array([7, 7, 8]), np.array([0, 1, 0]), np.array([0.2, 0.1, 0.5]))
    assert acc.loc[7].tolist() == [1, 0.2]


def test_evaluate_reports_base_rates() -> None:
    out = evaluate(np.array([1, 0, 0, 0]), np.array([0.9, 0.1, 0.2, 0.3]), (0.25,), np.array([1, 1, 2, 3]))
    assert out["base_rate"] == 0.25 and out["accounts"]["base_rate"] == pytest.approx(1 / 3)
    assert out["at_budget"][0].recall == 1.0


def test_recall_interval_brackets_the_estimate_and_counts_blocks() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    ts = pd.Series(pd.Timestamp("2022-09-08") + pd.to_timedelta(rng.integers(0, 3 * 24 * 60, n), unit="min"))
    y = (rng.random(n) < 0.01).astype(int)
    score = y * rng.random(n) + rng.random(n) * 0.8  # informative but imperfect
    ci = recall_interval(y, score, ts, 0.01, "6h", n_resamples=200)
    assert ci.n_blocks == 12
    assert ci.low <= ci.estimate <= ci.high
    assert ci.estimate == pytest.approx(at_budget(y, score, 0.01).recall, abs=1e-9)
