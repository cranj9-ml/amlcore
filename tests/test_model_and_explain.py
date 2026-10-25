"""Tests for amlcore.model and amlcore.explain: early stopping, exact SHAP, readable reasons."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from amlcore.explain import alert_queue, describe, raw_scores, shap_contributions, top_reasons
from amlcore.model import fit_lightgbm, positive_weight, predict_scores


@pytest.fixture(scope="module")
def fitted(config):
    """A small model where src_out_count_24h carries the signal and amount_usd is noise."""
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.05).astype(int)
    x = pd.DataFrame({
        "src_out_count_24h": y * 20 + rng.poisson(3, n),
        "amount_usd": rng.lognormal(7, 1, n),
        "payment_format_code": rng.integers(0, 7, n),
    })
    model = fit_lightgbm(x[:3000], pd.Series(y[:3000]), x[3000:], pd.Series(y[3000:]), config.model)
    return model, x, y


def test_positive_weight() -> None:
    y = np.array([1] + [0] * 99)
    assert positive_weight(y, "none") == 1.0
    assert positive_weight(y, "sqrt") == pytest.approx(math.sqrt(99))
    with pytest.raises(ValueError, match="no positives"):
        positive_weight(np.zeros(5), "none")


def test_early_stopping_keeps_fewer_trees_and_ranks_the_signal(fitted, config) -> None:
    model, x, y = fitted
    assert model.best_iteration_ < config.model.n_estimators
    # The small tuned learning rate keeps scores near the base rate, so check the ranking.
    assert roc_auc_score(y, predict_scores(model, x)) > 0.97


def test_shap_values_add_up_to_the_raw_score(fitted) -> None:
    """TreeSHAP is exact: contributions plus the expected value equal the log-odds per row."""
    model, x, _ = fitted
    contrib, expected = shap_contributions(model, x)
    np.testing.assert_allclose(contrib.sum(axis=1) + expected, raw_scores(model, x), atol=1e-6)


def test_the_signal_feature_explains_the_positives(fitted) -> None:
    model, x, y = fitted
    contrib, _ = shap_contributions(model, x[y == 1])
    assert contrib.abs().mean().idxmax() == "src_out_count_24h"


@pytest.mark.parametrize("feature, value, typical, expected", [
    ("dst_out_count_7d", 42, 3, "receiver made 42 payments in the past 7 days (typical: 3)"),
    ("src_in_sum_24h", 12345.6, 800, "sender received $12,346 in the past 24 hours (typical: $800)"),
    ("src_out_degree", 57, 2, "sender had paid 57 distinct accounts before this (typical: 2)"),
    ("src_secs_since_prev", 90, 7200, "sender's previous activity was 2 minutes earlier (typical: 2 hours)"),
    ("dst_age_secs", math.nan, 3600, "receiver's age unknown this early in the data"),
    ("src_age_secs", 0, 172800, "sender first seen with this payment"),
    ("pass_through_ratio_24h", 0.97, None, "pays out 97% of what the sender received in the past 24 hours"),
    ("payment_format_code", 0, None, "paid by ACH"),
    ("currency_mismatch", 1, None, "paid and received in different currencies"),
    ("weekday", 5, None, "made on a Saturday"),
])
def test_describe(feature: str, value: float, typical: float | None, expected: str) -> None:
    assert describe(feature, value, typical) == expected


def test_capped_age_reads_at_least() -> None:
    assert describe("src_age_secs", 172800, None, age_cap=172800) == "sender first seen at least 2 days earlier"


def test_top_reasons_keep_only_positive_contributions_in_order() -> None:
    values = pd.Series({"amount_usd": 9500.0, "hour": 3, "cross_bank": 1})
    contrib = pd.Series({"amount_usd": 0.4, "hour": -0.2, "cross_bank": 0.9})
    reasons = top_reasons(values, contrib, pd.Series(dtype=float))
    assert [r.feature for r in reasons] == ["cross_bank", "amount_usd"]


def test_alert_queue_has_reasons_and_rules(fitted, config) -> None:
    model, x, y = fitted
    part = x.assign(
        txn_id=np.arange(len(x)), timestamp=pd.Timestamp("2022-09-08"), src_id=0, dst_id=1,
        dst_in_count_24h=0, pass_through_ratio_24h=np.nan, currency_mismatch=0, src_age_secs=np.nan,
        is_laundering=y, typology="none",
    )
    columns = list(x.columns)
    queue = alert_queue(model, columns, part, predict_scores(model, x), x.median(), config.rules, 0.01)
    assert len(queue) == 40 and queue["rank"].tolist() == list(range(1, 41))
    assert queue["score"].is_monotonic_decreasing
    assert queue["reason_1"].str.contains("payments in the past 24 hours").mean() > 0.9
    assert {"rules_fired", "is_laundering", "typology"} <= set(queue.columns)
