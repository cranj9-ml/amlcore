"""Tests for amlcore.features. The leakage tests here are the most important in the project.

If test_future_perturbation ever fails, some feature is using information from the
future. Fix that before anything else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amlcore.config import load_config, parse_window
from amlcore.data import add_split
from amlcore.features import (
    META_COLUMNS,
    build_features,
    degree_features,
    feature_columns,
    point_in_time_features,
)

START = pd.Timestamp("2022-09-01")
WINDOWS = ("1h", "24h", "7d")
CURRENCIES = sorted(load_config().fx_to_usd)


def make_txns(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """Transactions from (minute offset, src_id, dst_id, USD amount); other columns are plain."""
    minute, src, dst, amount = zip(*rows)
    return pd.DataFrame({
        "txn_id": np.arange(len(rows)),
        "timestamp": [START + pd.Timedelta(minutes=m) for m in minute],
        "src_id": src, "dst_id": dst, "from_bank": "001", "to_bank": "002",
        "amount_paid": amount, "amount_paid_usd": amount, "amount_received_usd": amount,
        "payment_currency": "US Dollar", "receiving_currency": "US Dollar",
        "payment_format": "ACH", "is_laundering": 0, "typology": "none",
    })


def features(rows: list[tuple[int, int, int, float]], age_cap: str = "48h") -> pd.DataFrame:
    return point_in_time_features(make_txns(rows), WINDOWS, CURRENCIES, age_cap)


# --- The point-in-time rule -----------------------------------------------------------------


def test_same_minute_transactions_do_not_see_each_other() -> None:
    f = features([
        (0, 1, 2, 100.0),  # account 1 pays twice at minute 0...
        (0, 1, 3, 50.0),
        (0, 2, 4, 30.0),  # ...and account 2 receives and pays in that same minute
        (1, 1, 5, 10.0),  # one minute later, account 1 sees both earlier payments
    ])
    assert list(f["src_out_count_1h"]) == [0, 0, 0, 2]
    assert list(f["src_out_sum_1h"]) == [0, 0, 0, 150]
    assert f.loc[2, "src_in_count_1h"] == 0  # account 2's inflow is in the same minute
    assert f["src_secs_since_prev"].isna().tolist() == [True, True, True, False]
    assert f.loc[3, "src_out_degree"] == 2  # receivers 2 and 3, both met at minute 0


@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75])
def test_future_perturbation(demo_txns: pd.DataFrame, quantile: float) -> None:
    """Rewriting everything after a cutoff must not change any earlier row's features."""
    cutoff = demo_txns["timestamp"].quantile(quantile)
    late = demo_txns["timestamp"] > cutoff
    before = point_in_time_features(demo_txns, WINDOWS, CURRENCIES)
    after = point_in_time_features(_perturb(demo_txns, late), WINDOWS, CURRENCIES)

    pd.testing.assert_frame_equal(before[~late], after[~late], check_exact=True)
    assert not before[late].equals(after[late])  # the perturbation really did something
    assert before.shape[1] == 45  # every feature is covered


def test_same_minute_twins_change_nothing(demo_txns: pd.DataFrame) -> None:
    """A twin payment in the same minute, between the same accounts, changes nothing for
    the originals in that minute.

    The future-perturbation test can't see same-minute leaks, because it only compares
    rows strictly before the cutoff. Here, for 20 sampled minutes, every row in the minute
    gets a twin with another amount; if any window included time t, the originals'
    counts and sums would move. (Later rows may change: the twins are real history to them.)
    """
    before = point_in_time_features(demo_txns, WINDOWS, CURRENCIES)
    rng = np.random.default_rng(3)
    for minute in rng.choice(demo_txns["timestamp"].unique(), 20, replace=False):
        in_minute = demo_txns["timestamp"] == minute
        twins = demo_txns[in_minute].assign(amount_paid_usd=1e6, amount_received_usd=1e6)
        # Twins go first, so a stable sort puts them before their originals: a window that
        # reached time t would then include them.
        after = point_in_time_features(pd.concat([twins, demo_txns], ignore_index=True), WINDOWS, CURRENCIES)
        originals = after.iloc[len(twins):].set_axis(before.index)
        pd.testing.assert_frame_equal(before[in_minute], originals[in_minute], check_exact=True)


def _perturb(txns: pd.DataFrame, rows: pd.Series, seed: int = 0) -> pd.DataFrame:
    """Scramble amounts and accounts of the given rows, using existing and new accounts."""
    rng = np.random.default_rng(seed)
    out = txns.copy()
    k = int(rows.sum())
    factor = rng.uniform(0.1, 10.0, k)
    for col in ["amount_paid", "amount_paid_usd", "amount_received_usd"]:
        out.loc[rows, col] = out.loc[rows, col] * factor
    n_ids = int(max(out["src_id"].max(), out["dst_id"].max())) + 1
    out.loc[rows, "src_id"] = rng.integers(0, n_ids + 50, k)
    out.loc[rows, "dst_id"] = rng.integers(0, n_ids + 50, k)
    return out


def test_window_boundary_at_60_and_61_minutes() -> None:
    f = features([
        (0, 1, 2, 10.0),
        (60, 1, 3, 10.0),  # 60 minutes later: [t - 1h, t) includes minute 0
        (0, 5, 6, 10.0),
        (61, 5, 7, 10.0),  # 61 minutes later: minute 0 is outside the 1h window
    ])
    assert f.loc[1, "src_out_count_1h"] == 1
    assert f.loc[3, "src_out_count_1h"] == 0
    assert f.loc[3, "src_out_count_24h"] == 1  # still inside the longer window


def test_inflow_and_outflow_per_role() -> None:
    f = features([(0, 1, 2, 100.0), (10, 2, 3, 40.0), (20, 3, 1, 25.0)])
    # Row 1: sender 2 has one inflow of 100 and no outflows; receiver 3 has no history.
    assert f.loc[1, ["src_in_count_1h", "src_in_sum_1h", "src_out_count_1h"]].tolist() == [1, 100, 0]
    assert f.loc[1, ["dst_in_count_1h", "dst_out_count_1h"]].tolist() == [0, 0]
    # Row 2: receiver 1 has one outflow of 100 and no inflows.
    assert f.loc[2, ["dst_out_count_1h", "dst_out_sum_1h", "dst_in_count_1h"]].tolist() == [1, 100, 0]


def test_rolling_counts_match_a_brute_force_loop(demo_txns: pd.DataFrame) -> None:
    f = point_in_time_features(demo_txns, WINDOWS, CURRENCIES)
    ts, src, dst = demo_txns["timestamp"], demo_txns["src_id"], demo_txns["dst_id"]
    rng = np.random.default_rng(1)
    for i in rng.choice(len(demo_txns), 40, replace=False):
        t = ts.iloc[i]
        in_window = (ts >= t - pd.Timedelta(hours=24)) & (ts < t)
        assert f["src_out_count_24h"].iloc[i] == (in_window & (src == src.iloc[i])).sum()
        assert f["dst_in_count_24h"].iloc[i] == (in_window & (dst == dst.iloc[i])).sum()


def test_degrees_match_brute_force(demo_txns: pd.DataFrame) -> None:
    deg = degree_features(demo_txns)
    ts, src, dst = demo_txns["timestamp"], demo_txns["src_id"], demo_txns["dst_id"]
    rng = np.random.default_rng(2)
    for i in rng.choice(len(demo_txns), 40, replace=False):
        earlier = ts < ts.iloc[i]
        assert deg["src_out_degree"].iloc[i] == dst[earlier & (src == src.iloc[i])].nunique()
        assert deg["dst_in_degree"].iloc[i] == src[earlier & (dst == dst.iloc[i])].nunique()


def test_row_order_does_not_change_features(demo_txns: pd.DataFrame) -> None:
    shuffled = demo_txns.sample(frac=1.0, random_state=0)
    a = point_in_time_features(demo_txns, WINDOWS, CURRENCIES)
    b = point_in_time_features(shuffled, WINDOWS, CURRENCIES).loc[demo_txns.index]
    pd.testing.assert_frame_equal(a, b)


# --- Account age ----------------------------------------------------------------------------


def test_age_is_capped_and_missing_while_the_data_warms_up() -> None:
    day = 24 * 60
    f = features([
        (0, 1, 2, 10.0),  # the data starts: no age yet
        (3 * day, 1, 3, 10.0),  # account 1 first seen 3 days ago: capped at 48h
        (3 * day, 7, 8, 10.0),  # account 7 is new
        (3 * day + 120, 7, 9, 10.0),  # account 7 two hours later
    ])
    assert np.isnan(f.loc[0, "src_age_secs"])
    assert f.loc[1, "src_age_secs"] == parse_window("48h").total_seconds()
    assert f.loc[2, "src_age_secs"] == 0
    assert f.loc[3, "src_age_secs"] == 7200


# --- Derived and transaction features --------------------------------------------------------


def test_pass_through_ratio_and_burst() -> None:
    f = features([(0, 1, 2, 1000.0), (30, 2, 3, 900.0), (31, 2, 4, 10.0)])
    assert f.loc[1, "pass_through_ratio_24h"] == pytest.approx(0.9)
    assert np.isnan(f.loc[0, "pass_through_ratio_24h"])  # nothing received yet
    # Row 2: one outflow in the past hour, against an expected 1 x (1h / 7d).
    assert f.loc[2, "src_out_burst_1h_7d"] == pytest.approx(168.0)


def test_transaction_features(tiny_txns: pd.DataFrame) -> None:
    f = point_in_time_features(tiny_txns, WINDOWS, CURRENCIES)
    row = f.loc[tiny_txns["from_bank"] == "001"].iloc[0]  # paid USD, received GBP
    assert row["currency_mismatch"] == 1 and row["cross_bank"] == 1
    self_pay = f.loc[tiny_txns["from_bank"] == "010"].iloc[0]
    assert self_pay["self_transfer"] == 1 and self_pay["payment_format_code"] == 5  # Reinvestment


def test_build_features_has_meta_then_45_features(demo_txns: pd.DataFrame) -> None:
    config = load_config()
    feats = build_features(add_split(demo_txns, config.split), config)
    assert feats.columns[: len(META_COLUMNS)].tolist() == META_COLUMNS
    assert len(feature_columns(feats)) == 45
