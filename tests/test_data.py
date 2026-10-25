"""Tests for amlcore.data: parsing, ids, typologies, the tail cut and the temporal split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amlcore.config import SplitConfig
from amlcore.data import (
    NO_TYPOLOGY,
    TEST,
    TRAIN,
    UNATTRIBUTED,
    VAL,
    add_split,
    build_transactions,
    cut_points,
    normalize_typology,
    parse_patterns,
)


def test_sorted_by_time_with_sequential_txn_ids(tiny_txns: pd.DataFrame) -> None:
    assert tiny_txns["timestamp"].is_monotonic_increasing
    assert tiny_txns["txn_id"].tolist() == list(range(6))


def test_stable_sort_keeps_file_order_within_a_minute(tiny_txns: pd.DataFrame) -> None:
    first_two = tiny_txns.iloc[:2]  # both at 00:06; the 021174 row comes first in the file
    assert first_two["from_bank"].tolist() == ["021174", "001"]


def test_same_account_gets_the_same_id_on_either_side(tiny_txns: pd.DataFrame) -> None:
    a = tiny_txns[tiny_txns["from_bank"] == "021174"].iloc[0]
    b = tiny_txns[tiny_txns["to_bank"] == "021174"].iloc[0]
    assert a["src_id"] == b["dst_id"]
    ids = np.concatenate([tiny_txns["src_id"], tiny_txns["dst_id"]])
    assert set(ids) == set(range(ids.max() + 1))  # dense


def test_typologies(tiny_txns: pd.DataFrame) -> None:
    by_bank = dict(zip(tiny_txns["from_bank"], tiny_txns["typology"]))
    assert by_bank["021174"] == "cycle"
    assert by_bank["011"] == UNATTRIBUTED  # laundering, but in no block
    assert by_bank["013"] == NO_TYPOLOGY  # matches a pattern row, but isn't laundering


def test_usd_amounts(tiny_txns: pd.DataFrame) -> None:
    row = tiny_txns[tiny_txns["from_bank"] == "001"].iloc[0]  # paid 12 USD, received 10 GBP
    assert row["amount_paid_usd"] == 12.0
    assert row["amount_received_usd"] == pytest.approx(11.3)


def test_unknown_currency_raises(tiny_raw) -> None:
    with pytest.raises(ValueError, match="no fx_to_usd rate"):
        build_transactions(*tiny_raw, {"US Dollar": 1.0})


def test_end_drops_rows_at_or_after_it(config, tiny_raw) -> None:
    txns = build_transactions(*tiny_raw, config.fx_to_usd, end=pd.Timestamp("2022-09-01 00:15"))
    assert txns["timestamp"].max() == pd.Timestamp("2022-09-01 00:10")


@pytest.mark.parametrize("name, expected", [("SCATTER-GATHER", "scatter_gather"), (" Fan-Out ", "fan_out")])
def test_normalize_typology(name: str, expected: str) -> None:
    assert normalize_typology(name) == expected


@pytest.mark.parametrize("text", [
    "2022/09/01 00:06,001,A,002,B,1,US Dollar,1,US Dollar,ACH,1",  # row outside a block
    "BEGIN LAUNDERING ATTEMPT - CYCLE\nBEGIN LAUNDERING ATTEMPT - CYCLE",  # nested BEGIN
    "BEGIN LAUNDERING ATTEMPT - CYCLE\nEND LAUNDERING ATTEMPT - STACK",  # mismatched END
    "BEGIN LAUNDERING ATTEMPT - CYCLE",  # never closed
])
def test_malformed_patterns_raise(text: str) -> None:
    with pytest.raises(ValueError):
        parse_patterns(text)


def test_patterns_keep_leading_zeros(tiny_raw) -> None:
    assert parse_patterns(tiny_raw[1])["from_bank"].iloc[0] == "021174"


# --- Temporal split -------------------------------------------------------------------------


def _times(minutes: list[int]) -> pd.Series:
    return pd.Series(pd.Timestamp("2022-09-01") + pd.to_timedelta(minutes, unit="min"))


def test_splits_are_ordered_in_time_and_sized_by_rows() -> None:
    txns = pd.DataFrame({"timestamp": _times(list(range(100)))})
    split = add_split(txns, SplitConfig(0.6, 0.2))["split"]
    assert split.value_counts().to_dict() == {TRAIN: 60, VAL: 20, TEST: 20}
    assert split.tolist() == sorted(split, key=[TRAIN, VAL, TEST].index)


def test_tied_timestamps_never_straddle_a_cut() -> None:
    txns = pd.DataFrame({"timestamp": _times([0, 1, 2, 3, 3, 3, 3, 4, 5, 6])})
    out = add_split(txns, SplitConfig(0.5, 0.3))
    assert out.groupby("timestamp")["split"].nunique().max() == 1


def test_ties_that_empty_a_split_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        cut_points(_times([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 0.5, 0.3)
