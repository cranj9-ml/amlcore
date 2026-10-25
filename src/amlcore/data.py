"""Raw AMLworld files → one clean, time-sorted transactions table with a temporal split.

Steps, in order:
  1. read the Trans CSV and the Patterns file with one shared schema (string ids)
  2. drop the simulator's artificial tail (rows at or after data.end)
  3. sort by time, number rows as txn_id, give every account a dense integer id
  4. label laundering rows with their typology from the Patterns file
  5. convert amounts to USD
  6. assign train / val / test by time

Typologies are never model features (that would leak the label). They only break
results down by laundering pattern, e.g. "we catch fan-outs but miss cycles".
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from amlcore.config import Config, SplitConfig

# --- Schema shared by both raw files ------------------------------------------------------

# Our own names, because the CSV header repeats "Account" for sender and receiver.
RAW_COLUMNS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid", "payment_currency",
    "payment_format", "is_laundering",
]
# Ids are strings: banks have leading zeros ("021") and accounts are hex ("8000EBD30").
# Reading both files the same way matters: typologies are attached by exact matching,
# and "021" read as the number 21 in one file would silently never match.
RAW_DTYPES = {
    "from_bank": "str", "from_account": "str", "to_bank": "str", "to_account": "str",
    "amount_received": "float64", "receiving_currency": "str", "amount_paid": "float64",
    "payment_currency": "str", "payment_format": "str", "is_laundering": "int8",
}
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
# Every field except the label identifies a transaction when matching the two files.
MATCH_COLUMNS = [col for col in RAW_COLUMNS if col != "is_laundering"]

NO_TYPOLOGY = "none"  # not laundering
UNATTRIBUTED = "unattributed"  # laundering, but in no documented scheme
TRAIN, VAL, TEST = "train", "val", "test"
SPLITS = (TRAIN, VAL, TEST)


def read_raw_rows(source: str | Path | IO[str], *, has_header: bool) -> pd.DataFrame:
    """Read AMLworld rows with our column names, string ids and parsed timestamps."""
    df = pd.read_csv(source, header=0 if has_header else None, names=RAW_COLUMNS, dtype=RAW_DTYPES)
    # An explicit format is much faster than guessing, and fails loudly on a bad timestamp.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT)
    return df


# --- Patterns file ------------------------------------------------------------------------

BEGIN_PREFIX = "BEGIN LAUNDERING ATTEMPT - "
END_PREFIX = "END LAUNDERING ATTEMPT - "


def normalize_typology(name: str) -> str:
    """'SCATTER-GATHER' -> 'scatter_gather', so it is safe as a column or file name."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def parse_patterns(text: str) -> pd.DataFrame:
    """Parse the Patterns file into its transaction rows, each tagged with its typology.

    The file is a sequence of blocks, one per laundering attempt:
        BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 4 hops
        <rows in the Trans CSV format, no header>
        END LAUNDERING ATTEMPT - CYCLE
    The text after ':' describes the attempt's size, not its type, so it is dropped.
    Broken structure raises with the line number rather than mislabelling rows.
    """
    rows: list[str] = []
    typologies: list[str] = []
    current: str | None = None
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(BEGIN_PREFIX):
            if current is not None:
                raise ValueError(f"line {line_no}: BEGIN inside an open {current!r} block")
            current = normalize_typology(line[len(BEGIN_PREFIX):].split(":", 1)[0])
        elif line.startswith(END_PREFIX):
            if normalize_typology(line[len(END_PREFIX):]) != current:
                raise ValueError(f"line {line_no}: END does not match the open {current!r} block")
            current = None
        elif current is None:
            raise ValueError(f"line {line_no}: transaction row outside any BEGIN/END block")
        else:
            rows.append(line)
            typologies.append(current)
    if current is not None:
        raise ValueError(f"patterns text ends inside an open {current!r} block")
    if not rows:
        raise ValueError("no laundering attempts found; is this a Patterns file?")
    return read_raw_rows(io.StringIO("\n".join(rows)), has_header=False).assign(typology=typologies)


# --- Building the transactions table -------------------------------------------------------


def load_transactions(config: Config) -> pd.DataFrame:
    """The clean transactions table for config.variant, with a split column."""
    txns = build_transactions(
        config.trans_path, config.patterns_path.read_text(encoding="utf-8"),
        config.fx_to_usd, end=config.data.end,
    )
    return add_split(txns, config.split)


def build_transactions(trans: str | Path | IO[str], patterns_text: str, fx_to_usd: dict[str, float],
                       end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Read, truncate, sort, key, label and convert the raw files."""
    txns = read_raw_rows(trans, has_header=True)
    txns = drop_from(txns, end)
    txns = sort_by_time(txns)
    txns = add_account_ids(txns)
    txns["typology"] = attach_typologies(txns, parse_patterns(patterns_text))
    return add_usd_amounts(txns, fx_to_usd)


def drop_from(txns: pd.DataFrame, end: pd.Timestamp | None) -> pd.DataFrame:
    """Keep only rows strictly before `end` (None keeps everything).

    AMLworld's normal activity stops after Sep 10, but laundering schemes keep finishing
    for a week afterwards, so the tail is 59% laundering. Anything that tracks activity
    levels would catch it for free and inflate test scores. It is dropped here, before
    ids exist, so nothing downstream ever sees it.
    """
    return txns if end is None else txns[txns["timestamp"] < end]


def sort_by_time(txns: pd.DataFrame) -> pd.DataFrame:
    """Sort by timestamp and number the rows 0..n-1 as txn_id.

    A stable sort keeps the file order within a minute, so txn_id is the same on every
    run. That order carries no meaning: features treat same-minute rows as simultaneous.
    """
    out = txns.sort_values("timestamp", kind="stable").reset_index(drop=True)
    out.insert(0, "txn_id", np.arange(len(out), dtype=np.int64))
    return out


def add_account_ids(txns: pd.DataFrame) -> pd.DataFrame:
    """Dense integer ids for '<bank>_<account>', shared by senders and receivers.

    An account number is only unique within its bank, so the key needs both parts.
    Factorizing both sides together gives an account the same id whichever side it is on.
    """
    src_key = txns["from_bank"] + "_" + txns["from_account"]
    dst_key = txns["to_bank"] + "_" + txns["to_account"]
    codes, _ = pd.factorize(pd.concat([src_key, dst_key], ignore_index=True))
    return txns.assign(src_id=codes[: len(txns)], dst_id=codes[len(txns):])


def attach_typologies(txns: pd.DataFrame, patterns: pd.DataFrame) -> pd.Series:
    """Label each transaction with a typology name, 'unattributed' or 'none'.

    Only laundering rows are matched, so a legitimate row that happens to share every
    field with a pattern row can't inherit a typology. Patterns are de-duplicated first
    (the first attempt wins), so a row listed twice can't duplicate a transaction.
    """
    typology = pd.Series(NO_TYPOLOGY, index=txns.index)
    laundering = txns.loc[txns["is_laundering"] == 1, MATCH_COLUMNS]
    lookup = patterns.drop_duplicates(MATCH_COLUMNS)[MATCH_COLUMNS + ["typology"]]
    # A left merge keeps the left rows' order, so the result lines up with `laundering`.
    matched = laundering.merge(lookup, on=MATCH_COLUMNS, how="left", validate="many_to_one")
    typology.loc[laundering.index] = matched["typology"].fillna(UNATTRIBUTED).to_numpy()
    return typology


def add_usd_amounts(txns: pd.DataFrame, fx_to_usd: dict[str, float]) -> pd.DataFrame:
    """Add amount_paid_usd and amount_received_usd so amounts are comparable.

    An unknown currency raises instead of becoming NaN, which would quietly drop those
    rows from every amount-based feature.
    """
    unknown = set(txns["payment_currency"]) | set(txns["receiving_currency"])
    unknown -= fx_to_usd.keys()
    if unknown:
        raise ValueError(f"no fx_to_usd rate for {sorted(unknown)}; add them to the config")
    return txns.assign(
        amount_paid_usd=txns["amount_paid"] * txns["payment_currency"].map(fx_to_usd),
        amount_received_usd=txns["amount_received"] * txns["receiving_currency"].map(fx_to_usd),
    )


# --- Temporal split -----------------------------------------------------------------------


@dataclass(frozen=True)
class CutPoints:
    """The first timestamp of the val split and of the test split."""

    val_start: pd.Timestamp
    test_start: pd.Timestamp


def cut_points(timestamps: pd.Series, train_frac: float, val_frac: float) -> CutPoints:
    """Cuts at transaction-count quantiles of time (by default 60% and 80% of rows).

    Why temporal: a model is deployed on the future, so it must be judged on the future.
    A random split trains on payments from after the ones it is tested on, including
    later hops of the very same scheme. The cut is a timestamp, so a minute of tied rows
    always starts the next split together.
    """
    ordered = timestamps.sort_values(ignore_index=True)
    n = len(ordered)
    if n == 0:
        raise ValueError("cannot split an empty table")

    def at(frac: float) -> pd.Timestamp:
        return pd.Timestamp(ordered[min(round(n * frac), n - 1)])

    cuts = CutPoints(val_start=at(train_frac), test_start=at(train_frac + val_frac))
    if not ordered[0] < cuts.val_start < cuts.test_start:
        raise ValueError(f"cuts {cuts} leave a split empty; too few rows or too many tied timestamps")
    return cuts


def add_split(txns: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Add a 'split' column: train before the first cut, val before the second, else test."""
    ts = txns["timestamp"]
    cuts = cut_points(ts, config.train_frac, config.val_frac)
    split = np.where(ts < cuts.val_start, TRAIN, np.where(ts < cuts.test_start, VAL, TEST))
    return txns.assign(split=split)
