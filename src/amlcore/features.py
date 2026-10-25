"""Point-in-time features: what a model may know about a transaction at time t.

The rule: a feature for a transaction at time t may use only transactions with a
timestamp strictly before t. Windows are [t - w, t), and transactions in the same minute
never see each other. AMLworld timestamps have minute resolution, so within a minute we
can't know which payment came first; treating them as simultaneous is the only safe
choice. tests/test_features.py proves it by rewriting the future and checking that no
earlier row's features change.

The 45 features, in four groups:
  * transaction (11): the row itself: amount, currencies, banks, hour, payment format, ...
  * account history (28), for the sender (src_) and the receiver (dst_): inflow and
    outflow counts and USD sums over 1h / 24h / 7d, seconds since the account's previous
    activity, and account age
  * derived (2): pass-through ratio and outflow burst, built from the two above
  * graph degree (4): distinct counterparties each account had met before t

How account history is computed: each transaction becomes two events, an outflow for the
sender and an inflow for the receiver. Events are sorted by (account, time), and a
per-account time-based rolling sum with closed="left" gives the window [t - w, t), which
also excludes every event at exactly time t.

Features are computed on the full table before splitting: earlier splits are real
history for later ones, and the split column says which rows to train and score on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from amlcore.config import Config, parse_window

# A fixed vocabulary keeps codes identical across runs and variants (unknown -> -1).
PAYMENT_FORMATS = ("ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire")
FLOW_COLUMNS = ["in_count", "in_sum", "out_count", "out_sum"]
# Identifiers and labels, never model inputs. Account ids are kept for account-level metrics.
META_COLUMNS = ["txn_id", "timestamp", "split", "src_id", "dst_id", "is_laundering", "typology"]
# Integer-coded categories; LightGBM splits on them as groups instead of as numbers.
CATEGORICAL_FEATURES = ["payment_format_code", "payment_currency_code", "receiving_currency_code"]


def build_features(txns: pd.DataFrame, config: Config) -> pd.DataFrame:
    """The full feature table: id / label / split columns first, then the 45 features."""
    pit = point_in_time_features(txns, config.features.windows, config.currencies,
                                 config.features.age_cap)
    return pd.concat([txns[META_COLUMNS], pit], axis=1)


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Every column of the feature table that is a model input."""
    return [col for col in feats.columns if col not in META_COLUMNS]


def point_in_time_features(txns: pd.DataFrame, windows: Sequence[str], currencies: Sequence[str],
                           age_cap: str = "48h") -> pd.DataFrame:
    """Every feature, in a fixed column order, all built from strictly earlier rows."""
    txn = transaction_features(txns, currencies)
    acct = account_features(txns, windows, age_cap)
    return pd.concat([txn, acct, derived_features(txn, acct, windows), degree_features(txns)], axis=1)


# --- Transaction features -------------------------------------------------------------------


def transaction_features(txns: pd.DataFrame, currencies: Sequence[str]) -> pd.DataFrame:
    """Features of the row itself; they look at no other transaction."""
    paid_usd = txns["amount_paid_usd"]
    pay_code = _codes(txns["payment_currency"], currencies)
    recv_code = _codes(txns["receiving_currency"], currencies)
    return pd.DataFrame(
        {
            "amount_usd": paid_usd,
            # ~1 for same-currency payments; differs when the payment converts currency.
            "amount_ratio": txns["amount_received_usd"] / paid_usd.where(paid_usd > 0),
            "currency_mismatch": (pay_code != recv_code).astype("int8"),
            "cross_bank": (txns["from_bank"] != txns["to_bank"]).astype("int8"),
            "self_transfer": (txns["src_id"] == txns["dst_id"]).astype("int8"),
            "hour": txns["timestamp"].dt.hour.astype("int8"),
            "weekday": txns["timestamp"].dt.day_of_week.astype("int8"),
            "round_amount": _is_round(txns["amount_paid"]).astype("int8"),
            "payment_format_code": _codes(txns["payment_format"], PAYMENT_FORMATS),
            "payment_currency_code": pay_code,
            "receiving_currency_code": recv_code,
        },
        index=txns.index,
    )


def _codes(values: pd.Series, vocabulary: Sequence[str]) -> np.ndarray:
    """Integer code of each value in a fixed vocabulary; -1 for values not in it."""
    code_of = {value: code for code, value in enumerate(vocabulary)}
    return values.astype(str).map(code_of).fillna(-1).astype("int8").to_numpy()


def _is_round(amount: pd.Series) -> np.ndarray:
    """Whole hundreds in the payer's own currency, typical of hand-picked amounts.

    Compared in integer cents, because float remainders like 9000.0 % 100 can be off by a
    tiny amount.
    """
    cents = np.round(amount.to_numpy() * 100).astype(np.int64)
    return cents % 10_000 == 0


# --- Account-history features ---------------------------------------------------------------


def account_features(txns: pd.DataFrame, windows: Sequence[str], age_cap: str = "48h") -> pd.DataFrame:
    """Sender (src_) and receiver (dst_) history features, aligned with txns' rows."""
    events = make_events(txns)
    per_event = pd.concat(
        [
            *(rolling_flows(events, window) for window in windows),
            pd.DataFrame({
                "secs_since_prev": seconds_since_previous(events),
                "age_secs": account_age_seconds(events, parse_window(age_cap), txns["timestamp"].min()),
            }),
        ],
        axis=1,
    )
    return pd.concat(
        [for_role(events, per_event, is_src=True, prefix="src_", index=txns.index),
         for_role(events, per_event, is_src=False, prefix="dst_", index=txns.index)],
        axis=1,
    )


def make_events(txns: pd.DataFrame) -> pd.DataFrame:
    """Two events per transaction: an outflow for the sender, an inflow for the receiver.

    'row' is the transaction's position in txns, used to map results back. Events are
    sorted by (account, time), which the rolling step relies on.
    """
    n = len(txns)
    zeros, ones = np.zeros(n), np.ones(n)
    common = {"row": np.arange(n), "ts": txns["timestamp"].to_numpy()}
    outflows = pd.DataFrame({
        **common, "is_src": True, "acct": txns["src_id"].to_numpy(),
        "in_count": zeros, "in_sum": zeros, "out_count": ones, "out_sum": txns["amount_paid_usd"].to_numpy(),
    })
    inflows = pd.DataFrame({
        **common, "is_src": False, "acct": txns["dst_id"].to_numpy(),
        "in_count": ones, "in_sum": txns["amount_received_usd"].to_numpy(), "out_count": zeros, "out_sum": zeros,
    })
    events = pd.concat([outflows, inflows], ignore_index=True)
    return events.sort_values(["acct", "ts"], kind="stable", ignore_index=True)


def rolling_flows(events: pd.DataFrame, window: str) -> pd.DataFrame:
    """For each event: its account's inflow/outflow counts and USD sums over [t - w, t).

    closed="left" excludes the right edge t itself, so events in the same minute,
    including the event's own transaction, are never counted.
    """
    roller = events.groupby("acct", sort=True).rolling(parse_window(window), on="ts", closed="left")
    rolled = roller[FLOW_COLUMNS].sum()
    _check_aligned(rolled, events)
    # An empty window sums to NaN; no history means zero flows.
    return pd.DataFrame(np.nan_to_num(rolled.to_numpy()), columns=[f"{c}_{window}" for c in FLOW_COLUMNS])


def _check_aligned(rolled: pd.DataFrame, events: pd.DataFrame) -> None:
    """Prove the rolled rows come back in event order, instead of assuming it.

    The rolled result is indexed by (acct, ts), and timestamps repeat, so it can't be
    joined back by index; it relies on events being pre-sorted by (acct, ts).
    """
    aligned = (np.array_equal(rolled.index.get_level_values(0), events["acct"].to_numpy())
               and np.array_equal(rolled.index.get_level_values(1), events["ts"].to_numpy()))
    if not aligned:
        raise RuntimeError("rolling output is not in event order; features would be misaligned")


def seconds_since_previous(events: pd.DataFrame) -> np.ndarray:
    """Seconds since the account's previous activity at a strictly earlier time (NaN if none).

    Working on distinct (account, timestamp) pairs means an event in the same minute never
    counts as 'previous'.
    """
    distinct = events[["acct", "ts"]].drop_duplicates()
    distinct = distinct.assign(prev_ts=distinct.groupby("acct")["ts"].shift())
    # A left merge keeps the left rows' order, so prev_ts lines up with events.
    prev_ts = events[["acct", "ts"]].merge(distinct, on=["acct", "ts"], how="left")["prev_ts"]
    return (events["ts"].to_numpy() - prev_ts.to_numpy()) / np.timedelta64(1, "s")


def account_age_seconds(events: pd.DataFrame, cap: pd.Timedelta, data_start: pd.Timestamp) -> np.ndarray:
    """Seconds since the account was first seen, capped at `cap`, and missing (NaN) while
    the data itself is younger than `cap`.

    Why the cap: age is "time since first seen in the data", not a true opening date, so
    raw age can never exceed the time since the data began. It grows with the calendar:
    training days never show the ages test days have, and drift monitoring flagged it as
    the most-shifted feature. Capped, it means the same on every day: "first seen under
    48h ago, and when", or "seen at least 48h ago".

    Why missing at the start: there every account looks young only because the data just
    began. 48h rather than 24h because one day of history is too little to recognise old
    accounts: with a 24h cap, 17% of senders still looked new on day 2.
    """
    first_seen = events.groupby("acct")["ts"].cummin()
    age = (events["ts"] - first_seen).dt.total_seconds().to_numpy()
    warming_up = events["ts"].to_numpy() < np.datetime64(data_start + cap)
    return np.where(warming_up, np.nan, np.minimum(age, cap.total_seconds()))


def for_role(events: pd.DataFrame, per_event: pd.DataFrame, *, is_src: bool, prefix: str,
             index: pd.Index) -> pd.DataFrame:
    """Pick one role's events and put them back in transaction order, with a prefix."""
    mask = (events["is_src"] == is_src).to_numpy()
    part = per_event[mask].set_axis(events.loc[mask, "row"].to_numpy()).sort_index()
    part.columns = [prefix + col for col in part.columns]
    return part.set_axis(index).astype("float32")


# --- Derived features -----------------------------------------------------------------------


def derived_features(txn: pd.DataFrame, acct: pd.DataFrame, windows: Sequence[str]) -> pd.DataFrame:
    """Ratios that describe how money moves through the sender.

    pass_through_ratio_24h: this payment / what the sender received in the last 24h. Near 1
      means money in, money straight out: the signature of a mule account.
    src_out_burst_<short>_<long>: the sender's outflows in the shortest window, relative to
      its average rate over the longest window. High means a sudden burst.
    Both are NaN when the denominator is zero (no history), which LightGBM handles.
    """
    inflow_24h = acct["src_in_sum_24h"]
    out = {"pass_through_ratio_24h": txn["amount_usd"] / inflow_24h.where(inflow_24h > 0)}
    ordered = sorted(windows, key=parse_window)
    if len(ordered) >= 2:
        short, long = ordered[0], ordered[-1]
        expected = acct[f"src_out_count_{long}"] * (parse_window(short) / parse_window(long))
        out[f"src_out_burst_{short}_{long}"] = acct[f"src_out_count_{short}"] / expected.where(expected > 0)
    return pd.DataFrame(out, index=txn.index).astype("float32")


# --- Graph degree features ------------------------------------------------------------------


def degree_features(txns: pd.DataFrame) -> pd.DataFrame:
    """Distinct counterparties each row's sender and receiver had met strictly before t.

    out-degree = distinct accounts it first paid before t; in-degree = distinct accounts
    that first paid it before t. A counterparty counts only once the first payment between
    the two has happened, and a first payment in the same minute doesn't count.
    """
    # Dense rank of each timestamp: same order as time, but small integers, so
    # (account, time) can be packed into one sortable int64 key.
    _, time_rank = np.unique(txns["timestamp"].to_numpy(), return_inverse=True)
    src, dst = txns["src_id"].to_numpy(), txns["dst_id"].to_numpy()
    n_ranks = int(time_rank.max()) + 1 if len(time_rank) else 1
    out_degree = _DegreeCounter(account=src, counterparty=dst, time_rank=time_rank, n_ranks=n_ranks)
    in_degree = _DegreeCounter(account=dst, counterparty=src, time_rank=time_rank, n_ranks=n_ranks)
    return pd.DataFrame(
        {
            "src_out_degree": out_degree.before(src, time_rank),
            "src_in_degree": in_degree.before(src, time_rank),
            "dst_out_degree": out_degree.before(dst, time_rank),
            "dst_in_degree": in_degree.before(dst, time_rank),
        },
        index=txns.index,
    ).astype("float32")


class _DegreeCounter:
    """Answers: how many distinct counterparties had `account` met before time rank r?

    Each (account, counterparty) pair is reduced to the time the two first transacted.
    The degree at time t is how many of the account's first-meet times are before t,
    answered for every row at once by binary search on keys sorted by account, then time.
    """

    def __init__(self, account: np.ndarray, counterparty: np.ndarray, time_rank: np.ndarray,
                 n_ranks: int) -> None:
        pairs = pd.DataFrame({"account": account, "counterparty": counterparty, "rank": time_rank})
        first_meet = pairs.groupby(["account", "counterparty"], sort=False)["rank"].min()
        accounts = first_meet.index.get_level_values("account").to_numpy()
        self._n_ranks = n_ranks
        self._keys = np.sort(self._key(accounts, first_meet.to_numpy()))

    def _key(self, account: np.ndarray, rank: np.ndarray) -> np.ndarray:
        """Pack (account, rank) into one int64 that sorts by account, then by time."""
        return account.astype(np.int64) * self._n_ranks + rank.astype(np.int64)

    def before(self, account: np.ndarray, rank: np.ndarray) -> np.ndarray:
        """Count of the account's first-meet times strictly earlier than `rank`.

        Searching on the left side stops before any key with the same rank, so same-minute
        first meetings are excluded.
        """
        start = np.searchsorted(self._keys, self._key(account, np.zeros_like(rank)), side="left")
        end = np.searchsorted(self._keys, self._key(account, rank), side="left")
        return end - start
