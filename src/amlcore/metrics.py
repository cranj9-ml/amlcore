"""Metrics for a ranked alert list. Never accuracy.

Investigators can only review a fixed number of alerts, so the questions are: of the top
k transactions, how many are laundering (precision), and what share of all laundering do
they catch (recall)? k = budget x N, e.g. 0.5% of transactions. At a fixed k, precision
and recall rise together, so maximising one maximises the other; the real trade-off is
the choice of budget, which is a staffing decision. PR-AUC (average precision) summarises
the whole ranking. With ~0.1% positives, a model that flags nothing is 99.9% accurate,
which is why accuracy is never used.

Ties at the cut-off: rule scores take only a few distinct values, so thousands of rows
can tie at the k-th place. Picking among them by row order would make the metric
arbitrary. Instead, metrics are the expected value if ties were broken at random: rows
above the cut-off are always alerted, and tied rows share the remaining slots equally.
With no ties this is the ordinary top-k.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class BudgetResult:
    """Metrics for one alert budget."""

    budget: float  # fraction of rows that can be alerted
    k: int  # number of alerts
    hits: float  # expected laundering rows among the alerts
    precision: float
    recall: float


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision; a random ranking scores about the base rate. NaN without positives."""
    y = np.asarray(y)
    return float(average_precision_score(y, score)) if y.sum() else float("nan")


def alert_count(n: int, budget: float) -> int:
    """k = budget x n, rounded down (never more alerts than the budget), at least 1."""
    return max(1, math.floor(budget * n + 1e-9))


def selection_probability(score: np.ndarray, k: int) -> np.ndarray:
    """Each row's chance of being among the top k, with ties broken at random."""
    score = np.asarray(score, dtype="float64")
    if k >= len(score):
        return np.ones(len(score))
    cutoff = np.partition(score, len(score) - k)[len(score) - k]  # the k-th largest score
    above, tied = score > cutoff, score == cutoff
    prob = above.astype("float64")
    prob[tied] = (k - above.sum()) / tied.sum()
    return prob


def at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> BudgetResult:
    """Expected precision and recall when alerting the top budget x N rows."""
    y = np.asarray(y)
    k = alert_count(len(y), budget)
    hits = float(selection_probability(score, k) @ y)
    n_pos = y.sum()
    return BudgetResult(budget, k, hits, hits / k, hits / n_pos if n_pos else float("nan"))


def recall_by_typology(y: np.ndarray, score: np.ndarray, typology: np.ndarray, budget: float) -> pd.DataFrame:
    """Per laundering typology: positives, expected number caught, and recall at the budget."""
    y = np.asarray(y)
    prob = selection_probability(score, alert_count(len(y), budget))
    positives = pd.DataFrame({"typology": np.asarray(typology)[y == 1], "caught": prob[y == 1]})
    table = positives.groupby("typology")["caught"].agg(["size", "sum"])
    table.columns = ["positives", "caught"]
    table["recall"] = table["caught"] / table["positives"]
    return table.sort_values("positives", ascending=False)


def account_level(src_ids: np.ndarray, y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Collapse transactions to sending accounts: label = any laundering, score = max.

    Investigators open cases on accounts, not single payments; an account is caught if any
    of its payments ranks high enough.
    """
    df = pd.DataFrame({"src_id": src_ids, "y": y, "score": score})
    return df.groupby("src_id").agg(y=("y", "max"), score=("score", "max"))


def evaluate(y: np.ndarray, score: np.ndarray, budgets: tuple[float, ...],
             src_ids: np.ndarray) -> dict[str, object]:
    """PR-AUC and per-budget results at transaction and account level, with base rates
    (precision means little without them)."""
    y = np.asarray(y)
    accounts = account_level(src_ids, y, score)
    ay, ascore = accounts["y"].to_numpy(), accounts["score"].to_numpy()
    return {
        "n": len(y),
        "base_rate": float(y.mean()),
        "pr_auc": pr_auc(y, score),
        "at_budget": [at_budget(y, score, b) for b in budgets],
        "accounts": {
            "n": len(accounts),
            "base_rate": float(ay.mean()),
            "pr_auc": pr_auc(ay, ascore),
            "at_budget": [at_budget(ay, ascore, b) for b in budgets],
        },
    }


# --- Block-bootstrap interval for recall at a budget -----------------------------------------


@dataclass(frozen=True)
class Interval:
    """A point estimate with its bootstrap percentile interval."""

    estimate: float
    low: float
    high: float
    n_blocks: int


def recall_interval(y: np.ndarray, score: np.ndarray, timestamps: pd.Series, budget: float,
                    block: str = "6h", n_resamples: int = 1000, level: float = 0.95,
                    seed: int = 0) -> Interval:
    """Recall@budget with a block-bootstrap percentile interval.

    Why blocks, not rows: one scheme's hops and busy accounts' payments cluster in time, so
    rows are not independent and a row bootstrap gives intervals that are too narrow.
    Whole time blocks are resampled instead. Validation and test each span about three
    days, so day blocks would give only three units; 6-hour blocks give about ten.

    A resample is a whole-number weight per block, so rows keep their weights instead of
    being copied; the alert count is budget x resampled rows, with ties shared as above.
    """
    blocks = pd.factorize(timestamps.dt.floor(block).to_numpy())[0]
    n_blocks = int(blocks.max()) + 1
    ranking = _WeightedRanking(y, score, blocks)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        weight = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks).astype("float64")
        samples[i] = ranking.recall(weight, budget)
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return Interval(ranking.recall(np.ones(n_blocks), budget), float(low), float(high), n_blocks)


class _WeightedRanking:
    """One ranking, prepared once so each resample costs a few O(n) passes."""

    def __init__(self, y: np.ndarray, score: np.ndarray, blocks: np.ndarray) -> None:
        score = np.asarray(score, dtype="float64")
        order = np.argsort(-score, kind="stable")
        ordered = score[order]
        # Rows with equal scores form one group; groups are in descending score order.
        self.group = np.concatenate([[0], np.cumsum(ordered[1:] != ordered[:-1])])
        self.n_groups = int(self.group[-1]) + 1
        self.y = np.asarray(y, dtype="float64")[order]
        self.blocks = blocks[order]

    def recall(self, block_weight: np.ndarray, budget: float) -> float:
        """Tie-aware recall@budget when each row counts block_weight[its block] times."""
        w = block_weight[self.blocks]
        group_w = np.bincount(self.group, weights=w, minlength=self.n_groups)
        group_pos = np.bincount(self.group, weights=w * self.y, minlength=self.n_groups)
        positives = group_pos.sum()
        if positives == 0:
            return float("nan")
        k = max(1, math.floor(budget * group_w.sum() + 1e-9))
        cum_w, cum_pos = np.cumsum(group_w), np.cumsum(group_pos)
        g = min(int(np.searchsorted(cum_w, k)), self.n_groups - 1)  # group holding the k-th slot
        before_w, before_pos = cum_w[g] - group_w[g], cum_pos[g] - group_pos[g]
        share = min(1.0, (k - before_w) / group_w[g])  # the cut-off group's share of slots
        return float((before_pos + share * group_pos[g]) / positives)
