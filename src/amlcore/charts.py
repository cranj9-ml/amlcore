"""The two report charts: recall against alert budget, and global SHAP importance.

Kept apart from pipeline.py so matplotlib is imported only when charts are written.
Colors: the model is always blue and the rules always orange, in every chart; the
"perfect ranking" ceiling is a recessive gray reference, not a series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
MODEL_COLOR, RULES_COLOR = "#2a78d6", "#eb6834"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _percent(digits: int) -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{v:.{digits}%}")


def chart_recall_vs_budget(curve: pd.DataFrame, primary: float, path: Path) -> None:
    """Recall (y) at each alert budget (x, log scale) for the model, the rules and a perfect ranking."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.plot(curve["budget"], curve["perfect"], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.plot(curve["budget"], curve["lightgbm"], color=MODEL_COLOR, linewidth=2)
    ax.plot(curve["budget"], curve["rules"], color=RULES_COLOR, linewidth=2)
    ax.axvline(primary, color=INK_2, linewidth=0.8)
    ax.text(primary * 1.05, 0.98, f"{primary:.1%} budget", color=INK_2, fontsize=9, va="top")

    # Direct labels at the right end of each line, in ink colors; the mark carries identity.
    last = curve.iloc[-1]
    for column, label in (("perfect", "perfect ranking"), ("lightgbm", "LightGBM"), ("rules", "rules")):
        ax.annotate(label, (last["budget"], last[column]), xytext=(6, 0), textcoords="offset points",
                    color=INK if column != "perfect" else INK_2, fontsize=9, va="center")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(_percent(1))
    ax.set_xticks([0.001, 0.005, 0.01, 0.05])
    ax.yaxis.set_major_formatter(_percent(0))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("alert budget (share of transactions reviewed)", color=INK_2, fontsize=9)
    ax.set_ylabel("recall (share of laundering caught)", color=INK_2, fontsize=9)
    ax.set_title("Recall at each alert budget, test", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def chart_importance(importance: pd.DataFrame, path: Path, top: int = 12) -> None:
    """Horizontal bars of mean |SHAP| for the top features."""
    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(data) + 1.1), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.barh(data["feature"], data["mean_abs_shap"], color=MODEL_COLOR, height=0.6)
    ax.set_xlabel("mean |SHAP| (log-odds)", color=INK_2, fontsize=9)
    ax.set_title("What moves the score", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
