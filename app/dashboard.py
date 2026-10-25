"""amlcore dashboard: the model's metrics, alert queue and explanations, in a browser.

Run from the project root:
    pip install -e '.[app]'
    streamlit run app/dashboard.py

Two modes, chosen in the sidebar:
  * Demo (live): trains on the bundled synthetic data in seconds. Toy-sized numbers;
    it shows the pipeline working end to end.
  * <variant> (saved run): the results `python -m amlcore.cli run` saved, e.g. HI-Small.

This file only lays things out. Every number comes from amlcore.showcase, which is
tested; AMLCORE_CONFIG points the app at another config (the tests use it).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from amlcore.config import Config, demo_config, load_config
from amlcore.data import TEST
from amlcore.pipeline import MODEL, RULES, report_dir, run_features, run_train
from amlcore import showcase as sc

MODEL_COLOR, RULES_COLOR, MUTED, INK_2 = "#2a78d6", "#eb6834", "#8a8984", "#52514e"
SPLIT_COLORS = {"train": "#8a8984", "val": "#1baf7a", "test": "#6250d6"}
BUDGETS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
QUEUE_ROWS_SHOWN = 1000


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{digits}%}"


def budget_label(budget: float) -> str:
    """0.005 -> "0.5%", 0.0005 -> "0.05%": as few digits as the budget needs."""
    return f"{budget * 100:g}%"


def md(text: str) -> str:
    """Escape dollar signs: Streamlit's markdown reads "$13,102 ... $940" as LaTeX."""
    return text.replace("$", "\\$")


def style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    """Title top left, legend top right, recessive axes and grid, so the marks carry the chart."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=70, b=10), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2),
                      title=dict(text=title, x=0, xanchor="left", y=0.97, yanchor="top"),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom"))
    fig.update_xaxes(showgrid=False, linecolor="#e6e5e0")
    fig.update_yaxes(gridcolor="#e6e5e0", zeroline=False)
    return fig


# --- Configs and loading -------------------------------------------------------------------------


def saved_config() -> Config:
    return load_config(os.environ.get("AMLCORE_CONFIG"))


@st.cache_resource(show_spinner="Loading the saved run...")
def cached_run(_config: Config, key: str) -> sc.Run:
    """`key` (the run folder and its metrics.json time) refreshes the cache after a retrain."""
    return sc.load_run(_config)


def run_key(config: Config) -> str:
    metrics = report_dir(config) / "metrics.json"
    return f"{metrics}:{metrics.stat().st_mtime if metrics.exists() else 0}"


@st.cache_data(show_spinner="Explaining a sample of alerts...")
def cached_reason_shares(_run: sc.Run, _config: Config, key: str, budget: float) -> pd.Series:
    return sc.reason_shares(_run, budget, _config)


def sidebar() -> tuple[Config, float]:
    base = saved_config()
    st.sidebar.title("amlcore")
    st.sidebar.caption("Ranking AML alerts for a fixed investigator budget")
    saved_label = f"{base.variant} (saved run)"
    mode = st.sidebar.radio("Data", [saved_label, "Demo (live)"],
                            help="The saved run is what `python -m amlcore.cli run` wrote. "
                                 "Demo trains on small bundled synthetic data in seconds.")
    config = base if mode == saved_label else demo_config(base)
    if mode == "Demo (live)":
        if st.sidebar.button("Train on demo data", type="primary", icon=":material/play_arrow:"):
            with st.sidebar.status("Training on demo data...", expanded=False) as status:
                run_train(config, run_features(config))
                status.update(label="Demo model trained", state="complete")
            st.cache_resource.clear()
            st.cache_data.clear()
    budget = st.sidebar.select_slider("Alert budget (share of test transactions reviewed)", BUDGETS,
                                      value=base.evaluation.primary_budget, format_func=budget_label)
    st.sidebar.divider()
    st.sidebar.caption("Model: LightGBM on 45 point-in-time features. Baseline: six textbook "
                       "rule scenarios. Both are judged on a later, held-out test period.")
    return config, budget


# --- Tabs ----------------------------------------------------------------------------------------


def overview(run: sc.Run, budget: float, primary: float) -> None:
    m = sc.budget_metrics(run, budget)
    model, rules = m[MODEL], m[RULES]
    c = st.columns(4)
    c[0].metric("LightGBM recall", pct(model["recall"]),
                f"{(model['recall'] - rules['recall']) * 100:+.1f} pts vs rules")
    c[1].metric("LightGBM precision", pct(model["precision"]),
                f"{(model['precision'] - rules['precision']) * 100:+.1f} pts vs rules")
    c[2].metric("Rules recall", pct(rules["recall"]), help="Six textbook scenarios, same alert count")
    c[3].metric("Alerts at this budget", f"{model['alerts']:,}",
                help=f"{model['hits']:,.0f} of them laundering (expected, ties shared)")

    test = run.metrics[TEST]
    ci = run.metrics["test_recall_interval"][MODEL]
    c = st.columns(4)
    c[0].metric("PR-AUC, LightGBM", f"{test[MODEL]['pr_auc']:.3f}", help="Whole ranking; random ≈ base rate")
    c[1].metric("PR-AUC, rules", f"{test[RULES]['pr_auc']:.4f}")
    c[2].metric(f"95% CI, recall at {primary:.1%}", f"{pct(ci['low'])} – {pct(ci['high'])}",
                help=f"Block bootstrap over {ci['n_blocks']} six-hour blocks of test")
    c[3].metric("Test base rate", pct(test[MODEL]["base_rate"], 3))

    curve = sc.recall_curve(run, np.geomspace(0.0005, 0.05, 41))
    fig = go.Figure()
    fig.add_scatter(x=curve["budget"], y=curve["perfect"], name="perfect ranking",
                    line=dict(color=MUTED, dash="dash", width=1.5))
    fig.add_scatter(x=curve["budget"], y=curve[MODEL], name="LightGBM", line=dict(color=MODEL_COLOR, width=2))
    fig.add_scatter(x=curve["budget"], y=curve[RULES], name="rules", line=dict(color=RULES_COLOR, width=2))
    fig.add_vline(x=budget, line=dict(color=INK_2, width=1), annotation_text=f"{budget:.2%} budget")
    fig.update_xaxes(type="log", range=[np.log10(0.0005), np.log10(0.05)], tickvals=BUDGETS,
                     ticktext=[budget_label(b) for b in BUDGETS],
                     title="alert budget (share of transactions reviewed)")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02], title="recall (share of laundering caught)")
    fig.update_traces(hovertemplate="%{x:.2%} budget: %{y:.1%} recall")
    st.plotly_chart(style(fig, "Recall at each alert budget, test period", 440), width="stretch")

    data = pd.DataFrame(run.metrics["data"])
    data["base_rate"] = data["base_rate"].map(lambda r: pct(r, 3))
    st.markdown("**Temporal split** (train on the past, judge on the future)")
    st.dataframe(data, hide_index=True, width="stretch")


def alert_queue(run: sc.Run, budget: float, config: Config) -> None:
    q = sc.queue(run, budget)
    st.markdown(f"**{len(q):,} alerts** at a {budget:.2%} budget, highest score first. "
                "Select a row to see why it was alerted.")
    shown = q.head(QUEUE_ROWS_SHOWN).rename(columns={"is_laundering": "label"})
    event = st.dataframe(
        shown[["rank", "timestamp", "amount_usd", "payment_format", "model_score", "rules_fired", "label",
               "typology"]],
        hide_index=True, on_select="rerun", selection_mode="single-row", height=320, width="stretch",
        column_config={"timestamp": st.column_config.DatetimeColumn("time", format="MMM D, HH:mm"),
                       "amount_usd": st.column_config.NumberColumn("amount (USD)", format="dollar"),
                       "model_score": st.column_config.NumberColumn("score", format="%.4f"),
                       "label": st.column_config.CheckboxColumn("laundering (label)")},
    )
    if len(q) > QUEUE_ROWS_SHOWN:
        st.caption(f"Showing the top {QUEUE_ROWS_SHOWN:,}.")
    rows = event.selection.rows if event and event.selection else []
    alert = shown.iloc[rows[0] if rows else 0]
    ex = sc.explain_alert(run, int(alert["txn_id"]), config)

    left, right = st.columns([2, 3])
    with left:
        st.subheader(f"Alert #{alert['rank']}")
        st.caption(md(f"Transaction {alert['txn_id']} · {pd.Timestamp(alert['timestamp']):%b %d %H:%M} · "
                      f"${alert['amount_usd']:,.0f} · {alert['payment_format']}"))
        st.markdown("**Why it was alerted**")
        for i, reason in enumerate(ex.reasons, start=1):
            st.markdown(f"{i}. {md(reason.text)}")
        st.markdown(f"**Rule scenarios fired:** {alert['rules_fired'] or 'none'}")
        verdict = f"laundering ({alert['typology']})" if alert["label"] else "legitimate"
        st.markdown(f"**Label:** {verdict} (known only because the data is synthetic)")
    with right:
        top = ex.contributions.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top.to_numpy(), y=top.index, orientation="h",
            marker_color=[MODEL_COLOR if v > 0 else MUTED for v in top.to_numpy()],
            hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
        fig.update_xaxes(title="contribution to the score (log-odds); right = towards laundering",
                         zeroline=True, zerolinecolor=INK_2)
        st.plotly_chart(style(fig, f"SHAP, top 10 features: average {ex.expected_value:+.2f} → this alert "
                                   f"{ex.raw_score:+.2f} log-odds", 400), width="stretch")


def drivers(run: sc.Run, budget: float, config: Config) -> None:
    left, right = st.columns(2)
    with left:
        imp = run.importance.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["feature"], orientation="h", marker_color=MODEL_COLOR,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="mean |SHAP| (log-odds), test sample")
        st.plotly_chart(style(fig, "What moves the score", 480), width="stretch")
    with right:
        shares = cached_reason_shares(run, config, run_key(config), budget).head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=shares.to_numpy(), y=shares.index, orientation="h", marker_color=MODEL_COLOR,
                               hovertemplate="%{x:.1%} of alerts<extra></extra>"))
        fig.update_traces(width=min(0.8, 0.12 * len(shares) + 0.1), texttemplate="%{x:.0%}", textposition="auto")
        fig.update_xaxes(tickformat=".0%", range=[0, 1], title="share of alerts (up to 300 top alerts)")
        st.plotly_chart(style(fig, "Most common first reason", 480), width="stretch")

    for by, title, note in (
        ("payment_format", "Recall by payment format",
         "Read with the reasons above: if laundering paid one way is never caught, the model has "
         "learned the format, not the behaviour."),
        ("typology", "Recall by laundering typology",
         "\"unattributed\" is laundering in no documented scheme: the honest weak spot."),
    ):
        table = sc.breakdown(run, budget, by)
        labels = [f"{g} ({n:,})" for g, n in zip(table[by], table["positives"])]
        fig = go.Figure()
        fig.add_bar(x=labels, y=table[f"recall_{MODEL}"], name="LightGBM", marker_color=MODEL_COLOR)
        fig.add_bar(x=labels, y=table[f"recall_{RULES}"], name="rules", marker_color=RULES_COLOR)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title=f"recall at {budget:.2%}")
        fig.update_traces(hovertemplate="%{x}: %{y:.1%}", texttemplate="%{y:.0%}", textposition="outside")
        st.plotly_chart(style(fig, f"{title} (positives in brackets)", 380).update_layout(bargap=0.3),
                        width="stretch")
        st.caption(note)


def rules_tab(run: sc.Run) -> None:
    st.markdown("Six textbook transaction-monitoring scenarios, thresholds set a priori. A weighted sum "
                "of the flags ranks the alerts, so the rules are judged at the same budget as the model.")
    table = sc.rule_table(run)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "fires_on": st.column_config.NumberColumn("fires on (share of test)", format="percent"),
        "precision_when_fired": st.column_config.NumberColumn("precision when fired", format="percent"),
        "laundering_flagged": st.column_config.NumberColumn("share of laundering flagged", format="percent"),
    })
    st.caption("A rule that fires on a large share of payments can't rank anything; one that never "
               "touches laundering adds noise. Each rule sees one payment's surroundings; the model "
               "learns how they combine.")


def data_tab(run: sc.Run) -> None:
    daily = run.daily.sort_values("day")
    fig = go.Figure()
    for split, color in SPLIT_COLORS.items():
        part = daily[daily["split"] == split]
        fig.add_bar(x=part["day"], y=part["transactions"], name=split, marker_color=color,
                    hovertemplate="%{x|%b %d}: %{y:,} transactions")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="transactions per day")
    st.plotly_chart(style(fig, "Daily volume by split", 350), width="stretch")

    per_day = daily.groupby("day")[["transactions", "laundering"]].sum()
    fig = go.Figure(go.Scatter(x=per_day.index, y=per_day["laundering"] / per_day["transactions"],
                               line=dict(color=INK_2, width=2), mode="lines+markers",
                               hovertemplate="%{x|%b %d}: %{y:.3%} laundering<extra></extra>"))
    fig.update_yaxes(tickformat=".2%", title="laundering rate", rangemode="tozero")
    st.plotly_chart(style(fig, "Daily laundering rate", 300), width="stretch")
    st.caption("AMLworld's activity stops after Sep 10, while unfinished schemes keep running: the "
               "tail after it is 59% laundering, a simulator artifact, so it is cut before anything else.")


def main() -> None:
    st.set_page_config(page_title="amlcore dashboard", page_icon=":material/monitoring:", layout="wide")
    config, budget = sidebar()
    demo = config.variant == "DEMO"
    try:
        run = cached_run(config, run_key(config))
    except FileNotFoundError as exc:
        st.title("amlcore: AML alert ranking")
        if demo:
            st.info("No demo run yet. Click **Train on demo data** in the sidebar (a few seconds).")
        else:
            st.info(f"No saved run for {config.variant}. From the project root:\n\n"
                    f"```\nscripts/download_data.sh {config.variant}\npython -m amlcore.cli run\n```\n\n"
                    "or switch to **Demo (live)** in the sidebar.")
        st.caption(str(exc))
        return

    test = next(d for d in run.metrics["data"] if d["split"] == TEST)
    st.title("amlcore: AML alert ranking")
    st.caption(f"{run.variant} test period, {test['start'][:10]} to {test['end'][:10]}: "
               f"{test['transactions']:,} transactions, {test['laundering']:,} laundering. "
               f"LightGBM ({run.metrics['model']['trees']:,} trees) against six textbook rules.")
    if demo:
        st.warning("Demo data: about 3,000 synthetic transactions with easy schemes. The numbers are "
                   "toy-sized; switch to the saved run for real results.", icon=":material/science:")

    tabs = st.tabs(["Overview", "Alert queue", "What drives it", "Rules baseline", "Data"])
    with tabs[0]:
        overview(run, budget, config.evaluation.primary_budget)
    with tabs[1]:
        alert_queue(run, budget, config)
    with tabs[2]:
        drivers(run, budget, config)
    with tabs[3]:
        rules_tab(run)
    with tabs[4]:
        data_tab(run)


main()
