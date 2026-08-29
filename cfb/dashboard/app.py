"""Streamlit dashboard: slates, distributions, model comparison, live tracker.

Run with ``cfb dashboard`` (or ``streamlit run cfb/dashboard/app.py``).

The dashboard is deliberately a *viewer*, not a second implementation: every
number on screen comes from the same ``Predictor`` the CLI uses, so there is no
chance of the two disagreeing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from cfb.config import CONFIG
from cfb.pipeline import Predictor, artifacts_dir, load_features

st.set_page_config(page_title="CFB Analytics", layout="wide",
                   initial_sidebar_state="expanded")

DIST_METHODS = ["lattice", "mc", "blend", "kde", "plain"]


# ---------------------------------------------------------------- cache --
@st.cache_resource(show_spinner=False)
def get_predictor(name: str):
    return Predictor.load(artifacts_dir(name))


@st.cache_data(show_spinner=False)
def get_features(name: str):
    pred = get_predictor(name)
    return load_features(cfg=pred.cfg.feature_cfg)


def available_models() -> list[str]:
    root = CONFIG.artifacts_dir
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "config.json").exists())


# --------------------------------------------------------------- charts --
def margin_chart(dist, home: str, away: str, market: float | None = None):
    import plotly.graph_objects as go

    f = dist.to_frame()
    f = f[(f["margin"] >= -49) & (f["margin"] <= 49)]
    colors = np.where(f["margin"] > 0, "#2b7bba", "#c4462f")
    fig = go.Figure(go.Bar(x=f["margin"], y=f["prob"], marker_color=colors,
                           hovertemplate="margin %{x}<br>%{y:.3%}<extra></extra>"))
    fig.add_vline(x=dist.mean(), line_dash="dash", line_color="#444",
                  annotation_text=f"model {dist.mean():+.1f}")
    if market is not None and np.isfinite(market):
        fig.add_vline(x=market, line_dash="dot", line_color="#e08c00",
                      annotation_text=f"market {market:+.1f}")
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=f"final margin  (positive = {home} wins)",
        yaxis_title="probability", showlegend=False,
        title=f"{away} @ {home}",
    )
    return fig


def cumulative_chart(dist):
    import plotly.graph_objects as go

    f = dist.to_frame()
    f = f[(f["margin"] >= -49) & (f["margin"] <= 49)]
    fig = go.Figure(go.Scatter(x=f["margin"], y=f["cum_prob"], mode="lines",
                               line=dict(color="#2b7bba", width=2)))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="margin", yaxis_title="P(margin <= x)",
                      title="cumulative distribution")
    return fig


# ----------------------------------------------------------------- main --
def main():
    st.sidebar.title("CFB Analytics")
    models = available_models()
    if not models:
        st.title("No trained model yet")
        st.markdown(
            "Train one first:\n\n"
            "```bash\n"
            "cfb synth            # or: cfb fetch --seasons 2015-2024\n"
            "cfb train --model ensemble\n"
            "```")
        st.stop()

    name = st.sidebar.selectbox("model artifact", models)
    method = st.sidebar.selectbox("distribution method", DIST_METHODS,
                                  help="lattice = t-density reshaped by key numbers; "
                                       "mc = drive-level simulation; "
                                       "plain = normal, for comparison only")
    page = st.sidebar.radio("view", ["Slate", "Game", "Model", "Backtest", "Live"])
    st.sidebar.caption(f"data: {CONFIG.data_dir}")

    predictor = get_predictor(name)
    feats = get_features(name)

    if page == "Slate":
        page_slate(predictor, feats, method)
    elif page == "Game":
        page_game(predictor, feats, method)
    elif page == "Model":
        page_model(predictor)
    elif page == "Backtest":
        page_backtest(predictor)
    else:
        page_live(predictor, feats)


def page_slate(predictor, feats, method):
    st.title("Slate")
    seasons = sorted(feats["season"].unique())
    c1, c2, c3 = st.columns([1, 1, 2])
    season = c1.selectbox("season", seasons, index=len(seasons) - 1)
    sub = feats[feats["season"] == season]
    weeks = sorted(sub["week"].unique())
    week = c2.selectbox("week", weeks, index=len(weeks) - 1)
    only_upcoming = c3.checkbox("only unplayed games", value=True)

    block = sub[sub["week"] == week]
    if only_upcoming:
        block = block[block["margin"].isna()]
    if block.empty:
        st.info("No games match. Untick 'only unplayed games' to score finished ones.")
        return

    with st.spinner("pricing slate ..."):
        slate, dists = predictor.predict_slate(block, method=method)

    show = slate[["away_team", "home_team", "pred_margin", "fair_spread_home",
                  "pred_total", "sigma", "p_home_win",
                  "p_home_by_1_3", "p_home_by_4_7", "p_home_by_8_plus"]].copy()
    show.columns = ["away", "home", "proj margin", "fair spread", "proj total",
                    "sigma", "P(home)", "home by 1-3", "home by 4-7", "home by 8+"]
    if "market_margin" in slate and slate["market_margin"].notna().any():
        show.insert(4, "market", slate["market_margin"].to_numpy())
        show.insert(5, "edge", slate["edge_vs_market"].round(2).to_numpy())
    st.dataframe(show.round(3), width="stretch", hide_index=True)

    if "edge_vs_market" in slate and slate["edge_vs_market"].notna().any():
        st.subheader("largest disagreements with the market")
        top = slate.reindex(slate["edge_vs_market"].abs()
                            .sort_values(ascending=False).index).head(6)
        for _, r in top.iterrows():
            side = r["home_team"] if r["edge_vs_market"] > 0 else r["away_team"]
            st.write(f"**{r['away_team']} @ {r['home_team']}** — model "
                     f"{r['pred_margin']:+.1f} vs market {r['market_margin']:+.1f} "
                     f"→ {abs(r['edge_vs_market']):.1f} pts toward **{side}**")

    st.caption("Positive margin = home team wins by that much. "
               "'Fair spread' is quoted the way a book would quote the home side.")


def page_game(predictor, feats, method):
    st.title("Game")
    teams = sorted(set(feats["home_team"]) | set(feats["away_team"]))
    upcoming = feats[feats["margin"].isna()]
    c1, c2 = st.columns(2)
    if not upcoming.empty:
        labels = [f"{r.away_team} @ {r.home_team}" for r in upcoming.itertuples()]
        choice = st.selectbox("upcoming game", ["(pick teams manually)"] + labels)
        if choice != "(pick teams manually)":
            row = upcoming.iloc[labels.index(choice)]
            home, away = row["home_team"], row["away_team"]
            block = upcoming.iloc[[labels.index(choice)]]
        else:
            home = c1.selectbox("home", teams)
            away = c2.selectbox("away", teams, index=min(1, len(teams) - 1))
            block = _manual_row(feats, home, away)
    else:
        home = c1.selectbox("home", teams)
        away = c2.selectbox("away", teams, index=min(1, len(teams) - 1))
        block = _manual_row(feats, home, away)

    slate, dists = predictor.predict_slate(block, method=method)
    d = dists[0]
    r = slate.iloc[0]
    market = r.get("market_margin", np.nan)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("projected margin", f"{d.mean():+.1f}")
    m2.metric(f"fair spread ({home})", f"{-d.mean():+.1f}")
    m3.metric(f"P({home} wins)", f"{d.p_home_win():.1%}")
    m4.metric("sigma", f"{float(r['sigma']):.1f}")

    st.plotly_chart(margin_chart(d, home, away, market), width="stretch")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("win-by buckets")
        tbl = d.bucket_table()
        tbl["bucket"] = tbl["bucket"].str.replace("home", home).str.replace("away", away)
        tbl["fair price (c)"] = (tbl["prob"] * 100).round(1)
        st.dataframe(tbl.round(4), width="stretch", hide_index=True)
    with right:
        st.subheader("most likely exact margins")
        top = d.to_frame().nlargest(12, "prob")[["margin", "prob"]].copy()
        top["fair price (c)"] = (top["prob"] * 100).round(1)
        st.dataframe(top.round(4), width="stretch", hide_index=True)
    st.plotly_chart(cumulative_chart(d), width="stretch")


def _manual_row(feats, home, away):
    from cfb.cli import _synthetic_matchup_row
    return _synthetic_matchup_row(feats, home, away, neutral=False, season=None)


def page_model(predictor):
    st.title("Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("model", predictor.model.name)
    c2.metric("features used", len(predictor.model.features_used))
    rep = predictor.model.report
    c3.metric("train MAE (in-sample)", f"{rep.margin_train_mae:.2f}" if rep else "-")

    st.subheader("feature importance")
    fi = predictor.model.feature_importance()
    if fi.empty:
        st.info("this estimator does not expose feature importances")
    else:
        st.bar_chart(fi.head(20).set_index("feature"))

    st.subheader("scale model")
    if predictor.sigma_model.fit_report:
        st.json(predictor.sigma_model.fit_report.__dict__)
    st.caption("z_sd should be ~1.00 and t_df is the fitted tail thickness "
               "(lower = fatter tails).")

    st.subheader("key numbers")
    kn = predictor.key_numbers.top(15)
    st.bar_chart(kn.set_index("margin"))
    st.caption("Multiplier on each integer margin relative to a smooth density. "
               "3 and 7 dominate because football scores are built from 3s and 7s.")


def page_backtest(predictor):
    st.title("Backtest")
    oos = predictor.oos
    if oos is None or oos.empty:
        st.info("no saved out-of-sample frame")
        return
    from cfb.evaluation.backtest import ats_by_threshold, backtest_report

    rep = backtest_report(oos)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("games", f"{rep.n:,}")
    c2.metric("margin MAE", f"{rep.margin_mae:.2f}")
    c3.metric("straight-up", f"{rep.su_accuracy:.1%}")
    if rep.market_mae:
        c4.metric("vs market MAE", f"{rep.margin_mae - rep.market_mae:+.2f}",
                  delta_color="inverse")
    st.text(str(rep))

    if rep.by_season is not None and not rep.by_season.empty:
        st.subheader("by season")
        st.dataframe(rep.by_season.round(3), width="stretch", hide_index=True)
        chart = rep.by_season.set_index("season")[
            [c for c in ("margin_mae", "market_mae") if c in rep.by_season.columns]]
        st.line_chart(chart)

    if "market_margin" in oos.columns and oos["market_margin"].notna().any():
        st.subheader("ATS vs the closing line, by disagreement threshold")
        st.dataframe(ats_by_threshold(oos).round(4), width="stretch", hide_index=True)
        st.caption("Break-even at -110 is 52.38%. Under a few thousand bets, "
                   "anything in the low 50s is indistinguishable from noise.")


def page_live(predictor, feats):
    st.title("Live")
    st.caption("Pulls the ESPN scoreboard (no API key needed).")
    date = st.text_input("date (YYYYMMDD, blank = today)", "")
    only_live = st.checkbox("only games in progress", value=True)
    if not st.button("refresh", type="primary"):
        st.info("Press refresh to pull the current scoreboard.")
        return

    from cfb.live.diffusion import LiveConfig, LiveMarginModel
    from cfb.live.tracker import LiveTracker

    upcoming = feats[feats["margin"].isna()]
    pregame = predictor.predict_frame(upcoming) if not upcoming.empty else pd.DataFrame()
    tracker = LiveTracker(
        pregame=pregame,
        live_model=LiveMarginModel(LiveConfig(), key_numbers=predictor.key_numbers),
    )
    try:
        df = tracker.frame(date=date or None, only_live=only_live)
    except Exception as exc:  # noqa: BLE001
        st.error(f"could not reach ESPN: {exc}")
        return
    if df.empty:
        st.info("no games in progress")
        return
    st.dataframe(df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
