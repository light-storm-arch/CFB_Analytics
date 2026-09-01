"""Streamlit app -- the primary interface when there is no terminal available.

Designed to run on Streamlit Community Cloud, where you cannot shell in. Every
operation the CLI offers (load data, fetch real seasons, train, backtest, price
Kalshi markets) is reachable as a button here, and the heavy lifting lives in
``cfb.bootstrap`` so the CLI and this app cannot drift apart.

Two hosting facts shape the design:

* **The filesystem is ephemeral.** Anything written here disappears when the app
  reboots. Processed data therefore lives in the git repo (refreshed by a
  scheduled GitHub Action) and models are retrained on boot and held in
  Streamlit's resource cache.
* **Boot time is user-visible.** Ridge trains in about ten seconds and is the
  default for an automatic first train; the ensemble takes roughly a minute and
  is opt-in from the Setup page.
"""
from __future__ import annotations

import datetime as dt
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from cfb import bootstrap
from cfb.config import get_config
from cfb.models.registry import MODEL_REGISTRY
from cfb.pipeline import load_features

st.set_page_config(page_title="CFB Analytics", layout="wide",
                   page_icon="🏈", initial_sidebar_state="expanded")

DIST_METHODS = ["lattice", "mc", "blend", "kde", "plain"]
DIST_HELP = ("lattice: Student-t reshaped by learned key numbers (default) · "
             "mc: drive-level Monte Carlo · blend: both · "
             "kde: empirical residuals · plain: a normal, for comparison only")
PAGES = ["Setup", "Slate", "Game", "Model", "Backtest", "Live", "Kalshi"]


# ---------------------------------------------------------------- caching --
# Caches are keyed on a fingerprint of what is actually on disk, not on a
# per-session counter.  Streamlit's caches are shared across every browser
# session, so a session-local token would let one visitor train a model that
# every other visitor's cache never notices -- and would go stale again after a
# restart.  Fingerprints are correct in both cases and need no bookkeeping.
def _data_fingerprint() -> tuple:
    root = get_config().data_dir / "processed"
    if not root.exists():
        return ()
    return tuple(sorted(
        (p.name, p.stat().st_size, int(p.stat().st_mtime_ns))
        for p in root.glob("*.parquet")))


def _model_fingerprint(name: str) -> tuple:
    d = get_config().artifacts_dir / name
    out = []
    for f in ("config.json", "model.joblib", "sigma.joblib", "key_numbers.json"):
        fp = d / f
        out.append((f, int(fp.stat().st_mtime_ns)) if fp.exists() else (f, 0))
    return tuple(out)


@st.cache_resource(show_spinner=False)
def _cached_predictor(name: str, fingerprint: tuple):
    return bootstrap.load_predictor(name)


@st.cache_data(show_spinner=False)
def _cached_features(name: str, data_fp: tuple, model_fp: tuple):
    pred = _cached_predictor(name, model_fp)
    from cfb.features.build import FeatureConfig
    cfg = pred.cfg.feature_cfg if pred else FeatureConfig()
    return load_features(cfg=cfg)


@st.cache_data(show_spinner=False)
def _cached_status(data_fp: tuple):
    return bootstrap.data_status()


def get_predictor(name: str):
    return _cached_predictor(name, _model_fingerprint(name))


def get_features(name: str):
    return _cached_features(name, _data_fingerprint(), _model_fingerprint(name))


def artifact_names() -> list[str]:
    root = get_config().artifacts_dir
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "config.json").exists())


# ------------------------------------------------------------------ utils --
def run_with_progress(label: str, fn, *args, **kwargs):
    """Run a long bootstrap action with a live progress bar and clean errors."""
    box = st.empty()
    bar = st.progress(0.0, text=label)

    def cb(frac: float, msg: str):
        bar.progress(float(np.clip(frac, 0.0, 1.0)), text=msg)

    try:
        result = fn(*args, progress=cb, **kwargs)
    except Exception as exc:  # noqa: BLE001 - surface, never traceback at the user
        bar.empty()
        box.error(f"**{label} failed**\n\n{exc}")
        with st.expander("details"):
            st.code(traceback.format_exc())
        return None
    bar.empty()
    box.success(f"{label} — done")
    return result


def chip(ok: bool, yes: str, no: str) -> str:
    return f"{'✅' if ok else '⚪'} {yes if ok else no}"


def margin_chart(dist, home: str, away: str, market: float | None = None):
    import plotly.graph_objects as go

    f = dist.to_frame()
    f = f[(f["margin"] >= -49) & (f["margin"] <= 49)]
    colors = np.where(f["margin"] > 0, "#2b7bba", "#c4462f")
    fig = go.Figure(go.Bar(x=f["margin"], y=f["prob"], marker_color=colors,
                           hovertemplate="margin %{x}<br>%{y:.2%}<extra></extra>"))
    fig.add_vline(x=dist.mean(), line_dash="dash", line_color="#444",
                  annotation_text=f"model {dist.mean():+.1f}")
    if market is not None and np.isfinite(market):
        fig.add_vline(x=market, line_dash="dot", line_color="#e08c00",
                      annotation_text=f"market {market:+.1f}")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title=f"final margin (positive = {home} wins)",
                      yaxis_title="probability", showlegend=False,
                      title=f"{away} @ {home}")
    return fig


# ------------------------------------------------------------------- main --
def main():
    st.sidebar.title("🏈 CFB Analytics")
    status = _cached_status(_data_fingerprint())
    creds = bootstrap.credentials_status()
    names = artifact_names()

    st.sidebar.caption(chip(status.has_data, status.label, "no data loaded"))
    st.sidebar.caption(chip(bool(names), f"model: {names[0] if names else ''}",
                            "no model trained"))
    st.sidebar.caption(chip(creds["cfbd"], "CFBD key set", "no CFBD key"))
    st.sidebar.divider()

    # Force the Setup page until there is something to work with.
    blocked = (not status.has_data) or (not names)
    if blocked:
        st.sidebar.info("Finish setup to unlock the other views.")
        page = "Setup"
        name = names[0] if names else "default"
        method = "lattice"
    else:
        name = st.sidebar.selectbox("model artifact", names)
        method = st.sidebar.selectbox("distribution method", DIST_METHODS,
                                      help=DIST_HELP)
        page = st.sidebar.radio("view", PAGES, index=1)

    st.sidebar.divider()
    st.sidebar.caption(f"data: `{get_config().data_dir}`")
    if st.sidebar.button("clear caches", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    if page == "Setup":
        page_setup(status, creds, names)
        return

    predictor = get_predictor(name)
    if predictor is None:
        st.warning("That model artifact could not be loaded. Retrain it on the "
                   "Setup page.")
        return
    features = get_features(name)

    if page == "Slate":
        page_slate(predictor, features, method)
    elif page == "Game":
        page_game(predictor, features, method)
    elif page == "Model":
        page_model(predictor)
    elif page == "Backtest":
        page_backtest(predictor)
    elif page == "Live":
        page_live(predictor, features)
    elif page == "Kalshi":
        page_kalshi(predictor, features, method)


# ------------------------------------------------------------------ setup --
def page_setup(status, creds, names):
    st.title("Setup")
    st.caption("Everything the command line does, as buttons. Nothing here "
               "needs a terminal.")

    for step in bootstrap.first_run_plan(status, creds):
        st.info(step)

    c1, c2, c3 = st.columns(3)
    c1.metric("data", status.label if status.has_data else "none")
    c2.metric("completed games", f"{status.n_completed:,}" if status.has_data else "0")
    c3.metric("models trained", len(names))

    if status.has_data and status.is_synthetic:
        st.warning(
            "**This is synthetic data.** The whole pipeline works on it and the "
            "tests rely on it, but the team names are fake and no number here "
            "says anything about a real team. Add a CFBD key below for real games.",
            icon="⚠️")

    st.divider()
    tab_data, tab_train, tab_creds = st.tabs(
        ["1 · Load data", "2 · Train a model", "Credentials"])

    # ---- data ----
    with tab_data:
        st.subheader("Real games")
        if not creds["cfbd"]:
            st.warning("No CFBD API key configured — see the Credentials tab.")
        this_year = dt.date.today().year
        c1, c2, c3 = st.columns([2, 1, 1])
        yrs = c1.slider("seasons", 2005, this_year, (this_year - 9, this_year))
        plays = c2.checkbox("play-by-play", value=False,
                            help="Needed for the trained in-game model. One "
                                 "request per week per season, so this is slow.")
        seasons = list(range(yrs[0], yrs[1] + 1))
        c3.metric("seasons", len(seasons))
        if st.button("Fetch from CollegeFootballData", type="primary",
                     disabled=not creds["cfbd"], width="stretch"):
            out = run_with_progress("Fetching seasons", bootstrap.fetch_real_data,
                                    seasons, include_plays=plays)
            if out is not None:
                st.json(out)
                st.rerun()
        st.caption("8–10 seasons is the sweet spot. The sport changed materially "
                   "around 2014 (pace) and 2021 (portal/NIL), and 2020 was "
                   "structurally odd.")

        st.divider()
        st.subheader("Sample data (no API key needed)")
        st.caption("A synthetic league generated from a drive-level scoring "
                   "process, so key numbers are real and the whole stack is "
                   "exercisable. Useful for learning the tool; meaningless as a "
                   "prediction.")
        c1, c2 = st.columns(2)
        n_teams = c1.number_input("teams", 40, 200, 130, step=10)
        n_seasons = c2.number_input("seasons", 3, 15, 8)
        if st.button("Load sample data", width="stretch"):
            out = run_with_progress("Generating sample data",
                                    bootstrap.build_sample_data,
                                    n_teams=int(n_teams), n_seasons=int(n_seasons))
            if out is not None:
                st.json(out)
                st.rerun()

        if status.has_data:
            st.divider()
            st.subheader("What is loaded")
            from cfb.data.store import Store
            st.dataframe(Store().summary(), width="stretch", hide_index=True)

    # ---- training ----
    with tab_train:
        if not status.has_data:
            st.info("Load data first.")
        else:
            st.subheader("Train")
            c1, c2 = st.columns(2)
            model_name = c1.selectbox(
                "model", sorted(MODEL_REGISTRY),
                index=sorted(MODEL_REGISTRY).index("ridge"),
                help="ridge trains in ~10s and is usually within a hair of the "
                     "best. ensemble takes about a minute.")
            dist_method = c2.selectbox("distribution", DIST_METHODS, help=DIST_HELP)
            c3, c4 = st.columns(2)
            members = c3.multiselect("ensemble members", sorted(MODEL_REGISTRY),
                                     default=["ridge", "xgboost", "forest"],
                                     disabled=model_name != "ensemble")
            refit = c4.selectbox("walk-forward refit", ["season", "week"],
                                 help="'week' is a more honest simulation of "
                                      "in-season operation, and much slower.")
            include_market = st.checkbox(
                "use the betting line as a feature", value=False,
                help="Much more accurate, much less useful for finding edges — "
                     "it learns to copy the market. Leave off to hunt for edges.")
            art_name = st.text_input("save as", "default")

            if st.button("Train", type="primary", width="stretch"):
                out = run_with_progress(
                    f"Training {model_name}", bootstrap.train_model,
                    model_name=model_name, members=tuple(members or ["ridge"]),
                    include_market=include_market, dist_method=dist_method,
                    refit=refit, name=art_name)
                if out is not None:
                    predictor, oos = out
                    from cfb.evaluation.backtest import backtest_report
                    st.text(str(backtest_report(oos)))
                    st.caption("Those are walk-forward out-of-sample numbers — "
                               "the model never saw a game before predicting it.")
                    if predictor.cfg.min_train_games < bootstrap.PREFERRED_MIN_TRAIN_GAMES:
                        st.warning(
                            f"Short history: the walk-forward only required "
                            f"{predictor.cfg.min_train_games} prior games instead of "
                            f"{bootstrap.PREFERRED_MIN_TRAIN_GAMES}, so the early "
                            "blocks were predicted off thin ratings and these "
                            "numbers are pessimistic. Fetch more seasons for a "
                            "fair read.", icon="⚠️")
                    st.rerun()

        if names:
            st.divider()
            st.subheader("Trained models")
            for n in names:
                st.json(bootstrap.model_status(n))

    # ---- credentials ----
    with tab_creds:
        st.subheader("Status")
        st.write(chip(creds["cfbd"], "CFBD_API_KEY", "CFBD_API_KEY missing (required "
                                                     "for real data)"))
        st.write(chip(creds["kalshi_key"], "KALSHI_ACCESS_KEY",
                      "KALSHI_ACCESS_KEY not set (optional)"))
        st.write(chip(creds["kalshi_pem"], "Kalshi private key readable",
                      "Kalshi private key not set (optional)"))
        st.divider()
        st.markdown("""
**On Streamlit Community Cloud**, credentials go in the app's own secrets
manager — never in the repo. Open your app → **⋮ → Settings → Secrets**, paste:

```toml
CFBD_API_KEY = "your-key-here"

# optional, only for the Kalshi EV table
KALSHI_ACCESS_KEY = "your-access-key-id"
KALSHI_PRIVATE_KEY = \"\"\"-----BEGIN RSA PRIVATE KEY-----
...paste the whole .pem here...
-----END RSA PRIVATE KEY-----\"\"\"
```

Save, and the app restarts with them loaded.

A free CFBD key takes about a minute: **https://collegefootballdata.com/key**

**Running elsewhere?** The same names work as environment variables, or in a
`.env` file at the repo root.
""")
        if st.button("Re-read credentials"):
            get_config(refresh=True)
            st.rerun()


# ------------------------------------------------------------------ slate --
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
                  "pred_total", "sigma", "p_home_win", "p_home_by_1_3",
                  "p_home_by_4_7", "p_home_by_8_plus"]].copy()
    show.columns = ["away", "home", "proj margin", "fair spread", "proj total",
                    "sigma", "P(home)", "home by 1-3", "home by 4-7", "home by 8+"]
    if "market_margin" in slate and slate["market_margin"].notna().any():
        show.insert(4, "market", slate["market_margin"].to_numpy())
        show.insert(5, "edge", slate["edge_vs_market"].round(2).to_numpy())
    st.dataframe(show.round(3), width="stretch", hide_index=True)

    if "edge_vs_market" in slate and slate["edge_vs_market"].notna().any():
        st.subheader("Largest disagreements with the market")
        top = slate.reindex(slate["edge_vs_market"].abs()
                            .sort_values(ascending=False).index).head(6)
        for _, r in top.iterrows():
            side = r["home_team"] if r["edge_vs_market"] > 0 else r["away_team"]
            st.write(f"**{r['away_team']} @ {r['home_team']}** — model "
                     f"{r['pred_margin']:+.1f} vs market {r['market_margin']:+.1f} "
                     f"→ {abs(r['edge_vs_market']):.1f} pts toward **{side}**")
        st.caption("A disagreement is not an edge. Closing lines are hard to beat, "
                   "and the biggest gaps are often games where the market knows "
                   "something the model does not — an injury, most commonly.")

    st.download_button("download slate as CSV", slate.to_csv(index=False),
                       file_name=f"slate_{season}_wk{week}.csv", mime="text/csv")


# ------------------------------------------------------------------- game --
def page_game(predictor, feats, method):
    st.title("Game")
    teams = sorted(set(feats["home_team"]) | set(feats["away_team"]))
    upcoming = feats[feats["margin"].isna()]

    block = None
    if not upcoming.empty:
        labels = [f"{r.away_team} @ {r.home_team}" for r in upcoming.itertuples()]
        choice = st.selectbox("upcoming game", ["(pick teams manually)"] + labels)
        if choice != "(pick teams manually)":
            i = labels.index(choice)
            block = upcoming.iloc[[i]]
            home = block.iloc[0]["home_team"]
            away = block.iloc[0]["away_team"]
    if block is None:
        c1, c2, c3 = st.columns([2, 2, 1])
        home = c1.selectbox("home", teams)
        away = c2.selectbox("away", teams, index=min(1, len(teams) - 1))
        neutral = c3.checkbox("neutral site")
        if home == away:
            st.warning("Pick two different teams.")
            return
        from cfb.cli import _synthetic_matchup_row
        block = _synthetic_matchup_row(feats, home, away, neutral, None)

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
        st.subheader("Win-by buckets")
        tbl = d.bucket_table()
        tbl["bucket"] = tbl["bucket"].str.replace("home", home).str.replace("away", away)
        tbl["fair price (¢)"] = (tbl["prob"] * 100).round(1)
        st.dataframe(tbl.round(4), width="stretch", hide_index=True)
    with right:
        st.subheader("Most likely exact margins")
        top = d.to_frame().nlargest(12, "prob")[["margin", "prob"]].copy()
        top["fair price (¢)"] = (top["prob"] * 100).round(1)
        st.dataframe(top.round(4), width="stretch", hide_index=True)
        st.caption("These are the numbers a smooth normal gets most wrong — "
                   "switch the distribution method to `plain` to see by how much.")

    st.subheader("Interval")
    st.write(f"80% of outcomes fall between **{d.quantile(0.10):+d}** and "
             f"**{d.quantile(0.90):+d}** · median **{d.median():+d}** · "
             f"most likely **{d.mode():+d}**")


# ------------------------------------------------------------------ model --
def page_model(predictor):
    st.title("Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("model", predictor.model.name)
    c2.metric("features used", len(predictor.model.features_used))
    rep = predictor.model.report
    c3.metric("train MAE (in-sample)", f"{rep.margin_train_mae:.2f}" if rep else "—")

    st.subheader("Feature importance")
    fi = predictor.model.feature_importance()
    if fi.empty:
        st.info("this estimator does not expose feature importances")
    else:
        st.bar_chart(fi.head(20).set_index("feature"))

    st.subheader("Scale model")
    if predictor.sigma_model.fit_report:
        st.json(predictor.sigma_model.fit_report.__dict__)
    st.caption("`z_sd` should be ≈1.00 and `fitted_df` is tail thickness — lower "
               "means fatter tails. This is fit on out-of-sample residuals; "
               "fitting it in-sample gives a sigma 10–15% too small and makes "
               "every tail market look like a bargain.")

    st.subheader("Key numbers")
    kn = predictor.key_numbers.top(15)
    st.bar_chart(kn.set_index("margin"))
    st.caption("Multiplier on each integer margin relative to a smooth density. "
               "3 and 7 dominate because football scores are built from 3s and 7s.")


# --------------------------------------------------------------- backtest --
def page_backtest(predictor):
    st.title("Backtest")
    oos = predictor.oos
    if oos is None or oos.empty:
        st.info("No saved out-of-sample frame. Retrain from the Setup page.")
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
        st.subheader("By season")
        st.dataframe(rep.by_season.round(3), width="stretch", hide_index=True)
        cols = [c for c in ("margin_mae", "market_mae") if c in rep.by_season.columns]
        st.line_chart(rep.by_season.set_index("season")[cols])

    if "market_margin" in oos.columns and oos["market_margin"].notna().any():
        st.subheader("ATS vs the closing line, by disagreement threshold")
        st.dataframe(ats_by_threshold(oos).round(4), width="stretch", hide_index=True)
        st.caption("Break-even at −110 is 52.38%. Win rate should rise with the "
                   "threshold; if it doesn't, the model has no edge. And be brutal "
                   "about sample size — 200 bets tells you nothing.")

    st.divider()
    st.subheader("Distribution calibration")
    st.caption("Slower than the numbers above — it re-prices a sample of history "
               "under each method.")
    methods = st.multiselect("methods", DIST_METHODS, default=["plain", "lattice", "mc"])
    n_sample = st.slider("games to score", 200, min(3000, len(oos)),
                         min(800, len(oos)), step=100)
    if st.button("Run calibration"):
        _run_calibration(predictor, oos, methods, n_sample)


def _run_calibration(predictor, oos, methods, n_sample):
    from cfb.evaluation.calibration import distribution_report

    sample = oos.sample(min(n_sample, len(oos)), random_state=11).sort_index()
    sample = sample.assign(sigma=predictor.sigma_model.predict(sample))
    y = sample["margin"].to_numpy(float)
    cache, rows = {}, []
    bar = st.progress(0.0, text="scoring ...")
    for i, m in enumerate(methods):
        bar.progress(i / max(len(methods), 1), text=f"scoring '{m}' ...")
        ds = [predictor.distribution(a, s, t, m)
              for a, s, t in zip(sample.pred_margin, sample.sigma, sample.pred_total)]
        cache[m] = ds
        rep = distribution_report(ds, y)
        rows.append({"method": m, "CRPS": rep["crps_mean"], "log loss": rep["log_loss"],
                     "Brier": rep["brier"], "PIT chi2": rep["pit_chi2"],
                     "mean sd": rep["mean_sd"]})
    bar.empty()
    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
    st.caption("CRPS is dominated by the bulk of the distribution and barely moves "
               "between methods. The table below is where the difference lives — "
               "and where your money is.")

    st.subheader("Exact-margin accuracy")
    tbl = []
    for k in (1, 3, 4, 5, 7, 10, 14):
        row = {"margin": k, "actual": float(np.mean(np.abs(y) == k))}
        for m, ds in cache.items():
            row[m] = float(np.mean([d.p_exact(k) + d.p_exact(-k) for d in ds]))
        tbl.append(row)
    st.dataframe(pd.DataFrame(tbl).round(4), width="stretch", hide_index=True)

    if "coverage" in rep:
        st.subheader("Interval coverage")
        st.dataframe(rep["coverage"].round(4), width="stretch", hide_index=True)
        st.subheader("Win-probability reliability")
        st.dataframe(rep["calibration"].round(4), width="stretch", hide_index=True)


# ------------------------------------------------------------------- live --
def page_live(predictor, feats):
    st.title("Live")
    st.caption("Reads the public ESPN scoreboard — no API key needed.")
    c1, c2, c3 = st.columns([2, 1, 1])
    date = c1.text_input("date (YYYYMMDD, blank = today)", "")
    only_live = c2.checkbox("in progress only", value=True)
    auto = c3.checkbox("auto-refresh 60s", value=False)

    if auto:
        st.caption("Auto-refreshing. Untick to stop.")
    if not (auto or st.button("Refresh", type="primary")):
        st.info("Press Refresh to pull the current scoreboard.")
        return

    from cfb.live.diffusion import LiveConfig, LiveMarginModel
    from cfb.live.tracker import LiveTracker

    upcoming = feats[feats["margin"].isna()]
    pregame = predictor.predict_frame(upcoming) if not upcoming.empty else pd.DataFrame()
    tracker = LiveTracker(
        pregame=pregame,
        live_model=LiveMarginModel(LiveConfig(), key_numbers=predictor.key_numbers))
    try:
        with st.spinner("fetching scoreboard ..."):
            df = tracker.frame(date=date or None, only_live=only_live)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not reach the ESPN scoreboard: {exc}")
        return
    if df.empty:
        st.info("No games in progress. Untick 'in progress only' to see the full day.")
        return

    st.dataframe(df, width="stretch", hide_index=True)
    if (df["source"] == "flat_prior").any():
        st.warning(
            "Some games fell back to a flat prior — the model has no pregame "
            "number for them. That happens when the store has no upcoming games "
            "for this slate, or when team names could not be matched.", icon="⚠️")
    st.caption("`source` says where each pregame prior came from: `model` (yours), "
               "`espn_line` (the book line ESPN ships), or `flat_prior` (nothing).")

    if auto:
        import time
        time.sleep(60)
        st.rerun()


# ----------------------------------------------------------------- kalshi --
def page_kalshi(predictor, feats, method):
    st.title("Kalshi")
    creds = bootstrap.credentials_status()
    if not (creds["kalshi_key"] and creds["kalshi_pem"]):
        st.warning("No Kalshi credentials configured — see Setup → Credentials.")
    st.caption("Read-only. This app never places, modifies, or cancels an order.")

    from cfb.data.kalshi_client import KalshiClient

    if st.button("Discover college football markets"):
        try:
            with st.spinner("scanning open events ..."):
                df = KalshiClient().discover_football()
            st.session_state["kalshi_events"] = df
        except Exception as exc:  # noqa: BLE001
            st.error(f"Kalshi request failed: {exc}")
    if "kalshi_events" in st.session_state:
        ev = st.session_state["kalshi_events"]
        if ev.empty:
            st.info("No matching open events found.")
        else:
            st.dataframe(ev.head(60), width="stretch", hide_index=True)
            st.caption("Kalshi renames series between seasons, so nothing is "
                       "hardcoded — copy a `series_ticker` below.")

    st.divider()
    c1, c2, c3 = st.columns(3)
    series = c1.text_input("series ticker", "")
    bankroll = c2.number_input("bankroll ($)", 50.0, 1_000_000.0, 1000.0, step=50.0)
    kelly = c3.slider("fraction of Kelly", 0.05, 1.0, 0.25, step=0.05)
    c4, c5 = st.columns(2)
    min_edge = c4.slider("minimum edge", 0.0, 0.20, 0.02, step=0.01)
    dry = c5.checkbox("dry run (show how every market was read)", value=True)

    if not st.button("Price markets", type="primary", disabled=not series):
        return

    upcoming = feats[feats["margin"].isna()]
    if upcoming.empty:
        st.error("No unplayed games in the store to price against.")
        return
    with st.spinner("pricing ..."):
        slate, dists = predictor.predict_slate(upcoming, method=method)
        try:
            markets = KalshiClient().markets_frame(series_ticker=series)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Kalshi request failed: {exc}")
            return
        if markets.empty:
            st.error("Kalshi returned no open markets for that series ticker.")
            return
        from cfb.betting.edge import edge_table
        from cfb.betting.kalshi_map import price_markets
        from cfb.data.teams import TeamMatcher

        teams = sorted(set(slate["home_team"]) | set(slate["away_team"]))
        priced = price_markets(markets, dict(zip(slate["game_id"], dists)),
                               slate[["game_id", "home_team", "away_team"]],
                               TeamMatcher(teams))

    if dry:
        cols = ["ticker", "game_label", "interpretation", "market_kind",
                "parse_confident", "model_prob", "yes_ask", "no_ask", "skip_reason"]
        st.dataframe(priced[[c for c in cols if c in priced.columns]].round(4),
                     width="stretch", hide_index=True)
        st.error("**Read the `interpretation` column before trading.** A market "
                 "read from the wrong side inverts your edge — it does not just "
                 "shrink it. `parse_confident` is False where the reading came "
                 "from parsing title text rather than Kalshi's structured "
                 "strike fields.", icon="🛑")
        return

    table = edge_table(priced, bankroll=bankroll, kelly_fraction_of=kelly,
                       min_edge=min_edge)
    if table.empty:
        st.info("No markets cleared the edge threshold.")
    else:
        st.dataframe(table, width="stretch", hide_index=True)
    n_unmapped = int(priced["model_prob"].isna().sum())
    if n_unmapped:
        st.caption(f"{n_unmapped} markets could not be mapped — tick 'dry run' "
                   "to see why.")
    st.caption(f"Prices are **asks**, not mids — you have to cross the spread. "
               f"EV is net of Kalshi's fee. Stakes are {kelly:g}× Kelly on "
               f"${bankroll:,.0f}.")


if __name__ == "__main__":
    main()
