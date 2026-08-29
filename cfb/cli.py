"""``cfb`` command line.

Typical first session:

    cfb setup                       # check what credentials are present
    cfb synth                       # a fake universe so you can try everything
    cfb train --model ensemble      # walk-forward train + save artifacts
    cfb backtest --models ridge,xgboost,forest,ensemble
    cfb game --home "Ohio State" --away "Michigan"

With a CFBD key, replace ``cfb synth`` with ``cfb fetch --seasons 2015-2024``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

from cfb.config import CONFIG, get_config
from cfb.data.store import Store

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_seasons(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _echo_df(df: pd.DataFrame, empty: str = "(nothing to show)"):
    if df is None or len(df) == 0:
        click.echo(empty)
    else:
        click.echo(df.to_string(index=False))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", is_flag=True, help="Log progress to stderr.")
@click.pass_context
def main(ctx, verbose):
    """College football spread, distribution and in-game modeling."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------- setup --
@main.command()
def setup():
    """Show configuration and what still needs doing."""
    cfg = get_config(refresh=True)
    cfg.ensure_dirs()
    click.echo("paths")
    click.echo(f"  data      {cfg.data_dir}")
    click.echo(f"  artifacts {cfg.artifacts_dir}")
    click.echo("\ncredentials")
    ok = "✓"
    click.echo(f"  CFBD_API_KEY              {ok if cfg.cfbd_api_key else 'MISSING'}")
    click.echo(f"  KALSHI_ACCESS_KEY         {ok if cfg.kalshi_access_key else 'not set'}")
    have_pem = bool(cfg.kalshi_private_key_path and cfg.kalshi_private_key_path.exists())
    click.echo(f"  KALSHI_PRIVATE_KEY_PATH   {ok if have_pem else 'not set'}")
    click.echo("\nlocal store")
    _echo_df(Store(cfg).summary(), "  (empty - run `cfb fetch` or `cfb synth`)")
    if not cfg.cfbd_api_key:
        click.echo("\nNext step: get a free key at https://collegefootballdata.com/key")
        click.echo("then copy .env.example to .env and paste it in as CFBD_API_KEY.")
        click.echo("Meanwhile `cfb synth` builds a fake universe so you can try everything.")


@main.command()
def store():
    """Summarise the local parquet store."""
    _echo_df(Store().summary(), "(empty store)")


# ----------------------------------------------------------------- data --
@main.command()
@click.option("--seasons", default="2015-2024", help="e.g. 2019-2024 or 2021,2023")
@click.option("--plays/--no-plays", default=False,
              help="Also pull play-by-play (slow: one request per week).")
@click.option("--play-weeks", default="1-15", help="Weeks to pull plays for.")
def fetch(seasons, plays, play_weeks):
    """Download seasons from CollegeFootballData into the local store."""
    from cfb.data.cfbd_client import fetch_seasons

    yrs = _parse_seasons(seasons)
    click.echo(f"fetching {len(yrs)} season(s): {yrs[0]}-{yrs[-1]}")
    try:
        counts = fetch_seasons(yrs, include_plays=plays,
                               play_weeks=_parse_seasons(play_weeks) if plays else None)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    for table, n in sorted(counts.items()):
        click.echo(f"  {table:<18} {n:>7} rows")
    _echo_df(Store().summary())


@main.command()
@click.option("--teams", default=130, help="Number of synthetic teams.")
@click.option("--seasons", default=9, help="Number of synthetic seasons.")
@click.option("--start", default=2016, help="First synthetic season.")
@click.option("--pbp/--no-pbp", default=True, help="Also generate play-by-play states.")
@click.option("--seed", default=7)
def synth(teams, seasons, start, pbp, seed):
    """Generate a synthetic universe so the whole stack runs without an API key."""
    from cfb.data.synth import SynthConfig, build_synthetic_store

    cfg = SynthConfig(n_teams=teams, n_seasons=seasons, start_season=start, seed=seed)
    click.echo(f"generating {teams} teams x {seasons} seasons ...")
    out = build_synthetic_store(cfg, with_pbp=pbp)
    for k, v in out.items():
        click.echo(f"  {k:<14} {v:>7} rows")
    click.echo("\nNote: synthetic data. Useful for learning the tooling and for tests,")
    click.echo("meaningless as a prediction about real teams.")


# ---------------------------------------------------------------- train --
@main.command()
@click.option("--model", "model_name", default="ensemble",
              help="ridge | elasticnet | huber | forest | xgboost | lightgbm | ensemble")
@click.option("--members", default="ridge,xgboost,forest",
              help="Ensemble members (only used when --model ensemble).")
@click.option("--market/--no-market", default=False,
              help="Include the betting line as a feature. Off by default: a "
                   "market-blind model is what finds edges.")
@click.option("--dist", "dist_method", default="lattice",
              type=click.Choice(["lattice", "kde", "mc", "blend", "plain"]))
@click.option("--refit", default="season", type=click.Choice(["season", "week"]))
@click.option("--name", default="default", help="Artifact directory name.")
@click.option("--seasons", default=None, help="Restrict to these seasons.")
def train(model_name, members, market, dist_method, refit, name, seasons):
    """Walk-forward train a model plus its distribution layer, and save it."""
    from cfb.evaluation.backtest import backtest_report
    from cfb.features.build import FeatureConfig
    from cfb.pipeline import PredictorConfig, artifacts_dir, load_features, train_predictor

    fcfg = FeatureConfig(include_market=market)
    click.echo("building features ...")
    feats = load_features(cfg=fcfg, seasons=_parse_seasons(seasons) if seasons else None)
    click.echo(f"  {len(feats)} games, {int(feats['margin'].notna().sum())} completed")

    pcfg = PredictorConfig(model_name=model_name, members=tuple(members.split(",")),
                           include_market=market, dist_method=dist_method,
                           refit=refit, feature_cfg=fcfg)
    click.echo(f"walk-forward training '{model_name}' (refit per {refit}) ...")
    predictor, oos = train_predictor(feats, pcfg)
    click.echo("\n" + str(backtest_report(oos)))
    out = artifacts_dir(name)
    predictor.save(out)
    click.echo(f"\nsaved -> {out}")
    click.echo(f"sigma model: {predictor.sigma_model.fit_report}")
    click.echo("\nkey numbers learned:")
    _echo_df(predictor.key_numbers.top(10))


@main.command()
@click.option("--models", default="ridge,forest,xgboost,ensemble")
@click.option("--market/--no-market", default=False)
@click.option("--refit", default="season", type=click.Choice(["season", "week"]))
@click.option("--thresholds/--no-thresholds", default=True,
              help="Also show ATS results by disagreement threshold.")
def backtest(models, market, refit, thresholds):
    """Compare models with a strict walk-forward backtest."""
    from cfb.evaluation.backtest import (
        ats_by_threshold, compare_models, walk_forward_predictions,
    )
    from cfb.features.build import FeatureConfig, feature_columns
    from cfb.pipeline import load_features

    fcfg = FeatureConfig(include_market=market)
    feats = load_features(cfg=fcfg)
    cols = feature_columns(fcfg)
    names = [m.strip() for m in models.split(",") if m.strip()]
    click.echo(f"walk-forward over {len(feats)} games, refit per {refit}\n")
    table = compare_models(feats, cols, names, refit=refit, progress=False)
    _echo_df(table.round(4))
    if thresholds and names:
        best = table.iloc[0]["model"] if not table.empty else names[0]
        pred = walk_forward_predictions(feats, cols, model_name=best, refit=refit,
                                        progress=False)
        if not pred.empty and "market_margin" in pred.columns:
            click.echo(f"\nATS vs closing line, model '{best}', by disagreement:")
            _echo_df(ats_by_threshold(pred).round(4))
            click.echo("\nBreak-even at -110 is 0.5238. Treat anything under ~2,000 "
                       "bets as noise.")


@main.command("calibration")
@click.option("--name", default="default")
@click.option("--methods", default="plain,lattice,mc,blend")
@click.option("--sample", default=1500,
              help="Games to score (0 = all). Monte Carlo methods are slow at "
                   "full history and the estimates are stable well before then.")
def calibration_cmd(name, methods, sample):
    """Score the saved model's distributions: CRPS, log loss, PIT, coverage."""
    from cfb.evaluation.calibration import distribution_report, summarize_report
    from cfb.pipeline import Predictor, artifacts_dir

    pred = Predictor.load(artifacts_dir(name))
    if pred.oos is None or pred.oos.empty:
        raise click.ClickException("no saved out-of-sample frame; re-run `cfb train`")
    oos = pred.oos.copy()
    oos["sigma"] = pred.sigma_model.predict(oos)
    if sample and len(oos) > sample:
        oos = oos.sample(sample, random_state=11).sort_index()
        click.echo(f"(scoring a {sample}-game sample of the out-of-sample history)")
    y = oos["margin"].to_numpy(float)

    cache: dict[str, list] = {}
    rows = []
    for method in [m.strip() for m in methods.split(",") if m.strip()]:
        dists = [pred.distribution(m_, s, t, method)
                 for m_, s, t in zip(oos.pred_margin, oos.sigma, oos.pred_total)]
        cache[method] = dists
        rep = distribution_report(dists, y)
        rows.append({"method": method, "CRPS": rep["crps_mean"],
                     "log_loss": rep["log_loss"], "brier": rep["brier"],
                     "PIT_chi2": rep["pit_chi2"], "mean_sd": rep["mean_sd"]})
        if method == pred.cfg.dist_method:
            detail = rep
    _echo_df(pd.DataFrame(rows).round(4))

    default_method = pred.cfg.dist_method
    if default_method not in cache:
        default_method = next(iter(cache))
        detail = distribution_report(cache[default_method], y)
    click.echo(f"\ndetail for '{default_method}':\n")
    click.echo(summarize_report(detail))

    click.echo("\nExact-margin accuracy -- where the scoring lattice earns its keep.")
    click.echo("A smooth normal has no idea that 3 and 7 are special:")
    tbl = []
    for k in (1, 3, 4, 5, 7, 10, 14):
        row = {"margin": k, "actual": float(np.mean(np.abs(y) == k))}
        for method, ds in cache.items():
            row[method] = float(np.mean([d.p_exact(k) + d.p_exact(-k) for d in ds]))
        tbl.append(row)
    _echo_df(pd.DataFrame(tbl).round(4))


# -------------------------------------------------------------- predict --
@main.command()
@click.option("--name", default="default")
@click.option("--season", type=int, default=None)
@click.option("--week", type=int, default=None)
@click.option("--dist", "dist_method", default=None,
              type=click.Choice(["lattice", "kde", "mc", "blend", "plain"]))
@click.option("--upcoming/--all", default=True, help="Only games without a result.")
@click.option("--csv", type=click.Path(), default=None, help="Write the slate to CSV.")
def predict(name, season, week, dist_method, upcoming, csv):
    """Predict a slate: spread, win probability, and margin buckets."""
    from cfb.pipeline import Predictor, artifacts_dir, load_features

    predictor = Predictor.load(artifacts_dir(name))
    feats = load_features(cfg=predictor.cfg.feature_cfg)
    if season:
        feats = feats[feats["season"] == season]
    if week:
        feats = feats[feats["week"] == week]
    if upcoming:
        feats = feats[feats["margin"].isna()]
        if feats.empty:
            click.echo("No unplayed games matched. Use --all to score finished games.")
            return
    if feats.empty:
        raise click.ClickException("no games matched that filter")

    slate, dists = predictor.predict_slate(feats, method=dist_method)
    show = slate[["season", "week", "away_team", "home_team", "pred_margin",
                  "fair_spread_home", "pred_total", "sigma", "p_home_win",
                  "p_home_by_1_3", "p_home_by_4_7", "p_home_by_8_plus"]].copy()
    if "market_margin" in slate.columns and slate["market_margin"].notna().any():
        show["market"] = slate["market_margin"]
        show["edge"] = slate["edge_vs_market"]
    _echo_df(show.round(3))
    if csv:
        slate.to_csv(csv, index=False)
        click.echo(f"\nwrote {csv}")


@main.command()
@click.option("--name", default="default")
@click.option("--home", required=True)
@click.option("--away", required=True)
@click.option("--neutral", is_flag=True)
@click.option("--dist", "dist_method", default=None,
              type=click.Choice(["lattice", "kde", "mc", "blend", "plain"]))
@click.option("--season", type=int, default=None)
def game(name, home, away, neutral, dist_method, season):
    """Full distribution for one matchup, including 'wins by X' buckets."""
    from cfb.data.teams import TeamMatcher
    from cfb.pipeline import Predictor, artifacts_dir, load_features

    predictor = Predictor.load(artifacts_dir(name))
    feats = load_features(cfg=predictor.cfg.feature_cfg)
    teams = sorted(set(feats["home_team"]) | set(feats["away_team"]))
    matcher = TeamMatcher(teams)
    h, a = matcher.match(home), matcher.match(away)
    if not h or not a:
        raise click.ClickException(
            f"could not resolve team(s): {home if not h else ''} {away if not a else ''}".strip())

    row = _synthetic_matchup_row(feats, h, a, neutral, season)
    slate, dists = predictor.predict_slate(row, method=dist_method)
    d = dists[0]
    click.echo(f"\n{a} @ {h}" + ("  (neutral site)" if neutral else ""))
    click.echo("-" * 58)
    s = d.summary(h, a)
    click.echo(f"  projected margin   {s['mean_margin']:+.2f}  (home)")
    click.echo(f"  fair spread        {h} {s['fair_spread_home']:+.1f}")
    click.echo(f"  projected total    {float(slate['pred_total'].iloc[0]):.1f}")
    click.echo(f"  sigma              {float(slate['sigma'].iloc[0]):.2f}")
    click.echo(f"  P({h} wins)        {s['p_home_win']:.4f}")
    click.echo(f"  P({a} wins)        {s['p_away_win']:.4f}")
    click.echo(f"  most likely margin {s['modal_margin']:+d}")
    click.echo(f"  80% interval       [{d.quantile(0.10):+d}, {d.quantile(0.90):+d}]")
    click.echo("\nwin-by buckets:")
    tbl = d.bucket_table()
    tbl["bucket"] = tbl["bucket"].str.replace("home", h).str.replace("away", a)
    _echo_df(tbl.round(4))
    click.echo("\nmost likely exact margins:")
    top = d.to_frame().nlargest(10, "prob")[["margin", "prob"]]
    top["fair_cents"] = (top["prob"] * 100).round(1)
    _echo_df(top.round(4))


def _synthetic_matchup_row(feats: pd.DataFrame, home: str, away: str,
                           neutral: bool, season: int | None) -> pd.DataFrame:
    """Build a feature row for a hypothetical matchup from the latest known state.

    Uses each team's most recent feature row, so the ratings/form values are the
    freshest ones in the store.  Anything genuinely game-specific (rest days,
    market line) is left missing and the models fall back to their medians.
    """
    f = feats if season is None else feats[feats["season"] == season]
    if f.empty:
        f = feats
    f = f.sort_values(["season", "week"], kind="stable")

    def latest(team, side):
        sub = f[f[f"{side}_team"] == team]
        return sub.iloc[-1] if not sub.empty else None

    hr = latest(home, "home")
    hr = latest(home, "away") if hr is None else hr
    ar = latest(away, "away")
    ar = latest(away, "home") if ar is None else ar
    base = f.iloc[-1].copy()
    for col in f.columns:
        if col.endswith("_home") and hr is not None:
            src = col if col in hr.index else col.replace("_home", "_away")
            base[col] = hr.get(src, np.nan)
        elif col.endswith("_away") and ar is not None:
            src = col if col in ar.index else col.replace("_away", "_home")
            base[col] = ar.get(src, np.nan)
    base["home_team"], base["away_team"] = home, away
    base["neutral_site"] = 1.0 if neutral else 0.0
    for col in ("rat_margin", "rat_net_diff", "rat_off_diff", "rat_def_diff",
                "elo_diff", "market_margin", "market_total", "margin", "total",
                "game_id"):
        if col in base.index:
            base[col] = np.nan
    row = pd.DataFrame([base])
    # Rebuild the differences that were just invalidated.
    for stem in ("rat_net", "talent", "returning", "sp_prev", "form_margin", "sos"):
        h, a, diff = f"{stem}_home", f"{stem}_away", f"{stem}_diff"
        if h in row.columns and a in row.columns:
            row[diff] = row[h] - row[a]
    if {"rat_off_home", "rat_off_away"} <= set(row.columns):
        row["rat_off_diff"] = row["rat_off_home"] - row["rat_off_away"]
        row["rat_def_diff"] = row["rat_def_home"] - row["rat_def_away"]
    if {"rat_net_diff", "rat_hfa"} <= set(row.columns):
        hfa = 0.0 if neutral else float(row["rat_hfa"].fillna(2.4).iloc[0])
        row["rat_hfa"] = hfa
        row["rat_margin"] = row["rat_net_diff"] + hfa
    if {"elo_home_pre", "elo_away_pre"} <= set(row.columns):
        row["elo_diff"] = row["elo_home_pre"] - row["elo_away_pre"] + (0 if neutral else 60)
        row["elo_home_wp"] = 1 / (1 + 10 ** (-row["elo_diff"] / 400))
    row["game_id"] = -1
    return row


# ----------------------------------------------------------------- live --
@main.command("live")
@click.option("--name", default="default")
@click.option("--date", default=None, help="YYYYMMDD (defaults to today's slate).")
@click.option("--all-games", is_flag=True, help="Include games not yet started / final.")
@click.option("--trained/--no-trained", default=False,
              help="Also score with the play-by-play model (needs `cfb live-train`).")
@click.option("--live-model", default="live_winprob.joblib")
def live_cmd(name, date, all_games, trained, live_model):
    """Live in-game spreads and win probabilities from the ESPN scoreboard."""
    from cfb.live.diffusion import LiveConfig, LiveMarginModel
    from cfb.live.tracker import LiveTracker
    from cfb.live.winprob import LiveWinProbModel
    from cfb.pipeline import Predictor, artifacts_dir, load_features

    pregame = pd.DataFrame()
    key_numbers = None
    try:
        predictor = Predictor.load(artifacts_dir(name))
        feats = load_features(cfg=predictor.cfg.feature_cfg)
        upcoming = feats[feats["margin"].isna()]
        if not upcoming.empty:
            pregame = predictor.predict_frame(upcoming)
        key_numbers = predictor.key_numbers
    except (FileNotFoundError, OSError):
        click.echo("(no saved model; falling back to the ESPN line as the prior)")

    tm = None
    if trained:
        path = CONFIG.artifacts_dir / live_model
        if path.exists():
            tm = LiveWinProbModel.load(path)
        else:
            click.echo(f"(no trained live model at {path}; run `cfb live-train`)")

    tracker = LiveTracker(pregame=pregame, trained_model=tm,
                          live_model=LiveMarginModel(LiveConfig(), key_numbers=key_numbers))
    try:
        df = (tracker.trained_frame(date=date, only_live=not all_games) if tm
              else tracker.frame(date=date, only_live=not all_games))
    except Exception as exc:  # noqa: BLE001 - a network failure should not traceback
        raise click.ClickException(
            f"could not reach the ESPN scoreboard: {exc}\n"
            "Check your connection; no API key is required for this endpoint."
        ) from exc
    _echo_df(df, "no games in progress right now")


@main.command("live-train")
@click.option("--name", default="default")
@click.option("--out", default="live_winprob.joblib")
def live_train(name, out):
    """Train the play-by-play in-game model on stored plays."""
    from cfb.live.winprob import LiveWinProbModel, attach_pregame, states_from_plays
    from cfb.pipeline import Predictor, artifacts_dir

    store = Store()
    states = store.read("pbp_states")
    if states.empty:
        plays = store.read("plays")
        games = store.read("games")
        if plays.empty:
            raise click.ClickException(
                "No play-by-play in the store. Run `cfb fetch --seasons ... --plays` "
                "(slow) or `cfb synth` to generate synthetic states.")
        states = states_from_plays(plays, games)

    if "pregame_mu" not in states.columns:
        lines = store.read("lines")
        if not lines.empty:
            pre = (lines.groupby("game_id")
                   .agg(market_margin=("market_margin", "median"),
                        market_total=("over_under", "median")).reset_index())
            states = attach_pregame(states, pre)

    key_numbers = None
    try:
        key_numbers = Predictor.load(artifacts_dir(name)).key_numbers
    except (FileNotFoundError, OSError):
        pass

    click.echo(f"training on {len(states)} play-by-play states ...")
    model = LiveWinProbModel(key_numbers=key_numbers).fit(states)
    path = CONFIG.artifacts_dir / out
    model.save(path)
    click.echo(f"saved -> {path}")
    click.echo(f"validation: {model.report}")


# --------------------------------------------------------------- kalshi --
@main.group()
def kalshi():
    """Kalshi market data: discovery and model-vs-market pricing (read-only)."""


@kalshi.command("discover")
@click.option("--keywords", default="football,ncaa,cfb,college")
def kalshi_discover(keywords):
    """List open Kalshi events that look like college football."""
    from cfb.data.kalshi_client import KalshiClient

    client = KalshiClient()
    if not client.authenticated:
        click.echo("(no Kalshi credentials found; trying unauthenticated read)")
    try:
        df = client.discover_football(tuple(k.strip() for k in keywords.split(",")))
    except RuntimeError as exc:
        raise click.ClickException(
            f"{exc}\nSee docs/DATA_SETUP.md section 3 for Kalshi API key setup."
        ) from exc
    _echo_df(df.head(60), "no matching open events found")
    if not df.empty:
        click.echo("\nUse a series_ticker from above with: cfb kalshi price --series <TICKER>")


@kalshi.command("price")
@click.option("--name", default="default")
@click.option("--series", default=None, help="Kalshi series ticker to price.")
@click.option("--event", default=None, help="Kalshi event ticker to price.")
@click.option("--bankroll", default=1000.0)
@click.option("--kelly", "kelly_frac", default=0.25, help="Fraction of Kelly to stake.")
@click.option("--min-edge", default=0.02)
@click.option("--min-volume", default=0)
@click.option("--dist", "dist_method", default=None,
              type=click.Choice(["lattice", "kde", "mc", "blend", "plain"]))
@click.option("--dry-run", is_flag=True,
              help="Show how every market was interpreted, before filtering on edge.")
def kalshi_price(name, series, event, bankroll, kelly_frac, min_edge, min_volume,
                 dist_method, dry_run):
    """Price open Kalshi markets against the model and rank by expected value."""
    from cfb.betting.edge import edge_table
    from cfb.betting.kalshi_map import price_markets
    from cfb.data.kalshi_client import KalshiClient
    from cfb.data.teams import TeamMatcher
    from cfb.pipeline import Predictor, artifacts_dir, load_features

    predictor = Predictor.load(artifacts_dir(name))
    feats = load_features(cfg=predictor.cfg.feature_cfg)
    upcoming = feats[feats["margin"].isna()]
    if upcoming.empty:
        raise click.ClickException("no unplayed games in the store to price against")
    slate, dists = predictor.predict_slate(upcoming, method=dist_method)
    dist_by_game = dict(zip(slate["game_id"], dists))

    client = KalshiClient()
    markets = client.markets_frame(series_ticker=series, event_ticker=event)
    if markets.empty:
        raise click.ClickException(
            "Kalshi returned no open markets. Run `cfb kalshi discover` to find a "
            "current series ticker (they change between seasons).")

    teams = sorted(set(slate["home_team"]) | set(slate["away_team"]))
    priced = price_markets(markets, dist_by_game,
                           slate[["game_id", "home_team", "away_team"]],
                           TeamMatcher(teams))
    if dry_run:
        cols = ["ticker", "game_label", "interpretation", "market_kind",
                "parse_confident", "model_prob", "yes_ask", "no_ask", "skip_reason"]
        _echo_df(priced[[c for c in cols if c in priced.columns]].round(4))
        click.echo("\nRead the `interpretation` column carefully before trading: "
                   "a market read from the wrong side inverts your edge.")
        return

    table = edge_table(priced, bankroll=bankroll, kelly_fraction_of=kelly_frac,
                       min_edge=min_edge, min_volume=min_volume)
    _echo_df(table, "no markets cleared the edge threshold")
    n_unmapped = int(priced["model_prob"].isna().sum())
    if n_unmapped:
        click.echo(f"\n({n_unmapped} markets could not be mapped; --dry-run shows why)")
    click.echo("\nPrices are asks and EV is net of Kalshi fees. Stakes are "
               f"{kelly_frac:g}x Kelly on a ${bankroll:g} bankroll.")


# ------------------------------------------------------------ dashboard --
@main.command()
@click.option("--port", default=8501)
def dashboard(port):
    """Launch the Streamlit dashboard."""
    import subprocess

    app = Path(__file__).parent / "dashboard" / "app.py"
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app),
                        "--server.port", str(port)], check=True)
    except FileNotFoundError as exc:
        raise click.ClickException(
            "streamlit is not installed: pip install 'cfb-analytics[dash]'") from exc


if __name__ == "__main__":
    main()
