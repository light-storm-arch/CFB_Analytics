"""Walk-forward, leak-free feature construction.

The single most important property of this module: **every feature attached to
a game is computable strictly before that game kicks off.**  Concretely,

* ratings are refit once per (season, week) on games that finished before the
  first kickoff of that week;
* Elo is a forward pass that records pregame values;
* rolling form uses ``shift(1)`` within team so a game never sees itself;
* SP+ is joined from the *prior* season (CFBD's SP+ for season Y reflects games
  played during Y, so joining it to Y's games would leak the season's results);
* recruiting talent and returning production are known before the season, so
  they join on the current season.

Market data (spread / total) is optional and off by default for training the
"market-blind" models you would use to find edges, and on for the market-aware
model that is a better in-game prior.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfb.features._rolling import group_codes, trailing_count, trailing_mean
from cfb.features.preseason import priors_for_season
from cfb.features.roster import build_roster_features
from cfb.features.ratings import RatingFit, elo_ratings, fit_off_def_ratings

log = logging.getLogger(__name__)

# Feature blocks. `MARKET_FEATURES` are appended only when include_market=True.
RATING_FEATURES = [
    "rat_margin", "rat_total", "rat_hfa",
    "rat_net_home", "rat_net_away", "rat_net_diff",
    "rat_off_home", "rat_def_home", "rat_off_away", "rat_def_away",
    "rat_off_diff", "rat_def_diff",
    "rat_games_home", "rat_games_away",
]
ELO_FEATURES = ["elo_home_pre", "elo_away_pre", "elo_diff", "elo_home_wp"]
CONTEXT_FEATURES = [
    "week", "neutral_site", "conference_game",
    "rest_days_home", "rest_days_away", "rest_diff",
    "played_home", "played_away",
]
FORM_FEATURES = [
    "form_margin_home", "form_margin_away", "form_margin_diff",
    "form_points_for_home", "form_points_for_away",
    "form_points_against_home", "form_points_against_away",
    "form_total_home", "form_total_away",
    "sos_home", "sos_away", "sos_diff",
]
PRIOR_FEATURES = [
    "talent_home", "talent_away", "talent_diff",
    "returning_home", "returning_away", "returning_diff",
    "sp_prev_home", "sp_prev_away", "sp_prev_diff",
]
#: Roster-continuity block. All preseason-known; see cfb.features.roster.
ROSTER_FEATURE_STEMS = [
    "portal_net_rating", "portal_in_count", "portal_out_count", "portal_in_best",
    "qb_prior_ppa", "qb_departed", "qb_transfer_in", "qb_continuity",
    "returning_off", "returning_def",
]
ROSTER_FEATURES = [f"{stem}_{side}" for stem in ROSTER_FEATURE_STEMS
                   for side in ("home", "away", "diff")]

MARKET_FEATURES = ["market_margin", "market_total", "market_margin_open", "line_move"]

TARGETS = ["margin", "total", "home_win"]


@dataclass
class FeatureConfig:
    half_life_days: float = 400.0
    ridge_lambda: float = 45.0
    cap_points: float | None = 52.0
    form_window: int = 3
    include_market: bool = False
    #: Roster-continuity feature block (portal flux, QB continuity).
    #: OFF by default: measured across four synthetic seasons-sets it costs
    #: +0.037 MAE (sd 0.022) -- thirty extra low-signal columns buy variance,
    #: not accuracy. Turn on and re-measure once you have real data;
    #: `cfb ablate` runs exactly that comparison.
    include_roster: bool = False
    #: Shrink early-season ratings toward a fitted preseason prior instead of
    #: toward the league mean (cfb.features.preseason). Improves the ratings
    #: themselves by ~0.11 MAE but is neutral downstream (+0.010, sd 0.013),
    #: because the model has other paths to the same information.
    use_roster_prior: bool = False
    prior_alpha: float = 5.0
    min_history_games: int = 150
    pool_non_fbs: bool = True


# ---------------------------------------------------------------------- #
# market consensus
# ---------------------------------------------------------------------- #
PROVIDER_PRIORITY = ["consensus", "Bovada", "DraftKings", "ESPN Bet", "William Hill (New Jersey)"]


def consensus_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse many books into one row per game (median is robust to outliers)."""
    if lines is None or lines.empty:
        return pd.DataFrame(columns=["game_id", "market_margin", "market_total",
                                     "market_margin_open"])
    df = lines.copy()
    if "market_margin" not in df.columns:
        df["market_margin"] = -df["spread"]
    if "spread_open" in df.columns:
        df["market_margin_open"] = -df["spread_open"]
    else:
        df["market_margin_open"] = np.nan
    agg = df.groupby("game_id").agg(
        market_margin=("market_margin", "median"),
        market_total=("over_under", "median"),
        market_margin_open=("market_margin_open", "median"),
        n_books=("provider", "nunique"),
    ).reset_index()
    agg["line_move"] = agg["market_margin"] - agg["market_margin_open"]
    return agg


# ---------------------------------------------------------------------- #
# per-team rolling history (vectorised, shift(1) => no self-leakage)
# ---------------------------------------------------------------------- #
def team_game_long(games: pd.DataFrame) -> pd.DataFrame:
    """Explode games into one row per (game, team) with that team's perspective."""
    base = games[["game_id", "season", "week", "start_date", "neutral_site",
                  "home_team", "away_team", "home_points", "away_points"]].copy()
    home = base.rename(columns={"home_team": "team", "away_team": "opponent",
                                "home_points": "points_for",
                                "away_points": "points_against"})
    home["is_home"] = True
    away = base.rename(columns={"away_team": "team", "home_team": "opponent",
                                "away_points": "points_for",
                                "home_points": "points_against"})
    away["is_home"] = False
    long = pd.concat([home, away], ignore_index=True)
    long["margin"] = long["points_for"] - long["points_against"]
    long["game_total"] = long["points_for"] + long["points_against"]
    return long.sort_values(["team", "start_date", "game_id"],
                            kind="stable").reset_index(drop=True)


def rolling_form(long: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Trailing form per team-season, always excluding the current game.

    ``long`` must already be sorted by (team, start_date), which
    ``team_game_long`` guarantees -- the prefix-sum helpers rely on it.
    """
    codes = group_codes(long["team"].to_numpy(), long["season"].to_numpy())

    out = long[["game_id", "team", "season", "start_date"]].copy()
    for src, dest in (("margin", "form_margin"),
                      ("points_for", "form_points_for"),
                      ("points_against", "form_points_against"),
                      ("game_total", "form_total")):
        out[dest] = trailing_mean(long[src].to_numpy(), codes, window)
    out["played"] = trailing_count(codes)

    prev_date = long.groupby(["team", "season"], sort=False)["start_date"].shift(1)
    rest = (long["start_date"] - prev_date).dt.total_seconds() / 86400.0
    out["rest_days"] = rest.fillna(14.0).clip(3, 30)
    return out


# ---------------------------------------------------------------------- #
# main builder
# ---------------------------------------------------------------------- #
def build_features(
    games: pd.DataFrame,
    lines: pd.DataFrame | None = None,
    talent: pd.DataFrame | None = None,
    returning: pd.DataFrame | None = None,
    sp_ratings: pd.DataFrame | None = None,
    portal: pd.DataFrame | None = None,
    player_ppa: pd.DataFrame | None = None,
    cfg: FeatureConfig | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    cfg = cfg or FeatureConfig()
    g = games.copy()
    g["start_date"] = pd.to_datetime(g["start_date"], utc=True)
    if "completed" not in g.columns:
        g["completed"] = g["home_points"].notna() & g["away_points"].notna()
    g = g.sort_values(["start_date", "game_id"], kind="stable").reset_index(drop=True)

    # ---- Elo forward pass (already leak-free) ----
    elo_df, _ = elo_ratings(g)

    # ---- roster table, needed by both the prior and the feature block ----
    roster = (build_roster_features(portal, player_ppa, returning)
              if (cfg.include_roster or cfg.use_roster_prior) else pd.DataFrame())

    # ---- walk-forward ratings, refit once per (season, week) ----
    rat_rows: list[dict] = []
    keys = g[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    prior_cache: dict[int, tuple[dict, dict]] = {}
    for season, week in keys.itertuples(index=False):
        block = g[(g["season"] == season) & (g["week"] == week)]
        cutoff = block["start_date"].min()
        history = g[(g["start_date"] < cutoff) & g["completed"]]
        if len(history) < 10:
            fit = None
        else:
            prior_off, prior_def = {}, {}
            if cfg.use_roster_prior:
                if season not in prior_cache:
                    # Priors depend only on seasons strictly before this one, so
                    # they are computed once per season rather than per week.
                    prior_cache[season] = priors_for_season(
                        g[g["completed"]], roster, int(season),
                        alpha=cfg.prior_alpha)[:2]
                prior_off, prior_def = prior_cache[season]
            fit = fit_off_def_ratings(
                history, asof=cutoff,
                half_life_days=cfg.half_life_days,
                ridge_lambda=cfg.ridge_lambda,
                cap_points=cfg.cap_points,
                prior_offense=prior_off or None,
                prior_defense=prior_def or None,
            )
        rat_rows.extend(_rating_features(block, fit, len(history)))
        if progress:
            log.info("ratings %s wk%-2d history=%d games=%d", season, week,
                     len(history), len(block))
    rat = pd.DataFrame(rat_rows)

    # ---- rolling form / rest ----
    long = team_game_long(g)
    form = rolling_form(long, cfg.form_window)
    form = form.merge(long[["game_id", "team", "opponent", "is_home"]],
                      on=["game_id", "team"], how="left")
    # Strength of schedule faced so far, using ratings known at kickoff.
    opp_net = _opponent_strength(long, rat, cfg.form_window)
    form = form.merge(opp_net, on=["game_id", "team"], how="left")

    home_form = form[form["is_home"]].set_index("game_id")
    away_form = form[~form["is_home"]].set_index("game_id")

    feats = g.copy()
    feats = feats.merge(rat, on="game_id", how="left")
    feats = feats.merge(elo_df, on="game_id", how="left")
    for side, src in (("home", home_form), ("away", away_form)):
        feats[f"form_margin_{side}"] = feats["game_id"].map(src["form_margin"])
        feats[f"form_points_for_{side}"] = feats["game_id"].map(src["form_points_for"])
        feats[f"form_points_against_{side}"] = feats["game_id"].map(src["form_points_against"])
        feats[f"form_total_{side}"] = feats["game_id"].map(src["form_total"])
        feats[f"rest_days_{side}"] = feats["game_id"].map(src["rest_days"])
        feats[f"played_{side}"] = feats["game_id"].map(src["played"])
        feats[f"sos_{side}"] = feats["game_id"].map(src["sos"])
    feats["form_margin_diff"] = feats["form_margin_home"] - feats["form_margin_away"]
    feats["rest_diff"] = feats["rest_days_home"] - feats["rest_days_away"]
    feats["sos_diff"] = feats["sos_home"] - feats["sos_away"]

    # ---- preseason-known priors ----
    feats = _join_team_season(feats, talent, "talent", "talent")
    feats = _join_team_season(feats, returning, "returning_ppa", "returning")
    if sp_ratings is not None and not sp_ratings.empty:
        prev = sp_ratings.copy()
        prev["season"] = prev["season"] + 1     # prior-season SP+ only
        feats = _join_team_season(feats, prev, "sp_overall", "sp_prev")
    for stem in ("talent", "returning", "sp_prev"):
        h, a = f"{stem}_home", f"{stem}_away"
        if h in feats.columns and a in feats.columns:
            feats[f"{stem}_diff"] = feats[h] - feats[a]
        else:
            feats[h] = feats[a] = feats[f"{stem}_diff"] = np.nan

    # ---- roster continuity (portal era) ----
    if cfg.include_roster:
        for stem in ROSTER_FEATURE_STEMS:
            if stem in roster.columns:
                feats = _join_team_season(feats, roster, stem, stem)
            else:
                feats[f"{stem}_home"] = np.nan
                feats[f"{stem}_away"] = np.nan
            h, a = f"{stem}_home", f"{stem}_away"
            feats[f"{stem}_diff"] = feats[h] - feats[a]

    # ---- market ----
    cons = consensus_lines(lines)
    if not cons.empty:
        feats = feats.merge(cons, on="game_id", how="left")
    for c in MARKET_FEATURES:
        if c not in feats.columns:
            feats[c] = np.nan

    # ---- targets ----
    feats["home_win"] = np.where(feats["margin"] > 0, 1.0,
                                 np.where(feats["margin"] < 0, 0.0, np.nan))
    feats["neutral_site"] = feats["neutral_site"].astype(float)
    feats["conference_game"] = feats["conference_game"].astype(float)
    feats["hist_games"] = feats["game_id"].map(rat.set_index("game_id")["hist_games"])
    return feats


def feature_columns(cfg: FeatureConfig | None = None) -> list[str]:
    cfg = cfg or FeatureConfig()
    cols = RATING_FEATURES + ELO_FEATURES + CONTEXT_FEATURES + FORM_FEATURES + PRIOR_FEATURES
    if cfg.include_roster:
        cols = cols + ROSTER_FEATURES
    if cfg.include_market:
        cols = cols + MARKET_FEATURES
    return cols


def _rating_features(block: pd.DataFrame, fit: RatingFit | None,
                     n_hist: int) -> list[dict]:
    rows = []
    for r in block.itertuples():
        if fit is None:
            rows.append({"game_id": r.game_id, "hist_games": n_hist,
                         **{c: np.nan for c in RATING_FEATURES}})
            continue
        neutral = bool(r.neutral_site)
        margin, total = fit.predict_game(r.home_team, r.away_team, neutral)
        oh = fit.offense.get(r.home_team, np.nan)
        dh = fit.defense.get(r.home_team, np.nan)
        oa = fit.offense.get(r.away_team, np.nan)
        da = fit.defense.get(r.away_team, np.nan)
        rows.append({
            "game_id": r.game_id,
            "hist_games": n_hist,
            "rat_margin": margin,
            "rat_total": total,
            "rat_hfa": 0.0 if neutral else fit.hfa,
            "rat_net_home": fit.rating(r.home_team),
            "rat_net_away": fit.rating(r.away_team),
            "rat_net_diff": fit.rating(r.home_team) - fit.rating(r.away_team),
            "rat_off_home": oh, "rat_def_home": dh,
            "rat_off_away": oa, "rat_def_away": da,
            "rat_off_diff": oh - oa, "rat_def_diff": dh - da,
            "rat_games_home": fit.n_games.get(r.home_team, 0),
            "rat_games_away": fit.n_games.get(r.away_team, 0),
        })
    return rows


def _opponent_strength(long: pd.DataFrame, rat: pd.DataFrame, window: int) -> pd.DataFrame:
    """Average net rating of the opponents a team has already faced."""
    net = rat[["game_id", "rat_net_home", "rat_net_away"]]
    tmp = long.merge(net, on="game_id", how="left")
    tmp["opp_net"] = np.where(tmp["is_home"], tmp["rat_net_away"], tmp["rat_net_home"])
    tmp = tmp.sort_values(["team", "start_date", "game_id"], kind="stable")
    codes = group_codes(tmp["team"].to_numpy(), tmp["season"].to_numpy())
    tmp["sos"] = trailing_mean(tmp["opp_net"].to_numpy(), codes, None)
    return tmp[["game_id", "team", "sos"]]


def _join_team_season(feats: pd.DataFrame, src: pd.DataFrame | None,
                      value_col: str, out_stem: str) -> pd.DataFrame:
    if src is None or src.empty or value_col not in src.columns:
        return feats
    s = src[["season", "team", value_col]].drop_duplicates(["season", "team"])
    for side in ("home", "away"):
        joined = feats.merge(
            s.rename(columns={"team": f"{side}_team", value_col: f"{out_stem}_{side}"}),
            on=["season", f"{side}_team"], how="left")
        feats[f"{out_stem}_{side}"] = joined[f"{out_stem}_{side}"].to_numpy()
    return feats
