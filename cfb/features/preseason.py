"""Preseason priors: what should we expect from this team before it plays?

The ratings model shrinks each team toward something when the evidence is thin.
Shrinking toward the league mean says "we know nothing in week 1", which is
wrong -- we know a great deal, just not from *this* season's games.

This module learns the mapping

    (last season's rating, who came back, who arrived, who plays quarterback)
        -> this season's rating

from history, and hands the result to ``fit_off_def_ratings`` as a per-team
prior.  The roster features matter here in a way they do not as ordinary model
inputs: on their own they barely correlate with margin, because they do not
predict *who wins* -- they predict *how much of last year still applies*.  That
is an interaction with the prior rating, and this is where it belongs.

Leak safety: the carryover mapping for season S is fit only on transitions that
finished before S.  ``priors_for_season`` refuses to use the target season's own
games, and there is a test that asserts it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cfb.features.ratings import RatingFit, fit_off_def_ratings

log = logging.getLogger(__name__)

#: Roster columns fed to the carryover model, if present.
CARRYOVER_ROSTER_COLS = [
    "returning_off", "returning_def", "portal_net_rating", "portal_in_best",
    "qb_departed", "qb_transfer_in", "qb_continuity", "qb_prior_ppa",
]


def season_ratings(games: pd.DataFrame, ridge_lambda: float = 30.0,
                   min_games: int = 60) -> dict[int, RatingFit]:
    """Fit each season's ratings from that season's games alone.

    Single-season fits are what the carryover model learns from: they are a
    clean "how good was this team that year", with no bleed from the year
    before, which is exactly the quantity we are trying to predict.
    """
    out: dict[int, RatingFit] = {}
    done = games[games["completed"].astype(bool)] if "completed" in games else games
    for season, block in done.groupby("season"):
        if len(block) < min_games:
            continue
        out[int(season)] = fit_off_def_ratings(
            block, half_life_days=10_000, ridge_lambda=ridge_lambda)
    return out


@dataclass
class CarryoverFit:
    n_train: int
    seasons_used: list[int]
    r2_offense: float
    r2_defense: float
    coefficients: dict[str, float] = field(default_factory=dict)


class CarryoverModel:
    """Predicts a season's ratings from the previous season plus roster facts."""

    def __init__(self, alpha: float = 5.0, use_interaction: bool = True,
                 use_roster: bool = True):
        self.alpha = alpha
        self.use_interaction = use_interaction
        self.use_roster = use_roster
        self.off_model = None
        self.def_model = None
        self.columns: list[str] = []
        self.report: CarryoverFit | None = None

    # -- design ------------------------------------------------------------
    def _rows(self, season_fits: dict[int, RatingFit], roster: pd.DataFrame,
              seasons: list[int]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        roster_idx = (roster.set_index(["season", "team"])
                      if roster is not None and not roster.empty else None)
        rows, y_off, y_def = [], [], []
        for season in seasons:
            prev = season_fits.get(season - 1)
            cur = season_fits.get(season)
            if prev is None or cur is None:
                continue
            for team in cur.teams:
                if team not in prev.offense:
                    continue
                rows.append(self._features(team, season, prev, roster_idx))
                y_off.append(cur.offense[team])
                y_def.append(cur.defense[team])
        if not rows:
            return pd.DataFrame(), np.array([]), np.array([])
        return pd.DataFrame(rows), np.asarray(y_off), np.asarray(y_def)

    def _features(self, team: str, season: int, prev: RatingFit,
                  roster_idx) -> dict[str, float]:
        prev_off = float(prev.offense.get(team, 0.0))
        prev_def = float(prev.defense.get(team, 0.0))
        rec: dict[str, float] = {"prev_off": prev_off, "prev_def": prev_def,
                                 "prev_games": float(prev.n_games.get(team, 0))}
        vals: dict[str, float] = {}
        if roster_idx is not None and self.use_roster:
            try:
                r = roster_idx.loc[(season, team)]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                for c in CARRYOVER_ROSTER_COLS:
                    if c in r.index:
                        vals[c] = float(r[c]) if pd.notna(r[c]) else np.nan
            except KeyError:
                pass
        if self.use_roster:
            for c in CARRYOVER_ROSTER_COLS:
                rec[c] = vals.get(c, np.nan)
        if self.use_interaction and self.use_roster:
            # The load-bearing term: returning production scales how much of
            # last season actually carries forward.
            ret = vals.get("returning_off", np.nan)
            rec["prev_off_x_returning"] = prev_off * ret if np.isfinite(ret) else np.nan
            retd = vals.get("returning_def", np.nan)
            rec["prev_def_x_returning"] = prev_def * retd if np.isfinite(retd) else np.nan
        return rec

    def _estimator(self):
        # Scaling matters here: the inputs range from ratings in points to PPA
        # in hundredths, and an unscaled ridge penalty would regularise them
        # wildly unevenly.
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("est", Ridge(alpha=self.alpha))])

    # -- fit / predict -----------------------------------------------------
    def fit(self, season_fits: dict[int, RatingFit], roster: pd.DataFrame,
            before_season: int) -> "CarryoverModel":
        """Fit on transitions that completed strictly before ``before_season``."""
        seasons = sorted(s for s in season_fits if s < before_season)
        X, y_off, y_def = self._rows(season_fits, roster, seasons)
        if len(X) < 40:
            self.off_model = self.def_model = None
            return self
        self.columns = list(X.columns)
        self.off_model = self._estimator().fit(X, y_off)
        self.def_model = self._estimator().fit(X, y_def)
        po, pd_ = self.off_model.predict(X), self.def_model.predict(X)
        self.report = CarryoverFit(
            n_train=len(X), seasons_used=seasons,
            r2_offense=float(1 - np.sum((y_off - po) ** 2) / max(np.sum((y_off - y_off.mean()) ** 2), 1e-9)),
            r2_defense=float(1 - np.sum((y_def - pd_) ** 2) / max(np.sum((y_def - pd_.mean()) ** 2), 1e-9)),
            coefficients=dict(zip(X.columns, self.off_model.named_steps["est"].coef_)),
        )
        return self

    def priors(self, season: int, season_fits: dict[int, RatingFit],
               roster: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
        """Per-team offence/defence priors for ``season``."""
        prev = season_fits.get(season - 1)
        if prev is None or self.off_model is None:
            return {}, {}
        roster_idx = (roster.set_index(["season", "team"])
                      if roster is not None and not roster.empty else None)
        teams = list(prev.teams)
        X = pd.DataFrame([self._features(t, season, prev, roster_idx) for t in teams])
        X = X.reindex(columns=self.columns)
        off = dict(zip(teams, self.off_model.predict(X)))
        dfn = dict(zip(teams, self.def_model.predict(X)))
        return off, dfn


def priors_for_season(games: pd.DataFrame, roster: pd.DataFrame, season: int,
                      ridge_lambda: float = 30.0, alpha: float = 5.0
                      ) -> tuple[dict[str, float], dict[str, float], CarryoverModel]:
    """Convenience: build leak-free priors for one season.

    Only games from seasons **before** ``season`` are used, both for the
    single-season ratings and for fitting the carryover mapping.
    """
    history = games[games["season"] < season]
    if history.empty:
        return {}, {}, CarryoverModel(alpha)
    fits = season_ratings(history, ridge_lambda=ridge_lambda)
    model = CarryoverModel(alpha).fit(fits, roster, before_season=season)
    off, dfn = model.priors(season, fits, roster)
    return off, dfn, model
