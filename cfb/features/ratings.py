"""Opponent-adjusted team ratings.

Two rating systems live here:

``fit_off_def_ratings``
    A weighted ridge regression on *points scored*.  Each game contributes two
    rows -- one per scoring team -- and the design matrix carries an offence
    column and a defence column per team plus separate home/away field-advantage
    terms.  Ridge shrinkage both regularises light-schedule teams toward the
    mean and resolves the offence/defence additive identifiability.  It yields a
    margin prediction *and* a total prediction, which the distribution layer
    needs (variance scales with total).

``elo_ratings``
    A margin-aware Elo, useful as an independent feature and as a sanity check
    that the ridge ratings have not been broken by a data issue.

Everything here is fit on a *subset of games chosen by the caller*.  Nothing in
this module knows about "the future"; the walk-forward discipline lives in
``cfb.features.build``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cfb.constants import DEFAULT_HFA

log = logging.getLogger(__name__)

FCS_TOKEN = "__FCS__"


@dataclass
class RatingFit:
    """Fitted offence/defence ratings, in points."""
    teams: list[str]
    offense: dict[str, float]
    defense: dict[str, float]
    hfa_home: float
    hfa_away: float
    intercept: float
    n_games: dict[str, int] = field(default_factory=dict)
    asof: pd.Timestamp | None = None

    @property
    def hfa(self) -> float:
        """Home-field advantage expressed in margin points."""
        return self.hfa_home - self.hfa_away

    def rating(self, team: str) -> float:
        """Net team strength in points vs. an average opponent."""
        return self.offense.get(team, 0.0) - self.defense.get(team, 0.0)

    def expected_points(self, offense_team: str, defense_team: str,
                        at_home: bool | None) -> float:
        adj = self.hfa_home if at_home is True else (self.hfa_away if at_home is False else 0.0)
        return (self.intercept
                + self.offense.get(offense_team, 0.0)
                + self.defense.get(defense_team, 0.0)
                + adj)

    def predict_game(self, home: str, away: str, neutral: bool = False
                     ) -> tuple[float, float]:
        """Return (expected home margin, expected total)."""
        hp = self.expected_points(home, away, None if neutral else True)
        ap = self.expected_points(away, home, None if neutral else False)
        return hp - ap, hp + ap

    def to_frame(self) -> pd.DataFrame:
        rows = [{"team": t,
                 "offense": self.offense.get(t, 0.0),
                 "defense": self.defense.get(t, 0.0),
                 "rating": self.rating(t),
                 "games": self.n_games.get(t, 0)} for t in self.teams]
        return (pd.DataFrame(rows)
                .sort_values("rating", ascending=False)
                .reset_index(drop=True))


def recency_weights(dates: pd.Series, asof: pd.Timestamp,
                    half_life_days: float = 400.0) -> np.ndarray:
    """Exponential decay: a game ``half_life_days`` old counts half as much."""
    age = (asof - pd.to_datetime(dates, utc=True)).dt.total_seconds() / 86400.0
    age = age.clip(lower=0).to_numpy()
    return np.exp(-np.log(2.0) * age / max(half_life_days, 1.0))


def fit_off_def_ratings(
    games: pd.DataFrame,
    asof: pd.Timestamp | None = None,
    half_life_days: float = 400.0,
    ridge_lambda: float = 45.0,
    cap_points: float | None = 52.0,
    min_weight: float = 0.01,
    prior_offense: dict[str, float] | None = None,
    prior_defense: dict[str, float] | None = None,
) -> RatingFit:
    """Weighted ridge on points scored -> per-team offence/defence ratings.

    ``ridge_lambda`` is in units of effective games: a team with roughly
    ``ridge_lambda`` weighted games is pulled halfway toward the prior.
    ``cap_points`` soft-caps blowout scoring so a 70-point game does not
    dominate; set to None to disable.

    ``prior_offense`` / ``prior_defense`` shrink each team toward a *specific*
    value instead of toward the league mean.  That is the difference between
    "we know nothing about this team in week 1" and "we expect this team to be
    about this good, because of who is on the roster".  Minimising
    ``|W^.5(y - XB)|^2 + lambda*|B - p|^2`` just adds ``lambda * p`` to the
    right-hand side of the normal equations.
    """
    g = games.loc[games["completed"].astype(bool)].copy() if "completed" in games \
        else games.copy()
    g = g.dropna(subset=["home_points", "away_points"])
    if g.empty:
        return RatingFit([], {}, {}, DEFAULT_HFA / 2, -DEFAULT_HFA / 2, 27.0)

    asof = asof or pd.to_datetime(g["start_date"], utc=True).max() + pd.Timedelta(days=1)
    w_game = recency_weights(g["start_date"], asof, half_life_days)
    keep = w_game > min_weight
    g, w_game = g.loc[keep], w_game[keep]
    if g.empty:
        return RatingFit([], {}, {}, DEFAULT_HFA / 2, -DEFAULT_HFA / 2, 27.0)

    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    # columns: [offense 0..n-1][defense n..2n-1][hfa_home][hfa_away][intercept]
    n_p = 2 * n_t + 3
    OFF, DEF, HH, HA, IC = 0, n_t, 2 * n_t, 2 * n_t + 1, 2 * n_t + 2

    hp = g["home_points"].to_numpy(float)
    ap = g["away_points"].to_numpy(float)
    if cap_points is not None:
        hp = np.minimum(hp, cap_points)
        ap = np.minimum(ap, cap_points)
    neutral = g["neutral_site"].to_numpy(bool) if "neutral_site" in g else np.zeros(len(g), bool)
    hi = g["home_team"].map(idx).to_numpy()
    ai = g["away_team"].map(idx).to_numpy()

    n_rows = 2 * len(g)
    y = np.concatenate([hp, ap])
    w = np.concatenate([w_game, w_game])

    # Accumulate the normal equations from a sparse design: each row has only
    # 3-4 non-zeros, so materialising X densely would waste an order of
    # magnitude of time and memory once history spans several seasons.
    from scipy import sparse

    rows = np.arange(n_rows)
    cols_off = np.concatenate([hi, ai])                    # scoring team's offence
    cols_def = np.concatenate([ai + n_t, hi + n_t])        # opponent's defence
    on_road = np.concatenate([np.zeros(len(g), bool), ~neutral])
    at_home = np.concatenate([~neutral, np.zeros(len(g), bool)])

    r_idx = [rows, rows, rows[at_home], rows[on_road], rows]
    c_idx = [cols_off, cols_def,
             np.full(at_home.sum(), HH), np.full(on_road.sum(), HA),
             np.full(n_rows, IC)]
    vals = [np.ones(n_rows), np.ones(n_rows),
            np.ones(at_home.sum()), np.ones(on_road.sum()), np.ones(n_rows)]
    X = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))),
        shape=(n_rows, n_p),
    )
    W = sparse.diags(w)
    A = np.asarray((X.T @ W @ X).todense())
    b = np.asarray(X.T @ (w * y)).ravel()

    penalty = np.zeros(n_p)
    penalty[OFF:DEF] = ridge_lambda
    penalty[DEF:HH] = ridge_lambda
    penalty[HH] = 1e-6
    penalty[HA] = 1e-6
    penalty[IC] = 1e-8
    A[np.diag_indices_from(A)] += penalty

    # Shrink toward the supplied prior rather than toward zero.
    if prior_offense or prior_defense:
        p_vec = np.zeros(n_p)
        for team, i in idx.items():
            if prior_offense:
                p_vec[OFF + i] = float(prior_offense.get(team, 0.0) or 0.0)
            if prior_defense:
                p_vec[DEF + i] = float(prior_defense.get(team, 0.0) or 0.0)
        b = b + penalty * p_vec

    beta = np.linalg.solve(A, b)

    counts = pd.concat([g["home_team"], g["away_team"]]).value_counts().to_dict()
    return RatingFit(
        teams=teams,
        offense={t: float(beta[OFF + i]) for t, i in idx.items()},
        defense={t: float(beta[DEF + i]) for t, i in idx.items()},
        hfa_home=float(beta[HH]),
        hfa_away=float(beta[HA]),
        intercept=float(beta[IC]),
        n_games={t: int(counts.get(t, 0)) for t in teams},
        asof=asof,
    )


# ------------------------------------------------------------------------ #
# Elo
# ------------------------------------------------------------------------ #
@dataclass
class EloConfig:
    k: float = 40.0
    hfa: float = 60.0            # Elo points
    start: float = 1500.0
    regress: float = 0.25        # fraction pulled to the mean each preseason
    mov_scale: float = 2.2


def elo_ratings(games: pd.DataFrame, cfg: EloConfig | None = None
                ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Elo forward through ``games``.

    Returns (per-game pregame Elo frame, final rating dict).  The per-game frame
    is inherently leak-free: each row records the ratings *before* that game.
    """
    cfg = cfg or EloConfig()
    g = games.sort_values(["season", "week", "start_date", "game_id"],
                          kind="stable").reset_index(drop=True)
    rat: dict[str, float] = {}
    last_season: int | None = None
    out = []
    for r in g.itertuples():
        if last_season is not None and r.season != last_season:
            for t in rat:
                rat[t] = cfg.start + (1 - cfg.regress) * (rat[t] - cfg.start)
        last_season = r.season

        rh = rat.setdefault(r.home_team, cfg.start)
        ra = rat.setdefault(r.away_team, cfg.start)
        hfa = 0.0 if bool(getattr(r, "neutral_site", False)) else cfg.hfa
        diff = (rh + hfa) - ra
        exp_h = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        out.append({"game_id": r.game_id, "elo_home_pre": rh, "elo_away_pre": ra,
                    "elo_diff": diff, "elo_home_wp": exp_h})

        hp, ap = getattr(r, "home_points", None), getattr(r, "away_points", None)
        if hp is None or ap is None or pd.isna(hp) or pd.isna(ap):
            continue
        margin = float(hp) - float(ap)
        s_h = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        winner_diff = diff if margin > 0 else -diff
        mult = np.log(abs(margin) + 1.0) * (cfg.mov_scale /
                                            (0.001 * winner_diff + cfg.mov_scale))
        delta = cfg.k * mult * (s_h - exp_h)
        rat[r.home_team] = rh + delta
        rat[r.away_team] = ra - delta
    return pd.DataFrame(out), rat
