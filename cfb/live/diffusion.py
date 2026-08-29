"""Analytic in-game model: a Brownian bridge on the score margin.

The idea is simple and holds up well.  Treat the remaining scoring as a random
walk whose drift is the pregame edge scaled by the time left, and whose spread
shrinks with the time left:

    final_margin = current_margin + possession_value
                 + Normal( mu_pre * f ,  (sigma_pre * f^alpha)^2 )

with ``f`` the fraction of regulation remaining.  Pure Brownian motion implies
``alpha = 0.5``; real football sits slightly below that early and above it late
(trailing teams speed up, leading teams bleed clock, and onside kicks put a
lump of variance in the last two minutes), so ``alpha`` is fitted from
play-by-play rather than assumed.

Two refinements matter more than they look:

* **Possession is worth points.**  A tie game with the ball at your own 25 is
  not a coin flip -- it is roughly a 1-point edge, more in the red zone.  The
  field-position term below is a linear expected-points approximation, adjusted
  for down.
* **The lattice never goes away.**  Down 4 with 5 minutes left, the relevant
  question is whether the game ends at -4, -3, -1, +3 or +7 -- not what a smooth
  density says.  ``LiveMarginModel`` therefore hands the final distribution to
  the same discrete machinery the pregame model uses, and prefers the drive
  simulator (which reconstructs the lattice from the *current* score) when it
  is available.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfb.constants import DEFAULT_DRIVES_PER_TEAM as DRIVES_PER_TEAM_FULL, GAME_SECONDS
from cfb.distribution.margin import KeyNumberProfile, MarginDistribution
from cfb.distribution.simulate import DriveSimulator, SimConfig

log = logging.getLogger(__name__)

# Linear expected-points-by-field-position approximation, first-and-ten.
# Anchored to the familiar values: own 1 ~ -0.5, own 20 ~ 0.9, midfield ~ 2.6,
# opponent 20 ~ 4.4.
EP_INTERCEPT = 5.95
EP_SLOPE = 0.065
DOWN_PENALTY = {1: 0.0, 2: -0.4, 3: -0.9, 4: -1.4}


def expected_points(yards_to_goal: float | None, down: int | None = 1,
                    distance: float | None = 10.0) -> float:
    """Expected points for the team with the ball, from this field position."""
    if yards_to_goal is None or not np.isfinite(yards_to_goal):
        yards_to_goal = 75.0
    ytg = float(np.clip(yards_to_goal, 1.0, 99.0))
    ep = EP_INTERCEPT - EP_SLOPE * ytg
    ep += DOWN_PENALTY.get(int(down) if down else 1, 0.0)
    if distance is not None and np.isfinite(distance):
        ep -= 0.05 * max(float(distance) - 10.0, 0.0)
    return float(ep)


@dataclass
class LiveConfig:
    alpha: float = 0.5             # variance decay exponent; fit from PBP
    sigma_floor: float = 1.2       # points; keeps the last snap from collapsing
    possession_seconds_scale: float = 60.0
    overtime_sigma: float = 6.5    # margin sd once regulation is tied
    use_simulator: bool = True
    mc_sims: int = 20_000


class LiveMarginModel:
    """Turns a live game state into a distribution over the *final* margin."""

    def __init__(self, cfg: LiveConfig | None = None,
                 key_numbers: KeyNumberProfile | None = None,
                 t_df: float = 7.0):
        self.cfg = cfg or LiveConfig()
        self.key_numbers = key_numbers
        self.t_df = t_df
        self._sim = DriveSimulator(SimConfig(n_sims=self.cfg.mc_sims))

    # -- components --------------------------------------------------------
    def possession_value(self, possession_home: bool | None,
                         yards_to_goal: float | None,
                         down: int | None, distance: float | None,
                         seconds_remaining: float) -> float:
        if possession_home is None:
            return 0.0
        ep = expected_points(yards_to_goal, down, distance)
        scale = float(np.clip(seconds_remaining / self.cfg.possession_seconds_scale,
                              0.0, 1.0))
        return (1.0 if possession_home else -1.0) * ep * scale

    def remaining_sigma(self, sigma_pre: float, fraction_remaining: float) -> float:
        f = float(np.clip(fraction_remaining, 0.0, 1.0))
        return max(sigma_pre * (f ** self.cfg.alpha), self.cfg.sigma_floor)

    # -- main API ----------------------------------------------------------
    def distribution(
        self,
        current_margin: float,
        seconds_remaining: float,
        mu_pregame: float,
        sigma_pregame: float,
        total_pregame: float | None = None,
        possession_home: bool | None = None,
        yards_to_goal: float | None = None,
        down: int | None = None,
        distance: float | None = None,
        period: int = 1,
        method: str | None = None,
    ) -> MarginDistribution:
        """Distribution over the final margin, given where the game stands."""
        secs = float(max(seconds_remaining, 0.0))
        f = float(np.clip(secs / GAME_SECONDS, 0.0, 1.0))

        # Regulation is over and it is tied -> overtime, which is close to a
        # coin flip with a small edge to the better team.
        if secs <= 0 and period >= 4:
            if current_margin == 0:
                edge = np.tanh(mu_pregame / 24.0)
                return self._overtime_distribution(edge)
            return MarginDistribution.from_samples([current_margin], laplace=0.0)

        drift = mu_pregame * f
        poss = self.possession_value(possession_home, yards_to_goal, down,
                                     distance, secs)
        mu_live = current_margin + drift + poss
        sigma_live = self.remaining_sigma(sigma_pregame, f)

        use_sim = (method == "mc") or (method is None and self.cfg.use_simulator)
        if use_sim and f > 0.02:
            return self._simulated(current_margin, drift + poss, f, sigma_live,
                                   total_pregame)
        return MarginDistribution.from_continuous(
            mu_live, sigma_live, dist="t", df=self.t_df,
            key_numbers=self.key_numbers,
            meta={"live": True, "fraction_remaining": f,
                  "possession_value": poss, "drift": drift})

    def _simulated(self, current_margin: float, delta_mean: float,
                   fraction_remaining: float, sigma_live: float,
                   total_pregame: float | None) -> MarginDistribution:
        """Simulate only the drives that are actually left to play.

        This is what makes live 'wins by exactly X' pricing work: the reachable
        final margins are the current margin plus sums of 3s and 7s, and that
        structure is completely invisible to a smooth density centred on the
        live mean.
        """
        total = float(total_pregame) if total_pregame and np.isfinite(total_pregame) else 54.0
        drives_left = max(DRIVES_PER_TEAM_FULL * fraction_remaining, 0.4)
        remaining_total = max(total * fraction_remaining, 1.0)
        clip = (max(int(np.floor(drives_left - 2)), 0), int(np.ceil(drives_left + 3)))
        home, away = self._sim.simulate_scores(
            mu=delta_mean, total=remaining_total, sigma=sigma_live,
            drives_per_team=drives_left, resolve_ties=False, drive_clip=clip)
        samples = current_margin + (home - away)
        # Regulation ties go to overtime.
        tied = samples == 0
        if tied.any():
            rng = np.random.default_rng(19)
            bump = rng.choice([3, 6, 7, 8], size=int(tied.sum()), p=[0.45, 0.08, 0.42, 0.05])
            sign = np.where(rng.random(int(tied.sum())) < 0.5, 1, -1)
            samples = samples.astype(float)
            samples[tied] = bump * sign
        return MarginDistribution.from_samples(
            samples, laplace=0.2,
            meta={"scaffold": "live_mc", "live": True,
                  "fraction_remaining": fraction_remaining,
                  "current_margin": float(current_margin)})

    def _overtime_distribution(self, edge: float) -> MarginDistribution:
        rng = np.random.default_rng(23)
        n = 20_000
        home_wins = rng.random(n) < (0.5 + 0.06 * edge)
        bump = rng.choice([3, 6, 7, 8, 10, 14], size=n,
                          p=[0.34, 0.07, 0.36, 0.06, 0.10, 0.07])
        samples = np.where(home_wins, bump, -bump)
        return MarginDistribution.from_samples(
            samples, meta={"scaffold": "overtime", "live": True})

    # -- fitting alpha -----------------------------------------------------
    def fit_variance_decay(self, pbp: pd.DataFrame,
                           mu_col: str | None = None,
                           sigma_pregame: float = 16.5,
                           bins: int = 14) -> float:
        """Estimate the variance-decay exponent from play-by-play snapshots.

        Regresses ``log(sd of remaining margin change)`` on ``log(f)``; the
        slope is alpha.  Needs ``seconds_remaining``, the score at the snapshot,
        and the final margin.
        """
        d = pbp.dropna(subset=["seconds_remaining", "final_margin"]).copy()
        if d.empty:
            return self.cfg.alpha
        cur = d["home_score"] - d["away_score"]
        f = (d["seconds_remaining"] / GAME_SECONDS).clip(0.001, 1.0)
        drift = (d[mu_col] * f) if mu_col and mu_col in d.columns else 0.0
        resid = d["final_margin"] - cur - drift
        d = d.assign(f=f, resid=resid)
        d["bin"] = pd.qcut(d["f"], bins, duplicates="drop")
        grp = d.groupby("bin", observed=True).agg(
            f_mid=("f", "mean"), sd=("resid", "std"), n=("resid", "size"))
        grp = grp[(grp["n"] > 30) & (grp["sd"] > 0) & (grp["f_mid"] > 0.01)]
        if len(grp) < 4:
            return self.cfg.alpha
        x = np.log(grp["f_mid"].to_numpy())
        y = np.log(grp["sd"].to_numpy() / sigma_pregame)
        alpha = float(np.polyfit(x, y, 1)[0])
        alpha = float(np.clip(alpha, 0.25, 0.9))
        log.info("fitted variance-decay alpha=%.3f from %d snapshots", alpha, len(d))
        self.cfg.alpha = alpha
        return alpha
