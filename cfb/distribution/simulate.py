"""Drive-level Monte Carlo scoring model.

The parametric route (``MarginDistribution.from_continuous``) gets the lattice
by *learning* multiplicative weights.  This module gets it for free by
simulating the thing that actually produces the lattice: a sequence of drives
that end in 7, 3 or 0.

Why keep both?  They fail differently.  The lattice weights are stable and
cheap but assume the shape of the key-number effect is the same for a 40-point
game and a 70-point game.  The simulator gets those right by construction, and
gives you the joint distribution of (home score, away score) -- so it can price
totals, team-total, and 'exact score'-style markets that a margin-only PMF
cannot touch.  The simulator is slower and its tails depend on the drive model
being right.

Calibration is anchored to the point model, not invented: the per-team points
per drive are solved so the simulated mean margin equals ``mu`` and the mean
total equals ``total``, and a per-simulation team-quality shock is scaled so the
simulated margin sd equals the sigma model's ``sigma``.  The simulator supplies
*shape*; the fitted models still supply location and scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from cfb.constants import DEFAULT_DRIVES_PER_TEAM
from cfb.distribution.margin import DEFAULT_MAX_MARGIN, MarginDistribution

# Multinomial-logit drive outcome model, league-average intercepts.
A_TD, B_TD = -0.954, 0.60
A_FG, B_FG = -1.370, 0.15
TD_POINTS_MEAN = 7.03          # 7 usually, occasionally 6 or 8
DEF_ST_TD_RATE = 0.11          # non-offensive touchdowns per team per game


def _drive_probs(q: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    e_td = np.exp(A_TD + B_TD * q)
    e_fg = np.exp(A_FG + B_FG * q)
    z = e_td + e_fg + 1.0
    return e_td / z, e_fg / z


def points_per_drive(q: float) -> float:
    p_td, p_fg = _drive_probs(q)
    return float(p_td * TD_POINTS_MEAN + p_fg * 3.0)


def quality_for_ppd(target_ppd: float) -> float:
    """Invert points-per-drive -> latent offensive quality."""
    lo, hi = -6.0, 6.0
    t = float(np.clip(target_ppd, points_per_drive(lo) + 1e-6,
                      points_per_drive(hi) - 1e-6))
    return float(brentq(lambda q: points_per_drive(q) - t, lo, hi, xtol=1e-8))


@dataclass
class SimConfig:
    n_sims: int = 40_000
    drives_per_team: float = DEFAULT_DRIVES_PER_TEAM
    drive_sd: float = 1.4          # game-to-game variation in possessions
    def_st_td_rate: float = DEF_ST_TD_RATE
    max_margin: int = DEFAULT_MAX_MARGIN
    seed: int = 11


class DriveSimulator:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        # Scales the non-offensive touchdown rate when only part of a game is
        # being simulated; reset on every call to simulate_scores.
        self._drive_fraction = 1.0

    # -- core --------------------------------------------------------------
    def simulate_scores(
        self,
        mu: float,
        total: float,
        sigma: float | None = None,
        n_sims: int | None = None,
        drives_per_team: float | None = None,
        seed: int | None = None,
        refine: bool = True,
        resolve_ties: bool = True,
        drive_clip: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (home_scores, away_scores) arrays of length ``n_sims``.

        ``resolve_ties`` should be False when simulating only the *remaining*
        drives of a game in progress: an equal number of points scored from here
        is not a tie, and breaking it would silently add several points of drift.
        ``drive_clip`` bounds the per-game drive count; the full-game default is
        far too high when only a quarter of the game is left.
        """
        cfg = self.cfg
        n = int(n_sims or cfg.n_sims)
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        d_mean = float(drives_per_team or cfg.drives_per_team)

        # Solve the per-team scoring rates implied by the point model.  Back
        # out the non-offensive touchdowns first, otherwise the simulated total
        # overshoots the requested one by ~1.5 points every time.
        self._drive_fraction = float(np.clip(d_mean / cfg.drives_per_team, 0.0, 1.5))
        non_off = 7.0 * cfg.def_st_td_rate * self._drive_fraction
        ppd_home = max(((total + mu) / 2.0 - non_off) / d_mean, 0.10)
        ppd_away = max(((total - mu) / 2.0 - non_off) / d_mean, 0.10)
        q_home = quality_for_ppd(ppd_home)
        q_away = quality_for_ppd(ppd_away)

        tau = 0.0 if sigma is None else self._shock_scale(q_home, q_away, d_mean, sigma)
        if tau > 0 and refine:
            tau = self._refine_shock(rng, tau, q_home, q_away, d_mean, sigma)
        shock = rng.normal(0.0, tau, n) if tau > 0 else np.zeros(n)

        lo, hi = drive_clip if drive_clip else (7, 20)
        drives = np.clip(
            np.rint(rng.normal(d_mean, cfg.drive_sd, n)).astype(int), lo, hi)
        home = self._score_batch(rng, q_home + shock, drives)
        away = self._score_batch(rng, q_away - shock, drives)
        if resolve_ties:
            home, away = self._resolve_overtime(rng, home, away, q_home, q_away)
        return home, away

    def _score_batch(self, rng, q: np.ndarray, drives: np.ndarray) -> np.ndarray:
        n = len(drives)
        max_d = int(drives.max())
        if max_d <= 0:
            return np.zeros(n, dtype=int)
        p_td, p_fg = _drive_probs(q)
        u = rng.random((n, max_d))
        active = np.arange(max_d)[None, :] < drives[:, None]
        is_td = (u < p_td[:, None]) & active
        is_fg = (u >= p_td[:, None]) & (u < (p_td + p_fg)[:, None]) & active
        n_td = is_td.sum(axis=1)
        # Split touchdowns into 7 / 6 / 8 point conversions.
        pts = np.zeros(n, dtype=int)
        if n_td.max() > 0:
            extra = rng.random((n, max_d))
            six = ((extra < 0.03) & is_td).sum(axis=1)
            eight = ((extra >= 0.03) & (extra < 0.06) & is_td).sum(axis=1)
            seven = n_td - six - eight
            pts += 7 * seven + 6 * six + 8 * eight
        pts += 3 * is_fg.sum(axis=1)
        pts += 7 * rng.poisson(self.cfg.def_st_td_rate * self._drive_fraction, n)
        return pts

    def _resolve_overtime(self, rng, home, away, q_home, q_away):
        tied = home == away
        if not tied.any():
            return home, away
        k = int(tied.sum())
        # College OT: each side gets the ball; repeat until someone wins.
        edge = 0.5 + 0.05 * np.tanh(q_home - q_away)
        home_wins = rng.random(k) < edge
        bump = rng.choice([3, 6, 7, 8], size=k, p=[0.45, 0.08, 0.42, 0.05])
        h, a = home.copy(), away.copy()
        idx = np.where(tied)[0]
        h[idx[home_wins]] += bump[home_wins]
        a[idx[~home_wins]] += bump[~home_wins]
        return h, a

    @staticmethod
    def _shock_scale(q_home: float, q_away: float, d_mean: float,
                     sigma_target: float) -> float:
        """Team-quality shock sd that lifts simulated margin sd to the target.

        The drive process alone produces roughly 13-14 points of margin sd; real
        pregame uncertainty is ~16-17 because we do not know how good the teams
        are *today*.  That extra variance belongs on team quality (where it
        preserves the scoring lattice), not bolted on as Gaussian noise
        afterwards (which would smear the lattice away).
        """
        # d margin / d shock, via the analytic derivative of points-per-drive.
        eps = 1e-4
        dppd_h = (points_per_drive(q_home + eps) - points_per_drive(q_home - eps)) / (2 * eps)
        dppd_a = (points_per_drive(q_away + eps) - points_per_drive(q_away - eps)) / (2 * eps)
        dm_dshock = d_mean * (dppd_h + dppd_a)
        # Binomial drive noise alone -- approximated from the per-drive variance.
        p_td_h, p_fg_h = _drive_probs(q_home)
        p_td_a, p_fg_a = _drive_probs(q_away)
        var_drive = 0.0
        for p_td, p_fg in ((p_td_h, p_fg_h), (p_td_a, p_fg_a)):
            ex = p_td * TD_POINTS_MEAN + p_fg * 3.0
            ex2 = p_td * TD_POINTS_MEAN ** 2 + p_fg * 9.0
            var_drive += d_mean * (ex2 - ex ** 2)
        residual = sigma_target ** 2 - var_drive
        if residual <= 0 or dm_dshock <= 1e-9:
            return 0.0
        return float(np.sqrt(residual) / dm_dshock)

    def _refine_shock(self, rng, tau: float, q_home: float, q_away: float,
                      d_mean: float, sigma_target: float, pilot: int = 6000) -> float:
        """One cheap correction pass: the analytic tau is a linearisation, and
        the drive process is mildly non-linear in quality."""
        drives = np.clip(np.rint(rng.normal(d_mean, self.cfg.drive_sd, pilot)).astype(int),
                         7, 20)
        shock = rng.normal(0.0, tau, pilot)
        m = (self._score_batch(rng, q_home + shock, drives)
             - self._score_batch(rng, q_away - shock, drives))
        sd_now = float(np.std(m))
        if sd_now <= 1e-6:
            return tau
        gap = sigma_target ** 2 - sd_now ** 2
        # Only the shock component is adjustable; the drive noise is fixed.
        shock_var = max(sd_now ** 2 - self._drive_only_var(q_home, q_away, d_mean), 1e-6)
        scaled = shock_var + gap
        if scaled <= 0:
            return 0.0
        return float(tau * np.sqrt(scaled / shock_var))

    @staticmethod
    def _drive_only_var(q_home: float, q_away: float, d_mean: float) -> float:
        var = 0.0
        for q in (q_home, q_away):
            p_td, p_fg = _drive_probs(q)
            ex = p_td * TD_POINTS_MEAN + p_fg * 3.0
            ex2 = p_td * TD_POINTS_MEAN ** 2 + p_fg * 9.0
            var += d_mean * (ex2 - ex ** 2)
        return float(var)

    # -- public API --------------------------------------------------------
    def distribution(self, mu: float, total: float, sigma: float | None = None,
                     n_sims: int | None = None, seed: int | None = None,
                     **kwargs) -> MarginDistribution:
        home, away = self.simulate_scores(mu, total, sigma, n_sims, seed=seed, **kwargs)
        dist = MarginDistribution.from_samples(
            home - away, max_margin=self.cfg.max_margin, laplace=0.25,
            meta={"scaffold": "monte_carlo", "mu_input": float(mu),
                  "total_input": float(total),
                  "sigma_input": None if sigma is None else float(sigma)})
        return dist

    def score_matrix(self, mu: float, total: float, sigma: float | None = None,
                     n_sims: int | None = None, max_score: int = 90,
                     seed: int | None = None):
        """Joint (home score, away score) probability grid -- for exact-score
        and team-total markets."""
        home, away = self.simulate_scores(mu, total, sigma, n_sims, seed=seed)
        h = np.clip(home, 0, max_score)
        a = np.clip(away, 0, max_score)
        grid = np.zeros((max_score + 1, max_score + 1), dtype=float)
        np.add.at(grid, (h, a), 1.0)
        return grid / grid.sum()

    def total_distribution(self, mu: float, total: float, sigma: float | None = None,
                           n_sims: int | None = None, seed: int | None = None):
        home, away = self.simulate_scores(mu, total, sigma, n_sims, seed=seed)
        return home + away
