"""Discrete margin distributions.

Everything the betting layer needs is a probability mass function over integer
home margins.  Continuous densities are only ever a *scaffold*: they get
integrated over each integer cell and then reshaped, because football margins
live on a lattice built out of 3s and 7s.

Three facts drive the design:

1. **Margins are integers and 0 is impossible.**  College football has had no
   ties since overtime arrived in 1996.  Any model that leaves mass on 0 is
   wrong, and around a pick'em that error is large.
2. **Key numbers are real.**  3 and 7 are roughly twice as likely as their
   neighbours; 10, 14, 4, 1 and 17 also spike.  A market for "wins by exactly
   3" priced off a smooth normal is mispriced by a factor of ~2.
3. **The lattice sits in margin space, the location sits in strength space.**
   A game is *3* more often than *5* whether the spread is 2 or 20.  So the PMF
   factorises as ``smooth(k - mu) x lattice_weight(k)``, renormalised.

The tails also matter: standardised residuals in football are mildly
leptokurtic, so Student-t with ~7 df is a better scaffold than a normal, and
you can also skip the parametric step entirely and use the empirical residual
KDE or the drive-level Monte Carlo in ``simulate.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_MAX_MARGIN = 80
StrikeType = Literal["greater", "greater_or_equal", "less", "less_or_equal", "between"]


# ---------------------------------------------------------------------- #
# Key-number lattice
# ---------------------------------------------------------------------- #
@dataclass
class KeyNumberProfile:
    """Multiplicative weights ``w(k)`` on integer margins.

    Fitted as ``observed_count(k) / expected_count_under_smooth_model(k)``, so
    ``w`` measures exactly the lattice structure that a smooth density misses.
    """
    weights: dict[int, float] = field(default_factory=dict)
    max_margin: int = DEFAULT_MAX_MARGIN
    shrink: float = 0.35
    allow_ties: bool = False
    n_games: int = 0

    def weight(self, k: int) -> float:
        if k == 0 and not self.allow_ties:
            return 0.0
        return float(self.weights.get(int(k), 1.0))

    def vector(self, grid: np.ndarray) -> np.ndarray:
        return np.array([self.weight(int(k)) for k in grid], dtype=float)

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit(
        cls,
        margins: Sequence[float],
        mus: Sequence[float] | None = None,
        sigmas: Sequence[float] | float = 16.5,
        max_margin: int = DEFAULT_MAX_MARGIN,
        df: float = 7.0,
        shrink: float = 0.35,
        symmetric: bool = True,
        allow_ties: bool = False,
        lattice_range: int = 30,
    ) -> "KeyNumberProfile":
        """Learn lattice weights from historical games.

        ``mus`` are the model/market expectations for those games; when omitted
        the sample mean is used, which is fine for a rough profile but much
        weaker than passing real predictions.
        """
        m = np.asarray(margins, dtype=float)
        m = m[np.isfinite(m)]
        if m.size == 0:
            return cls({}, max_margin, shrink, allow_ties, 0)
        mu = (np.full_like(m, float(np.mean(m))) if mus is None
              else np.asarray(mus, dtype=float)[: m.size])
        sd = (np.full_like(m, float(sigmas)) if np.isscalar(sigmas)
              else np.asarray(sigmas, dtype=float)[: m.size])
        sd = np.where(np.isfinite(sd) & (sd > 1), sd, 16.5)

        grid = np.arange(-max_margin, max_margin + 1)
        # Expected counts under the smooth scaffold: sum over games of the
        # probability that game lands on each integer cell.
        z_hi = (grid[None, :] + 0.5 - mu[:, None]) / sd[:, None]
        z_lo = (grid[None, :] - 0.5 - mu[:, None]) / sd[:, None]
        cell = stats.t.cdf(z_hi, df) - stats.t.cdf(z_lo, df)
        expected = cell.sum(axis=0)

        obs = np.zeros_like(grid, dtype=float)
        vals, counts = np.unique(np.rint(m).astype(int), return_counts=True)
        inside = (vals >= -max_margin) & (vals <= max_margin)
        obs[vals[inside] + max_margin] = counts[inside]

        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(expected > 1e-9, obs / expected, 1.0)

        # Shrink toward 1 by how much evidence the cell actually has.  Using
        # the *expected* count as the sample-size proxy is what makes this
        # behave: margin 3 is expected ~120 times in 5,000 games and earns its
        # weight, margin 41 is expected 3 times and should not move a price.
        alpha = expected / (expected + max(shrink, 1e-6) * 85.0)
        w = 1.0 + alpha * (raw - 1.0)
        w = np.clip(w, 0.15, 4.0)

        # Beyond the lattice range the scoring structure has washed out and any
        # remaining deviation is tail thickness, which belongs in the scaffold's
        # degrees of freedom rather than in per-cell weights.
        outside = np.abs(grid) > lattice_range
        w[outside] = 1.0

        if symmetric:
            w = 0.5 * (w + w[::-1])
        if not allow_ties:
            w[max_margin] = 0.0

        weights = {int(k): float(v) for k, v in zip(grid, w)}
        return cls(weights, max_margin, shrink, allow_ties, int(m.size))

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "weights": {str(k): v for k, v in self.weights.items()},
            "max_margin": self.max_margin, "shrink": self.shrink,
            "allow_ties": self.allow_ties, "n_games": self.n_games,
        }))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "KeyNumberProfile":
        d = json.loads(Path(path).read_text())
        return cls({int(k): float(v) for k, v in d["weights"].items()},
                   int(d["max_margin"]), float(d["shrink"]),
                   bool(d.get("allow_ties", False)), int(d.get("n_games", 0)))

    def top(self, n: int = 12) -> pd.DataFrame:
        rows = [{"margin": k, "weight": v} for k, v in self.weights.items() if k > 0]
        return (pd.DataFrame(rows).sort_values("weight", ascending=False)
                .head(n).reset_index(drop=True))


# ---------------------------------------------------------------------- #
# The distribution itself
# ---------------------------------------------------------------------- #
class MarginDistribution:
    """A PMF over integer home margins, with the query methods Kalshi needs."""

    def __init__(self, grid: np.ndarray, pmf: np.ndarray, meta: dict | None = None):
        grid = np.asarray(grid, dtype=int)
        pmf = np.asarray(pmf, dtype=float)
        if grid.shape != pmf.shape:
            raise ValueError("grid and pmf must have the same shape")
        pmf = np.clip(pmf, 0.0, None)
        s = pmf.sum()
        if s <= 0:
            raise ValueError("pmf sums to zero")
        self.grid = grid
        self.pmf = pmf / s
        self.meta = dict(meta or {})

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_continuous(
        cls,
        mu: float,
        sigma: float,
        dist: Literal["normal", "t"] = "t",
        df: float = 7.0,
        key_numbers: KeyNumberProfile | None = None,
        max_margin: int = DEFAULT_MAX_MARGIN,
        match_moments: bool = True,
        meta: dict | None = None,
    ) -> "MarginDistribution":
        """Discretise a continuous scaffold and impose the lattice.

        With ``match_moments`` (the default) the scaffold's location and scale
        are solved for so the *final* PMF has mean ``mu`` and sd ``sigma``.
        That matters: multiplying by lattice weights and zeroing the tie cell
        both perturb the moments, and without this correction a distribution
        asked for sigma=16 can quietly come back at 15, which is a systematic
        overconfidence bias on every tail market.
        """
        sigma = max(float(sigma), 0.5)
        grid = np.arange(-max_margin, max_margin + 1)
        cdf = ((lambda z: stats.norm.cdf(z)) if dist == "normal"
               else (lambda z: stats.t.cdf(z, df)))
        unit_sd = 1.0 if dist == "normal" else (
            np.sqrt(df / (df - 2.0)) if df > 2 else 1.0)
        kn_vec = None if key_numbers is None else key_numbers.vector(grid)

        def build(loc: float, scale: float) -> np.ndarray:
            raw = cdf((grid + 0.5 - loc) / scale) - cdf((grid - 0.5 - loc) / scale)
            if kn_vec is not None:
                raw = raw * kn_vec
            total = raw.sum()
            return raw / total if total > 0 else raw

        loc, scale = float(mu), sigma / unit_sd
        if match_moments:
            for _ in range(24):
                pmf = build(loc, scale)
                m = float(np.sum(grid * pmf))
                sd = float(np.sqrt(np.sum((grid - m) ** 2 * pmf)))
                if abs(m - mu) < 1e-3 and abs(sd - sigma) < 1e-3:
                    break
                loc += (mu - m)
                if sd > 1e-6:
                    scale *= float(np.clip(sigma / sd, 0.5, 2.0))
                scale = float(np.clip(scale, 0.25, 200.0))
        pmf = build(loc, scale)

        m = {"mu_input": float(mu), "sigma_input": float(sigma),
             "scaffold": dist, "df": float(df),
             "scaffold_loc": float(loc), "scaffold_scale": float(scale),
             "lattice": key_numbers is not None}
        m.update(meta or {})
        return cls(grid, pmf, m)

    @classmethod
    def from_samples(cls, samples: Iterable[float], max_margin: int = DEFAULT_MAX_MARGIN,
                     key_numbers: KeyNumberProfile | None = None,
                     laplace: float = 0.0, allow_ties: bool = False,
                     meta: dict | None = None) -> "MarginDistribution":
        """Empirical PMF from simulated or observed margins.

        ``laplace`` adds a small floor to every cell so a market on a margin the
        simulation happened never to produce is not priced at exactly zero.  The
        tie cell is exempt and forced to zero -- college football has had
        overtime since 1996, so a final margin of 0 is genuinely impossible, not
        merely unobserved.
        """
        s = np.rint(np.asarray(list(samples), dtype=float)).astype(int)
        s = s[np.isfinite(s)]
        grid = np.arange(-max_margin, max_margin + 1)
        pmf = np.full(grid.shape, float(laplace))
        vals, counts = np.unique(np.clip(s, -max_margin, max_margin), return_counts=True)
        pmf[vals + max_margin] += counts
        ties_ok = allow_ties if key_numbers is None else key_numbers.allow_ties
        if not ties_ok:
            pmf[max_margin] = 0.0
        return cls(grid, pmf, {"scaffold": "samples", "n_samples": int(s.size), **(meta or {})})

    @classmethod
    def from_residual_kde(
        cls,
        mu: float,
        residuals: Sequence[float],
        sigma: float | None = None,
        bandwidth: float = 2.0,
        key_numbers: KeyNumberProfile | None = None,
        max_margin: int = DEFAULT_MAX_MARGIN,
        max_pool: int = 1500,
        meta: dict | None = None,
    ) -> "MarginDistribution":
        """Empirical scaffold: smooth the historical residual cloud, then shift.

        Optionally rescale the residual cloud to a game-specific ``sigma`` --
        that is how a heteroskedastic scale model gets combined with a
        non-parametric shape.
        """
        r = np.asarray(residuals, dtype=float)
        r = r[np.isfinite(r)]
        if r.size > max_pool:
            # Thin by even strides rather than randomly: keeps the shape of the
            # residual cloud exactly and keeps the result reproducible.
            r = r[:: int(np.ceil(r.size / max_pool))]
        if r.size < 50:
            return cls.from_continuous(mu, sigma or float(np.std(r) if r.size else 16.5),
                                       key_numbers=key_numbers, max_margin=max_margin, meta=meta)
        if sigma is not None and np.std(r) > 1e-6:
            r = r * (float(sigma) / float(np.std(r)))
        grid = np.arange(-max_margin, max_margin + 1)
        kn_vec = None if key_numbers is None else key_numbers.vector(grid)

        def build(shift: float) -> np.ndarray:
            z = (grid[:, None] - shift - r[None, :]) / bandwidth
            raw = np.exp(-0.5 * z ** 2).sum(axis=1)
            if kn_vec is not None:
                raw = raw * kn_vec
            total = raw.sum()
            return raw / total if total > 0 else raw

        # The residual cloud is already scaled; only recentre so the PMF mean
        # lands on mu after the lattice and tie-cell adjustments.
        shift = float(mu)
        for _ in range(12):
            pmf = build(shift)
            m_now = float(np.sum(grid * pmf))
            if abs(m_now - mu) < 1e-3:
                break
            shift += (mu - m_now)
        pmf = build(shift)
        m = {"scaffold": "kde", "mu_input": float(mu), "n_residuals": int(r.size),
             "lattice": key_numbers is not None}
        m.update(meta or {})
        return cls(grid, pmf, m)

    # -- summary stats -----------------------------------------------------
    def mean(self) -> float:
        return float(np.sum(self.grid * self.pmf))

    def var(self) -> float:
        m = self.mean()
        return float(np.sum((self.grid - m) ** 2 * self.pmf))

    def sd(self) -> float:
        return float(np.sqrt(self.var()))

    def cdf(self, k: float) -> float:
        """P(margin <= k)."""
        return float(self.pmf[self.grid <= k].sum())

    def quantile(self, q: float) -> int:
        c = np.cumsum(self.pmf)
        i = int(np.searchsorted(c, min(max(q, 0.0), 1.0)))
        return int(self.grid[min(i, len(self.grid) - 1)])

    def median(self) -> int:
        return self.quantile(0.5)

    def mode(self) -> int:
        return int(self.grid[int(np.argmax(self.pmf))])

    # -- market queries ----------------------------------------------------
    def p_exact(self, k: int) -> float:
        idx = np.where(self.grid == int(k))[0]
        return float(self.pmf[idx][0]) if idx.size else 0.0

    def p_greater(self, x: float) -> float:
        """P(margin > x). ``x`` may be a half-point."""
        return float(self.pmf[self.grid > x].sum())

    def p_at_least(self, k: float) -> float:
        return float(self.pmf[self.grid >= k].sum())

    def p_less(self, x: float) -> float:
        return float(self.pmf[self.grid < x].sum())

    def p_at_most(self, k: float) -> float:
        return float(self.pmf[self.grid <= k].sum())

    def p_between(self, lo: float, hi: float, inclusive: bool = True) -> float:
        if inclusive:
            mask = (self.grid >= lo) & (self.grid <= hi)
        else:
            mask = (self.grid > lo) & (self.grid < hi)
        return float(self.pmf[mask].sum())

    def p_home_win(self) -> float:
        return self.p_greater(0.0)

    def p_away_win(self) -> float:
        return self.p_less(0.0)

    def p_tie(self) -> float:
        return self.p_exact(0)

    def cover_probability(self, spread: float, side: str = "home") -> dict[str, float]:
        """Probability of covering a spread quoted in *home* terms.

        ``spread=-7`` means home is laying 7.  Home covers when
        ``margin > 7``; a push happens when ``margin == 7``.
        """
        line = -float(spread)
        win = self.p_greater(line)
        push = self.p_exact(int(line)) if float(line).is_integer() else 0.0
        lose = 1.0 - win - push
        if side == "away":
            win, lose = lose, win
        return {"win": win, "push": push, "lose": lose,
                "win_excl_push": win / (win + lose) if (win + lose) > 0 else 0.5}

    def probability_for_strike(self, strike_type: str | None,
                               floor_strike: float | None = None,
                               cap_strike: float | None = None) -> float | None:
        """Map a Kalshi-style strike specification onto this PMF."""
        st = (strike_type or "").lower()
        if st in ("greater", "gt") and floor_strike is not None:
            return self.p_greater(floor_strike)
        if st in ("greater_or_equal", "gte") and floor_strike is not None:
            return self.p_at_least(floor_strike)
        if st in ("less", "lt") and cap_strike is not None:
            return self.p_less(cap_strike)
        if st in ("less_or_equal", "lte") and cap_strike is not None:
            return self.p_at_most(cap_strike)
        if st == "between" and floor_strike is not None and cap_strike is not None:
            return self.p_between(floor_strike, cap_strike)
        return None

    # -- transforms --------------------------------------------------------
    def flip(self) -> "MarginDistribution":
        """Same game, away-team perspective."""
        return MarginDistribution(-self.grid[::-1], self.pmf[::-1],
                                  {**self.meta, "flipped": True})

    def shift(self, delta: float) -> "MarginDistribution":
        """Translate the distribution by ``delta`` points (integer rounding)."""
        d = int(round(delta))
        pmf = np.zeros_like(self.pmf)
        src_lo = max(0, -d)
        src_hi = min(len(self.grid), len(self.grid) - d)
        pmf[src_lo + d: src_hi + d] = self.pmf[src_lo:src_hi]
        return MarginDistribution(self.grid, pmf, {**self.meta, "shifted_by": d})

    def blend(self, other: "MarginDistribution", weight: float = 0.5
              ) -> "MarginDistribution":
        """Linear opinion pool with another distribution on the same grid."""
        if not np.array_equal(self.grid, other.grid):
            raise ValueError("grids must match to blend")
        w = float(np.clip(weight, 0.0, 1.0))
        return MarginDistribution(self.grid, (1 - w) * self.pmf + w * other.pmf,
                                  {**self.meta, "blended_with": other.meta.get("scaffold"),
                                   "blend_weight": w})

    # -- output ------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"margin": self.grid, "prob": self.pmf,
                             "cum_prob": np.cumsum(self.pmf)})

    def summary(self, home: str = "home", away: str = "away") -> dict:
        mean = self.mean()
        return {
            "home_team": home, "away_team": away,
            "mean_margin": round(mean, 2),
            "median_margin": self.median(),
            "modal_margin": self.mode(),
            "sd": round(self.sd(), 2),
            "fair_spread_home": round(-mean, 1),
            "p_home_win": round(self.p_home_win(), 4),
            "p_away_win": round(self.p_away_win(), 4),
            "p_home_win_by_3_or_less": round(self.p_between(1, 3), 4),
            "p_home_win_by_7_plus": round(self.p_at_least(7), 4),
            "q05": self.quantile(0.05), "q25": self.quantile(0.25),
            "q75": self.quantile(0.75), "q95": self.quantile(0.95),
            "scaffold": self.meta.get("scaffold"),
        }

    def bucket_table(self, edges: Sequence[float] | None = None) -> pd.DataFrame:
        """Probabilities for the 'wins by X' bucket shapes Kalshi tends to list."""
        edges = list(edges or [1, 3, 7, 10, 14, 21, 28])
        rows = []
        prev = 0.0
        for e in edges:
            rows.append({"bucket": f"home by {int(prev) + 1}-{int(e)}",
                         "prob": self.p_between(prev + 1, e)})
            prev = e
        rows.append({"bucket": f"home by {int(prev) + 1}+", "prob": self.p_greater(prev)})
        prev = 0.0
        for e in edges:
            rows.append({"bucket": f"away by {int(prev) + 1}-{int(e)}",
                         "prob": self.p_between(-e, -(prev + 1))})
            prev = e
        rows.append({"bucket": f"away by {int(prev) + 1}+", "prob": self.p_less(-prev)})
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (f"MarginDistribution(mean={self.mean():.2f}, sd={self.sd():.2f}, "
                f"P(home)={self.p_home_win():.3f}, scaffold={self.meta.get('scaffold')})")
