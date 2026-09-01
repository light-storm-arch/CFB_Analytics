"""Price conversions and vig removal.

Kalshi quotes in cents where a contract settles at 100 cents, so a price *is* a
probability -- no conversion needed, which is the main reason it is a nicer
venue to model against than a sportsbook.  Sportsbook lines still matter as a
reference for what the sharp consensus thinks, and those need the vig taken out
before they mean anything.

Three de-vig methods are provided because they disagree most exactly where it
matters -- on longshots:

* ``multiplicative`` divides by the overround.  Simple, and biases longshots up.
* ``additive`` splits the overround evenly in probability terms.
* ``power`` solves for the exponent that makes the implied probabilities sum to
  one.  Best-behaved on lopsided markets, and the default here for that reason.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def american_to_prob(odds: float) -> float:
    """Raw (vig-inclusive) implied probability from American odds."""
    o = float(odds)
    return (-o) / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def prob_to_american(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return -100.0 * p / (1 - p) if p >= 0.5 else 100.0 * (1 - p) / p


def decimal_to_prob(dec: float) -> float:
    return 1.0 / float(dec)


def fair_cents(prob: float) -> float:
    """Model probability as a Kalshi contract price, in cents."""
    return round(100.0 * float(np.clip(prob, 0.0, 1.0)), 1)


def devig(raw_probs, method: str = "power") -> np.ndarray:
    """Remove the bookmaker's margin from a set of mutually exclusive outcomes."""
    p = np.asarray(list(raw_probs), dtype=float)
    p = np.clip(p, 1e-9, None)
    s = p.sum()
    if s <= 0:
        return p
    if method == "multiplicative":
        return p / s
    if method == "additive":
        out = p - (s - 1.0) / len(p)
        return np.clip(out, 1e-9, None) / np.clip(out, 1e-9, None).sum()
    if method != "power":
        raise ValueError("method must be multiplicative, additive or power")
    if abs(s - 1.0) < 1e-9:
        return p
    try:
        k = brentq(lambda kk: np.sum(p ** kk) - 1.0, 0.2, 5.0, xtol=1e-10)
    except ValueError:
        return p / s
    out = p ** k
    return out / out.sum()


def two_way_devig(home_odds: float, away_odds: float,
                  method: str = "power") -> tuple[float, float]:
    p = devig([american_to_prob(home_odds), american_to_prob(away_odds)], method)
    return float(p[0]), float(p[1])


def spread_to_probability(spread: float, sigma: float = 16.5) -> float:
    """Normal-approximation win probability implied by a point spread.

    Deliberately crude -- for anything you are betting, use a real
    ``MarginDistribution`` instead.  This exists for quick sanity checks and
    for filling gaps when only a spread is available.
    """
    from scipy import stats
    return float(stats.norm.cdf(-float(spread) / float(sigma)))
