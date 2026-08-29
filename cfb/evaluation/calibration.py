"""Distribution scoring and calibration.

A spread number is only half the product.  What decides whether a Kalshi
'wins by 4-7' price is any good is whether the *distribution* is calibrated,
and MAE says nothing about that.  So:

* **CRPS** -- the proper scoring rule for a full predictive distribution.
  Lower is better, and it is in points, so it is comparable across models.
* **PIT / reliability** -- if the distribution is honest, the probability
  integral transform of the outcomes is uniform, and events assigned 30%
  happen 30% of the time.  A model that is overconfident shows up here as a
  U-shaped PIT histogram long before it shows up in profit.
* **Log loss / Brier** on the binary win probability.
* **Interval coverage** -- does the 80% interval contain the result 80% of
  the time?

Randomised PIT is used because the outcome is discrete; without the
randomisation the histogram is biased even for a perfectly calibrated model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.distribution.margin import MarginDistribution


def crps_discrete(dist: MarginDistribution, outcome: float) -> float:
    """CRPS = sum over the grid of (F(k) - 1{outcome <= k})^2."""
    cdf = np.cumsum(dist.pmf)
    step = (dist.grid >= outcome).astype(float)
    return float(np.sum((cdf - step) ** 2))


def crps_normal(mu: float, sigma: float, outcome: float) -> float:
    """Closed-form CRPS for a normal -- handy as a reference baseline."""
    from scipy import stats
    z = (outcome - mu) / sigma
    return float(sigma * (z * (2 * stats.norm.cdf(z) - 1)
                          + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)))


def pit_values(dists: list[MarginDistribution], outcomes: np.ndarray,
               seed: int = 3) -> np.ndarray:
    """Randomised probability integral transform, uniform iff calibrated."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(outcomes), dtype=float)
    for i, (d, y) in enumerate(zip(dists, outcomes)):
        lo = d.p_less(y)
        p_at = d.p_exact(int(round(y)))
        out[i] = lo + rng.random() * p_at
    return out


def calibration_table(probs: np.ndarray, outcomes: np.ndarray,
                      bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted probability vs realised frequency."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size == 0:
        return pd.DataFrame()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "pred_mean": float(p[m].mean()),
            "actual_rate": float(y[m].mean()),
            "gap": float(y[m].mean() - p[m].mean()),
        })
    return pd.DataFrame(rows)


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    y = np.asarray(outcomes, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    return float(-np.mean(y[ok] * np.log(p[ok]) + (1 - y[ok]) * np.log(1 - p[ok])))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[ok] - y[ok]) ** 2))


def interval_coverage(dists: list[MarginDistribution], outcomes: np.ndarray,
                      levels=(0.5, 0.8, 0.9)) -> pd.DataFrame:
    rows = []
    for lvl in levels:
        lo_q, hi_q = (1 - lvl) / 2, 1 - (1 - lvl) / 2
        hits = 0
        widths = []
        for d, y in zip(dists, outcomes):
            lo, hi = d.quantile(lo_q), d.quantile(hi_q)
            widths.append(hi - lo)
            hits += int(lo <= y <= hi)
        rows.append({"level": lvl, "coverage": hits / max(len(outcomes), 1),
                     "mean_width": float(np.mean(widths))})
    return pd.DataFrame(rows)


def distribution_report(dists: list[MarginDistribution], outcomes: np.ndarray,
                        bins: int = 10) -> dict:
    """Everything you need to decide whether the distribution can be traded."""
    outcomes = np.asarray(outcomes, dtype=float)
    crps = np.array([crps_discrete(d, y) for d, y in zip(dists, outcomes)])
    p_home = np.array([d.p_home_win() for d in dists])
    y_home = (outcomes > 0).astype(float)
    pit = pit_values(dists, outcomes)
    # Uniformity of the PIT via a chi-square on equal-width bins.
    counts, _ = np.histogram(pit, bins=bins, range=(0, 1))
    expected = len(pit) / bins
    chi2 = float(np.sum((counts - expected) ** 2) / expected) if expected > 0 else np.nan
    return {
        "n": int(len(dists)),
        "crps_mean": float(np.mean(crps)),
        "log_loss": log_loss(p_home, y_home),
        "brier": brier(p_home, y_home),
        "pit_chi2": chi2,
        "pit_chi2_df": bins - 1,
        "mean_sd": float(np.mean([d.sd() for d in dists])),
        "calibration": calibration_table(p_home, y_home, bins),
        "coverage": interval_coverage(dists, outcomes),
        "pit": pit,
    }


def summarize_report(rep: dict) -> str:
    lines = [
        f"games        {rep['n']}",
        f"CRPS         {rep['crps_mean']:.4f}   (lower is better)",
        f"log loss     {rep['log_loss']:.4f}",
        f"Brier        {rep['brier']:.4f}",
        f"mean sd      {rep['mean_sd']:.2f}",
        f"PIT chi2     {rep['pit_chi2']:.1f} on {rep['pit_chi2_df']} df "
        f"(>~{rep['pit_chi2_df'] * 2} suggests miscalibration)",
        "",
        "interval coverage:",
        rep["coverage"].to_string(index=False),
        "",
        "win-probability reliability:",
        rep["calibration"].to_string(index=False),
    ]
    return "\n".join(lines)
