"""Heteroskedastic scale model: how wrong is the spread likely to be?

A single league-wide sigma (~16.5 points) is a decent first approximation, but
it is measurably wrong at the edges:

* **Scoring environment.** A projected 75-point track meet has a wider margin
  distribution than a projected 38-point rock fight.  Variance scales with
  possessions and points per possession.
* **Mismatches.** Big favourites have fatter right tails -- a 35-point favourite
  can win by 60, but cannot lose by 60 in any realistic sense.  This shows up as
  higher residual variance for large |mu|.
* **Early season.** Week 1-3 residuals are wider because the ratings have less
  to work with.

The model is deliberately simple and fit on **out-of-sample** residuals: predict
``log|residual|`` from a handful of features, exponentiate, then apply a single
global calibration constant so the standardised residuals have unit variance.
Fitting on in-sample residuals would understate sigma badly -- the most common
way a distribution model ends up overconfident.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cfb.constants import DEFAULT_MARGIN_SD

log = logging.getLogger(__name__)

SIGMA_FEATURES = [
    "pred_total", "abs_pred_margin", "week", "neutral_site",
    "rat_games_home", "rat_games_away", "early_season",
]


@dataclass
class SigmaFit:
    n: int
    baseline_sd: float
    calibration: float
    z_sd: float
    z_kurtosis: float
    fitted_df: float


class SigmaModel:
    """Predicts the standard deviation of (actual margin - predicted margin)."""

    def __init__(self, floor: float = 8.0, ceiling: float = 26.0,
                 alpha: float = 3.0, use_gbm: bool = False):
        self.floor = floor
        self.ceiling = ceiling
        self.alpha = alpha
        self.use_gbm = use_gbm
        self.model = None
        self.calibration = 1.0
        self.baseline_sd = DEFAULT_MARGIN_SD
        self.fit_report: SigmaFit | None = None
        self.residual_pool: np.ndarray = np.array([])
        self.fitted_df: float = 7.0

    # -- feature prep ------------------------------------------------------
    @staticmethod
    def make_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["pred_total"] = df.get("pred_total", pd.Series(np.nan, index=df.index))
        out["abs_pred_margin"] = df.get("pred_margin",
                                        pd.Series(np.nan, index=df.index)).abs()
        out["week"] = df.get("week", pd.Series(np.nan, index=df.index))
        out["neutral_site"] = pd.to_numeric(
            df.get("neutral_site", pd.Series(0.0, index=df.index)), errors="coerce")
        out["rat_games_home"] = df.get("rat_games_home", pd.Series(np.nan, index=df.index))
        out["rat_games_away"] = df.get("rat_games_away", pd.Series(np.nan, index=df.index))
        out["early_season"] = (out["week"].fillna(9) <= 3).astype(float)
        return out[SIGMA_FEATURES]

    def _estimator(self):
        if self.use_gbm:
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                 max_iter=250, random_state=7)
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("est", Ridge(alpha=self.alpha)),
        ])

    # -- fit ---------------------------------------------------------------
    def fit(self, oos: pd.DataFrame) -> "SigmaModel":
        """Fit on an out-of-sample frame with ``margin`` and ``pred_margin``."""
        d = oos.dropna(subset=["margin", "pred_margin"]).copy()
        if d.empty:
            raise ValueError("SigmaModel needs out-of-sample predictions to fit")
        resid = (d["margin"] - d["pred_margin"]).to_numpy(float)
        self.baseline_sd = float(np.std(resid))
        self.residual_pool = resid

        X = self.make_frame(d)
        y = np.log(np.abs(resid) + 1.0)
        self.model = self._estimator()
        self.model.fit(X, y)

        raw = np.exp(self.model.predict(X))
        raw = np.clip(raw, 1e-3, None)
        # One global constant makes E[(resid/sigma)^2] == 1.
        self.calibration = float(np.sqrt(np.mean((resid / raw) ** 2)))
        sigma = np.clip(raw * self.calibration, self.floor, self.ceiling)
        z = resid / sigma
        self.fitted_df = _fit_t_df(z)
        self.fit_report = SigmaFit(
            n=len(d), baseline_sd=self.baseline_sd, calibration=self.calibration,
            z_sd=float(np.std(z)), z_kurtosis=float(pd.Series(z).kurtosis()),
            fitted_df=self.fitted_df,
        )
        log.info("sigma fit n=%d baseline_sd=%.2f z_sd=%.3f kurt=%.2f t_df=%.1f",
                 len(d), self.baseline_sd, self.fit_report.z_sd,
                 self.fit_report.z_kurtosis, self.fitted_df)
        return self

    # -- predict -----------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(df), self.baseline_sd)
        raw = np.exp(self.model.predict(self.make_frame(df)))
        return np.clip(raw * self.calibration, self.floor, self.ceiling)

    def standardized_residuals(self, oos: pd.DataFrame) -> np.ndarray:
        d = oos.dropna(subset=["margin", "pred_margin"])
        return ((d["margin"] - d["pred_margin"]).to_numpy(float) / self.predict(d))

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        if self.fit_report:
            path.with_suffix(".json").write_text(json.dumps(
                self.fit_report.__dict__, indent=2))
        return path

    @staticmethod
    def load(path: str | Path) -> "SigmaModel":
        import joblib
        return joblib.load(Path(path))


def _fit_t_df(z: np.ndarray, lo: float = 3.0, hi: float = 40.0) -> float:
    """MLE-ish degrees of freedom for standardised residuals (grid search)."""
    from scipy import stats

    z = z[np.isfinite(z)]
    if z.size < 100:
        return 7.0
    best, best_ll = 7.0, -np.inf
    for df in np.linspace(lo, hi, 60):
        scale = np.sqrt(df / (df - 2.0))
        ll = float(np.sum(stats.t.logpdf(z * scale, df) + np.log(scale)))
        if ll > best_ll:
            best, best_ll = float(df), ll
    return best
