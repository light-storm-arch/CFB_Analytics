"""Common interface for every spread model.

A model consumes the feature frame produced by ``cfb.features.build`` and emits
two point estimates per game:

* ``pred_margin`` -- expected home margin (home_points - away_points)
* ``pred_total``  -- expected combined points

The total matters as much as the margin: game variance scales with scoring
environment, so the distribution layer conditions its spread parameter on the
predicted total.

Models are deliberately thin wrappers.  Everything shared -- feature selection,
recency weighting, NaN policy, persistence -- lives here so that flipping
between ridge / XGBoost / random forest changes only the estimator, never the
data contract.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    game_id: Any
    pred_margin: float
    pred_total: float

    def as_dict(self) -> dict:
        return {"game_id": self.game_id, "pred_margin": self.pred_margin,
                "pred_total": self.pred_total}


@dataclass
class TrainReport:
    n_train: int
    features_used: list[str]
    dropped_features: list[str]
    margin_train_mae: float
    total_train_mae: float
    extra: dict = field(default_factory=dict)


class SpreadModel(ABC):
    """Base class: fit two heads (margin, total) on the same feature matrix."""

    name: str = "base"
    #: Whether the estimator copes with NaN internally (tree learners do).
    handles_nan: bool = False

    def __init__(self, features: list[str], half_life_days: float | None = 900.0,
                 min_feature_coverage: float = 0.5, **params):
        self.features = list(features)
        self.half_life_days = half_life_days
        self.min_feature_coverage = min_feature_coverage
        self.params = params
        self.features_used: list[str] = []
        self.margin_model = None
        self.total_model = None
        self.report: TrainReport | None = None
        self._medians: pd.Series | None = None

    # -- subclass hooks ---------------------------------------------------
    @abstractmethod
    def _make_estimator(self, target: str):
        """Return an unfitted sklearn-compatible regressor for ``target``."""

    def _fit_estimator(self, est, X: np.ndarray, y: np.ndarray, w: np.ndarray | None):
        """Fit, routing sample weights to the final step of a Pipeline."""
        if w is None:
            est.fit(X, y)
            return est
        steps = getattr(est, "steps", None)
        kwargs = {f"{steps[-1][0]}__sample_weight": w} if steps else {"sample_weight": w}
        try:
            est.fit(X, y, **kwargs)
        except (TypeError, ValueError):
            log.debug("%s does not accept sample_weight; fitting unweighted", type(est).__name__)
            est.fit(X, y)
        return est

    # -- shared machinery -------------------------------------------------
    def _select_features(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        present = [f for f in self.features if f in df.columns]
        coverage = df[present].notna().mean()
        used = [f for f in present if coverage[f] >= self.min_feature_coverage]
        dropped = [f for f in self.features if f not in used]
        return used, dropped

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.features_used).astype(float)
        if not self.handles_nan:
            X = X.fillna(self._medians)
            X = X.fillna(0.0)
        return X.to_numpy()

    def _weights(self, df: pd.DataFrame) -> np.ndarray | None:
        if not self.half_life_days or "start_date" not in df.columns:
            return None
        dates = pd.to_datetime(df["start_date"], utc=True)
        age = (dates.max() - dates).dt.total_seconds() / 86400.0
        return np.exp(-np.log(2.0) * age.to_numpy() / self.half_life_days)

    def fit(self, df: pd.DataFrame) -> "SpreadModel":
        train = df.dropna(subset=["margin", "total"]).copy()
        if train.empty:
            raise ValueError("no completed games to train on")
        self.features_used, dropped = self._select_features(train)
        if not self.features_used:
            raise ValueError("no usable features (all below coverage threshold)")
        self._medians = train[self.features_used].astype(float).median()
        X = self._matrix(train)
        w = self._weights(train)

        self.margin_model = self._fit_estimator(
            self._make_estimator("margin"), X, train["margin"].to_numpy(float), w)
        self.total_model = self._fit_estimator(
            self._make_estimator("total"), X, train["total"].to_numpy(float), w)

        pm = self.margin_model.predict(X)
        pt = self.total_model.predict(X)
        self.report = TrainReport(
            n_train=len(train),
            features_used=list(self.features_used),
            dropped_features=dropped,
            margin_train_mae=float(np.mean(np.abs(pm - train["margin"].to_numpy()))),
            total_train_mae=float(np.mean(np.abs(pt - train["total"].to_numpy()))),
        )
        log.info("%s fit n=%d feats=%d train MAE margin=%.2f total=%.2f",
                 self.name, len(train), len(self.features_used),
                 self.report.margin_train_mae, self.report.total_train_mae)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.margin_model is None:
            raise RuntimeError(f"{self.name} is not fitted")
        X = self._matrix(df)
        out = pd.DataFrame({
            "game_id": df["game_id"].to_numpy() if "game_id" in df else np.arange(len(df)),
            "pred_margin": self.margin_model.predict(X),
            "pred_total": self.total_model.predict(X),
        }, index=df.index)
        return out

    # -- introspection ----------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        est = _unwrap(self.margin_model)
        vals = None
        if hasattr(est, "feature_importances_"):
            vals = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            vals = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
        if vals is None or len(vals) != len(self.features_used):
            return pd.DataFrame(columns=["feature", "importance"])
        return (pd.DataFrame({"feature": self.features_used, "importance": vals})
                .sort_values("importance", ascending=False).reset_index(drop=True))

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        meta = path.with_suffix(".json")
        meta.write_text(json.dumps({
            "name": self.name,
            "features_used": self.features_used,
            "params": {k: str(v) for k, v in self.params.items()},
            "report": None if self.report is None else {
                "n_train": self.report.n_train,
                "margin_train_mae": self.report.margin_train_mae,
                "total_train_mae": self.report.total_train_mae,
                "dropped_features": self.report.dropped_features,
            },
        }, indent=2))
        return path

    @staticmethod
    def load(path: str | Path) -> "SpreadModel":
        import joblib
        return joblib.load(Path(path))


def _unwrap(model):
    """Reach the estimator inside an sklearn Pipeline, if there is one."""
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model
