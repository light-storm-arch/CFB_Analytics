"""Blend several fitted models.

Two modes:

``fixed``    -- user-supplied weights (default: equal).
``stacked``  -- weights learned by non-negative least squares on out-of-sample
                predictions, which is the honest way to combine correlated
                models.  Falls back to equal weights if no OOS frame is given.

Ensembling matters more than model choice in this domain: ridge and boosted
trees make different *kinds* of errors (ridge is biased toward the mean on
mismatches, trees are noisy on sparse regions), and averaging is a reliable
half-point of MAE.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cfb.models.base import SpreadModel, TrainReport

log = logging.getLogger(__name__)


class EnsembleSpreadModel(SpreadModel):
    name = "ensemble"
    handles_nan = True

    def __init__(self, features, members: list[SpreadModel] | None = None,
                 weights: list[float] | None = None, **params):
        super().__init__(features, **params)
        self.members = members or []
        self.weights = weights

    def _make_estimator(self, target: str):  # pragma: no cover - unused
        raise NotImplementedError("EnsembleSpreadModel composes fitted members")

    def fit(self, df: pd.DataFrame) -> "EnsembleSpreadModel":
        if not self.members:
            raise ValueError("ensemble needs members")
        for m in self.members:
            m.fit(df)
        if self.weights is None:
            self.weights = [1.0 / len(self.members)] * len(self.members)
        self.features_used = sorted({f for m in self.members for f in m.features_used})
        train = df.dropna(subset=["margin", "total"])
        pred = self.predict(train)
        self.report = TrainReport(
            n_train=len(train),
            features_used=self.features_used,
            dropped_features=[],
            margin_train_mae=float(np.mean(np.abs(pred["pred_margin"] - train["margin"]))),
            total_train_mae=float(np.mean(np.abs(pred["pred_total"] - train["total"]))),
            extra={"weights": dict(zip([m.name for m in self.members], self.weights))},
        )
        return self

    def fit_weights(self, oos: pd.DataFrame) -> "EnsembleSpreadModel":
        """Learn non-negative weights on an out-of-sample prediction frame.

        ``oos`` must contain the true ``margin`` plus one column per member
        named ``pred_margin__<member name>``.
        """
        from scipy.optimize import nnls

        cols = [f"pred_margin__{m.name}" for m in self.members]
        sub = oos.dropna(subset=cols + ["margin"])
        if sub.empty:
            log.warning("no OOS rows; keeping equal weights")
            return self
        A = sub[cols].to_numpy(float)
        y = sub["margin"].to_numpy(float)
        w, _ = nnls(A, y)
        if w.sum() <= 0:
            log.warning("degenerate NNLS solution; keeping equal weights")
            return self
        self.weights = list(w / w.sum())
        log.info("stacked weights: %s",
                 dict(zip([m.name for m in self.members], np.round(self.weights, 3))))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [m.predict(df) for m in self.members]
        w = np.asarray(self.weights, dtype=float)
        w = w / w.sum()
        margin = sum(wi * p["pred_margin"].to_numpy() for wi, p in zip(w, parts))
        total = sum(wi * p["pred_total"].to_numpy() for wi, p in zip(w, parts))
        out = pd.DataFrame({
            "game_id": df["game_id"].to_numpy() if "game_id" in df else np.arange(len(df)),
            "pred_margin": margin, "pred_total": total,
        }, index=df.index)
        for m, p in zip(self.members, parts):
            out[f"pred_margin__{m.name}"] = p["pred_margin"].to_numpy()
            out[f"pred_total__{m.name}"] = p["pred_total"].to_numpy()
        return out

    def feature_importance(self) -> pd.DataFrame:
        frames = []
        for m, w in zip(self.members, self.weights or []):
            fi = m.feature_importance()
            if fi.empty:
                continue
            fi = fi.copy()
            fi["importance"] = fi["importance"] / fi["importance"].sum() * w
            frames.append(fi)
        if not frames:
            return pd.DataFrame(columns=["feature", "importance"])
        return (pd.concat(frames).groupby("feature", as_index=False)["importance"].sum()
                .sort_values("importance", ascending=False).reset_index(drop=True))
