"""Tree ensembles: random forest, XGBoost, LightGBM.

All three take NaN in stride (the forest via an imputer, the boosters natively),
which matters because early-season games genuinely have no trailing form.

If xgboost / lightgbm are not installed the corresponding classes fall back to
scikit-learn's HistGradientBoostingRegressor rather than failing, so the CLI
still works on a bare install.
"""
from __future__ import annotations

import logging

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from cfb.models.base import SpreadModel

log = logging.getLogger(__name__)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


class RandomForestSpreadModel(SpreadModel):
    name = "forest"
    handles_nan = False

    def _make_estimator(self, target: str):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("est", RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 600),
                max_depth=self.params.get("max_depth", 12),
                min_samples_leaf=self.params.get("min_samples_leaf", 20),
                max_features=self.params.get("max_features", 0.4),
                n_jobs=self.params.get("n_jobs", -1),
                random_state=self.params.get("random_state", 7),
            )),
        ])


class XGBoostSpreadModel(SpreadModel):
    """Gradient-boosted trees.

    Defaults are deliberately conservative: college football has ~800 FBS games
    a season and a signal-to-noise ratio where an over-fit booster will happily
    memorise blowouts.  Shallow trees, heavy subsampling, and strong L2 keep it
    honest.
    """
    name = "xgboost"
    handles_nan = True

    def _make_estimator(self, target: str):
        if not _has("xgboost"):
            log.warning("xgboost not installed; using HistGradientBoostingRegressor")
            return HistGradientBoostingRegressor(
                max_depth=self.params.get("max_depth", 4),
                learning_rate=self.params.get("learning_rate", 0.03),
                max_iter=self.params.get("n_estimators", 600),
                l2_regularization=self.params.get("reg_lambda", 3.0),
                random_state=self.params.get("random_state", 7),
            )
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=self.params.get("n_estimators", 700),
            learning_rate=self.params.get("learning_rate", 0.028),
            max_depth=self.params.get("max_depth", 4),
            min_child_weight=self.params.get("min_child_weight", 12),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.7),
            reg_lambda=self.params.get("reg_lambda", 3.0),
            reg_alpha=self.params.get("reg_alpha", 0.2),
            objective=self.params.get("objective", "reg:squarederror"),
            tree_method=self.params.get("tree_method", "hist"),
            n_jobs=self.params.get("n_jobs", -1),
            random_state=self.params.get("random_state", 7),
        )


class LightGBMSpreadModel(SpreadModel):
    name = "lightgbm"
    handles_nan = True

    def _make_estimator(self, target: str):
        if not _has("lightgbm"):
            log.warning("lightgbm not installed; using HistGradientBoostingRegressor")
            return HistGradientBoostingRegressor(
                max_depth=self.params.get("max_depth", 4),
                learning_rate=self.params.get("learning_rate", 0.03),
                max_iter=self.params.get("n_estimators", 600),
                random_state=self.params.get("random_state", 7),
            )
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=self.params.get("n_estimators", 700),
            learning_rate=self.params.get("learning_rate", 0.028),
            num_leaves=self.params.get("num_leaves", 15),
            min_child_samples=self.params.get("min_child_samples", 25),
            subsample=self.params.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=self.params.get("colsample_bytree", 0.7),
            reg_lambda=self.params.get("reg_lambda", 3.0),
            n_jobs=self.params.get("n_jobs", -1),
            random_state=self.params.get("random_state", 7),
            verbose=-1,
        )
