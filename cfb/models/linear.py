"""Regularised linear models: the interpretable baseline.

Ridge is the right default here.  The rating features are strongly collinear by
construction (rat_net_diff is nearly rat_off_diff - rat_def_diff), and ridge
handles that gracefully where OLS would produce wild, unstable coefficients.
``ElasticNetSpreadModel`` is offered when you want sparsity for interpretation.
"""
from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor  # noqa: F401  (handy for experiments)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cfb.models.base import SpreadModel

ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


class RidgeSpreadModel(SpreadModel):
    name = "ridge"
    handles_nan = False

    def _make_estimator(self, target: str):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("est", RidgeCV(alphas=self.params.get("alphas", ALPHAS))),
        ])


class ElasticNetSpreadModel(SpreadModel):
    name = "elasticnet"
    handles_nan = False

    def _make_estimator(self, target: str):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("est", ElasticNetCV(
                l1_ratio=self.params.get("l1_ratio", [0.1, 0.5, 0.9, 1.0]),
                cv=self.params.get("cv", 4),
                max_iter=self.params.get("max_iter", 5000),
                n_jobs=self.params.get("n_jobs", -1),
            )),
        ])


class HuberSpreadModel(SpreadModel):
    """Robust linear fit -- down-weights blowouts instead of chasing them."""
    name = "huber"
    handles_nan = False

    def _make_estimator(self, target: str):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("est", HuberRegressor(epsilon=self.params.get("epsilon", 1.35),
                                   alpha=self.params.get("alpha", 1.0),
                                   max_iter=self.params.get("max_iter", 500))),
        ])
