"""End-to-end pregame pipeline: data -> features -> model -> distribution.

``Predictor`` is the object you actually trade off.  It bundles four fitted
pieces that must stay in sync:

1. a **point model** for expected margin and total;
2. a **sigma model** for how uncertain that point estimate is, fit on
   out-of-sample residuals;
3. a **key-number profile** describing the scoring lattice;
4. the **standardised residual pool**, so the empirical-shape route is available
   without refitting anything.

Training deliberately runs the walk-forward backtest *first* and fits (2)-(4) on
its out-of-sample residuals.  Fitting the scale model on in-sample residuals is
the classic way to end up with a distribution that is 10-15% too narrow, which
is precisely the error that turns a small edge into a losing one.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.config import CONFIG
from cfb.data.store import Store
from cfb.distribution.margin import KeyNumberProfile, MarginDistribution
from cfb.distribution.sigma import SigmaModel
from cfb.distribution.simulate import DriveSimulator, SimConfig
from cfb.evaluation.backtest import walk_forward_predictions
from cfb.features.build import FeatureConfig, build_features, feature_columns
from cfb.models import build_model
from cfb.models.base import SpreadModel

log = logging.getLogger(__name__)

DIST_METHODS = ("lattice", "kde", "mc", "blend", "plain")


@dataclass
class PredictorConfig:
    model_name: str = "ensemble"
    members: tuple[str, ...] = ("ridge", "xgboost", "forest")
    include_market: bool = False
    dist_method: str = "lattice"
    t_df: float | None = None          # None -> use the df fitted on residuals
    blend_weight: float = 0.5          # weight on the MC side when blending
    refit: str = "season"
    min_train_games: int = 800
    half_life_days: float | None = 900.0
    mc_sims: int = 30_000
    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["members"] = list(self.members)
        return d


class Predictor:
    def __init__(self, model: SpreadModel, sigma_model: SigmaModel,
                 key_numbers: KeyNumberProfile, feature_cols: list[str],
                 cfg: PredictorConfig, residual_pool: np.ndarray | None = None,
                 oos: pd.DataFrame | None = None):
        self.model = model
        self.sigma_model = sigma_model
        self.key_numbers = key_numbers
        self.feature_cols = list(feature_cols)
        self.cfg = cfg
        self.residual_pool = np.asarray(residual_pool if residual_pool is not None else [])
        self.oos = oos
        self._sim = DriveSimulator(SimConfig(n_sims=cfg.mc_sims))

    # -- point predictions -------------------------------------------------
    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        pred = self.model.predict(features)
        out = features[[c for c in ("game_id", "season", "week", "start_date",
                                    "home_team", "away_team", "neutral_site",
                                    "market_margin", "market_total",
                                    "margin", "total")
                        if c in features.columns]].copy()
        out["pred_margin"] = pred["pred_margin"].to_numpy()
        out["pred_total"] = pred["pred_total"].to_numpy()
        for c in pred.columns:
            if c.startswith("pred_margin__") or c.startswith("pred_total__"):
                out[c] = pred[c].to_numpy()
        out["sigma"] = self.sigma_model.predict(out)
        out["fair_spread_home"] = -out["pred_margin"].round(1)
        if "market_margin" in out.columns:
            out["edge_vs_market"] = out["pred_margin"] - out["market_margin"]
        return out

    # -- distributions -----------------------------------------------------
    def distribution(self, mu: float, sigma: float, total: float | None = None,
                     method: str | None = None) -> MarginDistribution:
        method = (method or self.cfg.dist_method).lower()
        if method not in DIST_METHODS:
            raise ValueError(f"dist_method must be one of {DIST_METHODS}")
        df = self.cfg.t_df or getattr(self.sigma_model, "fitted_df", 7.0)
        total = float(total) if total is not None and np.isfinite(total) else 54.0

        if method == "plain":
            return MarginDistribution.from_continuous(mu, sigma, dist="normal",
                                                      key_numbers=None)
        if method == "lattice":
            return MarginDistribution.from_continuous(
                mu, sigma, dist="t", df=df, key_numbers=self.key_numbers)
        if method == "kde":
            return MarginDistribution.from_residual_kde(
                mu, self.residual_pool * sigma if self.residual_pool.size else [],
                sigma=sigma, key_numbers=self.key_numbers)
        if method == "mc":
            return self._sim.distribution(mu, total, sigma=sigma)
        # blend: parametric lattice + drive-level Monte Carlo
        a = MarginDistribution.from_continuous(mu, sigma, dist="t", df=df,
                                               key_numbers=self.key_numbers)
        b = self._sim.distribution(mu, total, sigma=sigma)
        return a.blend(b, self.cfg.blend_weight)

    def distributions(self, pred: pd.DataFrame, method: str | None = None
                      ) -> list[MarginDistribution]:
        return [
            self.distribution(float(r.pred_margin), float(r.sigma),
                              float(getattr(r, "pred_total", np.nan)), method)
            for r in pred.itertuples()
        ]

    def predict_slate(self, features: pd.DataFrame, method: str | None = None
                      ) -> tuple[pd.DataFrame, list[MarginDistribution]]:
        pred = self.predict_frame(features)
        dists = self.distributions(pred, method)
        pred = pred.copy()
        pred["p_home_win"] = [d.p_home_win() for d in dists]
        pred["dist_sd"] = [d.sd() for d in dists]
        pred["modal_margin"] = [d.mode() for d in dists]
        pred["p_home_by_1_3"] = [d.p_between(1, 3) for d in dists]
        pred["p_home_by_4_7"] = [d.p_between(4, 7) for d in dists]
        pred["p_home_by_8_plus"] = [d.p_at_least(8) for d in dists]
        return pred, dists

    # -- persistence -------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        import joblib
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, d / "model.joblib")
        joblib.dump(self.sigma_model, d / "sigma.joblib")
        self.key_numbers.save(d / "key_numbers.json")
        np.save(d / "residual_pool.npy", self.residual_pool)
        (d / "config.json").write_text(json.dumps({
            "cfg": self.cfg.to_dict(),
            "feature_cols": self.feature_cols,
            "model_name": self.model.name,
            "features_used": self.model.features_used,
        }, indent=2, default=str))
        if self.oos is not None:
            self.oos.to_parquet(d / "oos_predictions.parquet", index=False)
        log.info("saved predictor -> %s", d)
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "Predictor":
        import joblib
        d = Path(directory)
        meta = json.loads((d / "config.json").read_text())
        raw = meta["cfg"]
        fc = raw.pop("feature_cfg", {}) or {}
        cfg = PredictorConfig(**{**raw, "members": tuple(raw.get("members", ())),
                                 "feature_cfg": FeatureConfig(**fc)})
        oos_path = d / "oos_predictions.parquet"
        return cls(
            model=joblib.load(d / "model.joblib"),
            sigma_model=joblib.load(d / "sigma.joblib"),
            key_numbers=KeyNumberProfile.load(d / "key_numbers.json"),
            feature_cols=meta["feature_cols"],
            cfg=cfg,
            residual_pool=np.load(d / "residual_pool.npy"),
            oos=pd.read_parquet(oos_path) if oos_path.exists() else None,
        )


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def train_predictor(features: pd.DataFrame, cfg: PredictorConfig | None = None,
                    progress: bool = True) -> tuple[Predictor, pd.DataFrame]:
    cfg = cfg or PredictorConfig()
    cols = feature_columns(cfg.feature_cfg)
    params = {"half_life_days": cfg.half_life_days}
    members = list(cfg.members) if cfg.model_name == "ensemble" else None

    log.info("walk-forward (%s, refit=%s) to generate out-of-sample residuals",
             cfg.model_name, cfg.refit)
    oos = walk_forward_predictions(
        features, cols, model_name=cfg.model_name, refit=cfg.refit,
        min_train_games=cfg.min_train_games,
        model_params={**params, **({"members": members} if members else {})},
        progress=progress,
    )
    if oos.empty:
        raise ValueError(
            "walk-forward produced no out-of-sample games. You likely have too "
            "little history -- lower min_train_games or fetch more seasons."
        )

    sigma_model = SigmaModel().fit(oos)
    sigmas = sigma_model.predict(oos)
    key_numbers = KeyNumberProfile.fit(
        oos["margin"].to_numpy(float), mus=oos["pred_margin"].to_numpy(float),
        sigmas=sigmas, df=sigma_model.fitted_df)
    resid_pool = sigma_model.standardized_residuals(oos)

    log.info("fitting final %s on all %d completed games", cfg.model_name,
             int(features["margin"].notna().sum()))
    model = build_model(cfg.model_name, cols, members=members, **params)
    model.fit(features)

    pred = Predictor(model, sigma_model, key_numbers, cols, cfg, resid_pool, oos)
    return pred, oos


# ---------------------------------------------------------------------- #
# Data assembly
# ---------------------------------------------------------------------- #
def load_features(store: Store | None = None, cfg: FeatureConfig | None = None,
                  seasons: list[int] | None = None,
                  progress: bool = False) -> pd.DataFrame:
    """Read the local store and build the feature table."""
    store = store or Store()
    games = store.read("games")
    if games.empty:
        raise FileNotFoundError(
            "No games in the local store. Run `cfb fetch --seasons ...` "
            "(needs CFBD_API_KEY) or `cfb synth` to generate a test universe."
        )
    if seasons:
        games = games[games["season"].isin(seasons)]
    return build_features(
        games,
        lines=store.read("lines"),
        talent=store.read("talent"),
        returning=store.read("returning"),
        sp_ratings=store.read("sp_ratings"),
        cfg=cfg or FeatureConfig(),
        progress=progress,
    )


def artifacts_dir(name: str = "default") -> Path:
    return CONFIG.artifacts_dir / name
