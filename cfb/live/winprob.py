"""Trained in-game model, learned from play-by-play states.

Where the Brownian bridge *assumes* a shape, this model learns one.  Given a
snapshot -- score, clock, down, distance, field position, who has the ball, and
the pregame line -- it predicts:

* ``p_home_win``  from a gradient-boosted classifier, and
* ``final_margin`` (plus its residual scale) from a regressor,

which together give a live margin distribution that respects things the
analytic model only approximates: fourth-down leverage, the value of field
position late, the asymmetry of trailing-team behaviour in the last five
minutes.

It needs play-by-play data to train, which is a heavy pull (one CFBD request
per week per season).  The analytic model works with none, so the two are
complementary rather than redundant -- and comparing them is a good check that
neither has drifted.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.constants import GAME_SECONDS, QUARTER_SECONDS
from cfb.distribution.margin import KeyNumberProfile, MarginDistribution

log = logging.getLogger(__name__)

STATE_FEATURES = [
    "score_diff", "seconds_remaining", "sqrt_seconds", "period",
    "down", "distance", "yards_to_goal", "possession_home",
    "pregame_mu", "pregame_total",
    "diff_per_sqrt_time", "expected_points_poss", "is_two_minute",
]


def build_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical live-state feature frame (works for training and inference)."""
    from cfb.live.diffusion import expected_points

    out = pd.DataFrame(index=df.index)
    hs = pd.to_numeric(df.get("home_score"), errors="coerce")
    as_ = pd.to_numeric(df.get("away_score"), errors="coerce")
    out["score_diff"] = hs - as_
    secs = pd.to_numeric(df.get("seconds_remaining"), errors="coerce").clip(0, GAME_SECONDS)
    out["seconds_remaining"] = secs
    out["sqrt_seconds"] = np.sqrt(secs.clip(lower=0))
    out["period"] = pd.to_numeric(df.get("period"), errors="coerce").fillna(1)
    out["down"] = pd.to_numeric(df.get("down"), errors="coerce").fillna(1).clip(1, 4)
    out["distance"] = pd.to_numeric(df.get("distance"), errors="coerce").fillna(10).clip(1, 40)
    ytg = pd.to_numeric(df.get("yards_to_goal"), errors="coerce").fillna(75).clip(1, 99)
    out["yards_to_goal"] = ytg
    poss = df.get("possession_home")
    out["possession_home"] = (pd.Series(poss, index=df.index).astype("float")
                              if poss is not None else np.nan)
    out["pregame_mu"] = pd.to_numeric(df.get("pregame_mu"), errors="coerce").fillna(0.0)
    out["pregame_total"] = pd.to_numeric(df.get("pregame_total"), errors="coerce").fillna(54.0)
    out["diff_per_sqrt_time"] = out["score_diff"] / np.sqrt(secs.clip(lower=1))
    ep = np.array([expected_points(y, d, dist) for y, d, dist
                   in zip(out["yards_to_goal"], out["down"], out["distance"])])
    sign = out["possession_home"].fillna(0.5) * 2 - 1
    out["expected_points_poss"] = ep * sign
    out["is_two_minute"] = ((secs <= 120) | ((secs > 1800) & (secs <= 1920))).astype(float)
    return out[STATE_FEATURES]


def states_from_plays(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Convert CFBD ``/plays`` rows into the canonical live-state schema.

    CFBD reports scores from the *offence's* perspective, so the first job is to
    put everything back into home-margin space.
    """
    if plays.empty:
        return pd.DataFrame()
    p = plays.copy()
    is_home_off = p["offense"] == p["home"]
    p["home_score"] = np.where(is_home_off, p["offense_score"], p["defense_score"])
    p["away_score"] = np.where(is_home_off, p["defense_score"], p["offense_score"])
    p["possession_home"] = is_home_off
    period = pd.to_numeric(p["period"], errors="coerce").fillna(1).clip(1, 4)
    clock = (pd.to_numeric(p["clock_minutes"], errors="coerce").fillna(0) * 60
             + pd.to_numeric(p["clock_seconds"], errors="coerce").fillna(0))
    p["seconds_remaining"] = ((4 - period) * QUARTER_SECONDS + clock).clip(0, GAME_SECONDS)

    finals = games[["game_id", "home_points", "away_points", "home_team", "away_team"]].copy()
    finals["final_margin"] = finals["home_points"] - finals["away_points"]
    finals["home_win"] = (finals["final_margin"] > 0).astype(int)
    out = p.merge(finals[["game_id", "final_margin", "home_win", "home_team", "away_team"]],
                  on="game_id", how="inner")
    keep = ["play_id", "game_id", "season", "week", "home_team", "away_team",
            "home_score", "away_score", "period", "seconds_remaining",
            "possession_home", "down", "distance", "yards_to_goal",
            "final_margin", "home_win"]
    return out[[c for c in keep if c in out.columns]].dropna(subset=["final_margin"])


def attach_pregame(states: pd.DataFrame, pregame: pd.DataFrame) -> pd.DataFrame:
    """Join each snapshot to its game's pregame expectation.

    ``pregame`` needs ``game_id`` plus ``pred_margin``/``pred_total`` (model) or
    ``market_margin``/``market_total`` (closing line).
    """
    src = pregame.copy()
    mu_col = "pred_margin" if "pred_margin" in src else "market_margin"
    tot_col = "pred_total" if "pred_total" in src else "market_total"
    src = src[["game_id", mu_col, tot_col]].rename(
        columns={mu_col: "pregame_mu", tot_col: "pregame_total"})
    src = src.drop_duplicates("game_id")
    return states.merge(src, on="game_id", how="left")


@dataclass
class WinProbReport:
    n_train: int
    n_valid: int
    log_loss: float
    brier: float
    margin_mae: float
    resid_sd: float


class LiveWinProbModel:
    def __init__(self, key_numbers: KeyNumberProfile | None = None, **params):
        self.params = params
        self.key_numbers = key_numbers
        self.clf = None
        self.reg = None
        self.sd_model = None
        self.report: WinProbReport | None = None

    # -- estimators --------------------------------------------------------
    def _classifier(self):
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=self.params.get("n_estimators", 500),
                learning_rate=self.params.get("learning_rate", 0.05),
                max_depth=self.params.get("max_depth", 6),
                min_child_weight=self.params.get("min_child_weight", 40),
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                eval_metric="logloss", tree_method="hist",
                n_jobs=self.params.get("n_jobs", -1), random_state=7)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05,
                                                  max_iter=400, random_state=7)

    def _regressor(self):
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=self.params.get("n_estimators", 500),
                learning_rate=self.params.get("learning_rate", 0.05),
                max_depth=self.params.get("max_depth", 6),
                min_child_weight=self.params.get("min_child_weight", 40),
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                tree_method="hist", n_jobs=self.params.get("n_jobs", -1),
                random_state=7)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05,
                                                 max_iter=400, random_state=7)

    # -- fit ---------------------------------------------------------------
    def fit(self, states: pd.DataFrame, valid_seasons: list[int] | None = None
            ) -> "LiveWinProbModel":
        d = states.dropna(subset=["home_win", "final_margin"]).copy()
        if d.empty:
            raise ValueError("no labelled play-by-play states to train on")
        if valid_seasons:
            train = d[~d["season"].isin(valid_seasons)]
            valid = d[d["season"].isin(valid_seasons)]
        else:
            # Hold out the most recent season so the report is honest.
            last = d["season"].max()
            train, valid = d[d["season"] < last], d[d["season"] == last]
        if train.empty:
            train, valid = d, d.iloc[:0]

        Xtr = build_state_features(train)
        self.clf = self._classifier()
        self.clf.fit(Xtr, train["home_win"].to_numpy(int))
        self.reg = self._regressor()
        self.reg.fit(Xtr, train["final_margin"].to_numpy(float))

        # Residual scale as a function of the same state -- late-game residuals
        # are far tighter than first-quarter ones.
        resid = train["final_margin"].to_numpy(float) - self.reg.predict(Xtr)
        self.sd_model = self._regressor()
        self.sd_model.fit(Xtr, np.log(np.abs(resid) + 1.0))
        raw = np.exp(self.sd_model.predict(Xtr))
        self._sd_calibration = float(np.sqrt(np.mean((resid / np.clip(raw, 1e-3, None)) ** 2)))

        if not valid.empty:
            Xv = build_state_features(valid)
            p = np.clip(self._proba(Xv), 1e-6, 1 - 1e-6)
            y = valid["home_win"].to_numpy(int)
            pm = self.reg.predict(Xv)
            self.report = WinProbReport(
                n_train=len(train), n_valid=len(valid),
                log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                brier=float(np.mean((p - y) ** 2)),
                margin_mae=float(np.mean(np.abs(pm - valid["final_margin"]))),
                resid_sd=float(np.std(valid["final_margin"] - pm)),
            )
            log.info("live model: logloss=%.4f brier=%.4f marginMAE=%.2f",
                     self.report.log_loss, self.report.brier, self.report.margin_mae)
        return self

    def _proba(self, X) -> np.ndarray:
        p = self.clf.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p

    # -- predict -----------------------------------------------------------
    def predict(self, states: pd.DataFrame) -> pd.DataFrame:
        X = build_state_features(states)
        sd = np.clip(np.exp(self.sd_model.predict(X)) * self._sd_calibration, 0.8, 30.0)
        return pd.DataFrame({
            "p_home_win": self._proba(X),
            "pred_final_margin": self.reg.predict(X),
            "pred_sd": sd,
        }, index=states.index)

    def distribution(self, state: pd.DataFrame | dict) -> MarginDistribution:
        df = pd.DataFrame([state]) if isinstance(state, dict) else state
        row = self.predict(df).iloc[0]
        dist = MarginDistribution.from_continuous(
            float(row["pred_final_margin"]), float(row["pred_sd"]), dist="t", df=6.0,
            key_numbers=self.key_numbers,
            meta={"scaffold": "live_trained", "live": True})
        return dist

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        if self.report:
            path.with_suffix(".json").write_text(json.dumps(self.report.__dict__, indent=2))
        return path

    @staticmethod
    def load(path: str | Path) -> "LiveWinProbModel":
        import joblib
        return joblib.load(Path(path))
