"""Walk-forward backtesting.

Every number this project reports about accuracy comes from here, and the rule
is absolute: a prediction for season/week *t* may only use games completed
before *t*.  There is no k-fold cross-validation anywhere in this codebase --
random folds leak future information through opponent-adjusted ratings and
would flatter every model by a point or more of MAE.

``walk_forward_predictions`` is also the engine that produces the out-of-sample
residuals the sigma model and the key-number profile are fit on, so getting it
right matters twice over.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cfb.models import build_model

log = logging.getLogger(__name__)

VIG_PRICE = 110.0  # standard -110 sportsbook price, used for ATS ROI


def walk_forward_predictions(
    features: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "ridge",
    refit: str = "season",
    min_train_games: int = 800,
    start_season: int | None = None,
    model_params: dict | None = None,
    keep_members: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Produce genuinely out-of-sample predictions across the history.

    ``refit='season'`` retrains once per season (fast, and what you would
    realistically do); ``refit='week'`` retrains before every week (slower,
    slightly more accurate, and the honest simulation of in-season operation).
    """
    df = features.dropna(subset=["margin"]).copy()
    df = df.sort_values(["season", "week", "start_date", "game_id"],
                        kind="stable").reset_index(drop=True)
    model_params = dict(model_params or {})

    if refit == "week":
        blocks = df.groupby(["season", "week"], sort=True)
    elif refit == "season":
        blocks = df.groupby(["season"], sort=True)
    else:
        raise ValueError("refit must be 'season' or 'week'")

    out_frames: list[pd.DataFrame] = []
    for key, block in blocks:
        season = key[0] if isinstance(key, tuple) else key
        if start_season is not None and season < start_season:
            continue
        cutoff = block["start_date"].min()
        train = df[df["start_date"] < cutoff]
        if len(train) < min_train_games:
            continue
        model = build_model(model_name, feature_cols, **model_params)
        try:
            model.fit(train)
        except ValueError as exc:
            log.warning("skip block %s: %s", key, exc)
            continue
        pred = model.predict(block)
        rec = block[["game_id", "season", "week", "start_date", "home_team",
                     "away_team", "margin", "total"]].copy()
        for c in ("market_margin", "market_total", "neutral_site",
                  "rat_games_home", "rat_games_away"):
            if c in block.columns:
                rec[c] = block[c].to_numpy()
        rec["pred_margin"] = pred["pred_margin"].to_numpy()
        rec["pred_total"] = pred["pred_total"].to_numpy()
        if keep_members:
            for c in pred.columns:
                if c.startswith("pred_margin__") or c.startswith("pred_total__"):
                    rec[c] = pred[c].to_numpy()
        rec["model"] = model_name
        rec["n_train"] = len(train)
        out_frames.append(rec)
        if progress:
            log.info("walk-forward %s: train=%d test=%d", key, len(train), len(block))

    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)


# ---------------------------------------------------------------------- #
# Reporting
# ---------------------------------------------------------------------- #
@dataclass
class BacktestReport:
    n: int
    margin_mae: float
    margin_rmse: float
    margin_bias: float
    total_mae: float
    su_accuracy: float
    market_mae: float | None = None
    ats_record: dict = field(default_factory=dict)
    by_season: pd.DataFrame | None = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "by_season"}
        return d

    def __str__(self) -> str:
        lines = [
            f"games              {self.n}",
            f"margin MAE         {self.margin_mae:.3f}",
            f"margin RMSE        {self.margin_rmse:.3f}",
            f"margin bias        {self.margin_bias:+.3f}",
            f"total MAE          {self.total_mae:.3f}",
            f"straight-up acc    {self.su_accuracy:.4f}",
        ]
        if self.market_mae is not None:
            lines.append(f"market MAE         {self.market_mae:.3f}  "
                         f"(model - market = {self.margin_mae - self.market_mae:+.3f})")
        if self.ats_record:
            r = self.ats_record
            lines.append(
                f"ATS vs close       {r['wins']}-{r['losses']}-{r['pushes']} "
                f"({r['win_pct']:.4f})  ROI {r['roi']:+.4f}  "
                f"(break-even {r['breakeven']:.4f})")
        return "\n".join(lines)


def backtest_report(pred: pd.DataFrame, ats_threshold: float = 0.0) -> BacktestReport:
    """Summarise a walk-forward prediction frame.

    ``ats_threshold`` filters to games where the model disagrees with the
    closing line by at least that many points -- the realistic way to bet,
    since a 0.3-point disagreement is noise.
    """
    d = pred.dropna(subset=["margin", "pred_margin"]).copy()
    err = d["pred_margin"] - d["margin"]
    rep = BacktestReport(
        n=len(d),
        margin_mae=float(err.abs().mean()),
        margin_rmse=float(np.sqrt((err ** 2).mean())),
        margin_bias=float(err.mean()),
        total_mae=float((d["pred_total"] - d["total"]).abs().mean())
        if "pred_total" in d and d["pred_total"].notna().any() else float("nan"),
        su_accuracy=float(((d["pred_margin"] > 0) == (d["margin"] > 0)).mean()),
    )
    if "market_margin" in d.columns and d["market_margin"].notna().any():
        m = d.dropna(subset=["market_margin"])
        rep.market_mae = float((m["market_margin"] - m["margin"]).abs().mean())
        rep.ats_record = ats_record(m, ats_threshold)
    rep.by_season = season_table(d)
    return rep


def ats_record(d: pd.DataFrame, threshold: float = 0.0) -> dict:
    """Bet the side the model likes vs the closing line; grade at -110."""
    m = d.dropna(subset=["market_margin"]).copy()
    m["edge"] = m["pred_margin"] - m["market_margin"]
    bets = m[m["edge"].abs() >= threshold]
    if bets.empty:
        return {"wins": 0, "losses": 0, "pushes": 0, "win_pct": float("nan"),
                "roi": float("nan"), "breakeven": VIG_PRICE / (100 + VIG_PRICE),
                "n_bets": 0, "threshold": threshold}
    # Backing home when the model is higher than the market, away when lower.
    back_home = bets["edge"] > 0
    diff = bets["margin"] - bets["market_margin"]
    result = np.where(back_home, diff, -diff)
    wins = int((result > 0).sum())
    losses = int((result < 0).sum())
    pushes = int((result == 0).sum())
    staked = wins + losses
    profit = wins * (100.0 / VIG_PRICE) - losses * 1.0
    return {
        "wins": wins, "losses": losses, "pushes": pushes,
        "n_bets": len(bets), "threshold": float(threshold),
        "win_pct": wins / staked if staked else float("nan"),
        "roi": profit / staked if staked else float("nan"),
        "breakeven": VIG_PRICE / (100 + VIG_PRICE),
    }


def ats_by_threshold(d: pd.DataFrame,
                     thresholds=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        r = ats_record(d, t)
        rows.append({"threshold": t, "n_bets": r["n_bets"], "win_pct": r["win_pct"],
                     "roi": r["roi"]})
    return pd.DataFrame(rows)


def season_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.copy()
    g["abs_err"] = (g["pred_margin"] - g["margin"]).abs()
    g["su_hit"] = ((g["pred_margin"] > 0) == (g["margin"] > 0)).astype(float)
    agg = g.groupby("season").agg(
        games=("game_id", "count"),
        margin_mae=("abs_err", "mean"),
        su_acc=("su_hit", "mean"),
    ).reset_index()
    if "market_margin" in g.columns:
        g["mkt_err"] = (g["market_margin"] - g["margin"]).abs()
        agg = agg.merge(g.groupby("season")["mkt_err"].mean().rename("market_mae")
                        .reset_index(), on="season", how="left")
    return agg


def compare_models(features: pd.DataFrame, feature_cols: list[str],
                   model_names: list[str], **kwargs) -> pd.DataFrame:
    """Run the same walk-forward for several models and line the results up."""
    rows = []
    for name in model_names:
        pred = walk_forward_predictions(features, feature_cols, model_name=name, **kwargs)
        if pred.empty:
            continue
        rep = backtest_report(pred)
        row = {"model": name, "n": rep.n, "margin_mae": rep.margin_mae,
               "margin_rmse": rep.margin_rmse, "su_acc": rep.su_accuracy,
               "total_mae": rep.total_mae}
        if rep.ats_record:
            row["ats_win_pct"] = rep.ats_record["win_pct"]
            row["ats_roi"] = rep.ats_record["roi"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("margin_mae").reset_index(drop=True)
