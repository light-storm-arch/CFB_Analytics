"""Getting from an empty repo to a usable model, without a terminal.

Everything the CLI does in `cfb synth` / `cfb fetch` / `cfb train` is available
here as plain functions with progress callbacks, so the Streamlit app can offer
the same operations as buttons.  Keeping the logic here rather than in the app
means it is testable without a Streamlit runtime, and means the CLI and the UI
cannot drift apart.

Hosting note: on Streamlit Community Cloud the filesystem is **ephemeral** -- it
resets whenever the app reboots or redeploys.  So:

* processed data is small enough to live in the git repo (a scheduled GitHub
  Action refreshes it), which is what makes the app usable on a cold boot;
* trained models are *not* committed -- a random forest pickle is tens of
  megabytes and pickles are fragile across library versions -- so the app
  retrains on boot and holds the result in Streamlit's resource cache.

Training is fast enough for that to be fine: roughly ten seconds for ridge,
about a minute for the full ensemble.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from cfb.config import get_config
from cfb.data.store import Store
from cfb.features.build import FeatureConfig
from cfb.pipeline import Predictor, PredictorConfig, artifacts_dir, load_features, \
    train_predictor

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]

#: Tables that must exist before anything can be trained.
REQUIRED_TABLES = ("games",)
#: Tables that improve the model but are not fatal if missing.
OPTIONAL_TABLES = ("lines", "talent", "returning", "sp_ratings", "pbp_states")


def _noop(_frac: float, _msg: str) -> None:
    pass


@dataclass
class DataStatus:
    has_data: bool
    is_synthetic: bool
    n_games: int
    n_completed: int
    seasons: tuple[int, int] | None
    tables: dict[str, int] = field(default_factory=dict)
    missing_optional: list[str] = field(default_factory=list)
    upcoming_games: int = 0

    @property
    def label(self) -> str:
        if not self.has_data:
            return "no data"
        kind = "synthetic" if self.is_synthetic else "real"
        span = f"{self.seasons[0]}-{self.seasons[1]}" if self.seasons else "?"
        return f"{kind} · {self.n_games:,} games · {span}"


def data_status(store: Store | None = None) -> DataStatus:
    store = store or Store()
    tables = {t: len(store.read(t)) for t in store.tables()}
    games = store.read("games")
    if games.empty:
        return DataStatus(False, False, 0, 0, None, tables,
                          [t for t in OPTIONAL_TABLES if t not in tables])
    completed = int(games["home_points"].notna().sum())
    return DataStatus(
        has_data=True,
        # The synthetic generator writes a truth table no real pull produces.
        is_synthetic="synth_truth" in tables,
        n_games=len(games),
        n_completed=completed,
        seasons=(int(games["season"].min()), int(games["season"].max())),
        tables=tables,
        missing_optional=[t for t in OPTIONAL_TABLES if t not in tables],
        upcoming_games=int(len(games) - completed),
    )


def model_status(name: str = "default") -> dict:
    d = artifacts_dir(name)
    cfg = d / "config.json"
    if not cfg.exists():
        return {"exists": False, "dir": str(d)}
    import json
    try:
        meta = json.loads(cfg.read_text())
    except json.JSONDecodeError:
        return {"exists": False, "dir": str(d)}
    return {"exists": True, "dir": str(d), "model": meta.get("model_name"),
            "features": len(meta.get("features_used", [])),
            "config": meta.get("cfg", {})}


def credentials_status() -> dict:
    cfg = get_config(refresh=True)
    pem = cfg.kalshi_private_key_path
    return {
        "cfbd": bool(cfg.cfbd_api_key),
        "kalshi_key": bool(cfg.kalshi_access_key),
        "kalshi_pem": bool(pem and Path(pem).exists()),
        "data_dir": str(cfg.data_dir),
        "artifacts_dir": str(cfg.artifacts_dir),
    }


# ---------------------------------------------------------------------- #
# Actions
# ---------------------------------------------------------------------- #
def build_sample_data(n_teams: int = 130, n_seasons: int = 8, with_pbp: bool = True,
                      seed: int = 7, progress: Progress | None = None) -> dict:
    """Generate the synthetic universe. No credentials required."""
    from cfb.data.synth import SynthConfig, build_synthetic_store

    progress = progress or _noop
    store = Store()
    if data_status(store).has_data:
        # Symmetric to fetch_real_data: never leave real games sitting beside
        # invented ones in the same store.
        progress(0.02, "clearing existing data ...")
        store.clear()
    progress(0.05, f"generating {n_teams} teams x {n_seasons} seasons ...")
    out = build_synthetic_store(
        SynthConfig(n_teams=n_teams, n_seasons=n_seasons, seed=seed),
        store=store, with_pbp=with_pbp)
    progress(1.0, "sample data ready")
    return out


def fetch_real_data(seasons: list[int], include_plays: bool = False,
                    progress: Progress | None = None,
                    replace_synthetic: bool = True) -> dict:
    """Pull real seasons from CollegeFootballData. Requires CFBD_API_KEY.

    If the store currently holds the synthetic league it is cleared first.  The
    two must not be mixed: ``Store.upsert`` concatenates, so the ratings model
    would end up fit over a league that is part real teams and part invented
    ones, quietly corrupting every downstream number.
    """
    from cfb.data.cfbd_client import fetch_seasons

    progress = progress or _noop
    if not get_config(refresh=True).cfbd_api_key:
        raise RuntimeError(
            "No CFBD API key configured. Get a free one at "
            "https://collegefootballdata.com/key and add it as CFBD_API_KEY "
            "(Streamlit: App settings -> Secrets)."
        )
    store = Store()
    if replace_synthetic and data_status(store).is_synthetic:
        progress(0.0, "clearing synthetic sample data ...")
        dropped = store.clear()
        log.info("cleared synthetic store before real fetch: %s", dropped)
    progress(0.0, f"fetching {len(seasons)} season(s) ...")
    counts = fetch_seasons(seasons, store=store, include_plays=include_plays,
                           progress=progress)
    progress(1.0, "fetch complete")
    return counts


#: Walk-forward will not predict a block until this many prior games exist.
#: Below it the ratings are too thin for the result to mean much.
PREFERRED_MIN_TRAIN_GAMES = 800
ABSOLUTE_MIN_TRAIN_GAMES = 200


def adaptive_min_train_games(n_completed: int) -> int:
    """How much history to demand before predicting a block.

    A fixed 800-game floor is right for a real 8-season pull but makes training
    impossible on a small store -- every block gets skipped and the walk-forward
    yields nothing.  Scaling with what is actually available keeps the app
    usable on a short history while still preferring the honest threshold once
    there is enough data to meet it.
    """
    if n_completed >= PREFERRED_MIN_TRAIN_GAMES * 3:
        return PREFERRED_MIN_TRAIN_GAMES
    return int(max(ABSOLUTE_MIN_TRAIN_GAMES, min(PREFERRED_MIN_TRAIN_GAMES,
                                                 n_completed * 0.35)))


def train_model(
    model_name: str = "ridge",
    members: tuple[str, ...] = ("ridge", "xgboost", "forest"),
    include_market: bool = False,
    include_roster: bool = False,
    use_roster_prior: bool = False,
    dist_method: str = "lattice",
    refit: str = "season",
    name: str = "default",
    save: bool = True,
    min_train_games: int | None = None,
    progress: Progress | None = None,
) -> tuple[Predictor, pd.DataFrame]:
    """Build features, walk-forward train, fit the distribution layer, save."""
    progress = progress or _noop
    fcfg = FeatureConfig(include_market=include_market,
                         include_roster=include_roster,
                         use_roster_prior=use_roster_prior)
    progress(0.05, "building features (walk-forward ratings) ...")
    feats = load_features(cfg=fcfg)
    n_completed = int(feats["margin"].notna().sum())
    if min_train_games is None:
        min_train_games = adaptive_min_train_games(n_completed)
    if min_train_games < PREFERRED_MIN_TRAIN_GAMES:
        log.warning("only %d completed games; walk-forward floor lowered to %d. "
                    "Accuracy numbers from this run will be pessimistic.",
                    n_completed, min_train_games)
    progress(0.35, f"{len(feats):,} games; walk-forward training '{model_name}' ...")
    pcfg = PredictorConfig(model_name=model_name, members=tuple(members),
                           include_market=include_market, dist_method=dist_method,
                           refit=refit, min_train_games=min_train_games,
                           feature_cfg=fcfg)
    predictor, oos = train_predictor(feats, pcfg, progress=False)
    progress(0.9, "fitting scale model and key numbers ...")
    if save:
        try:
            predictor.save(artifacts_dir(name))
        except OSError as exc:
            # A read-only or full filesystem should not lose the trained model;
            # the caller still has it in memory.
            log.warning("could not save predictor: %s", exc)
    progress(1.0, "model ready")
    return predictor, oos


def roster_ablation(models: tuple[str, ...] = ("ridge",),
                    progress: Progress | None = None) -> pd.DataFrame:
    """Does the portal-era work help *this* data? Four-way walk-forward.

    Compares baseline against the roster feature block, the roster-aware prior,
    and both.  On the bundled synthetic league both come out neutral-to-worse,
    which is why they ship off; this is how you check whether real data says
    something different.
    """
    from cfb.data.store import Store
    from cfb.evaluation.backtest import walk_forward_predictions
    from cfb.features.build import build_features, feature_columns

    progress = progress or _noop
    store = Store()
    games = store.read("games")
    if games.empty:
        raise RuntimeError("no games in the store")
    kw = dict(lines=store.read("lines"), talent=store.read("talent"),
              returning=store.read("returning"), sp_ratings=store.read("sp_ratings"),
              portal=store.read("portal"), player_ppa=store.read("player_ppa"))
    variants = {
        "baseline": FeatureConfig(include_roster=False, use_roster_prior=False),
        "roster features": FeatureConfig(include_roster=True, use_roster_prior=False),
        "roster prior": FeatureConfig(include_roster=False, use_roster_prior=True),
        "both": FeatureConfig(include_roster=True, use_roster_prior=True),
    }
    rows, step, total = [], 0, len(models) * len(variants)
    for model in models:
        for label, cfg in variants.items():
            step += 1
            progress(step / total, f"{model} / {label} ...")
            feats = build_features(games, cfg=cfg, **kw)
            oos = walk_forward_predictions(feats, feature_columns(cfg),
                                           model_name=model, progress=False)
            if oos.empty:
                continue
            err = (oos["pred_margin"] - oos["margin"]).abs()
            rows.append({"model": model, "variant": label, "games": len(oos),
                         "wk1_4_mae": float(err[oos["week"] <= 4].mean()),
                         "wk5plus_mae": float(err[oos["week"] >= 5].mean()),
                         "mae": float(err.mean())})
    out = pd.DataFrame(rows)
    if not out.empty:
        base = out[out.variant == "baseline"].set_index("model")["mae"]
        out["vs_baseline"] = [r.mae - base.get(r.model, float("nan"))
                              for r in out.itertuples()]
    progress(1.0, "done")
    return out


def load_predictor(name: str = "default") -> Predictor | None:
    """Load a saved predictor, or None if there is not a usable one on disk."""
    try:
        return Predictor.load(artifacts_dir(name))
    except (FileNotFoundError, OSError, KeyError, ValueError, ModuleNotFoundError) as exc:
        log.info("no usable saved predictor at %s (%s)", artifacts_dir(name), exc)
        return None


def load_or_train(name: str = "default", model_name: str = "ridge",
                  progress: Progress | None = None, **kwargs) -> Predictor:
    """Return a saved predictor if there is one, otherwise train a fresh one."""
    existing = load_predictor(name)
    if existing is not None:
        return existing
    predictor, _ = train_model(model_name=model_name, name=name,
                               progress=progress, **kwargs)
    return predictor


def first_run_plan(status: DataStatus, creds: dict) -> list[str]:
    """Human-readable next steps for whatever state the app is in."""
    steps: list[str] = []
    if not status.has_data:
        if creds["cfbd"]:
            steps.append("Fetch real seasons from CollegeFootballData.")
        else:
            steps.append(
                "Add a CFBD API key to load real games — or press "
                "**Load sample data** to try everything on a synthetic league.")
    elif status.is_synthetic and creds["cfbd"]:
        steps.append("You have a key now: fetch real seasons to replace the sample data.")
    if status.has_data and status.n_completed < 800:
        steps.append(
            f"Only {status.n_completed:,} completed games. Walk-forward training "
            "needs a few thousand — fetch more seasons.")
    if status.has_data and status.upcoming_games == 0:
        steps.append(
            "Every game in the store is finished, so there is no slate to price. "
            "Fetch the current season to get upcoming games.")
    return steps
