"""Shared fixtures.

The data directory is redirected to a throwaway temp dir **before any cfb module
is imported**, so the suite never reads or writes a developer's real store and
behaves identically on a clean CI runner.
"""
import os
import tempfile
import warnings
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="cfb-tests-"))
os.environ["CFB_DATA_DIR"] = str(_TMP / "data")
os.environ["CFB_ARTIFACTS_DIR"] = str(_TMP / "artifacts")

import pytest  # noqa: E402

from cfb.data.synth import SynthConfig, SyntheticLeague  # noqa: E402

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def league():
    return SyntheticLeague(SynthConfig(n_teams=60, n_seasons=4, seed=3,
                                       unplayed_last_week=False))


@pytest.fixture(scope="session")
def universe(league):
    return league.generate()


@pytest.fixture(scope="session")
def games(universe):
    return universe["games"]


@pytest.fixture(scope="session")
def features(universe):
    from cfb.features.build import build_features
    return build_features(universe["games"], universe["lines"],
                          talent=universe["talent"],
                          returning=universe["returning"],
                          sp_ratings=universe["sp_ratings"])


@pytest.fixture(scope="session")
def key_numbers(games):
    from cfb.distribution.margin import KeyNumberProfile
    resid = games["margin"] - games["true_mu"]
    return KeyNumberProfile.fit(games["margin"], mus=games["true_mu"],
                                sigmas=float(resid.std()))


@pytest.fixture(scope="session")
def populated_store():
    """A real on-disk store, small but with an unplayed final week."""
    from cfb.data.store import Store
    from cfb.data.synth import build_synthetic_store

    build_synthetic_store(SynthConfig(n_teams=60, n_seasons=5, seed=5),
                          with_pbp=False, pbp_games=0)
    return Store()


@pytest.fixture(scope="session")
def trained_artifact(populated_store):
    """A saved ridge predictor on disk, as the app expects to find."""
    from cfb import bootstrap
    predictor, _ = bootstrap.train_model(model_name="ridge", name="default")
    return predictor
