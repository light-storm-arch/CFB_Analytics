import warnings

import pytest

from cfb.data.synth import SynthConfig, SyntheticLeague

warnings.filterwarnings("ignore")


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
