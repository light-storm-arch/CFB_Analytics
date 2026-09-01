import numpy as np
import pytest

from cfb.distribution.margin import MarginDistribution
from cfb.distribution.simulate import DriveSimulator, SimConfig, points_per_drive, \
    quality_for_ppd


def test_pmf_normalised_and_no_ties(key_numbers):
    d = MarginDistribution.from_continuous(-3.5, 16.0, key_numbers=key_numbers)
    assert d.pmf.sum() == pytest.approx(1.0)
    assert d.p_tie() == 0.0
    assert d.p_home_win() + d.p_away_win() == pytest.approx(1.0)


@pytest.mark.parametrize("mu,sigma", [(-3.5, 16.0), (0.0, 12.0), (21.0, 18.5)])
def test_moment_matching(key_numbers, mu, sigma):
    d = MarginDistribution.from_continuous(mu, sigma, key_numbers=key_numbers)
    assert d.mean() == pytest.approx(mu, abs=0.05)
    assert d.sd() == pytest.approx(sigma, abs=0.05)


def test_key_numbers_lift_three_and_seven(key_numbers):
    """The whole point: 3 and 7 must be far likelier than their neighbours."""
    d = MarginDistribution.from_continuous(0.0, 16.0, key_numbers=key_numbers)
    flat = MarginDistribution.from_continuous(0.0, 16.0, dist="normal")
    assert d.p_exact(3) > 1.4 * d.p_exact(5)
    assert d.p_exact(7) > 1.4 * d.p_exact(5)
    assert d.p_exact(3) > 1.3 * flat.p_exact(3)
    assert flat.p_exact(3) == pytest.approx(flat.p_exact(5), rel=0.15)


def test_cover_probability_sign_convention(key_numbers):
    """spread=-7 means home lays 7: home covers only by winning by 8+."""
    d = MarginDistribution.from_continuous(10.0, 15.0, key_numbers=key_numbers)
    cov = d.cover_probability(-7.0, side="home")
    assert cov["win"] == pytest.approx(d.p_greater(7.0))
    assert cov["push"] == pytest.approx(d.p_exact(7))
    assert cov["win"] + cov["push"] + cov["lose"] == pytest.approx(1.0)
    away = d.cover_probability(-7.0, side="away")
    assert away["win"] == pytest.approx(cov["lose"])


def test_strike_mapping(key_numbers):
    d = MarginDistribution.from_continuous(4.0, 15.0, key_numbers=key_numbers)
    assert d.probability_for_strike("greater", floor_strike=7) == pytest.approx(d.p_greater(7))
    assert d.probability_for_strike("greater_or_equal", floor_strike=7) == \
        pytest.approx(d.p_at_least(7))
    assert d.probability_for_strike("less", cap_strike=0) == pytest.approx(d.p_less(0))
    assert d.probability_for_strike("between", floor_strike=4, cap_strike=7) == \
        pytest.approx(d.p_between(4, 7))
    assert d.probability_for_strike("nonsense") is None
    # greater vs greater_or_equal must actually differ at an integer strike
    assert d.p_at_least(7) > d.p_greater(7)


def test_flip_is_a_mirror(key_numbers):
    d = MarginDistribution.from_continuous(6.0, 15.0, key_numbers=key_numbers)
    f = d.flip()
    assert f.mean() == pytest.approx(-d.mean(), abs=1e-6)
    assert f.p_home_win() == pytest.approx(d.p_away_win())
    assert f.p_exact(-3) == pytest.approx(d.p_exact(3))


def test_bucket_table_is_exhaustive(key_numbers):
    d = MarginDistribution.from_continuous(-2.0, 16.0, key_numbers=key_numbers)
    assert d.bucket_table()["prob"].sum() == pytest.approx(1.0, abs=1e-6)


def test_quantiles_monotone(key_numbers):
    d = MarginDistribution.from_continuous(3.0, 16.0, key_numbers=key_numbers)
    qs = [d.quantile(q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert qs == sorted(qs)


def test_drive_simulator_hits_its_targets():
    sim = DriveSimulator(SimConfig(n_sims=30_000, seed=5))
    for mu, total, sigma in [(-3.5, 54, 16.0), (17.0, 62, 17.5), (0.0, 44, 14.0)]:
        home, away = sim.simulate_scores(mu, total, sigma=sigma)
        assert np.mean(home - away) == pytest.approx(mu, abs=0.8)
        assert np.mean(home + away) == pytest.approx(total, abs=1.5)
        assert np.std(home - away) == pytest.approx(sigma, abs=0.6)


def test_simulator_produces_the_lattice():
    sim = DriveSimulator(SimConfig(n_sims=40_000, seed=5))
    d = sim.distribution(0.0, 54, sigma=16.0)
    assert d.p_exact(3) > 1.5 * d.p_exact(5)
    assert d.p_exact(7) > 1.5 * d.p_exact(5)
    assert d.p_tie() == 0.0


def test_points_per_drive_inverts():
    for target in (1.2, 2.1, 3.4):
        assert points_per_drive(quality_for_ppd(target)) == pytest.approx(target, abs=1e-4)
