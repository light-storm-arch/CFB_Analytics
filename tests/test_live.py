import numpy as np
import pytest

from cfb.data.espn_client import _clock_to_seconds, seconds_remaining_in_regulation
from cfb.live.diffusion import LiveConfig, LiveMarginModel, expected_points


@pytest.fixture(scope="module")
def live(key_numbers):
    return LiveMarginModel(LiveConfig(mc_sims=8000), key_numbers=key_numbers)


def test_clock_parsing():
    assert _clock_to_seconds("12:34") == 754
    assert _clock_to_seconds("0:07") == 7
    assert _clock_to_seconds(None) == 0
    assert seconds_remaining_in_regulation(1, 900) == 3600
    assert seconds_remaining_in_regulation(4, 0) == 0
    assert seconds_remaining_in_regulation(5, 300) == 0      # overtime


def test_expected_points_increases_downfield():
    assert expected_points(80) < expected_points(50) < expected_points(20)
    assert expected_points(50, down=4) < expected_points(50, down=1)


def test_win_probability_rises_with_the_lead(live):
    probs = [
        live.distribution(m, 900, mu_pregame=0.0, sigma_pregame=16.0,
                          total_pregame=54, period=4).p_home_win()
        for m in (-14, -7, 0, 7, 14)
    ]
    assert probs == sorted(probs)
    assert probs[0] < 0.25 and probs[-1] > 0.75


def test_certainty_grows_as_the_clock_runs_down(live):
    wide = live.distribution(7, 3000, 0.0, 16.0, 54, period=1)
    tight = live.distribution(7, 200, 0.0, 16.0, 54, period=4)
    assert tight.sd() < wide.sd()
    assert tight.p_home_win() > wide.p_home_win()


def test_possession_is_worth_points(live):
    with_ball = live.distribution(0, 600, 0.0, 16.0, 54, possession_home=True,
                                  yards_to_goal=75, down=1, distance=10, period=4)
    without = live.distribution(0, 600, 0.0, 16.0, 54, possession_home=False,
                                yards_to_goal=75, down=1, distance=10, period=4)
    assert with_ball.p_home_win() > without.p_home_win()
    # ... and being in the red zone is worth more than being backed up.
    red = live.distribution(0, 600, 0.0, 16.0, 54, possession_home=True,
                            yards_to_goal=10, down=1, distance=10, period=4)
    assert red.p_home_win() > with_ball.p_home_win()


def test_live_mean_stays_near_the_current_margin_late(live):
    """A late-game distribution must not drift away from the actual score."""
    d = live.distribution(4, 300, mu_pregame=-2.5, sigma_pregame=16.0,
                          total_pregame=54, possession_home=True,
                          yards_to_goal=75, down=1, distance=10, period=4)
    assert abs(d.mean() - 4) < 3.0
    assert d.sd() < 9.0


def test_tied_at_the_final_whistle_goes_to_overtime(live):
    d = live.distribution(0, 0, mu_pregame=3.0, sigma_pregame=16.0,
                          total_pregame=54, period=4)
    assert d.p_tie() == 0.0
    assert 0.4 < d.p_home_win() < 0.6
    assert d.p_exact(3) + d.p_exact(-3) + d.p_exact(7) + d.p_exact(-7) > 0.5


def test_finished_game_is_a_point_mass(live):
    d = live.distribution(10, 0, 0.0, 16.0, 54, period=4)
    assert d.p_exact(10) == pytest.approx(1.0)


def test_variance_decay_is_fitted_in_range(league, games):
    pbp = league.generate_pbp(games, max_games=250)
    lm = LiveMarginModel(LiveConfig())
    alpha = lm.fit_variance_decay(pbp)
    assert 0.25 <= alpha <= 0.9


def test_trained_live_model_learns(league, games):
    from cfb.live.winprob import LiveWinProbModel, build_state_features

    pbp = league.generate_pbp(games, max_games=600)
    pbp["pregame_mu"] = 0.0
    pbp["pregame_total"] = 54.0
    model = LiveWinProbModel().fit(pbp)
    assert model.report.log_loss < 0.69   # better than a coin flip
    out = model.predict(pbp.head(200))
    assert out["p_home_win"].between(0, 1).all()
    assert np.isfinite(out["pred_final_margin"]).all()
    assert (out["pred_sd"] > 0).all()
    assert set(build_state_features(pbp.head(5)).columns)
