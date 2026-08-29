import numpy as np
import pytest

from cfb.features.ratings import EloConfig, elo_ratings, fit_off_def_ratings


def test_ratings_recover_synthetic_truth(games, universe):
    fit = fit_off_def_ratings(games, ridge_lambda=45)
    truth = universe["truth"]
    truth = truth[truth["season"] == games["season"].max()].set_index("team")
    joined = fit.to_frame().set_index("team").join(truth[["true_off", "true_def"]])
    # Fitted `defense` is points allowed, so it moves opposite to defensive skill.
    assert joined["offense"].corr(joined["true_off"]) > 0.6
    assert joined["defense"].corr(joined["true_def"]) < -0.6


def test_home_field_advantage_is_positive_and_plausible(games):
    fit = fit_off_def_ratings(games)
    assert 0.5 < fit.hfa < 5.0


def test_predict_game_matches_expected_points(games):
    fit = fit_off_def_ratings(games)
    h, a = fit.teams[0], fit.teams[1]
    margin, total = fit.predict_game(h, a, neutral=False)
    hp = fit.expected_points(h, a, at_home=True)
    ap = fit.expected_points(a, h, at_home=False)
    assert margin == pytest.approx(hp - ap)
    assert total == pytest.approx(hp + ap)
    neutral_margin, _ = fit.predict_game(h, a, neutral=True)
    assert margin > neutral_margin  # home field must help the home team


def test_elo_is_pregame_only(games):
    elo, final = elo_ratings(games, EloConfig())
    assert len(elo) == len(games)
    assert elo["elo_home_wp"].between(0, 1).all()
    # Ratings should be zero-sum-ish around the starting value.
    assert np.mean(list(final.values())) == pytest.approx(1500, abs=60)


def test_recency_weighting_changes_ratings(games):
    short = fit_off_def_ratings(games, half_life_days=120)
    long = fit_off_def_ratings(games, half_life_days=3000)
    diffs = [abs(short.rating(t) - long.rating(t)) for t in short.teams[:25]]
    assert max(diffs) > 0.5
