"""The leakage tests. If these fail, every accuracy number in the repo is a lie."""
import pandas as pd
import pytest

from cfb.features.build import FeatureConfig, build_features, feature_columns, \
    consensus_lines, rolling_form, team_game_long


def test_no_leakage_from_a_games_own_result(universe):
    """Changing one game's score must not change that game's own features.

    This is the test that catches an opponent-adjusted rating accidentally being
    fit on the full season -- the single easiest way to build a model that looks
    brilliant in backtest and loses money live.
    """
    games = universe["games"].copy()
    base = build_features(games, universe["lines"])

    target = games.index[len(games) // 2]
    tampered = games.copy()
    tampered.loc[target, "home_points"] = 99.0
    tampered.loc[target, "away_points"] = 0.0
    tampered["margin"] = tampered["home_points"] - tampered["away_points"]
    tampered["total"] = tampered["home_points"] + tampered["away_points"]
    after = build_features(tampered, universe["lines"])

    gid = games.loc[target, "game_id"]
    cols = [c for c in feature_columns() if c in base.columns]
    row_before = base.loc[base["game_id"] == gid, cols].iloc[0]
    row_after = after.loc[after["game_id"] == gid, cols].iloc[0]
    pd.testing.assert_series_equal(row_before, row_after, check_names=False)


def test_rolling_form_excludes_current_game(universe):
    long = team_game_long(universe["games"])
    form = rolling_form(long, window=3)
    merged = long.merge(form, on=["game_id", "team"], how="left")
    # A team's first game of a season has no prior games, so form must be null.
    first = merged[merged["played"] == 0]
    assert first["form_margin"].isna().all()
    assert (merged["played"] >= 0).all()


def test_early_season_has_no_form_but_late_does(features):
    wk1 = features[features["week"] == 1]
    late = features[features["week"] >= 6]
    assert wk1["form_margin_home"].isna().mean() > 0.9
    assert late["form_margin_home"].notna().mean() > 0.9


def test_market_features_are_opt_in(universe):
    blind = feature_columns(FeatureConfig(include_market=False))
    aware = feature_columns(FeatureConfig(include_market=True))
    assert "market_margin" not in blind
    assert "market_margin" in aware
    assert set(blind) < set(aware)


def test_ratings_features_are_informative(features):
    mature = features[(features["hist_games"] > 400) & features["margin"].notna()]
    assert mature["rat_margin"].corr(mature["margin"]) > 0.25


def test_consensus_lines_uses_the_median_across_books():
    lines = pd.DataFrame({
        "game_id": [1, 1, 1], "provider": ["a", "b", "c"],
        "spread": [-7.0, -6.5, -20.0],       # one bad book
        "spread_open": [-7.0, -7.0, -7.0],
        "over_under": [54.0, 55.0, 56.0],
    })
    out = consensus_lines(lines)
    assert out.loc[0, "market_margin"] == pytest.approx(7.0)   # median, not mean
    assert out.loc[0, "market_total"] == pytest.approx(55.0)
    assert out.loc[0, "n_books"] == 3


def test_sp_plus_is_joined_from_the_prior_season(universe):
    """SP+ for season Y reflects season Y's games, so it may only be used in Y+1."""
    sp = universe["sp_ratings"]
    feats = build_features(universe["games"], universe["lines"], sp_ratings=sp)
    season = int(feats["season"].min())
    early = feats[feats["season"] == season]
    # No prior season exists for the first season, so the join must be empty.
    assert early["sp_prev_home"].isna().all()
    later = feats[feats["season"] == season + 1]
    assert later["sp_prev_home"].notna().mean() > 0.8
