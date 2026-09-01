"""Roster-continuity features and the roster-aware ratings prior.

The load-bearing test here is the leak one: the obvious way to identify a
team's starting quarterback is "whoever took the most snaps this season", which
is only knowable once the season is over.
"""
import numpy as np
import pandas as pd
import pytest

from cfb.features.preseason import CarryoverModel, priors_for_season, season_ratings
from cfb.features.ratings import fit_off_def_ratings
from cfb.features.roster import build_roster_features, portal_flux, quarterback_continuity


@pytest.fixture(scope="module")
def roster(universe):
    return build_roster_features(universe["portal"], universe["player_ppa"],
                                 universe["returning"])


def test_portal_flux_nets_in_against_out():
    portal = pd.DataFrame([
        {"season": 2024, "player_id": "1", "position": "WR",
         "origin": "B", "destination": "A", "rating": 0.9},
        {"season": 2024, "player_id": "2", "position": "RB",
         "origin": "C", "destination": "A", "rating": 0.8},
        {"season": 2024, "player_id": "3", "position": "DB",
         "origin": "A", "destination": "D", "rating": 0.5},
    ])
    out = portal_flux(portal).set_index("team")
    assert out.loc["A", "portal_in_count"] == 2
    assert out.loc["A", "portal_out_count"] == 1
    assert out.loc["A", "portal_net_rating"] == pytest.approx(0.9 + 0.8 - 0.5)
    assert out.loc["A", "portal_in_best"] == pytest.approx(0.9)


def test_quarterback_continuity_never_reads_the_current_season(universe):
    """Scrambling season S's QB stats must not change season S's own features."""
    ppa = universe["player_ppa"]
    portal = universe["portal"]
    target = int(ppa["season"].max())

    before = quarterback_continuity(ppa, portal)
    before = before[before["season"] == target].set_index("team").sort_index()

    tampered = ppa.copy()
    mask = tampered["season"] == target
    rng = np.random.default_rng(0)
    tampered.loc[mask, "avg_ppa_all"] = rng.normal(5.0, 1.0, int(mask.sum()))
    tampered.loc[mask, "plays"] = rng.integers(1, 50, int(mask.sum()))

    after = quarterback_continuity(tampered, portal)
    after = after[after["season"] == target].set_index("team").sort_index()

    cols = ["qb_prior_ppa", "qb_departed", "qb_transfer_in", "qb_continuity"]
    pd.testing.assert_frame_equal(before[cols], after[cols], check_names=False)


def test_qb_status_is_derived_from_ids_across_seasons(universe):
    """Continuity must come from player ids, not name matching."""
    qb = quarterback_continuity(universe["player_ppa"], universe["portal"])
    truth = universe["truth"][["season", "team", "true_qb_status"]]
    chk = qb.merge(truth, on=["season", "team"])
    chk = chk[chk["season"] > chk["season"].min()]
    # A team whose starter returned must not show an incoming portal QB.
    returning = chk[chk["true_qb_status"] == "returning"]
    assert returning["qb_transfer_in"].fillna(0).max() == 0
    assert returning["qb_departed"].fillna(0).max() == 0
    # A team that took a transfer must show one.
    transfers = chk[chk["true_qb_status"] == "transfer"]
    assert transfers["qb_transfer_in"].fillna(0).min() >= 1


def test_roster_features_degrade_gracefully_when_sources_are_missing(universe):
    assert build_roster_features(None, None, None).empty
    only_portal = build_roster_features(universe["portal"], None, None)
    assert "portal_net_rating" in only_portal.columns
    assert "qb_continuity" not in only_portal.columns


def test_prior_shrinks_ratings_toward_it(games):
    base = fit_off_def_ratings(games, ridge_lambda=45)
    team = base.teams[0]
    high = fit_off_def_ratings(games, ridge_lambda=45,
                               prior_offense={team: base.offense[team] + 4.0})
    low = fit_off_def_ratings(games, ridge_lambda=45,
                              prior_offense={team: base.offense[team] - 4.0})
    assert high.offense[team] > base.offense[team] > low.offense[team]
    # Other teams should barely move.
    others = base.teams[1:15]
    assert max(abs(high.offense[t] - base.offense[t]) for t in others) < 0.2


def test_no_prior_reproduces_the_original_fit(games):
    a = fit_off_def_ratings(games, ridge_lambda=45)
    b = fit_off_def_ratings(games, ridge_lambda=45, prior_offense=None,
                            prior_defense=None)
    assert a.offense == b.offense


def test_carryover_model_is_fit_only_on_earlier_seasons(games, roster):
    seasons = sorted(games["season"].unique())
    target = seasons[-1]
    fits = season_ratings(games)
    model = CarryoverModel().fit(fits, roster, before_season=target)
    assert model.report is not None
    assert max(model.report.seasons_used) < target


def test_priors_ignore_the_target_seasons_games(games, roster):
    """Scrambling the target season's results must not change its priors."""
    seasons = sorted(games["season"].unique())
    target = seasons[-1]
    off_a, def_a, _ = priors_for_season(games, roster, target)

    tampered = games.copy()
    mask = tampered["season"] == target
    tampered.loc[mask, "home_points"] = 70.0
    tampered.loc[mask, "away_points"] = 0.0
    off_b, def_b, _ = priors_for_season(tampered, roster, target)

    assert set(off_a) == set(off_b)
    for team in off_a:
        assert off_a[team] == pytest.approx(off_b[team])
        assert def_a[team] == pytest.approx(def_b[team])


def test_prior_beats_raw_carryover_at_predicting_next_season(games, roster):
    """A fitted prior must beat 'just reuse last season's number'."""
    seasons = sorted(games["season"].unique())
    target = seasons[-1]
    fits_hist = season_ratings(games[games["season"] < target])
    actual = season_ratings(games[games["season"] == target]).get(target)
    prev = fits_hist.get(target - 1)
    assert actual is not None and prev is not None

    off_prior, _, _ = priors_for_season(games, roster, target)
    common = [t for t in off_prior if t in actual.offense and t in prev.offense]
    act = np.array([actual.offense[t] for t in common])
    prior_mae = np.mean(np.abs(np.array([off_prior[t] for t in common]) - act))
    naive_mae = np.mean(np.abs(np.array([prev.offense[t] for t in common]) - act))
    assert prior_mae < naive_mae


def test_feature_config_defaults_keep_the_roster_work_off():
    """Both options measured neutral-to-worse on synthetic data, so they ship off."""
    from cfb.features.build import FeatureConfig, feature_columns

    cfg = FeatureConfig()
    assert cfg.include_roster is False
    assert cfg.use_roster_prior is False
    assert not any(c.startswith("qb_") for c in feature_columns(cfg))
    on = feature_columns(FeatureConfig(include_roster=True))
    assert "qb_continuity_diff" in on and "portal_net_rating_diff" in on


def test_roster_features_are_leak_free_end_to_end(universe):
    """The full feature build must not let a game see its own result."""
    from cfb.features.build import FeatureConfig, build_features, feature_columns

    cfg = FeatureConfig(include_roster=True, use_roster_prior=True)
    kw = dict(lines=universe["lines"], talent=universe["talent"],
              returning=universe["returning"], sp_ratings=universe["sp_ratings"],
              portal=universe["portal"], player_ppa=universe["player_ppa"], cfg=cfg)
    games = universe["games"].copy()
    base = build_features(games, **kw)

    target = games.index[len(games) // 2]
    tampered = games.copy()
    tampered.loc[target, "home_points"] = 99.0
    tampered.loc[target, "away_points"] = 0.0
    tampered["margin"] = tampered["home_points"] - tampered["away_points"]
    tampered["total"] = tampered["home_points"] + tampered["away_points"]
    after = build_features(tampered, **kw)

    gid = games.loc[target, "game_id"]
    cols = [c for c in feature_columns(cfg) if c in base.columns]
    pd.testing.assert_series_equal(
        base.loc[base["game_id"] == gid, cols].iloc[0],
        after.loc[after["game_id"] == gid, cols].iloc[0], check_names=False)
