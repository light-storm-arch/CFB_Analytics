import numpy as np
import pandas as pd
import pytest

from cfb.betting.edge import (
    best_side, edge_table, evaluate_contract, fee_per_contract, kelly_fraction,
)
from cfb.betting.kalshi_map import parse_market, price_markets, split_matchup
from cfb.betting.pricing import american_to_prob, devig, prob_to_american
from cfb.distribution.margin import MarginDistribution


def test_fee_matches_kalshis_formula():
    # ceil(0.07 * P * (1-P) * 100), peaking at 50c.
    assert fee_per_contract(50) == 2
    assert fee_per_contract(10) == 1
    assert fee_per_contract(99) == 1
    assert fee_per_contract(0) == 0


def test_kelly_is_zero_without_an_edge():
    assert kelly_fraction(0.50, 50, 2) == 0.0
    assert kelly_fraction(0.40, 50, 2) == 0.0
    assert kelly_fraction(0.60, 50, 2) > 0.0


def test_ev_is_net_of_fees():
    ev = evaluate_contract(0.55, 52, "yes")
    assert ev.fee_cents == 2
    assert ev.ev_cents == pytest.approx(0.55 * 100 - 54)
    assert ev.edge == pytest.approx(0.55 - 0.54)


def test_no_side_uses_the_complement():
    ev = evaluate_contract(0.30, 60, "no")
    assert ev.model_prob == pytest.approx(0.70)
    # fee at 60c = ceil(0.07 * 0.6 * 0.4 * 100) = 2, so all-in cost is 62c
    assert ev.fee_cents == 2
    assert ev.ev_cents == pytest.approx(70 - 62)


def test_best_side_picks_the_better_of_the_two():
    assert best_side(0.80, yes_ask=90, no_ask=12).side == "no"
    assert best_side(0.80, yes_ask=60, no_ask=45).side == "yes"


def test_american_odds_roundtrip():
    for p in (0.25, 0.5, 0.73):
        assert american_to_prob(prob_to_american(p)) == pytest.approx(p, abs=1e-9)


def test_devig_normalises_and_shrinks_the_favourite_less():
    raw = [american_to_prob(-200), american_to_prob(+170)]
    assert sum(raw) > 1.0
    for method in ("power", "multiplicative", "additive"):
        out = devig(raw, method)
        assert out.sum() == pytest.approx(1.0)
        assert (out > 0).all()


def test_edge_table_filters_and_ranks():
    df = pd.DataFrame([
        {"ticker": "GOOD", "title": "t", "yes_sub_title": "A", "yes_ask": 40,
         "no_ask": 62, "volume": 500, "model_prob": 0.60},
        {"ticker": "BAD", "title": "t", "yes_sub_title": "B", "yes_ask": 60,
         "no_ask": 42, "volume": 500, "model_prob": 0.58},
    ])
    out = edge_table(df, min_edge=0.05)
    assert list(out["ticker"]) == ["GOOD"]
    assert out.iloc[0]["side"] == "yes"
    assert out.iloc[0]["ev_cents"] > 0


def test_market_parsing_reads_structured_strikes():
    spec = parse_market({"title": "x", "yes_sub_title": "Michigan",
                         "strike_type": "between", "floor_strike": 4, "cap_strike": 7})
    assert spec.kind == "bucket" and spec.floor == 4 and spec.cap == 7
    assert spec.confident

    spec = parse_market({"title": "x", "yes_sub_title": "Michigan",
                         "strike_type": "greater_or_equal", "floor_strike": 7})
    assert spec.kind == "spread" and spec.strike_type == "greater_or_equal"


def test_market_parsing_falls_back_to_titles_but_flags_it():
    spec = parse_market({"title": "Michigan wins by more than 10.5",
                         "yes_sub_title": "Michigan"})
    assert spec.kind == "spread" and spec.floor == 10.5
    assert not spec.confident

    spec = parse_market({"title": "Michigan wins by exactly 3 points",
                         "yes_sub_title": "Michigan"})
    assert spec.kind == "exact" and spec.floor == 3


def test_unparseable_market_is_refused_not_guessed():
    spec = parse_market({"title": "Something entirely unrelated", "yes_sub_title": ""})
    assert spec.kind == "unknown"
    assert not spec.confident


def test_split_matchup():
    assert split_matchup("Michigan vs Ohio State") == ("Michigan", "Ohio State")
    assert split_matchup("Will Alabama beat Georgia?") is None


def test_pricing_respects_which_team_the_market_names(key_numbers):
    """The most expensive possible bug: pricing the wrong side."""
    dist = MarginDistribution.from_continuous(-6.5, 16.0, key_numbers=key_numbers)
    slate = pd.DataFrame([{"game_id": 1, "home_team": "Ohio State",
                           "away_team": "Michigan"}])
    markets = pd.DataFrame([
        {"ticker": "H", "title": "Michigan vs Ohio State", "yes_sub_title": "Ohio State",
         "strike_type": None, "floor_strike": None, "cap_strike": None},
        {"ticker": "A", "title": "Michigan vs Ohio State", "yes_sub_title": "Michigan",
         "strike_type": None, "floor_strike": None, "cap_strike": None},
    ])
    out = price_markets(markets, {1: dist}, slate)
    home_p = out.loc[out["ticker"] == "H", "model_prob"].iloc[0]
    away_p = out.loc[out["ticker"] == "A", "model_prob"].iloc[0]
    assert home_p == pytest.approx(dist.p_home_win())
    assert away_p == pytest.approx(dist.p_away_win())
    assert home_p + away_p == pytest.approx(1.0)
    assert away_p > home_p          # the away team is a 6.5-point favourite


def test_unmappable_market_gets_no_probability(key_numbers):
    dist = MarginDistribution.from_continuous(0.0, 16.0, key_numbers=key_numbers)
    slate = pd.DataFrame([{"game_id": 1, "home_team": "A", "away_team": "B"}])
    markets = pd.DataFrame([{"ticker": "X", "title": "Totally unrelated market",
                             "yes_sub_title": "Nobody", "strike_type": None,
                             "floor_strike": None, "cap_strike": None}])
    out = price_markets(markets, {1: dist}, slate)
    assert np.isnan(out.iloc[0]["model_prob"])
    assert out.iloc[0]["skip_reason"]
