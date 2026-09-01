"""Concurrency and caching behaviour of the data layer.

No network: the throttle is tested directly and the fetch orchestration is
tested against a stub client, so these are deterministic on CI.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from cfb.data.http import JsonClient


def test_throttle_is_shared_across_threads():
    """Concurrency must overlap latency without raising the request rate."""
    client = JsonClient("https://example.invalid", min_interval=0.05)
    n = 12
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda _: client._throttle(), range(n)))
    elapsed = time.monotonic() - start
    # n calls at min_interval apart cannot finish faster than (n-1)*interval,
    # however many threads are asking.
    assert elapsed >= (n - 1) * 0.05 * 0.9


def test_throttle_is_a_noop_when_disabled():
    client = JsonClient("https://example.invalid", min_interval=0.0)
    start = time.monotonic()
    for _ in range(50):
        client._throttle()
    assert time.monotonic() - start < 0.2


class _StubClient:
    """Mimics CFBDClient's surface with a fixed per-call delay."""

    def __init__(self, delay=0.15):
        self.delay = delay
        self.calls = []

    def _frame(self, table, year):
        self.calls.append((table, year))
        time.sleep(self.delay)
        return pd.DataFrame({"season": [year], "team": [f"T{year}"], "value": [1.0]})

    def games(self, year):
        self.calls.append(("games", year))
        time.sleep(self.delay)
        return pd.DataFrame({
            "game_id": [year * 10], "season": [year], "week": [1],
            "start_date": pd.to_datetime([f"{year}-09-01"], utc=True),
            "home_team": ["A"], "away_team": ["B"],
            "home_points": [21.0], "away_points": [14.0],
            "neutral_site": [False], "conference_game": [True],
            "completed": [True], "margin": [7.0], "total": [35.0],
        })

    def lines(self, year):
        return self._frame("lines", year)

    def sp_ratings(self, year):
        return self._frame("sp_ratings", year)

    def talent(self, year):
        return self._frame("talent", year)

    def returning_production(self, year):
        return self._frame("returning", year)

    def advanced_season_stats(self, year):
        return self._frame("advanced_season", year)

    def team_game_stats(self, year):
        return self._frame("team_games", year)

    def transfer_portal(self, year):
        return self._frame("portal", year)

    def player_season_ppa(self, year):
        return self._frame("player_ppa", year)


def test_fetch_seasons_runs_concurrently(monkeypatch, tmp_path):
    from cfb.config import get_config
    from cfb.data.cfbd_client import fetch_seasons
    from cfb.data.store import Store

    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    get_config(refresh=True)
    try:
        years = [2021, 2022, 2023]
        stub = _StubClient(delay=0.15)
        store = Store()
        start = time.monotonic()
        counts = fetch_seasons(years, store=store, client=stub, max_workers=6)
        elapsed = time.monotonic() - start

        n_calls = len(years) * 9
        serial = n_calls * 0.15
        assert elapsed < serial * 0.6, f"took {elapsed:.2f}s, serial would be {serial:.2f}s"
        assert len(stub.calls) == n_calls
        assert counts["games"] == len(years)
        assert len(store.read("games")) == len(years)
    finally:
        monkeypatch.undo()
        get_config(refresh=True)


def test_one_failing_endpoint_does_not_abort_the_pull(monkeypatch, tmp_path):
    from cfb.config import get_config
    from cfb.data.cfbd_client import fetch_seasons
    from cfb.data.store import Store

    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    get_config(refresh=True)
    try:
        stub = _StubClient(delay=0.0)
        stub.talent = lambda year: (_ for _ in ()).throw(RuntimeError("404"))
        counts = fetch_seasons([2023], store=Store(), client=stub)
        assert "games" in counts and "talent" not in counts
    finally:
        monkeypatch.undo()
        get_config(refresh=True)


def test_feature_cache_hits_and_invalidates(monkeypatch, tmp_path):
    from cfb import bootstrap
    from cfb.config import get_config
    from cfb.data.store import Store
    from cfb.features.build import FeatureConfig
    from cfb.pipeline import feature_cache_key, load_features

    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    get_config(refresh=True)
    try:
        bootstrap.build_sample_data(n_teams=40, n_seasons=3, with_pbp=False)
        store, cfg = Store(), FeatureConfig()
        key_before = feature_cache_key(store, cfg, None)

        cold = load_features()
        warm = load_features()
        pd.testing.assert_frame_equal(cold, warm)

        # A different feature config must not reuse the cached frame.
        assert feature_cache_key(store, FeatureConfig(include_market=True), None) != key_before

        # New data must invalidate.
        games = store.read("games")
        store.write("games", games)
        assert feature_cache_key(store, cfg, None) != key_before
    finally:
        monkeypatch.undo()
        get_config(refresh=True)
