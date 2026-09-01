"""CollegeFootballData.com API client.

Free API key: https://collegefootballdata.com/key  ->  put it in .env as CFBD_API_KEY.

SIGN CONVENTIONS (the #1 source of bugs in this domain, so they are stated once
here and honoured everywhere downstream):

* ``margin``       = home_points - away_points.  Positive => home team won.
* CFBD ``spread``  = the *home team's* spread.  -7 means home is a 7-point
                     favourite.  We therefore define
                     ``market_margin = -spread`` = the market's expected
                     home margin, which is directly comparable to ``margin``.
* Everything the models emit (``pred_margin``, distributions, live margins) is
  in home-margin space.  Convert to a specific team's perspective only at the
  presentation / betting layer.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from cfb.config import Config, get_config
from cfb.data.http import JsonClient
from cfb.data.store import Store

log = logging.getLogger(__name__)

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def snake(name: str) -> str:
    return _CAMEL.sub("_", name).lower().replace("__", "_")


def snake_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {snake(k): snake_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [snake_keys(v) for v in obj]
    return obj


def _first(d: dict, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


class CFBDClient:
    """Typed-ish wrapper returning tidy DataFrames."""

    def __init__(self, cfg: Config | None = None, cache_ttl: float | None = None,
                 min_interval: float = 0.35):
        self.cfg = cfg or get_config()
        if not self.cfg.cfbd_api_key:
            raise RuntimeError(
                "CFBD_API_KEY is not set. Get a free key at "
                "https://collegefootballdata.com/key and add it to .env "
                "(see .env.example)."
            )
        self.cache_ttl = cache_ttl
        self.http = JsonClient(
            base_url=self.cfg.cfbd_base,
            headers={"Authorization": f"Bearer {self.cfg.cfbd_api_key}"},
            cache_dir=self.cfg.raw_dir / "cfbd",
            min_interval=min_interval,
        )

    def _get(self, path: str, **params) -> list[dict]:
        data = self.http.get(path, params=params, cache_ttl=self.cache_ttl)
        if isinstance(data, dict):
            data = [data]
        return [snake_keys(r) for r in (data or [])]

    # ------------------------------------------------------------------ #
    # Core tables
    # ------------------------------------------------------------------ #
    def games(self, year: int, season_type: str = "both",
              division: str | None = "fbs") -> pd.DataFrame:
        raw = self._get("/games", year=year, seasonType=season_type, division=division)
        rows = []
        for g in raw:
            hp, ap = _first(g, "home_points"), _first(g, "away_points")
            rows.append({
                "game_id": int(_first(g, "id", "game_id")),
                "season": int(_first(g, "season", default=year)),
                "week": int(_first(g, "week", default=0)),
                "season_type": _first(g, "season_type", default="regular"),
                "start_date": _first(g, "start_date"),
                "home_team": _first(g, "home_team"),
                "away_team": _first(g, "away_team"),
                "home_conference": _first(g, "home_conference"),
                "away_conference": _first(g, "away_conference"),
                "home_points": None if hp is None else float(hp),
                "away_points": None if ap is None else float(ap),
                "neutral_site": bool(_first(g, "neutral_site", default=False)),
                "conference_game": bool(_first(g, "conference_game", default=False)),
                "venue_id": _first(g, "venue_id"),
                "home_pregame_elo": _first(g, "home_pregame_elo"),
                "away_pregame_elo": _first(g, "away_pregame_elo"),
                "excitement_index": _first(g, "excitement_index"),
            })
        df = pd.DataFrame(rows)
        return finalize_games(df)

    def lines(self, year: int, season_type: str = "both") -> pd.DataFrame:
        raw = self._get("/lines", year=year, seasonType=season_type)
        rows = []
        for g in raw:
            gid = _first(g, "id", "game_id")
            if gid is None:
                continue
            for ln in g.get("lines") or []:
                rows.append({
                    "game_id": int(gid),
                    "season": int(_first(g, "season", default=year)),
                    "week": int(_first(g, "week", default=0)),
                    "home_team": _first(g, "home_team"),
                    "away_team": _first(g, "away_team"),
                    "provider": _first(ln, "provider", default="unknown"),
                    "spread": _num(_first(ln, "spread")),
                    "spread_open": _num(_first(ln, "spread_open")),
                    "over_under": _num(_first(ln, "over_under")),
                    "over_under_open": _num(_first(ln, "over_under_open")),
                    "home_moneyline": _num(_first(ln, "home_moneyline")),
                    "away_moneyline": _num(_first(ln, "away_moneyline")),
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            # market_margin = market's expected home margin.
            df["market_margin"] = -df["spread"]
        return df

    def team_game_stats(self, year: int, week: int | None = None,
                        season_type: str = "both") -> pd.DataFrame:
        raw = self._get("/games/teams", year=year, week=week, seasonType=season_type)
        rows = []
        for g in raw:
            gid = _first(g, "id", "game_id")
            for team in g.get("teams") or []:
                rec = {
                    "game_id": int(gid),
                    "season": year,
                    "team": _first(team, "school", "team"),
                    "conference": _first(team, "conference"),
                    "home_away": _first(team, "home_away"),
                    "points": _num(_first(team, "points")),
                }
                for st in team.get("stats") or []:
                    cat = snake(str(_first(st, "category", default="")))
                    if cat:
                        rec[f"st_{cat}"] = _first(st, "stat")
                rows.append(rec)
        return pd.DataFrame(rows)

    def advanced_season_stats(self, year: int, exclude_garbage_time: bool = True) -> pd.DataFrame:
        raw = self._get("/stats/season/advanced", year=year,
                        excludeGarbageTime=str(bool(exclude_garbage_time)).lower())
        rows = []
        for r in raw:
            rec = {"season": int(_first(r, "season", default=year)),
                   "team": _first(r, "team"),
                   "conference": _first(r, "conference")}
            for side in ("offense", "defense"):
                blob = r.get(side) or {}
                rec.update(_flatten(blob, prefix=side))
            rows.append(rec)
        return pd.DataFrame(rows)

    def sp_ratings(self, year: int) -> pd.DataFrame:
        raw = self._get("/ratings/sp", year=year)
        rows = []
        for r in raw:
            team = _first(r, "team")
            if not team or str(team).lower() == "nationalaverages":
                continue
            rec = {
                "season": int(_first(r, "year", "season", default=year)),
                "team": team,
                "conference": _first(r, "conference"),
                "sp_overall": _num(_first(r, "rating")),
                "sp_second_order_wins": _num(_first(r, "second_order_wins")),
                "sp_sos": _num(_first(r, "sos")),
                "sp_offense": _num((r.get("offense") or {}).get("rating")),
                "sp_defense": _num((r.get("defense") or {}).get("rating")),
                "sp_special_teams": _num((r.get("special_teams") or {}).get("rating")),
            }
            rows.append(rec)
        return pd.DataFrame(rows)

    def srs_ratings(self, year: int) -> pd.DataFrame:
        raw = self._get("/ratings/srs", year=year)
        return pd.DataFrame([
            {"season": int(_first(r, "year", "season", default=year)),
             "team": _first(r, "team"),
             "srs": _num(_first(r, "rating"))}
            for r in raw if _first(r, "team")
        ])

    def talent(self, year: int) -> pd.DataFrame:
        raw = self._get("/talent", year=year)
        return pd.DataFrame([
            {"season": int(_first(r, "year", "season", default=year)),
             "team": _first(r, "school", "team"),
             "talent": _num(_first(r, "talent"))}
            for r in raw
        ])

    def returning_production(self, year: int) -> pd.DataFrame:
        raw = self._get("/player/returning", year=year)
        return pd.DataFrame([
            {"season": int(_first(r, "season", default=year)),
             "team": _first(r, "team"),
             "returning_ppa": _num(_first(r, "total_ppa")),
             "returning_offense_ppa": _num(_first(r, "total_offense_ppa")),
             "returning_defense_ppa": _num(_first(r, "total_defense_ppa")),
             "usage": _num(_first(r, "usage")),
             "percent_ppa": _num(_first(r, "percent_ppa"))}
            for r in raw
        ])

    def fbs_teams(self, year: int) -> pd.DataFrame:
        raw = self._get("/teams/fbs", year=year)
        return pd.DataFrame([
            {"season": year, "team": _first(r, "school", "team"),
             "conference": _first(r, "conference"),
             "venue_id": _first(r, "venue_id")}
            for r in raw
        ])

    def calendar(self, year: int) -> pd.DataFrame:
        return pd.DataFrame(self._get("/calendar", year=year))

    # ------------------------------------------------------------------ #
    # Play-by-play (heavy: one call per week)
    # ------------------------------------------------------------------ #
    def plays(self, year: int, week: int, season_type: str = "regular") -> pd.DataFrame:
        raw = self._get("/plays", year=year, week=week, seasonType=season_type)
        rows = []
        for p in raw:
            clock = p.get("clock") or {}
            rows.append({
                "play_id": str(_first(p, "id", default="")),
                "game_id": _int(_first(p, "game_id")),
                "season": year,
                "week": week,
                "season_type": season_type,
                "offense": _first(p, "offense"),
                "defense": _first(p, "defense"),
                "home": _first(p, "home"),
                "away": _first(p, "away"),
                "offense_score": _num(_first(p, "offense_score")),
                "defense_score": _num(_first(p, "defense_score")),
                "period": _int(_first(p, "period")),
                "clock_minutes": _int(clock.get("minutes"), 0),
                "clock_seconds": _int(clock.get("seconds"), 0),
                "yard_line": _num(_first(p, "yard_line")),
                "yards_to_goal": _num(_first(p, "yards_to_goal")),
                "down": _int(_first(p, "down")),
                "distance": _num(_first(p, "distance")),
                "yards_gained": _num(_first(p, "yards_gained")),
                "play_type": _first(p, "play_type"),
                "ppa": _num(_first(p, "ppa")),
                "scoring": bool(_first(p, "scoring", default=False)),
            })
        return pd.DataFrame(rows)

    def plays_season(self, year: int, weeks: Iterable[int] | None = None,
                     season_type: str = "regular") -> pd.DataFrame:
        weeks = list(weeks) if weeks is not None else range(1, 16)
        frames = []
        for w in weeks:
            try:
                df = self.plays(year, w, season_type)
            except Exception as exc:  # noqa: BLE001 - one bad week shouldn't kill the pull
                log.warning("plays %s wk%s failed: %s", year, w, exc)
                continue
            if not df.empty:
                frames.append(df)
                log.info("plays %s wk%-2d rows=%d", year, w, len(df))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v, default=None):
    n = _num(v)
    return default if n is None else int(n)


def _flatten(blob: dict, prefix: str) -> dict:
    out = {}
    for k, v in (blob or {}).items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}_{k}"))
        else:
            n = _num(v)
            if n is not None:
                out[f"{prefix}_{k}"] = n
    return out


def finalize_games(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns + sort. Safe to call on synthetic frames too."""
    if df.empty:
        return df
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    df["completed"] = df["home_points"].notna() & df["away_points"].notna()
    df["margin"] = df["home_points"] - df["away_points"]
    df["total"] = df["home_points"] + df["away_points"]
    df["neutral_site"] = df["neutral_site"].fillna(False).astype(bool)
    order = ["season", "week", "start_date", "game_id"]
    return df.sort_values([c for c in order if c in df.columns],
                          kind="stable").reset_index(drop=True)


def fetch_seasons(
    years: Sequence[int],
    store: Store | None = None,
    client: CFBDClient | None = None,
    include_plays: bool = False,
    play_weeks: Iterable[int] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, int]:
    """Pull every season-level table for ``years`` and upsert into the store.

    ``progress`` is called as ``progress(fraction_done, message)`` so a UI with
    no console can show what is happening during a multi-minute pull.
    """
    store = store or Store()
    client = client or CFBDClient()
    counts: dict[str, int] = {}
    total = max(len(years), 1)

    def _report(i: int, msg: str):
        if progress:
            progress(min(i / total, 1.0), msg)

    def _add(table: str, df: pd.DataFrame):
        if df is not None and not df.empty:
            store.upsert(table, df)
            counts[table] = counts.get(table, 0) + len(df)
            log.info("%-16s +%d rows", table, len(df))

    for i, yr in enumerate(years):
        log.info("=== season %d ===", yr)
        _report(i, f"season {yr}: games and lines")
        _add("games", client.games(yr))
        _add("lines", client.lines(yr))
        _report(i, f"season {yr}: ratings, talent, box scores")
        for name, fn in (
            ("sp_ratings", client.sp_ratings),
            ("talent", client.talent),
            ("returning", client.returning_production),
            ("advanced_season", client.advanced_season_stats),
            ("team_games", client.team_game_stats),
        ):
            try:
                _add(name, fn(yr))
            except Exception as exc:  # noqa: BLE001
                log.warning("%s %s failed: %s", name, yr, exc)
        if include_plays:
            _report(i, f"season {yr}: play-by-play (slow)")
            _add("plays", client.plays_season(yr, weeks=play_weeks))
        _report(i + 1, f"season {yr}: done")
    return counts
