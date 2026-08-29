"""ESPN public scoreboard client -- free, no API key, used for LIVE game state.

ESPN's `site.api.espn.com` endpoints are undocumented but public and stable
enough for in-game use.  We only read from them.

The important output is :class:`LiveGameState`, the normalized snapshot the
in-game models consume.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from cfb.config import CONFIG, Config
from cfb.constants import GAME_SECONDS, QUARTER_SECONDS
from cfb.data.http import JsonClient

log = logging.getLogger(__name__)

FBS_GROUP = "80"  # ESPN group id for FBS (I-A)


@dataclass
class LiveGameState:
    """A single in-game snapshot, in home-margin space."""
    event_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: int
    clock_seconds: int          # seconds left in the current period
    seconds_remaining: int      # seconds left in regulation (0 once in OT)
    state: str                  # pre | in | post
    completed: bool
    possession_home: bool | None = None
    down: int | None = None
    distance: float | None = None
    yards_to_goal: float | None = None
    home_timeouts: int | None = None
    away_timeouts: int | None = None
    is_red_zone: bool | None = None
    venue_neutral: bool = False
    market_spread: float | None = None   # ESPN "details" spread, home perspective
    market_total: float | None = None
    fetched_at: str = ""

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score

    @property
    def fraction_remaining(self) -> float:
        return max(0.0, min(1.0, self.seconds_remaining / GAME_SECONDS))

    @property
    def is_overtime(self) -> bool:
        return self.period > 4

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["margin"] = self.margin
        d["fraction_remaining"] = self.fraction_remaining
        return d


def _clock_to_seconds(display: str | None) -> int:
    if not display:
        return 0
    txt = str(display).strip()
    if ":" not in txt:
        try:
            return int(float(txt))
        except ValueError:
            return 0
    mm, _, ss = txt.partition(":")
    try:
        return int(float(mm)) * 60 + int(float(ss))
    except ValueError:
        return 0


def seconds_remaining_in_regulation(period: int, clock_seconds: int) -> int:
    """Regulation seconds left. Overtime returns 0 (handled separately)."""
    if period <= 0:
        return GAME_SECONDS
    if period > 4:
        return 0
    return int((4 - period) * QUARTER_SECONDS + clock_seconds)


class ESPNClient:
    def __init__(self, cfg: Config | None = None, cache_ttl: float | None = 0.0):
        self.cfg = cfg or CONFIG
        # cache_ttl=0 => always refetch (live data); pass a number to cache.
        self.cache_ttl = cache_ttl
        self.http = JsonClient(
            base_url=self.cfg.espn_base,
            cache_dir=self.cfg.raw_dir / "espn",
            min_interval=0.2,
            max_retries=3,
            timeout=20.0,
        )

    def scoreboard_raw(self, date: str | None = None, groups: str = FBS_GROUP,
                       limit: int = 300) -> dict:
        params = {"groups": groups, "limit": limit}
        if date:
            params["dates"] = date  # YYYYMMDD or YYYYMMDD-YYYYMMDD
        use_cache = bool(self.cache_ttl)
        return self.http.get("/scoreboard", params=params,
                             use_cache=use_cache, cache_ttl=self.cache_ttl)

    def live_states(self, date: str | None = None,
                    groups: str = FBS_GROUP) -> list[LiveGameState]:
        payload = self.scoreboard_raw(date=date, groups=groups)
        now = datetime.now(timezone.utc).isoformat()
        out: list[LiveGameState] = []
        for ev in payload.get("events", []) or []:
            try:
                out.append(self._parse_event(ev, now))
            except Exception as exc:  # noqa: BLE001
                log.debug("skip event %s: %s", ev.get("id"), exc)
        return out

    # -- parsing ----------------------------------------------------------
    @staticmethod
    def _parse_event(ev: dict, fetched_at: str) -> LiveGameState:
        comp = (ev.get("competitions") or [{}])[0]
        status = comp.get("status") or ev.get("status") or {}
        stype = status.get("type") or {}
        period = int(status.get("period") or 0)
        clock_s = _clock_to_seconds(status.get("displayClock"))

        home = away = None
        for c in comp.get("competitors") or []:
            side = c.get("homeAway")
            if side == "home":
                home = c
            elif side == "away":
                away = c
        if home is None or away is None:
            raise ValueError("missing competitors")

        def _name(c):
            t = c.get("team") or {}
            return t.get("location") or t.get("displayName") or t.get("abbreviation") or "?"

        def _score(c):
            try:
                return int(float(c.get("score") or 0))
            except (TypeError, ValueError):
                return 0

        situation = comp.get("situation") or {}
        poss_id = situation.get("possession")
        poss_home: bool | None = None
        if poss_id is not None:
            poss_home = str(poss_id) == str(home.get("id"))

        # ESPN gives yardLine as distance from the possessing team's own goal
        # in some payloads and "yard line" text in others; prefer the explicit
        # distance-to-opponent-endzone when derivable.
        ytg = situation.get("distanceToGoal")
        if ytg is None and situation.get("yardLine") is not None:
            try:
                yl = float(situation["yardLine"])
                ytg = 100.0 - yl if 0 <= yl <= 100 else None
            except (TypeError, ValueError):
                ytg = None

        odds = (comp.get("odds") or [{}])[0]
        spread = odds.get("spread")
        try:
            spread = float(spread) if spread is not None else None
        except (TypeError, ValueError):
            spread = None
        total = odds.get("overUnder")
        try:
            total = float(total) if total is not None else None
        except (TypeError, ValueError):
            total = None

        return LiveGameState(
            event_id=str(ev.get("id")),
            home_team=_name(home),
            away_team=_name(away),
            home_score=_score(home),
            away_score=_score(away),
            period=period,
            clock_seconds=clock_s,
            seconds_remaining=seconds_remaining_in_regulation(period, clock_s),
            state=str(stype.get("state") or "pre"),
            completed=bool(stype.get("completed")),
            possession_home=poss_home,
            down=_maybe_int(situation.get("down")),
            distance=_maybe_float(situation.get("distance")),
            yards_to_goal=_maybe_float(ytg),
            home_timeouts=_maybe_int(situation.get("homeTimeouts")),
            away_timeouts=_maybe_int(situation.get("awayTimeouts")),
            is_red_zone=situation.get("isRedZone"),
            venue_neutral=bool(comp.get("neutralSite", False)),
            market_spread=spread,
            market_total=total,
            fetched_at=fetched_at,
        )

    def live_frame(self, date: str | None = None) -> pd.DataFrame:
        states = self.live_states(date=date)
        if not states:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in states])


def _maybe_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _maybe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
