"""Map Kalshi markets onto probability queries against a margin distribution.

Kalshi's structured fields (``strike_type`` + ``floor_strike``/``cap_strike``)
are the reliable path and are always preferred.  Titles are parsed only as a
fallback, because Kalshi renames series between seasons and not every series
populates the structured fields.

**Check the mapping before you trade it.**  Run ``cfb kalshi discover`` and then
``cfb kalshi price --dry-run`` and read the ``interpretation`` column: it states
in words what probability each row is being priced as.  A silently inverted
side is the single most expensive bug available in this codebase, so the code
returns ``None`` rather than guessing whenever it cannot resolve a market
confidently.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfb.data.teams import TeamMatcher
from cfb.distribution.margin import MarginDistribution

log = logging.getLogger(__name__)

# --- title patterns (fallback only) ------------------------------------- #
RE_MORE_THAN = re.compile(r"(?:by (?:more than|over)|by at least|wins by more than)\s+"
                          r"(\d+(?:\.\d+)?)", re.I)
RE_AT_LEAST = re.compile(r"by at least\s+(\d+(?:\.\d+)?)", re.I)
RE_RANGE = re.compile(r"by\s+(\d+)\s*(?:-|to|–|—)\s*(\d+)", re.I)
RE_FEWER = re.compile(r"by (?:fewer than|less than|under)\s+(\d+(?:\.\d+)?)", re.I)
RE_EXACT = re.compile(r"by exactly\s+(\d+)", re.I)
RE_VS = re.compile(r"\s+(?:vs\.?|v\.?|at|@|versus)\s+", re.I)


@dataclass
class MarketSpec:
    """What probability question a market is asking."""
    kind: str                    # winner | spread | bucket | exact | unknown
    team: str | None             # the team the YES side refers to
    strike_type: str | None = None
    floor: float | None = None
    cap: float | None = None
    interpretation: str = ""
    confident: bool = True

    def describe(self) -> str:
        return self.interpretation


def parse_market(market: dict | pd.Series) -> MarketSpec:
    """Work out what a Kalshi market's YES side pays on."""
    g = (lambda k, d=None: market.get(k, d) if isinstance(market, dict)
         else market.get(k, d))
    title = str(g("title", "") or "")
    yes_sub = str(g("yes_sub_title", "") or "")
    subtitle = str(g("subtitle", "") or "")
    strike_type = g("strike_type")
    floor = _f(g("floor_strike"))
    cap = _f(g("cap_strike"))
    # Search the most specific field first; concatenating them lets a mascot
    # or a subtitle swallow an anchor and mis-parse the market.
    text = " | ".join(x for x in (title, subtitle, yes_sub) if x)

    team = yes_sub.strip() or None
    if team and RE_VS.search(team):
        team = None  # a matchup label, not a single team

    # 1. Structured strikes -- the reliable path.
    if strike_type:
        st = str(strike_type).lower()
        if st == "between" and floor is not None and cap is not None:
            return MarketSpec("bucket", team, st, floor, cap,
                              f"{team or 'YES side'} wins by {floor:g}-{cap:g}")
        if st in ("greater", "greater_or_equal") and floor is not None:
            word = "more than" if st == "greater" else "at least"
            return MarketSpec("spread", team, st, floor, None,
                              f"{team or 'YES side'} wins by {word} {floor:g}")
        if st in ("less", "less_or_equal") and cap is not None:
            word = "fewer than" if st == "less" else "at most"
            return MarketSpec("spread", team, st, None, cap,
                              f"{team or 'YES side'} margin {word} {cap:g}")

    # 2. Title parsing -- best effort, flagged as not confident.
    m = RE_RANGE.search(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return MarketSpec("bucket", team, "between", lo, hi,
                          f"{team or 'YES side'} wins by {lo:g}-{hi:g}", confident=False)
    m = RE_AT_LEAST.search(text)
    if m:
        return MarketSpec("spread", team, "greater_or_equal", float(m.group(1)), None,
                          f"{team or 'YES side'} wins by at least {m.group(1)}",
                          confident=False)
    m = RE_MORE_THAN.search(text)
    if m:
        return MarketSpec("spread", team, "greater", float(m.group(1)), None,
                          f"{team or 'YES side'} wins by more than {m.group(1)}",
                          confident=False)
    m = RE_FEWER.search(text)
    if m:
        return MarketSpec("spread", team, "less", None, float(m.group(1)),
                          f"{team or 'YES side'} wins by fewer than {m.group(1)}",
                          confident=False)
    m = RE_EXACT.search(text)
    if m:
        k = float(m.group(1))
        return MarketSpec("exact", team, "between", k, k,
                          f"{team or 'YES side'} wins by exactly {k:g}", confident=False)

    # 3. A plain winner market: the YES side names one team and nothing else.
    if team and not re.search(r"\d", yes_sub):
        return MarketSpec("winner", team, None, None, None, f"{team} wins")

    return MarketSpec("unknown", team, None, None, None,
                      "could not interpret this market", confident=False)


def probability_for(spec: MarketSpec, dist: MarginDistribution,
                    home_team: str, away_team: str,
                    matcher: TeamMatcher | None = None) -> float | None:
    """Evaluate a parsed market against a home-margin distribution.

    Returns ``None`` when the referenced team cannot be resolved -- never a
    coin-flip guess, because a flipped side turns a positive edge negative.
    """
    if spec.kind == "unknown":
        return None
    side_home = _resolve_side(spec.team, home_team, away_team, matcher)
    if side_home is None:
        return None
    d = dist if side_home else dist.flip()

    if spec.kind == "winner":
        return d.p_home_win()
    if spec.kind == "exact" and spec.floor is not None:
        return d.p_exact(int(spec.floor))
    if spec.kind == "bucket" and spec.floor is not None and spec.cap is not None:
        return d.p_between(spec.floor, spec.cap)
    if spec.kind == "spread":
        return d.probability_for_strike(spec.strike_type, spec.floor, spec.cap)
    return None


def _resolve_side(team: str | None, home_team: str, away_team: str,
                  matcher: TeamMatcher | None) -> bool | None:
    """True if the market's team is the home team, False if away, None if unclear."""
    if not team:
        return None
    if matcher is not None:
        canon = matcher.match(team)
        if canon == home_team:
            return True
        if canon == away_team:
            return False
    local = TeamMatcher([home_team, away_team], cutoff=0.75)
    canon = local.match(team)
    if canon == home_team:
        return True
    if canon == away_team:
        return False
    return None


def split_matchup(title: str) -> tuple[str, str] | None:
    """Pull two team names out of an event title like 'Michigan vs Ohio State'."""
    if not title:
        return None
    parts = RE_VS.split(str(title).strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = (p.strip(" ?.:") for p in parts)
    # Strip a leading "Will ".
    left = re.sub(r"^will\s+", "", left, flags=re.I).strip()
    return (left, right) if left and right else None


def attach_games(markets: pd.DataFrame, slate: pd.DataFrame,
                 matcher: TeamMatcher | None = None) -> pd.DataFrame:
    """Attach each market to a game on the slate via its event title.

    ``slate`` needs ``game_id``, ``home_team``, ``away_team``.  Kalshi event
    titles list the matchup but not reliably in home/away order, so both
    orderings are tried and the frame records which one hit.
    """
    if markets.empty or slate.empty:
        return markets.assign(game_id=pd.NA, game_label="")
    teams = sorted(set(slate["home_team"]) | set(slate["away_team"]))
    matcher = matcher or TeamMatcher(teams)
    index: dict[frozenset, tuple] = {}
    for r in slate.itertuples():
        index[frozenset({r.home_team, r.away_team})] = (r.game_id, r.home_team, r.away_team)

    # A team plays once a week, so naming one team is usually enough to
    # identify the game -- but only when it is unambiguous on this slate.
    by_team: dict[str, list] = {}
    for r in slate.itertuples():
        for t in (r.home_team, r.away_team):
            by_team.setdefault(t, []).append((r.game_id, r.home_team, r.away_team))

    gids, labels = [], []
    for r in markets.itertuples():
        gid, label = pd.NA, ""
        pair = (split_matchup(getattr(r, "title", ""))
                or split_matchup(getattr(r, "event_title", "") or ""))
        if pair:
            a, b = (matcher.match(pair[0]), matcher.match(pair[1]))
            if a and b:
                hit = index.get(frozenset({a, b}))
                if hit:
                    gid, label = hit[0], f"{hit[2]} @ {hit[1]}"
        if pd.isna(gid):
            for field in ("yes_sub_title", "title", "event_title"):
                raw = getattr(r, field, "") or ""
                canon = matcher.match(_leading_team(str(raw)))
                hits = by_team.get(canon or "", [])
                if len(hits) == 1:
                    gid, label = hits[0][0], f"{hits[0][2]} @ {hits[0][1]}"
                    break
        gids.append(gid)
        labels.append(label)
    out = markets.copy()
    out["game_id"] = gids
    out["game_label"] = labels
    return out


def price_markets(
    markets: pd.DataFrame,
    dists: dict,
    slate: pd.DataFrame,
    matcher: TeamMatcher | None = None,
) -> pd.DataFrame:
    """Add ``model_prob`` and a human-readable ``interpretation`` to each market.

    ``dists`` maps game_id -> MarginDistribution.  Rows that cannot be mapped
    keep a null probability and say why in ``skip_reason`` rather than being
    dropped, so you can see what the model is *not* covering.
    """
    if markets.empty:
        return markets
    if "game_id" not in markets.columns:
        markets = attach_games(markets, slate, matcher)
    lookup = slate.set_index("game_id")[["home_team", "away_team"]].to_dict("index")

    probs, interps, kinds, reasons, confident = [], [], [], [], []
    for r in markets.itertuples():
        spec = parse_market(r._asdict() if hasattr(r, "_asdict") else {})
        kinds.append(spec.kind)
        interps.append(spec.interpretation)
        confident.append(spec.confident)
        gid = getattr(r, "game_id", None)
        if gid is None or (isinstance(gid, float) and np.isnan(gid)) or pd.isna(gid):
            probs.append(np.nan)
            reasons.append("no matching game on the slate")
            continue
        dist = dists.get(gid) or dists.get(int(gid)) if dists else None
        teams = lookup.get(gid) or lookup.get(int(gid))
        if dist is None or teams is None:
            probs.append(np.nan)
            reasons.append("no distribution for this game")
            continue
        p = probability_for(spec, dist, teams["home_team"], teams["away_team"], matcher)
        probs.append(np.nan if p is None else float(p))
        reasons.append("" if p is not None else
                       f"could not resolve market ({spec.kind})")
    out = markets.copy()
    out["market_kind"] = kinds
    out["interpretation"] = interps
    out["parse_confident"] = confident
    out["model_prob"] = probs
    out["skip_reason"] = reasons
    return out


def _leading_team(text: str) -> str:
    """Strip the market-language tail so only the team name is left."""
    t = re.sub(r"^\s*will\s+", "", str(text), flags=re.I)
    t = re.split(r"\b(?:wins?|to win|beats?|by|margin|vs\.?|at|@)\b", t, maxsplit=1,
                 flags=re.I)[0]
    return t.strip(" ?.:-")


def _f(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
