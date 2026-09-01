"""Roster-continuity features for the transfer-portal era.

The team ratings learn who a team is *from results*, which works once games
have been played but leaves the preseason estimate as "last year's team,
regressed toward the mean". In the portal era that is a materially worse
assumption than it used to be: rosters turn over hard, and a team can lose 40%
of its production and replace it with proven starters or with walk-ons, and the
ratings cannot tell those apart until October.

Two things this module adds that the model previously could not see at all:

* **Who arrived.** Returning production measures departures only. Portal
  arrivals were completely invisible.
* **Who plays quarterback.** The single biggest lever in college football, and
  there was no feature for it anywhere.

**The leak trap.** The obvious way to identify a team's starting quarterback is
"whoever took the most snaps this season" -- which is only knowable once the
season is over, and would leak a season's outcome into its own week-1 game.
Everything here is instead built from *last* season's player stats plus the
*offseason* portal, both of which are genuinely known before kickoff.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Columns produced per (season, team). All are preseason-known.
ROSTER_COLUMNS = [
    "portal_in_count", "portal_out_count", "portal_net_count",
    "portal_in_rating", "portal_out_rating", "portal_net_rating",
    "portal_in_best",
    "qb_prior_ppa", "qb_departed", "qb_transfer_in", "qb_transfer_in_ppa",
    "qb_continuity",
    "returning_off", "returning_def",
]


def _norm_name(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.lower()
            .str.replace(r"[^a-z ]", "", regex=True).str.strip())


def portal_flux(portal: pd.DataFrame) -> pd.DataFrame:
    """Net talent moving in and out of each team in an offseason.

    A portal row for season S is an offseason move *into* season S, so joining
    it to season S is correct and leak-free.
    """
    if portal is None or portal.empty:
        return pd.DataFrame(columns=["season", "team"])
    p = portal.copy()
    p["rating"] = pd.to_numeric(p.get("rating"), errors="coerce")

    incoming = (p.dropna(subset=["destination"])
                .groupby(["season", "destination"])
                .agg(portal_in_count=("player_id", "size"),
                     portal_in_rating=("rating", "sum"),
                     portal_in_best=("rating", "max"))
                .reset_index().rename(columns={"destination": "team"}))
    outgoing = (p.dropna(subset=["origin"])
                .groupby(["season", "origin"])
                .agg(portal_out_count=("player_id", "size"),
                     portal_out_rating=("rating", "sum"))
                .reset_index().rename(columns={"origin": "team"}))

    out = incoming.merge(outgoing, on=["season", "team"], how="outer")
    for col in ("portal_in_count", "portal_out_count",
                "portal_in_rating", "portal_out_rating"):
        out[col] = out[col].fillna(0.0)
    out["portal_net_count"] = out["portal_in_count"] - out["portal_out_count"]
    out["portal_net_rating"] = out["portal_in_rating"] - out["portal_out_rating"]
    return out


def quarterback_continuity(player_ppa: pd.DataFrame,
                           portal: pd.DataFrame | None = None) -> pd.DataFrame:
    """Quarterback situation entering each season, from prior-season data only.

    For a team entering season S this reports:

    * ``qb_prior_ppa``       -- how good last season's primary quarterback was
    * ``qb_departed``        -- whether he entered the portal in the offseason
    * ``qb_transfer_in``     -- whether a quarterback arrived via the portal
    * ``qb_transfer_in_ppa`` -- how good that arrival was at his previous school
    * ``qb_continuity``      -- the best available estimate of who will play

    Nothing here reads season S's own statistics.
    """
    if player_ppa is None or player_ppa.empty:
        return pd.DataFrame(columns=["season", "team"])
    ppa = player_ppa.copy()
    if "position" not in ppa.columns:
        return pd.DataFrame(columns=["season", "team"])
    qbs = ppa[ppa["position"].astype(str).str.upper() == "QB"].copy()
    if qbs.empty:
        return pd.DataFrame(columns=["season", "team"])
    qbs["plays"] = pd.to_numeric(qbs.get("plays"), errors="coerce").fillna(0)
    qbs["avg_ppa_all"] = pd.to_numeric(qbs.get("avg_ppa_all"), errors="coerce")

    # Each team's primary quarterback in each season it is observed.
    primary = (qbs.sort_values("plays", ascending=False)
               .drop_duplicates(["season", "team"])
               [["season", "team", "player_id", "name", "avg_ppa_all", "plays"]])

    # Shift forward one season: what the team knew going into the next one.
    prior = primary.copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={"player_id": "prior_qb_id",
                                  "name": "prior_qb_name",
                                  "avg_ppa_all": "qb_prior_ppa",
                                  "plays": "qb_prior_plays"})
    out = prior[["season", "team", "prior_qb_id", "prior_qb_name",
                 "qb_prior_ppa", "qb_prior_plays"]].copy()

    # A lookup of every quarterback's prior-season value, for arrivals.
    value_by_id = (primary.assign(season=primary["season"] + 1)
                   .set_index(["season", "player_id"])["avg_ppa_all"])
    all_qb_prior = qbs[["season", "player_id", "name", "avg_ppa_all"]].copy()
    all_qb_prior["season"] = all_qb_prior["season"] + 1
    value_by_name = (all_qb_prior.assign(key=_norm_name(all_qb_prior["name"]))
                     .dropna(subset=["avg_ppa_all"])
                     .groupby(["season", "key"])["avg_ppa_all"].max())

    out["qb_departed"] = np.nan
    out["qb_transfer_in"] = np.nan
    out["qb_transfer_in_ppa"] = np.nan

    if portal is not None and not portal.empty and "position" in portal.columns:
        p = portal.copy()
        p["is_qb"] = p["position"].astype(str).str.upper() == "QB"
        qb_moves = p[p["is_qb"]]

        left = qb_moves.dropna(subset=["origin"])[["season", "origin", "player_id", "name"]]
        left = left.rename(columns={"origin": "team"})
        left["left_id"] = left["player_id"].astype(str)
        left["left_key"] = _norm_name(left["name"])
        out["_pid"] = out["prior_qb_id"].astype(str)
        out["_pkey"] = _norm_name(out["prior_qb_name"])
        by_id = set(zip(left["season"], left["team"], left["left_id"]))
        by_key = set(zip(left["season"], left["team"], left["left_key"]))
        out["qb_departed"] = [
            float((s, t, i) in by_id or (s, t, k) in by_key)
            for s, t, i, k in zip(out["season"], out["team"], out["_pid"], out["_pkey"])
        ]
        out = out.drop(columns=["_pid", "_pkey"])

        arrivals = qb_moves.dropna(subset=["destination"]).copy()
        if not arrivals.empty:
            arrivals = arrivals.rename(columns={"destination": "team"})
            arrivals["by_id"] = [
                value_by_id.get((s, str(i)), np.nan)
                for s, i in zip(arrivals["season"], arrivals["player_id"].astype(str))
            ]
            keys = _norm_name(arrivals["name"])
            arrivals["by_name"] = [
                value_by_name.get((s, k), np.nan)
                for s, k in zip(arrivals["season"], keys)
            ]
            # Prefer the id join; fall back to the name join, because CFBD's
            # portal feed does not always carry a player id.
            arrivals["val"] = arrivals["by_id"].fillna(arrivals["by_name"])
            agg = (arrivals.groupby(["season", "team"])
                   .agg(qb_transfer_in=("player_id", "size"),
                        qb_transfer_in_ppa=("val", "max")).reset_index())
            out = out.drop(columns=["qb_transfer_in", "qb_transfer_in_ppa"])
            out = out.merge(agg, on=["season", "team"], how="left")
            out["qb_transfer_in"] = out["qb_transfer_in"].fillna(0.0).clip(upper=3.0)

    # Best available guess at this season's quarterback value: the returning
    # starter if he stayed, otherwise the best arrival, otherwise unknown.
    stayed = out["qb_departed"].fillna(0.0) < 0.5
    out["qb_continuity"] = np.where(stayed, out["qb_prior_ppa"],
                                    out["qb_transfer_in_ppa"])
    return out.drop(columns=["prior_qb_id", "prior_qb_name"])


def build_roster_features(portal: pd.DataFrame | None,
                          player_ppa: pd.DataFrame | None,
                          returning: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (season, team) with every roster-continuity column.

    Sources are optional: whatever is missing simply does not appear, and the
    models drop columns that are entirely null rather than imputing a fiction.
    """
    frames: list[pd.DataFrame] = []
    flux = portal_flux(portal)
    if not flux.empty:
        frames.append(flux)
    qb = quarterback_continuity(player_ppa, portal)
    if not qb.empty:
        frames.append(qb)
    if returning is not None and not returning.empty:
        cols = {"returning_offense_ppa": "returning_off",
                "returning_defense_ppa": "returning_def"}
        have = [c for c in cols if c in returning.columns]
        if have:
            frames.append(returning[["season", "team", *have]]
                          .rename(columns=cols))

    if not frames:
        return pd.DataFrame(columns=["season", "team"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["season", "team"], how="outer")
    out = out.drop_duplicates(["season", "team"])
    log.info("roster features: %d team-seasons, %d columns", len(out), len(out.columns))
    return out
