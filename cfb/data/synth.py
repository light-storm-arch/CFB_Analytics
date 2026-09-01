"""Synthetic college-football universe.

Purpose: let the entire pipeline (features -> models -> distributions ->
backtests -> live tracker) be developed and tested without network access,
and let a new user kick the tyres before they have an API key.

Scores are generated from a *drive-level* process rather than by adding
Gaussian noise to a mean, so the synthetic margin distribution has the real
thing's lumpiness around 3 / 7 / 10 / 14 -- which is exactly the structure the
distribution layer is built to capture.  A synthetic-data test of key-number
handling is therefore meaningful rather than circular.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfb.data.cfbd_client import finalize_games

CONFERENCES = [
    "SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "American",
    "Mountain West", "Sun Belt", "MAC", "Conference USA",
]

# Multinomial-logit drive-outcome intercepts, tuned so a league-average
# offence scores ~2.1 points per drive.
A_TD, B_TD = -0.954, 0.60
A_FG, B_FG = -1.370, 0.15

DRIVES_MEAN = 12.0
HFA_PPD = 0.15          # points-per-drive edge at home => ~2.4 pt HFA
YEAR_CARRYOVER = 0.72   # rating persistence season over season
STRENGTH_LOADING = 0.90  # how much of off/def comes from shared program strength


@dataclass
class SynthConfig:
    n_teams: int = 130
    start_season: int = 2016
    n_seasons: int = 9
    games_per_team: int = 12
    strength_sd: float = 0.36
    off_sd: float = 0.22
    def_sd: float = 0.20
    market_noise: float = 1.6
    unplayed_last_week: bool = True
    # Roster churn. `*_effect` control how much of the year-over-year rating
    # change roster facts explain; deliberately partial, since coaching and
    # development explain plenty that no roster feature can see.
    portal_sd: float = 1.0
    portal_effect: float = 0.075
    qb_sd: float = 0.55
    qb_effect: float = 0.30
    seed: int = 7


def _softmax3(u_td: np.ndarray, u_fg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_td, e_fg, e_no = np.exp(u_td), np.exp(u_fg), 1.0
    z = e_td + e_fg + e_no
    return e_td / z, e_fg / z, e_no / z


def simulate_score(rng: np.random.Generator, q_off: float, n_drives: int) -> int:
    """Points scored by one team over ``n_drives`` drives at quality ``q_off``."""
    p_td, p_fg, _ = _softmax3(np.array(A_TD + B_TD * q_off), np.array(A_FG + B_FG * q_off))
    draws = rng.random(n_drives)
    n_td = int(np.sum(draws < p_td))
    n_fg = int(np.sum((draws >= p_td) & (draws < p_td + p_fg)))
    pts = 0
    if n_td:
        # 94% straight 7, 3% missed XP (6), 3% two-point conversion (8)
        kinds = rng.choice([7, 6, 8], size=n_td, p=[0.94, 0.03, 0.03])
        pts += int(kinds.sum())
    pts += 3 * n_fg
    pts += 7 * int(rng.poisson(0.11))            # defensive / special-teams TD
    return pts


class SyntheticLeague:
    def __init__(self, cfg: SynthConfig | None = None):
        self.cfg = cfg or SynthConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.teams = self._make_teams()
        self.ratings = self._make_ratings()

    # -- setup ------------------------------------------------------------
    def _make_teams(self) -> pd.DataFrame:
        n = self.cfg.n_teams
        names = [f"Team {i:03d}" for i in range(n)]
        confs = [CONFERENCES[i % len(CONFERENCES)] for i in range(n)]
        return pd.DataFrame({"team": names, "conference": confs})

    def _make_ratings(self) -> pd.DataFrame:
        """Latent offence/defence ratings in points-per-drive units.

        Both are driven by a shared program-strength factor, because in real
        college football the good teams are good at *both* phases.  That
        correlation is what makes team strength vary a lot (a 17-point margin
        sd) while game totals stay in a narrow band (~54 +/- 14): when an elite
        offence meets an elite defence the two effects cancel in the total but
        add in the margin.

        **Year-over-year carryover is roster-driven**, which is the whole point
        of this generator for the portal-era work.  A team that returns most of
        its production and keeps its quarterback carries nearly all of last
        year's rating forward; a team that is gutted and starts a freshman
        regresses hard toward its recruiting baseline.  Roster facts explain
        only part of the change -- the rest is coaching, development and luck --
        so a model that finds *all* of this signal is over-fitting.
        """
        cfg, rng = self.cfg, self.rng
        n = cfg.n_teams
        # Power conferences (first 5 in the list) get a talent bump.
        conf_bump = self.teams["conference"].map(
            lambda c: 0.20 if c in CONFERENCES[:5] else -0.16).to_numpy()

        def fresh():
            strength = conf_bump + rng.normal(0, cfg.strength_sd, n)
            return (STRENGTH_LOADING * strength + rng.normal(0, cfg.off_sd, n),
                    STRENGTH_LOADING * strength + rng.normal(0, cfg.def_sd, n))

        off, dfn = fresh()
        rows, roster_rows = [], []
        prev_qb_quality = rng.normal(0, cfg.qb_sd, n)

        for s in range(cfg.n_seasons):
            season = cfg.start_season + s
            if s == 0:
                ret_frac = rng.beta(6, 3, n)
                portal_net = np.zeros(n)
                qb_status = np.array(["returning"] * n, dtype=object)
                qb_quality = prev_qb_quality
            else:
                # --- roster churn for this offseason ---
                ret_frac = rng.beta(5, 3, n)                     # ~0.63 mean
                portal_net = rng.normal(0, cfg.portal_sd, n)
                roll = rng.random(n)
                qb_status = np.where(roll < 0.55, "returning",
                                     np.where(roll < 0.85, "transfer", "freshman"))
                qb_quality = np.where(
                    qb_status == "returning",
                    prev_qb_quality + rng.normal(0.05, 0.25, n),   # small development
                    np.where(qb_status == "transfer",
                             rng.normal(0.10, cfg.qb_sd, n),       # portal QBs skew up
                             rng.normal(-0.25, cfg.qb_sd, n)))     # freshmen skew down

                # Carryover is a function of what actually came back.
                carry = np.clip(
                    0.34 + 0.42 * ret_frac + 0.10 * (qb_status == "returning"),
                    0.20, 0.92)
                base_off, base_def = fresh()
                shock = np.sqrt(np.clip(1 - carry ** 2, 0.01, None))
                off = (carry * off + shock * base_off
                       + cfg.portal_effect * portal_net
                       + cfg.qb_effect * (qb_quality - prev_qb_quality))
                dfn = carry * dfn + shock * base_def + cfg.portal_effect * portal_net

            for i, team in enumerate(self.teams["team"]):
                rows.append({
                    "season": season, "team": team,
                    "true_off": float(off[i]), "true_def": float(dfn[i]),
                    "true_rating": float((off[i] + dfn[i]) * DRIVES_MEAN * 0.82),
                    "true_ret_frac": float(ret_frac[i]),
                    "true_portal_net": float(portal_net[i]),
                    "true_qb_status": str(qb_status[i]),
                    "true_qb_quality": float(qb_quality[i]),
                })
            prev_qb_quality = qb_quality

        self._roster_rows = roster_rows
        return pd.DataFrame(rows)

    # -- schedule ---------------------------------------------------------
    def _schedule_season(self, season: int) -> list[dict]:
        rng = self.rng
        teams = self.teams["team"].tolist()
        conf_of = dict(zip(self.teams["team"], self.teams["conference"]))
        n_games = self.cfg.games_per_team
        home_count = {t: 0 for t in teams}
        played: set[tuple[str, str]] = set()
        games: list[dict] = []

        by_conf: dict[str, list[str]] = {}
        for t in teams:
            by_conf.setdefault(conf_of[t], []).append(t)

        week = 1
        for _ in range(n_games):
            pool = teams[:]
            rng.shuffle(pool)
            used: set[str] = set()
            for t in pool:
                if t in used:
                    continue
                # Prefer an unplayed conference opponent, else anyone.
                cands = [o for o in by_conf[conf_of[t]]
                         if o not in used and o != t
                         and (min(t, o), max(t, o)) not in played]
                if not cands:
                    cands = [o for o in pool if o not in used and o != t
                             and (min(t, o), max(t, o)) not in played]
                if not cands:
                    continue
                opp = cands[int(rng.integers(len(cands)))]
                used.update({t, opp})
                played.add((min(t, opp), max(t, opp)))
                # Whoever has fewer home games so far hosts.
                if home_count[t] <= home_count[opp]:
                    home, away = t, opp
                else:
                    home, away = opp, t
                home_count[home] += 1
                neutral = bool(rng.random() < 0.03)
                games.append({
                    "season": season, "week": week,
                    "home_team": home, "away_team": away,
                    "neutral_site": neutral,
                    "conference_game": conf_of[home] == conf_of[away],
                })
            week += 1
        return games

    # -- generation -------------------------------------------------------
    def generate(self) -> dict[str, pd.DataFrame]:
        rng = self.rng
        rating_idx = self.ratings.set_index(["season", "team"])
        all_games: list[dict] = []
        gid = 1_000_000
        for s in range(self.cfg.n_seasons):
            season = self.cfg.start_season + s
            for g in self._schedule_season(season):
                h, a = g["home_team"], g["away_team"]
                rh = rating_idx.loc[(season, h)]
                ra = rating_idx.loc[(season, a)]
                hfa = 0.0 if g["neutral_site"] else HFA_PPD
                q_h = float(rh.true_off - ra.true_def) + hfa
                q_a = float(ra.true_off - rh.true_def) - hfa
                n_drives = int(np.clip(rng.poisson(DRIVES_MEAN), 8, 18))
                hp = simulate_score(rng, q_h, n_drives)
                ap = simulate_score(rng, q_a, n_drives)
                if hp == ap:  # overtime: coin flip plus a field goal or TD
                    bump = int(rng.choice([3, 6, 7, 8], p=[0.45, 0.08, 0.42, 0.05]))
                    if rng.random() < 0.5 + 0.05 * np.sign(q_h - q_a):
                        hp += bump
                    else:
                        ap += bump
                true_mu = self._true_mu(q_h, q_a, n_drives=DRIVES_MEAN)
                gid += 1
                all_games.append({
                    **g,
                    "game_id": gid,
                    "season_type": "regular",
                    "start_date": pd.Timestamp(f"{season}-08-28", tz="UTC")
                    + pd.Timedelta(days=7 * (g["week"] - 1)),
                    "home_conference": self.teams.set_index("team").loc[h, "conference"],
                    "away_conference": self.teams.set_index("team").loc[a, "conference"],
                    "home_points": float(hp), "away_points": float(ap),
                    "venue_id": None,
                    "home_pregame_elo": None, "away_pregame_elo": None,
                    "excitement_index": None,
                    "true_mu": true_mu,
                    "n_drives": n_drives,
                })
        games = finalize_games(pd.DataFrame(all_games))
        lines = self._make_lines(games)
        if self.cfg.unplayed_last_week:
            # Leave the final week unplayed so there is always a live slate to
            # predict -- otherwise `cfb predict` has nothing to show.
            last_season = games["season"].max()
            last_week = games.loc[games["season"] == last_season, "week"].max()
            mask = (games["season"] == last_season) & (games["week"] == last_week)
            games.loc[mask, ["home_points", "away_points", "margin", "total"]] = np.nan
            games.loc[mask, "completed"] = False
        return {"games": games, "lines": lines, "truth": self.ratings.copy(),
                "teams": self.teams.copy(), **self._preseason_tables(),
                **self._roster_tables()}

    def _roster_tables(self) -> dict[str, pd.DataFrame]:
        """Portal / player-PPA / returning tables in the shapes CFBD returns.

        Generated *from* the latent churn that drove the rating dynamics, so the
        feature code exercises exactly the path it will with real data and the
        signal it finds is genuinely there rather than assumed.

        Transfer quarterbacks reuse a real player id from whichever team did not
        bring its starter back, because that is how continuity is actually
        detected downstream: the same id appearing for a *different* team last
        season.  Matching on names would be fragile on real data and would make
        this generator a poor rehearsal for it.
        """
        rng = np.random.default_rng(self.cfg.seed + 3)
        POSITIONS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]
        truth = self.ratings
        seasons = sorted(truth["season"].unique())

        portal_rows, ppa_rows, returning_rows = [], [], []
        prev_qb: dict[str, str] = {}         # team -> last season's QB1 id
        next_id = [900_000]

        def new_id() -> str:
            next_id[0] += 1
            return str(next_id[0])

        for season in seasons:
            block = truth[truth["season"] == season]
            statuses = dict(zip(block["team"], block["true_qb_status"]))
            # Starters whose teams are not bringing them back become the pool
            # available to teams taking a portal quarterback.
            pool = [(prev_qb[t], t) for t, st in statuses.items()
                    if st != "returning" and t in prev_qb]
            rng.shuffle(pool)
            pool_idx = 0
            qb_now: dict[str, str] = {}

            for r in block.itertuples():
                team = r.team
                status = r.true_qb_status
                qb_from = None
                if status == "returning" and team in prev_qb:
                    qb_id, via = prev_qb[team], None
                elif status == "transfer" and pool_idx < len(pool):
                    # Never hand a team back its own departing starter.
                    while pool_idx < len(pool) and pool[pool_idx][0] == prev_qb.get(team):
                        pool_idx += 1
                    if pool_idx < len(pool):
                        qb_id, qb_from = pool[pool_idx]
                        via = "portal"
                        pool_idx += 1
                    else:
                        qb_id, via = new_id(), "freshman"
                else:
                    qb_id, via = new_id(), "freshman"
                qb_now[team] = qb_id

                qb_ppa = 0.12 + 0.18 * r.true_qb_quality + rng.normal(0, 0.04)
                plays = int(abs(rng.normal(520, 90))) + 60
                ppa_rows.append({
                    "season": int(season), "player_id": qb_id, "name": f"QB {qb_id}",
                    "position": "QB", "team": team, "plays": plays,
                    "avg_ppa_all": float(qb_ppa),
                    "total_ppa_all": float(qb_ppa * plays),
                })
                for pos in rng.choice(POSITIONS[1:], size=6, replace=True):
                    v = 0.05 + 0.10 * r.true_rating / 12.0 + rng.normal(0, 0.06)
                    pl = int(abs(rng.normal(300, 120))) + 20
                    ppa_rows.append({
                        "season": int(season), "player_id": new_id(),
                        "name": f"{pos} {next_id[0]}", "position": str(pos),
                        "team": team, "plays": pl, "avg_ppa_all": float(v),
                        "total_ppa_all": float(v * pl),
                    })

                n_in = int(np.clip(rng.poisson(5) + 1, 1, 14))
                n_out = int(np.clip(rng.poisson(5) + 1, 1, 14))
                in_rating = 0.80 + 0.06 * r.true_portal_net + rng.normal(0, 0.05, n_in)
                out_rating = 0.80 - 0.06 * r.true_portal_net + rng.normal(0, 0.05, n_out)
                for k in range(n_in):
                    is_qb = (k == 0 and via == "portal")
                    portal_rows.append({
                        "season": int(season),
                        "player_id": qb_id if is_qb else new_id(),
                        "name": f"In {season}-{team}-{k}",
                        "position": "QB" if is_qb else str(rng.choice(POSITIONS[1:])),
                        "origin": (qb_from if is_qb and qb_from else "Team OTHER"),
                        "destination": team,
                        "rating": float(np.clip(in_rating[k], 0.4, 1.0)),
                        "stars": int(np.clip(round(in_rating[k] * 5), 2, 5)),
                    })
                for k in range(n_out):
                    portal_rows.append({
                        "season": int(season), "player_id": new_id(),
                        "name": f"Out {season}-{team}-{k}",
                        "position": str(rng.choice(POSITIONS)),
                        "origin": team, "destination": "Team OTHER",
                        "rating": float(np.clip(out_rating[k], 0.4, 1.0)),
                        "stars": int(np.clip(round(out_rating[k] * 5), 2, 5)),
                    })

                ret = float(np.clip(r.true_ret_frac + rng.normal(0, 0.05), 0.05, 0.98))
                returning_rows.append({
                    "season": int(season), "team": team, "returning_ppa": ret,
                    "returning_offense_ppa": float(np.clip(ret + rng.normal(0, 0.08), 0.02, 0.99)),
                    "returning_defense_ppa": float(np.clip(ret + rng.normal(0, 0.08), 0.02, 0.99)),
                    "usage": float(np.clip(ret + rng.normal(0, 0.06), 0.02, 0.99)),
                    "percent_ppa": ret,
                })
            prev_qb = qb_now

        return {
            "portal": pd.DataFrame(portal_rows),
            "player_ppa": pd.DataFrame(ppa_rows),
            "returning": pd.DataFrame(returning_rows),
        }

    def _preseason_tables(self) -> dict[str, pd.DataFrame]:
        """Noisy stand-ins for recruiting talent, returning production and SP+."""
        rng = np.random.default_rng(self.cfg.seed + 2)
        r = self.ratings.copy()
        n = len(r)
        talent = pd.DataFrame({
            "season": r["season"], "team": r["team"],
            "talent": 700 + 55 * r["true_rating"] + rng.normal(0, 90, n),
        })
        sp = pd.DataFrame({
            "season": r["season"], "team": r["team"],
            "sp_overall": r["true_rating"] + rng.normal(0, 3.0, n),
            "sp_offense": 28 + r["true_off"] * 8 + rng.normal(0, 2.5, n),
            "sp_defense": 28 - r["true_def"] * 8 + rng.normal(0, 2.5, n),
        })
        return {"talent": talent, "sp_ratings": sp}

    @staticmethod
    def _true_mu(q_h: float, q_a: float, n_drives: float = DRIVES_MEAN) -> float:
        """Expected margin implied by the drive process (analytic, not sampled)."""
        def ppd(q):
            p_td, p_fg, _ = _softmax3(np.array(A_TD + B_TD * q), np.array(A_FG + B_FG * q))
            return float(p_td * 7.03 + p_fg * 3.0)
        return float(n_drives * (ppd(q_h) - ppd(q_a)))

    def _make_lines(self, games: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        noise = rng.normal(0, self.cfg.market_noise, len(games))
        market_margin = np.round((games["true_mu"].to_numpy() + noise) * 2) / 2
        total_true = games["total"].mean()
        ou = np.round((total_true + rng.normal(0, 4.0, len(games))) * 2) / 2
        return pd.DataFrame({
            "game_id": games["game_id"].to_numpy(),
            "season": games["season"].to_numpy(),
            "week": games["week"].to_numpy(),
            "home_team": games["home_team"].to_numpy(),
            "away_team": games["away_team"].to_numpy(),
            "provider": "synthetic",
            "spread": -market_margin,
            "spread_open": -market_margin,
            "over_under": ou,
            "over_under_open": ou,
            "home_moneyline": np.nan,
            "away_moneyline": np.nan,
            "market_margin": market_margin,
        })

    # -- play-by-play ------------------------------------------------------
    def generate_pbp(self, games: pd.DataFrame, max_games: int | None = None) -> pd.DataFrame:
        """Drive-start snapshots with clock/score, for the live win-prob model."""
        rng = np.random.default_rng(self.cfg.seed + 1)
        rating_idx = self.ratings.set_index(["season", "team"])
        sub = games if max_games is None else games.head(max_games)
        rows = []
        for g in sub.itertuples():
            rh = rating_idx.loc[(g.season, g.home_team)]
            ra = rating_idx.loc[(g.season, g.away_team)]
            hfa = 0.0 if g.neutral_site else HFA_PPD
            q = {True: float(rh.true_off - ra.true_def) + hfa,
                 False: float(ra.true_off - rh.true_def) - hfa}
            clock = 3600
            hs = as_ = 0
            home_ball = bool(rng.random() < 0.5)
            k = 0
            while clock > 0:
                period = min(4, 1 + (3600 - clock) // 900)
                rows.append({
                    "play_id": f"{g.game_id}-{k}",
                    "game_id": int(g.game_id), "season": int(g.season),
                    "week": int(g.week),
                    "home_team": g.home_team, "away_team": g.away_team,
                    "home_score": hs, "away_score": as_,
                    "period": int(period),
                    "seconds_remaining": int(clock),
                    "possession_home": home_ball,
                    "down": 1, "distance": 10.0,
                    "yards_to_goal": float(np.clip(rng.normal(72, 12), 20, 99)),
                })
                p_td, p_fg, _ = _softmax3(np.array(A_TD + B_TD * q[home_ball]),
                                          np.array(A_FG + B_FG * q[home_ball]))
                u = rng.random()
                pts = 7 if u < p_td else (3 if u < p_td + p_fg else 0)
                if home_ball:
                    hs += pts
                else:
                    as_ += pts
                clock -= int(np.clip(rng.gamma(4.0, 38.0), 15, 600))
                home_ball = not home_ball
                k += 1
            for r in rows[-k:]:
                r["final_home_score"] = hs
                r["final_away_score"] = as_
                r["final_margin"] = hs - as_
                r["home_win"] = int(hs > as_)
        return pd.DataFrame(rows)


def build_synthetic_store(cfg: SynthConfig | None = None, store=None,
                          with_pbp: bool = True, pbp_games: int | None = 2600):
    """Generate a full synthetic universe and write it into a Store."""
    from cfb.data.store import Store  # local import avoids a cycle

    store = store or Store()
    league = SyntheticLeague(cfg)
    data = league.generate()
    store.write("games", data["games"])
    store.write("lines", data["lines"])
    store.write("synth_truth", data["truth"])
    for name in ("talent", "returning", "sp_ratings", "portal", "player_ppa"):
        store.write(name, data[name])
    out = {"games": len(data["games"]), "lines": len(data["lines"])}
    if with_pbp:
        pbp = league.generate_pbp(data["games"], max_games=pbp_games)
        store.write("pbp_states", pbp)
        out["pbp_states"] = len(pbp)
    return out
