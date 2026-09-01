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
        add in the margin.  Drawing offence and defence independently -- or
        worse, anti-correlated -- produces 190-point games and a total model
        with nothing to learn.
        """
        cfg, rng = self.cfg, self.rng
        n = cfg.n_teams
        # Power conferences (first 5 in the list) get a talent bump.
        conf_bump = self.teams["conference"].map(
            lambda c: 0.20 if c in CONFERENCES[:5] else -0.16).to_numpy()

        def draw(prev_off=None, prev_def=None):
            strength = conf_bump + rng.normal(0, cfg.strength_sd, n)
            off = STRENGTH_LOADING * strength + rng.normal(0, cfg.off_sd, n)
            dfn = STRENGTH_LOADING * strength + rng.normal(0, cfg.def_sd, n)
            if prev_off is None:
                return off, dfn
            k = np.sqrt(1 - YEAR_CARRYOVER ** 2)
            return (YEAR_CARRYOVER * prev_off + k * off,
                    YEAR_CARRYOVER * prev_def + k * dfn)

        off, dfn = draw()
        rows = []
        for s in range(cfg.n_seasons):
            season = cfg.start_season + s
            if s > 0:
                off, dfn = draw(off, dfn)
            for i, team in enumerate(self.teams["team"]):
                rows.append({"season": season, "team": team,
                             "true_off": float(off[i]), "true_def": float(dfn[i]),
                             "true_rating": float((off[i] + dfn[i]) * DRIVES_MEAN * 0.82)})
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
                "teams": self.teams.copy(), **self._preseason_tables()}

    def _preseason_tables(self) -> dict[str, pd.DataFrame]:
        """Noisy stand-ins for recruiting talent, returning production and SP+."""
        rng = np.random.default_rng(self.cfg.seed + 2)
        r = self.ratings.copy()
        n = len(r)
        talent = pd.DataFrame({
            "season": r["season"], "team": r["team"],
            "talent": 700 + 55 * r["true_rating"] + rng.normal(0, 90, n),
        })
        returning = pd.DataFrame({
            "season": r["season"], "team": r["team"],
            "returning_ppa": np.clip(rng.beta(5, 4, n), 0.05, 0.95),
            "usage": np.clip(rng.beta(5, 4, n), 0.05, 0.95),
        })
        sp = pd.DataFrame({
            "season": r["season"], "team": r["team"],
            "sp_overall": r["true_rating"] + rng.normal(0, 3.0, n),
            "sp_offense": 28 + r["true_off"] * 8 + rng.normal(0, 2.5, n),
            "sp_defense": 28 - r["true_def"] * 8 + rng.normal(0, 2.5, n),
        })
        return {"talent": talent, "returning": returning, "sp_ratings": sp}

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
    for name in ("talent", "returning", "sp_ratings"):
        store.write(name, data[name])
    out = {"games": len(data["games"]), "lines": len(data["lines"])}
    if with_pbp:
        pbp = league.generate_pbp(data["games"], max_games=pbp_games)
        store.write("pbp_states", pbp)
        out["pbp_states"] = len(pbp)
    return out
