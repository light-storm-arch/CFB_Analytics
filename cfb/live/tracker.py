"""Live game tracker: ESPN scoreboard -> live distributions -> market comparison."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfb.data.espn_client import ESPNClient, LiveGameState
from cfb.data.teams import TeamMatcher
from cfb.distribution.margin import MarginDistribution
from cfb.live.diffusion import LiveConfig, LiveMarginModel
from cfb.live.winprob import LiveWinProbModel

log = logging.getLogger(__name__)


@dataclass
class LiveGameView:
    state: LiveGameState
    home_canonical: str | None
    away_canonical: str | None
    pregame_mu: float
    pregame_sigma: float
    pregame_total: float
    dist: MarginDistribution
    source: str

    def row(self) -> dict:
        s = self.state
        return {
            "event_id": s.event_id,
            "home": self.home_canonical or s.home_team,
            "away": self.away_canonical or s.away_team,
            "score": f"{s.home_score}-{s.away_score}",
            "margin": s.margin,
            "period": s.period,
            "clock": f"{s.clock_seconds // 60}:{s.clock_seconds % 60:02d}",
            "secs_left": s.seconds_remaining,
            "poss": ("HOME" if s.possession_home else "AWAY")
                    if s.possession_home is not None else "",
            "pregame_mu": round(self.pregame_mu, 2),
            "live_mean": round(self.dist.mean(), 2),
            "live_spread_home": round(-self.dist.mean(), 1),
            "p_home_win": round(self.dist.p_home_win(), 4),
            "p_away_win": round(self.dist.p_away_win(), 4),
            "live_sd": round(self.dist.sd(), 2),
            "source": self.source,
        }


class LiveTracker:
    """Fetch live states and price them against a pregame prior."""

    def __init__(
        self,
        pregame: pd.DataFrame | None = None,
        live_model: LiveMarginModel | None = None,
        trained_model: LiveWinProbModel | None = None,
        espn: ESPNClient | None = None,
        default_sigma: float = 16.5,
        default_total: float = 54.0,
    ):
        self.pregame = pregame if pregame is not None else pd.DataFrame()
        self.live_model = live_model or LiveMarginModel(LiveConfig())
        self.trained_model = trained_model
        self.espn = espn or ESPNClient()
        self.default_sigma = default_sigma
        self.default_total = default_total
        teams: list[str] = []
        if not self.pregame.empty:
            teams = sorted(set(self.pregame.get("home_team", pd.Series(dtype=str)))
                           | set(self.pregame.get("away_team", pd.Series(dtype=str))))
        self.matcher = TeamMatcher(teams) if teams else None

    # -- prior lookup ------------------------------------------------------
    def _prior(self, state: LiveGameState) -> tuple[str | None, str | None,
                                                    float, float, float, str]:
        home = self.matcher.match(state.home_team) if self.matcher else None
        away = self.matcher.match(state.away_team) if self.matcher else None
        if home and away and not self.pregame.empty:
            hit = self.pregame[(self.pregame["home_team"] == home)
                               & (self.pregame["away_team"] == away)]
            if not hit.empty:
                r = hit.iloc[-1]
                mu = float(r.get("pred_margin", np.nan))
                sigma = float(r.get("sigma", self.default_sigma))
                total = float(r.get("pred_total", self.default_total))
                if np.isfinite(mu):
                    return home, away, mu, sigma, total, "model"
        # Fall back to the book line ESPN ships with the scoreboard.
        if state.market_spread is not None:
            total = state.market_total if state.market_total else self.default_total
            return home, away, -float(state.market_spread), self.default_sigma, \
                float(total), "espn_line"
        return home, away, 0.0, self.default_sigma, self.default_total, "flat_prior"

    # -- main --------------------------------------------------------------
    def views(self, date: str | None = None, only_live: bool = True,
              method: str | None = None) -> list[LiveGameView]:
        out = []
        for st in self.espn.live_states(date=date):
            if only_live and st.state != "in":
                continue
            home, away, mu, sigma, total, source = self._prior(st)
            dist = self.live_model.distribution(
                current_margin=st.margin,
                seconds_remaining=st.seconds_remaining,
                mu_pregame=mu, sigma_pregame=sigma, total_pregame=total,
                possession_home=st.possession_home,
                yards_to_goal=st.yards_to_goal,
                down=st.down, distance=st.distance,
                period=st.period, method=method,
            )
            out.append(LiveGameView(st, home, away, mu, sigma, total, dist, source))
        return out

    def frame(self, date: str | None = None, only_live: bool = True) -> pd.DataFrame:
        views = self.views(date=date, only_live=only_live)
        if not views:
            return pd.DataFrame()
        return pd.DataFrame([v.row() for v in views])

    def trained_frame(self, date: str | None = None, only_live: bool = True
                      ) -> pd.DataFrame:
        """Same slate scored by the trained play-by-play model, for comparison."""
        if self.trained_model is None:
            return pd.DataFrame()
        views = self.views(date=date, only_live=only_live)
        if not views:
            return pd.DataFrame()
        rows = []
        for v in views:
            s = v.state
            rows.append({
                "home_score": s.home_score, "away_score": s.away_score,
                "seconds_remaining": s.seconds_remaining, "period": s.period,
                "down": s.down, "distance": s.distance,
                "yards_to_goal": s.yards_to_goal,
                "possession_home": s.possession_home,
                "pregame_mu": v.pregame_mu, "pregame_total": v.pregame_total,
            })
        states = pd.DataFrame(rows)
        pred = self.trained_model.predict(states)
        out = pd.DataFrame([v.row() for v in views])
        out["trained_p_home_win"] = pred["p_home_win"].to_numpy()
        out["trained_final_margin"] = pred["pred_final_margin"].round(2).to_numpy()
        out["wp_gap"] = (out["trained_p_home_win"] - out["p_home_win"]).round(4)
        return out
