"""Domain constants used across the modeling stack."""
from __future__ import annotations

# Regulation game length in seconds (4 x 15:00 quarters).
GAME_SECONDS = 3600
HALF_SECONDS = 1800
QUARTER_SECONDS = 900

# Historical FBS-vs-FBS baselines. Overridden by fitted values wherever a
# fitted value is available; these are only priors / cold-start fallbacks.
DEFAULT_HFA = 2.35            # points, home field advantage (post-2020 era)
DEFAULT_MARGIN_SD = 16.4      # sd of (actual margin - closing spread)
DEFAULT_TOTAL = 53.0
DEFAULT_POINTS_PER_DRIVE = 2.05
DEFAULT_DRIVES_PER_TEAM = 12.2

# Margins that occur far more often than a smooth density implies, because
# football scores are built from 3s and 7s. Used to reshape the discrete PMF.
KEY_NUMBERS = (1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 17, 18, 20, 21, 24, 28)

# Column contract for the canonical games table.
GAME_COLUMNS = [
    "game_id", "season", "week", "season_type", "start_date",
    "home_team", "away_team", "home_conference", "away_conference",
    "home_points", "away_points", "neutral_site", "conference_game",
    "completed", "venue_id",
]

MARKET_COLUMNS = [
    "game_id", "season", "week", "provider",
    "spread", "over_under", "home_moneyline", "away_moneyline",
    "spread_open", "over_under_open",
]
