# Getting live data

Everything in this repo runs on synthetic data with no credentials
(`cfb synth`). This page covers what you need to do to point it at real games.

There is exactly **one required key** — CollegeFootballData — and it is free.

---

## 1. CollegeFootballData (required)

This is the backbone: schedules, scores, betting lines, advanced stats,
recruiting, and play-by-play, going back decades.

**What you need to do:**

1. Go to **https://collegefootballdata.com/key**
2. Enter your email. The key arrives by email in a minute or two.
3. Copy `.env.example` to `.env` and paste it in:

   ```bash
   cp .env.example .env
   ```
   ```
   CFBD_API_KEY=your_key_here
   ```

4. Verify:
   ```bash
   cfb setup          # should show CFBD_API_KEY ✓
   cfb fetch --seasons 2023
   ```

**About rate limits.** The free tier is generous for everything except
play-by-play. If you hit limits, the client already backs off and retries, and
every response is cached on disk under `data/raw/cfbd/`, so re-running a fetch
costs nothing. CFBD is a one-person passion project funded by
[Patreon](https://www.patreon.com/collegefootballdata) — if you end up leaning on
it, the paid tiers raise the limits and are worth it.

**What to pull, and how long it takes:**

```bash
# The core pull. ~10 seasons, a few minutes.
cfb fetch --seasons 2015-2024

# Play-by-play as well. One request per week per season, so this is slow
# (tens of minutes) and large. Only needed for the trained in-game model.
cfb fetch --seasons 2019-2024 --plays
```

| Endpoint | Table | Used for |
|---|---|---|
| `/games` | `games` | scores, schedule — the spine of everything |
| `/lines` | `lines` | closing spreads/totals from multiple books |
| `/ratings/sp` | `sp_ratings` | SP+ — joined from the **prior** season only |
| `/talent` | `talent` | recruiting composite (known preseason, safe to use) |
| `/player/returning` | `returning` | returning production (known preseason) |
| `/stats/season/advanced` | `advanced_season` | efficiency stats |
| `/games/teams` | `team_games` | per-game team box scores |
| `/plays` | `plays` | play-by-play, for the trained live model |

How much history? **8–10 seasons is the sweet spot.** More is not automatically
better: the sport changed materially around 2014 (pace) and 2021 (transfer portal,
NIL), and the recency weighting already down-weights old games. Start with
2015–2024.

> **A note on 2020.** The COVID season had partial conference schedules, missing
> non-conference games, and unusual home-field effects. It is not excluded
> automatically. If your backtest looks strange around 2020, that is why —
> consider `--seasons 2015-2019,2021-2024`.

---

## 2. ESPN (nothing to do)

The live in-game tracker reads ESPN's public scoreboard endpoint. No key, no
account, no signup. It gives score, clock, quarter, down, distance, field
position, possession, and often the current book line.

```bash
cfb live                    # games in progress right now
cfb live --date 20241123    # a specific Saturday
```

If ESPN is unreachable the command says so rather than silently returning stale
numbers. This is an undocumented endpoint — it is stable in practice but ESPN
owes nobody notice if they change it.

---

## 3. Kalshi (optional — needed for the EV table)

Only required if you want the model's prices compared against live Kalshi markets.
**This client is read-only. It never places, modifies, or cancels an order.**

**What you need to do:**

1. Log in to Kalshi → **Account → API Keys → Create New API Key**.
2. Kalshi shows an **Access Key ID** and downloads a **private key `.pem`**.
   The private key is shown **once** — save it somewhere safe.
3. Add both to `.env`:

   ```
   KALSHI_ACCESS_KEY=your-access-key-id
   KALSHI_PRIVATE_KEY_PATH=/absolute/path/to/kalshi-private-key.pem
   ```

4. Install the signing dependency and check it:
   ```bash
   pip install -e ".[kalshi]"
   cfb kalshi discover
   ```

**Find the current series ticker first.** Kalshi renames and reshuffles its
college football series between seasons, so nothing is hardcoded. `cfb kalshi
discover` scans open events and prints the ones that look like college football,
with their series tickers. Then:

```bash
cfb kalshi price --series <TICKER> --dry-run     # ALWAYS do this first
cfb kalshi price --series <TICKER> --bankroll 500 --kelly 0.25
```

### Read `--dry-run` before you trade

`--dry-run` prints an `interpretation` column stating, in words, what probability
each market is being priced as — *"Michigan wins by at least 7"*, *"Ohio State
wins by 1-3"*. **Read it.** A market resolved from the wrong side turns a positive
edge into a negative one of the same size, and it is the single most expensive
mistake available in this codebase.

The `parse_confident` column tells you which reading came from Kalshi's structured
strike fields (trustworthy) versus from parsing the title text (best effort). Be
sceptical of the latter. Markets that cannot be resolved are shown with a null
probability and a reason, never a guess.

### About the EV numbers

- Prices used are **asks**, not mids — you have to cross the spread to trade.
- EV is **net of Kalshi's fee**: `ceil(0.07 × contracts × P × (1−P) × 100)` cents,
  which peaks at 1.75c per contract near 50c. On a 3-cent edge that is over half
  your expected profit.
- Stakes are **fractional Kelly** (default a quarter). Full Kelly on a model whose
  probabilities are themselves uncertain is how bankrolls die.

---

## 4. Team names

Three sources, three naming conventions. CFBD says `Miami`, ESPN says
`Miami (FL)`, Kalshi says whatever fits the title. `cfb.data.teams.TeamMatcher`
handles exact → normalised → alias → fuzzy matching, and returns `None` rather
than guessing when it is not confident.

To teach it a mapping permanently, create `data/aliases.json`:

```json
{
  "Louisiana Ragin Cajuns": "Louisiana",
  "Hawaii Rainbow Warriors": "Hawai'i"
}
```

---

## Suggested first session

```bash
cp .env.example .env                       # paste your CFBD key
pip install -e ".[all]"
cfb setup                                  # confirm the key is picked up

cfb fetch --seasons 2015-2024              # a few minutes
cfb train --model ensemble                 # walk-forward train + save
cfb backtest --models ridge,forest,xgboost,ensemble
cfb calibration                            # is the distribution honest?

cfb predict --week 12                      # this week's slate
cfb game --home "Ohio State" --away "Michigan"
cfb dashboard                              # the visual version
```

Then, when you want the in-game model:

```bash
cfb fetch --seasons 2021-2024 --plays      # slow
cfb live-train
cfb live --trained
```

---

## Where things live

```
data/
  raw/cfbd/       cached API responses (safe to delete; costs a refetch)
  raw/espn/
  processed/      the parquet store: games, lines, sp_ratings, plays, ...
  aliases.json    your team-name overrides
artifacts/
  default/        the trained predictor (model, sigma, key numbers, OOS frame)
  live_winprob.joblib
```

Both are gitignored. `CFB_DATA_DIR` and `CFB_ARTIFACTS_DIR` override the paths.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CFBD_API_KEY is not set` | `.env` missing or in the wrong directory. `cfb setup` shows what is loaded. |
| 401 from CFBD | Key typo, or a stray quote in `.env`. |
| 429 from CFBD | Rate limited. It retries automatically; fetch fewer seasons at once. |
| `cfb train` says "no out-of-sample games" | Not enough history. Fetch more seasons or lower `min_train_games`. |
| Kalshi 401 | Check `KALSHI_PRIVATE_KEY_PATH` is the `.pem` itself and the key is still active. |
| `cfb kalshi price` finds no markets | Series ticker is stale — re-run `cfb kalshi discover`. |
| Live tracker shows no games | Nothing in progress. `--all-games` includes scheduled and finished. |
