# CFB Analytics

College football spread modeling for Kalshi: produces a point spread **and a full
distribution over the final margin**, so "Team A wins by 4–7" can be priced as
directly as "Team A wins". Also does in-game (live) probabilities.

Several models are available and interchangeable behind one flag — ridge, random
forest, XGBoost, LightGBM, and a stacked ensemble.

**No terminal required.** It deploys to Streamlit Community Cloud, and GitHub
Actions does the data fetching for you — see **[docs/HOSTING.md](docs/HOSTING.md)**.

```
Deploy → add your free CFBD key → press Fetch → press Train
```

Prefer a terminal? The `cfb` CLI does everything the app does:

```bash
cfb synth                                  # sample data, no API key needed
cfb train --model ensemble                 # walk-forward train + save
cfb game --home "Ohio State" --away "Michigan"
```

## What it produces

`cfb game --home "Ohio State" --away "Michigan"` prints:

```
Michigan @ Ohio State
----------------------------------------------------------
  projected margin   +6.20  (home)
  fair spread        Ohio State -6.2
  projected total    52.8
  sigma              15.9
  P(Ohio State wins) 0.6478
  most likely margin +7
  80% interval       [-14, +27]

win-by buckets:                  most likely exact margins:
  Ohio State by 1-1    0.0230     margin   prob   fair_cents
  Ohio State by 2-3    0.0621          7 0.0402          4.0
  Ohio State by 4-7    0.1210          3 0.0361          3.6
  Ohio State by 8-10   0.0738         10 0.0338          3.4
  ...
```

*(layout is real; the values above are illustrative — run it on your own data)*

Every one of those numbers is a Kalshi contract you can price.

## Why the distribution is the hard part

A point spread is the easy half. What makes or breaks a "wins by exactly X"
market is the *shape*, and football margins are not smooth:

| final margin | actually happened | plain normal said | this model said |
|---|---|---|---|
| 3 | 7.7% | 4.5% | 6.9% |
| 5 | 2.2% | 4.4% | 3.4% |
| 7 | 6.7% | 4.2% | 6.9% |
| 14 | 5.3% | 3.3% | 4.6% |

*Measured on out-of-sample backtest games from the bundled **synthetic** league,
because this repo ships without real data — reproduce with `cfb synth && cfb train
&& cfb calibration`. Real FBS margins spike a little less sharply (3 and 7 each
land around 5–6% of games rather than 7%), but the phenomenon and the size of the
normal's error are the same. Re-run `cfb calibration` after `cfb fetch` to get the
real-data version of this table.*

A normal distribution underprices "wins by exactly 3" by a third or more, and
overprices "wins by exactly 5" by about as much. Scores are built out of 3s and 7s, and a margin of 0 is impossible because overtime exists.
Two independent routes to that structure are implemented, and you can flip between
them:

- **`lattice`** — a Student-t scaffold reshaped by multiplicative key-number
  weights learned from history. Fast, stable, the default.
- **`mc`** — a drive-level Monte Carlo that simulates possessions ending in 7, 3
  or 0. Slower, but reconstructs the lattice from first principles and gives the
  joint distribution of both team scores.
- **`kde`**, **`blend`**, and **`plain`** (a normal, for comparison only).

## Getting real data

Everything runs on bundled sample data out of the box. For real games you need
**one free API key**.

| Source | What for | What you must do |
|---|---|---|
| [CollegeFootballData](https://collegefootballdata.com/key) | games, lines, ratings, play-by-play | **Get a free key** |
| ESPN scoreboard | live in-game state | nothing — public, no key |
| [Kalshi](https://kalshi.com) | market prices, EV table | optional: create an API key pair |

**Hosted (no terminal):** put the key in your Streamlit app's Secrets, then use
the Setup page's Fetch and Train buttons. Add the same key as a GitHub Actions
secret and run the **Refresh data** workflow so the app boots with real data
after every restart. Full walkthrough: **[docs/HOSTING.md](docs/HOSTING.md)**.

**Locally:**

```bash
cp .env.example .env      # paste your CFBD key in
cfb setup                 # confirms what is configured
cfb fetch --seasons 2016-2025
cfb train --model ensemble
```

Step-by-step for both, including rate limits and what each endpoint gives you:
**[docs/DATA_SETUP.md](docs/DATA_SETUP.md)**.

## Commands

| Command | Does |
|---|---|
| `cfb setup` | Show config and what is still missing |
| `cfb synth` | Generate a synthetic universe (no key needed) |
| `cfb fetch --seasons 2015-2024 [--plays]` | Pull CollegeFootballData into the local store |
| `cfb train --model ensemble` | Walk-forward train the model + distribution layer, save it |
| `cfb backtest --models ridge,forest,xgboost,ensemble` | Compare models honestly |
| `cfb calibration` | CRPS, log loss, PIT, interval coverage, exact-margin accuracy |
| `cfb predict --week 12` | Price a slate |
| `cfb game --home X --away Y` | One matchup, full distribution and buckets |
| `cfb live` | Live win probabilities from the ESPN scoreboard |
| `cfb live-train` | Train the play-by-play in-game model |
| `cfb kalshi discover` / `cfb kalshi price` | Find CFB markets, rank them by EV |
| `cfb dashboard` | Streamlit UI (same app that runs on Cloud) |

Every one of those is also a button in the app, so none of it requires a shell.

## Install

**Hosted:** nothing to install — see [docs/HOSTING.md](docs/HOSTING.md).
Streamlit Cloud reads `requirements.txt` and `packages.txt` automatically.

**Locally:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"        # or: pip install -e .   for the lean install
```

Python 3.10+. The lean install gives you ridge, random forest, and everything
except XGBoost/LightGBM, the dashboard, and Kalshi signing.

## How it fits together

```
CFBD ──┐
ESPN ──┼──> data/store (parquet) ──> features ──┬──> point model  ──> pred_margin
Kalshi ┘                                        │    (ridge/xgb/rf/ensemble)   pred_total
                                                │
                                                ├──> sigma model  ──> per-game uncertainty
                                                │    (fit on OUT-OF-SAMPLE residuals)
                                                │
                                                └──> key numbers  ──> the 3/7 lattice
                                                                 │
                                    MarginDistribution <─────────┘
                                              │
                        ┌─────────────────────┼──────────────────────┐
                   spread / buckets      live model            Kalshi EV table
                                     (Brownian bridge +        (net of fees,
                                      trained PBP model)        Kelly sizing)
```

The core object is `MarginDistribution` — a PMF over integer home margins with
the query methods the markets need (`p_between`, `p_at_least`, `p_exact`,
`cover_probability`, `probability_for_strike`).

## Sign conventions

Stated once, honoured everywhere, because sign errors are the most expensive bug
available here:

- **`margin` = `home_points - away_points`.** Positive means the home team won.
- **CFBD `spread` is the home team's spread.** `-7` means home is favoured by 7.
  The code derives `market_margin = -spread` so it is directly comparable to
  `margin`.
- Everything the models emit is in home-margin space. Conversion to a specific
  team's perspective happens only at the presentation and betting layers.

## Reading the results honestly

`cfb backtest` reports out-of-sample results only — the walk-forward refits the
model before each season (or week) and never lets a game see its own result.
There is no k-fold cross-validation anywhere in this repo, because random folds
leak the future through opponent-adjusted ratings.

A few things worth internalising before you bet:

- Beating the closing line is genuinely hard. Closing spreads have a mean
  absolute error around 10.5–11 points, and a model within half a point of that
  is doing well.
- Break-even at −110 is 52.38%. On Kalshi the equivalent hurdle is your edge
  minus fees minus the bid-ask spread you have to cross.
- Anything under a few thousand graded bets is noise. A 55% ATS record over 200
  bets is a coin flip that went well.
- `cfb calibration` matters more than `cfb backtest` for the markets you are
  actually trading. A model can have a great MAE and still misprice every bucket.

## Docs

- **[docs/HOSTING.md](docs/HOSTING.md)** — deploying to Streamlit Cloud with no terminal
- **[docs/DATA_SETUP.md](docs/DATA_SETUP.md)** — exactly what to do to get live data
- **[docs/MODELING.md](docs/MODELING.md)** — how each piece works and why

## Testing

```bash
pytest -q       # 95 tests, ~60s, all on synthetic data
```

CI runs them on every push against Python 3.11 and 3.12.

Two of them carry most of the weight:

- **Leakage.** Changing one game's score must not change that game's own
  features. If that fails, every accuracy number in the repo is wrong.
- **The app runs.** `AppTest` executes the real Streamlit script and walks every
  page. When the app *is* the product and there is no terminal to debug from, a
  runtime error in it is the whole thing being down.
