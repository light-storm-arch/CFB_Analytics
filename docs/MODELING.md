# How the model works

Written to be read start to finish. Each section says what the piece does, why it
is built that way, and where it can mislead you.

---

## 0. The one thing to get right

You are pricing markets like *"Team A wins by 4–7"*. That is a question about the
**shape** of the margin distribution, not just its centre. A model that nails the
spread and assumes a normal distribution around it will misprice those buckets by
30–50% — reliably, in the same direction, every week.

So this repo treats the point estimate and the distribution as two separate
modeling problems, and spends about as much effort on the second.

---

## 1. Ratings — `cfb/features/ratings.py`

**What.** A weighted ridge regression on points scored. Each game contributes two
rows (one per scoring team); the design matrix has an offence column and a
defence column per team, plus separate home/away field-advantage terms:

```
points_scored = intercept + offence[scoring team] + defence[opponent] + field_advantage
```

**Why ridge.** Two reasons.

1. *Identifiability.* Offence and defence are only defined up to an additive
   constant — add 5 to every offence and subtract 5 from every defence and the
   fit is identical. Ridge resolves this by shrinking both toward zero.
2. *Schedule strength.* A 2–0 team that has played two cupcakes has almost no
   information in it. Ridge pulls thin résumés toward the league mean by exactly
   the right amount, where an unregularised fit would rank them first in the
   country in September.

`ridge_lambda` is roughly "how many weighted games before a team is trusted on its
own". The default of 45 pulls a team with ~45 weighted games halfway to the mean.

**Recency.** Games are weighted `exp(-ln2 · age_days / half_life)`. The 400-day
default means last season counts about half as much as this one — enough
continuity to survive September, not so much that a team is judged on last year's
roster in November.

**Why also model the total.** Because variance scales with scoring environment.
A projected 75-point shootout has a genuinely wider margin distribution than a
projected 38-point rock fight, and the sigma model needs the total to know that.

**Elo** is also computed, with a margin-of-victory multiplier and preseason
regression to the mean. It is a feature, and a canary — if the ridge ratings and
Elo ever disagree wildly, something upstream broke.

**Where it misleads.** Scores are a noisy measure of quality. Garbage-time
touchdowns, defensive scores, and kneel-downs all move the rating. `cap_points`
soft-caps blowout scoring, but if you want to go further, the honest upgrade is to
fit these ratings on drive-level or EPA data instead of final scores.

---

## 2. Features — `cfb/features/build.py`

**The rule: every feature must be computable before kickoff.** Concretely:

- Ratings are refit **once per (season, week)**, on games that finished before
  that week's first kickoff.
- Elo is a forward pass recording pregame values.
- Rolling form uses `shift(1)` within team, so a game never sees itself.
- **SP+ is joined from the prior season.** CFBD's SP+ for season Y reflects games
  played *during* Y. Joining it to Y's games leaks the season's results and is
  worth a fraudulent point of MAE.
- Recruiting talent and returning production are known before the season, so they
  join on the current season.

There is a test for this (`tests/test_features.py::test_no_leakage_from_a_games_own_result`)
that tampers with one game's score and asserts that game's own features do not
move. If it ever fails, every accuracy number in the repo is a lie.

**Market features are opt-in.** `--market` adds the closing spread and total as
features. This makes the model much more accurate and much less useful for
betting: it learns to copy the market. Keep it **off** to hunt for edges; turn it
**on** when you want the best possible prior for the in-game model.

---

## 3. Point models — `cfb/models/`

All expose the same interface and are interchangeable via `--model`:

| Model | Character |
|---|---|
| `ridge` | Interpretable, hard to overfit, usually within a hair of the best |
| `huber` | Robust — down-weights blowouts rather than chasing them |
| `elasticnet` | Sparse, for working out which features actually matter |
| `forest` | Captures interactions, no extrapolation |
| `xgboost` / `lightgbm` | Best on real data if tuned; over-fits fast if not |
| `ensemble` | Weighted blend; NNLS-stacked weights available |

**Expect the ridge to be competitive, and do not be disappointed by that.**
College football gives you ~800 FBS games a year, most of the signal is linear in
opponent-adjusted ratings, and the irreducible noise is enormous. The boosters'
defaults here are deliberately conservative — shallow trees, heavy subsampling,
strong L2 — because the failure mode is memorising blowouts, not underfitting.

Ensembling is worth more than model selection. Ridge and trees make *different*
errors, and averaging reliably picks up a fraction of a point of MAE.

---

## 4. The scale model — `cfb/distribution/sigma.py`

Predicts the standard deviation of `actual margin − predicted margin`, per game,
from predicted total, |predicted margin|, week, and how much history each team
has. Roughly 8 to 26 points depending on the matchup, versus a league-wide ~16.5.

**It is fit on out-of-sample residuals.** This is not a detail. Fitting a scale
model on in-sample residuals produces a sigma that is 10–15% too small, which
makes every tail market look like a bargain and is exactly how a small edge turns
into a losing one. `train_predictor` therefore runs the walk-forward backtest
*first* and fits the scale model on its residuals.

A single global constant then forces `E[(residual/sigma)²] = 1`, and the
degrees-of-freedom of a Student-t are fit to the standardised residuals to
capture tail thickness. Real football residuals are mildly fat-tailed; expect a
fitted `t_df` somewhere around 8–15.

---

## 5. The distribution — `cfb/distribution/margin.py`

`MarginDistribution` is a PMF over integer home margins. Three facts drive it:

1. **Margins are integers, and 0 is impossible.** No ties since overtime arrived
   in 1996. Any mass on 0 is wrong, and near a pick'em that error is large.
2. **Key numbers are real.** 3 and 7 are roughly twice as likely as their
   neighbours; 4, 10, 14, 1 and 17 also spike.
3. **The lattice sits in margin space; the location sits in strength space.**
   A game ends *3* more often than *5* whether the spread is 2 or 20.

So the PMF factorises:

```
P(margin = k)  ∝  smooth(k − mu) × lattice_weight(k)
```

The lattice weights are fit as `observed_count(k) / expected_count(k)` under the
smooth scaffold — which is exactly "what does a smooth density miss?" — then
shrunk toward 1 by how much evidence each cell has, symmetrised, and forced to 1
outside ±30 (beyond which the structure has washed out and what is left is tail
thickness, which belongs in the degrees of freedom instead).

**Moment matching.** Multiplying by lattice weights and zeroing the tie cell both
perturb the mean and variance. The scaffold's location and scale are therefore
solved so the *final* PMF has exactly the requested mean and sigma. Without this,
asking for sigma = 16 quietly returns 15, which is a systematic overconfidence
bias on every tail market.

### The five methods

| `--dist` | What it is | Use when |
|---|---|---|
| `lattice` | Student-t reshaped by learned key numbers | Default. Fast, stable. |
| `mc` | Drive-level Monte Carlo | You want the lattice from first principles, or the joint score distribution |
| `blend` | Average of the two | Hedging between them |
| `kde` | Empirical residual cloud, rescaled and shifted | You distrust the parametric shape |
| `plain` | A normal, no lattice | **Comparison only.** Never trade off this. |

### The drive simulator — `cfb/distribution/simulate.py`

Simulates possessions ending in 7, 3 or 0 via a multinomial logit on team
quality. Per-team points-per-drive are solved so the simulated mean margin equals
`mu` and mean total equals `total`; a per-simulation team-quality shock is then
scaled so the simulated sd equals the sigma model's `sigma`.

That shock placement matters. Real pregame uncertainty (~16–17 points of sd) is
larger than the drive process alone produces (~13–14) because we do not know how
good the teams are *today*. Putting that extra variance on **team quality**
preserves the scoring lattice. Bolting it on afterwards as Gaussian noise would
smear the lattice away — which would defeat the entire purpose.

---

## 6. In-game — `cfb/live/`

Two models, deliberately kept both.

**Brownian bridge (`diffusion.py`).** Treat remaining scoring as a random walk:

```
final_margin = current_margin + possession_value + Normal(mu_pre·f, (sigma_pre·f^α)²)
```

with `f` the fraction of regulation left. Pure Brownian motion implies α = 0.5;
real football sits slightly below early and above late (trailing teams speed up,
leaders bleed clock, onside kicks lump variance into the last two minutes), so α
is **fitted from play-by-play** rather than assumed.

Two refinements that matter more than they look:

- *Possession is worth points.* A tied game with the ball at your own 25 is not a
  coin flip. A linear expected-points-by-field-position term, adjusted for down,
  handles this, and it fades out as the clock hits zero.
- *The lattice never goes away.* Down 4 with five minutes left, what matters is
  whether the game ends at −4, −3, −1, +3 or +7. So the live model prefers
  simulating **only the remaining drives** and adding the current margin — which
  reproduces exactly the reachable final scores.

**Trained model (`winprob.py`).** Gradient-boosted classifier + regressor on
play-by-play snapshots: score, clock, down, distance, field position, possession,
and the pregame line. Learns what the analytic model approximates — fourth-down
leverage, late-game asymmetry in trailing-team behaviour.

Keeping both is the point. They fail differently, and a large disagreement
between them is a signal that one of them is broken.

**Where in-game misleads.** Injuries, weather, and a quarterback who just left the
game are invisible to both. If the live model says 80% and the market says 55%,
the market probably knows something you do not.

---

## 7. Backtesting — `cfb/evaluation/`

Walk-forward only. Refit per season (fast, realistic) or per week (slower, the
honest simulation of in-season operation). **No k-fold anywhere** — random folds
leak the future through opponent-adjusted ratings.

**Two different questions, two different reports:**

- `cfb backtest` — is the *point estimate* good? MAE, RMSE, straight-up accuracy,
  ATS record vs the closing line, ROI at −110.
- `cfb calibration` — is the *distribution* honest? CRPS, log loss, Brier,
  randomised PIT, interval coverage, and per-exact-margin accuracy.

The second matters more for the markets you are trading, and MAE says nothing
about it. Note that CRPS is dominated by the bulk of the distribution and will
barely move between `plain` and `lattice` — the exact-margin table is where the
difference shows up, and that is where your money is.

**Interpreting the ATS table.** `ats_by_threshold` bets only when the model
disagrees with the closing line by at least N points. Win rate should rise with
the threshold; if it does not, the model has no edge. Break-even at −110 is
52.38%. And be brutal about sample size: 200 bets tells you nothing, 2,000 tells
you a little.

---

## 8. Kalshi — `cfb/betting/`

Kalshi prices *are* probabilities, which is why it is a nicer venue to model
against than a sportsbook. Three things the EV numbers get right that naive
calculations do not:

1. **Fees are on the trade, not the profit.**
   `ceil(0.07 × contracts × P × (1−P) × 100)` cents, peaking at 1.75c near 50c.
   On a 3-cent edge that is more than half your expected profit.
2. **You have to cross the spread.** Edge is computed against the **ask**, not the
   mid. Edge against the mid is the most common way a paper edge fails to exist.
3. **Kelly is scaled down.** Default is a quarter. Your probabilities are
   themselves uncertain, and full Kelly on an uncertain edge is how bankrolls die.

Markets are mapped to probability queries via Kalshi's structured strike fields
where available, title parsing as a flagged fallback, and **`None` rather than a
guess** when neither resolves. Always run `--dry-run` and read the
`interpretation` column first.

---

## 9. Known limitations

Things this model does not know, in rough order of how much they cost you:

- **Injuries and personnel.** No quarterback availability, no depth chart, no
  transfer-portal movement. This is the single biggest gap, and it is exactly what
  moves lines in the 48 hours before kickoff.
- **Motivation.** Rivalry games, lookahead spots, bowl opt-outs, coaching changes,
  a team already eliminated in week 12.
- **Weather.** Wind in particular has a large, well-documented effect on totals.
- **Travel and altitude.** Rest days are in there; distance and altitude are not.
- **In-season improvement.** A team with a new quarterback in October is treated
  as the average of its season.
- **Pace and possession count.** Modelled crudely — the drive simulator assumes a
  league-average pace rather than a team-specific one.
- **FCS opponents.** Non-FBS teams get individual ratings from very few games.
  Pooling them into one pseudo-team would be more stable.

Each of those is a reasonable next thing to build. Injuries first.

---

## 10. Where to start reading the code

```
cfb/pipeline.py                  the Predictor object — start here
cfb/distribution/margin.py       the PMF and its market queries
cfb/features/build.py            the leak-free feature construction
cfb/features/ratings.py          the ridge ratings
cfb/evaluation/backtest.py       walk-forward
cfb/live/diffusion.py            the in-game model
cfb/betting/edge.py              EV, fees, Kelly
```
