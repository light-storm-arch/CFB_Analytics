# Hosting this on Streamlit Community Cloud

For running the whole thing without ever opening a terminal.

Two pieces do the work:

| Piece | Role |
|---|---|
| **Streamlit Community Cloud** | Runs the app: slates, distributions, backtests, live tracking, Kalshi pricing |
| **GitHub Actions** | Your terminal. Fetches data on GitHub's machines and commits it back to the repo |

---

## 1. Deploy the app (about two minutes)

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `light-storm-arch/CFB_Analytics`
   - **Branch:** `main` (or `claude/college-football-betting-swn62m` to try it before merging)
   - **Main file path:** `streamlit_app.py`
4. Click **Deploy**.

First boot takes a few minutes while it installs dependencies. When it comes up
you'll land on the **Setup** page with the bundled sample data already loaded —
a synthetic league, clearly labelled as such, so you can click through every
screen before wiring up real data.

---

## 2. Add your CFBD key

In the app: **⋮ (top right) → Settings → Secrets**, and paste:

```toml
CFBD_API_KEY = "your-key-here"
```

Save. The app restarts with the key loaded. A free key takes about a minute to
get: **https://collegefootballdata.com/key**

Secrets live in Streamlit's own encrypted store, never in the repo. There is a
copyable template at `.streamlit/secrets.toml.example`.

---

## 3. Load real data

Two ways. **Do both** — they solve different problems.

### In the app (quick, but temporary)

**Setup → 1 · Load data → Fetch from CollegeFootballData.** Pick a season range
(8–10 seasons is the sweet spot) and go. Takes a few minutes with a progress bar.

Then **Setup → 2 · Train a model → Train**.

> This works immediately, but **Streamlit Cloud's filesystem is wiped whenever
> the app restarts** — a redeploy, a settings change, or just going idle. When
> that happens you're back to the sample data and have to fetch again. Which is
> why you also want:

### Via GitHub Actions (permanent)

This is the part that makes the hosted app actually stable.

1. In GitHub: **Settings → Secrets and variables → Actions → New repository
   secret**. Name it `CFBD_API_KEY`, paste the same key.
2. Go to the **Actions** tab → **Refresh data** → **Run workflow**.
   - `seasons`: `2016-2025`
   - `replace`: **tick this the first time** (it clears the sample data)
3. Wait for it to go green — around 5–10 minutes.

The workflow fetches on GitHub's machines and commits the parquet store back to
the repo. Streamlit Cloud notices the commit, redeploys, and now the app **boots
with real data already loaded**. It re-runs Tuesdays and Sundays during the
season, so results and lines stay current on their own.

You still press **Train** in the app after a data refresh — models aren't
committed (a random forest pickle is ~20 MB and pickles break across library
versions). Ridge trains in about ten seconds; the full ensemble takes about a
minute.

---

## 4. Optional: Kalshi

Only needed for the EV table. **Read-only — the app never places, modifies, or
cancels an order.**

Kalshi → **Account → API Keys → Create New API Key**. You get an Access Key ID
and a private `.pem` **shown once**. In Streamlit Secrets:

```toml
KALSHI_ACCESS_KEY = "your-access-key-id"
KALSHI_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
...paste the whole .pem here...
-----END RSA PRIVATE KEY-----"""
```

Paste the key *contents*, not a path — there's no filesystem you control on
Cloud. The app writes it to a temp file at startup for the signing library.

Then: **Kalshi page → Discover** to find the current series ticker (Kalshi
renames them between seasons, so nothing is hardcoded), then **Price markets**
with **dry run** ticked and read the `interpretation` column before trading
anything.

---

## Things that will bite you

**Your app is public by default.** Anyone with the URL can see your model's
numbers and its disagreements with the market. If that bothers you, Streamlit
Cloud lets you restrict viewers: **Settings → Sharing → "Only specific people
can view this app"**. Note that a public *app* and a public *repo* are separate
settings — and the repo has to be readable by Streamlit either way.

**The filesystem is ephemeral.** Covered above. Anything the app writes —
fetched data, trained models — is gone on the next restart. The committed
parquet store is the only thing that survives, which is what step 3 is for.

**Memory is capped at about 1 GB.** Enough for everything here, but:
- Ridge is the default for a reason; it's fast and light.
- The ensemble (600-tree forest) is the heaviest thing in the app. If it gets
  killed mid-train, use `ridge` or `xgboost` instead.
- Don't fetch 25 seasons. 8–10 is better modeling *and* better on memory.
- If the app gets wedged, **⋮ → Reboot app** clears everything.

**Training happens on every cold boot.** First visit after a restart spends ten
seconds or so training ridge. That's the trade for not committing model files.

**A "Refresh data" run that fails on the key.** The workflow checks for
`CFBD_API_KEY` first and tells you exactly where to add it. GitHub Actions
secrets and Streamlit secrets are two separate places — you need the key in
both.

---

## Alternatives, briefly

- **GitHub Pages won't work.** It serves static files only; no Python runs
  there. It could host a pre-rendered snapshot of numbers, but not the app.
- **Hugging Face Spaces** runs the same Streamlit app with a persistent disk on
  paid tiers, if the ephemeral filesystem becomes annoying.
- **Running locally** is still fully supported and faster for backtests — the
  `cfb` CLI does everything the app does. See the README.

---

## Troubleshooting

| Symptom | What's happening |
|---|---|
| App shows synthetic data after a restart | Expected — the filesystem was wiped. Set up the Actions refresh (step 3). |
| "No CFBD API key configured" | Key missing from **Streamlit** secrets (the Actions secret is separate). |
| Fetch button greyed out | Same — the app only enables it when it can see a key. |
| Deploy fails installing packages | Check the deploy logs. `libgomp1` for LightGBM is handled by `packages.txt`. |
| App restarts mid-train | Out of memory. Train `ridge` instead of `ensemble`, or fetch fewer seasons. |
| Actions run fails on push | Another commit landed first; the workflow retries with rebase four times. Just re-run it. |
| Kalshi page returns 401 | Access key or `.pem` wrong, or the key was revoked. |
| Live page shows `flat_prior` | No pregame number for that game — usually because the store has no upcoming games, or team names didn't match. |
