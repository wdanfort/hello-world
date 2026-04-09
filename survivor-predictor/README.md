# 🏝️ Survivor Prediction Model

An LLM-powered prediction market model for reality TV. Generates per-contestant win and elimination probabilities after each episode, compares them against Kalshi market odds, and flags mispricings.

Currently tracking: **Survivor 50**

---

## Architecture

```
Reddit / YouTube / Manual Paste
        ↓
Episode Context Builder (src/ingest/)
        ↓
Claude Sonnet — Episode Analysis (src/analysis/)
        ↓
Bayesian Prediction Engine (src/prediction/)
        ↓
Kalshi Market Comparison (src/market/)
        ↓
CLI Output + Streamlit Dashboard + Social Posts (src/output/, dashboard/)
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env

# Run for the latest episode (auto-detects episode number)
python -m src.main --show survivor --season 50 --episode auto --sources reddit,youtube

# Manually paste a recap (useful for first run)
python -m src.main --episode 1 --paste "Your recap text here..."

# Provide manual market odds if Kalshi API isn't configured
python -m src.main --episode 5 --market-odds '{"Name": 0.35, "Other Name": 0.12}'

# Re-run if already processed
python -m src.main --episode 5 --force
```

---

## Backfill (Calibration on Completed Seasons)

```bash
# Backfill full Survivor 49 season
python -m src.backfill.runner --season 49 --source reddit --comment-window-hours 24

# Results saved to backfill/survivor/s49/
# Calibration summary: backfill/survivor/s49/calibration_summary.json
```

> **First:** populate `data/survivor/s49/contestants.json` and `data/survivor/s49/outcomes.json`
> from the [Survivor wiki](https://survivor.fandom.com/wiki/Survivor_49).

---

## Bet Simulator

```bash
python -m src.simulation.simulator --show survivor --season 49 --bankroll 100
```

Uses Kelly criterion (fractional) on model vs Kalshi odds. Requires:
- `data/survivor/s49/predictions/` (from backfill)
- `data/survivor/s49/market/` (Kalshi snapshots)
- `data/survivor/s49/outcomes.json`

---

## Live Mode (During Episode Airing)

```bash
python -m src.live.live_runner --show survivor --season 50 --episode 8
```

Interactive CLI — enter act-break summaries, get real-time updates using Claude Haiku.
Kalshi odds polled every 60 seconds in the background.

---

## Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

**Tabs:**
1. Current Odds — win probability bar chart, market comparison
2. Elimination Forecast — next boot predictions
3. Trends — win probability and edit visibility over episodes
4. Track Record — calibration accuracy vs market
5. Episode Analysis — per-episode LLM scores and narrative
6. Bet Simulator — Kelly bankroll simulation
7. Live Mode Replay — Kalshi timeline animation

**Hosting:** Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) (free).

---

## Multi-Show Support

Show configs live in `shows/`. Currently supported:
- `shows/survivor.yaml`
- `shows/traitors.yaml` (ready for data + backfill)

```bash
python -m src.main --show traitors --season 3 --episode auto
```

---

## Configuration

```bash
cp .env.example .env
```

Required for core functionality:
- `ANTHROPIC_API_KEY` — Claude API key

Optional (graceful degradation if missing):
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` — Reddit API credentials
- `KALSHI_API_KEY` — Kalshi trading API key
- `TWITTER_BEARER_TOKEN` / `BLUESKY_HANDLE` + `BLUESKY_APP_PASSWORD` — social posting

---

## Data Layout

```
data/
└── survivor/
    ├── s50/
    │   ├── contestants.json    # Cast list with archetypes
    │   ├── episodes/           # Per-episode LLM analysis results
    │   ├── predictions/        # Per-episode probabilities (+ latest.json, history.json)
    │   ├── market/             # Kalshi snapshots
    │   └── live/               # Live mode Kalshi timelines
    └── s49/
        ├── contestants.json
        ├── outcomes.json       # Ground truth (from wiki)
        ├── episodes/
        └── predictions/
```

---

## GitHub Actions

The pipeline runs automatically on a Thursday morning schedule (after Wednesday airings):

```yaml
# .github/workflows/update-predictions.yml
schedule:
  - cron: '0 14 * * 4'  # 2pm UTC = 10am ET Thursday
```

Requires these GitHub Secrets:
- `ANTHROPIC_API_KEY`
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET`
- `KALSHI_API_KEY`

---

## Cost

Each episode run costs approximately **$0.10–0.30** in Claude API fees (3 calls: analysis + elimination + winner prediction).

Running cost is displayed in the CLI after each run.

---

## Project Structure

```
survivor-predictor/
├── src/
│   ├── main.py              # Orchestrator
│   ├── config.py            # Settings + path helpers
│   ├── ingest/              # Reddit, YouTube, manual
│   ├── analysis/            # LLM analysis, prompts, scoring
│   ├── prediction/          # Bayesian model + historical priors
│   ├── market/              # Kalshi API client
│   ├── backfill/            # Retroactive season analysis
│   ├── simulation/          # Kelly criterion bet simulator
│   ├── live/                # Real-time episode mode
│   └── output/              # CLI, social, export
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── shows/                   # Show YAML configs
├── data/                    # All data (committed to repo)
├── backfill/                # Backfill results + calibration
└── tests/                   # Unit tests
```
