"""Configuration: loads .env and show YAML, exposes typed settings."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Repo root is two levels up from this file
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SHOWS_DIR = ROOT / "shows"

load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
REDDIT_CLIENT_ID: str = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.environ.get("REDDIT_USER_AGENT", "survivor-predictor/1.0")
KALSHI_API_KEY: str = os.environ.get("KALSHI_API_KEY", "")
TWITTER_BEARER_TOKEN: str = os.environ.get("TWITTER_BEARER_TOKEN", "")
BLUESKY_HANDLE: str = os.environ.get("BLUESKY_HANDLE", "")
BLUESKY_APP_PASSWORD: str = os.environ.get("BLUESKY_APP_PASSWORD", "")


# ---------------------------------------------------------------------------
# Show config loader
# ---------------------------------------------------------------------------

_show_cache: dict[str, dict] = {}


def load_show_config(show_slug: str) -> dict:
    """Load and cache the YAML config for a given show slug."""
    if show_slug not in _show_cache:
        config_path = SHOWS_DIR / f"{show_slug}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No show config found at {config_path}")
        with open(config_path) as f:
            _show_cache[show_slug] = yaml.safe_load(f)
    return _show_cache[show_slug]


def get_scoring_keys(show_slug: str) -> list[str]:
    """Return the list of scoring dimension keys for a show."""
    config = load_show_config(show_slug)
    return [d["key"] for d in config["scoring_dimensions"]]


# ---------------------------------------------------------------------------
# Data path helpers
# ---------------------------------------------------------------------------

def season_dir(show_slug: str, season: int) -> Path:
    return DATA_DIR / show_slug / f"s{season}"


def episodes_dir(show_slug: str, season: int) -> Path:
    return season_dir(show_slug, season) / "episodes"


def predictions_dir(show_slug: str, season: int) -> Path:
    return season_dir(show_slug, season) / "predictions"


def market_dir(show_slug: str, season: int) -> Path:
    return season_dir(show_slug, season) / "market"


def live_dir(show_slug: str, season: int) -> Path:
    return season_dir(show_slug, season) / "live"


def backfill_dir(show_slug: str, season: int) -> Path:
    return ROOT / "backfill" / show_slug / f"s{season}"


# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------

LLM_MODEL_ANALYSIS = "claude-sonnet-4-20250514"
LLM_MODEL_LIVE = "claude-haiku-4-5-20251001"
LLM_TEMP_ANALYSIS = 0.3
LLM_TEMP_PREDICTION = 0.5
LLM_MAX_RETRIES = 2

# Scoring
SCORE_DECAY_HALF_LIFE_EPISODES = 4  # exponential decay for historical scores

# Market comparison
MISPRICING_THRESHOLD = 0.10  # |model - market| > 10% flags a mispricing

# Bet simulator defaults
SIM_STARTING_BANKROLL = 100.0
SIM_EDGE_THRESHOLD = 0.10
SIM_KELLY_FRACTION = 0.25
