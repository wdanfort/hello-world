"""Kalshi API client: fetch live market odds for reality TV prediction markets."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from src import config


KALSHI_API_BASE = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiClient:
    def __init__(self, api_key: Optional[str] = None, demo: bool = False):
        self.api_key = api_key or config.KALSHI_API_KEY
        self.base_url = KALSHI_DEMO_BASE if demo else KALSHI_API_BASE
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def search_markets(self, query: str, limit: int = 50) -> list[dict]:
        """Search for active markets matching a query string."""
        data = self._get("/markets", params={"limit": limit, "status": "open"})
        markets = data.get("markets", [])
        query_lower = query.lower()
        return [m for m in markets if query_lower in m.get("title", "").lower()
                or query_lower in m.get("ticker", "").lower()]

    def get_market(self, ticker: str) -> dict:
        """Fetch a single market by ticker."""
        return self._get(f"/markets/{ticker}")

    def get_market_orderbook(self, ticker: str) -> dict:
        """Fetch the current orderbook for a market."""
        return self._get(f"/markets/{ticker}/orderbook")


def _fuzzy_name_match(kalshi_title: str, contestant_names: list[str]) -> Optional[str]:
    """Simple fuzzy match between a Kalshi market title and contestant names."""
    title_lower = kalshi_title.lower()
    for name in contestant_names:
        # Check last name or full name
        parts = name.lower().split()
        if any(part in title_lower for part in parts if len(part) > 3):
            return name
    return None


def fetch_survivor_odds(
    show_slug: str,
    season: int,
    episode_number: int,
    contestant_names: list[str],
    manual_odds: Optional[dict[str, float]] = None,
    cache: bool = True,
) -> dict[str, float]:
    """
    Fetch Kalshi winner market odds for the given show/season.

    Returns { contestant_name: implied_probability }

    If manual_odds are provided (via --market-odds CLI flag), use those directly.
    Falls back gracefully if API fails.
    """
    cache_path = config.market_dir(show_slug, season) / f"episode_{episode_number:02d}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache and cache_path.exists():
        print(f"[kalshi] Cache hit: {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        return data.get("winner_odds", {})

    if manual_odds:
        print("[kalshi] Using manually provided market odds.")
        snapshot = {
            "episode_number": episode_number,
            "season": season,
            "show": show_slug,
            "source": "manual",
            "winner_odds": manual_odds,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if cache:
            with open(cache_path, "w") as f:
                json.dump(snapshot, f, indent=2)
        return manual_odds

    if not config.KALSHI_API_KEY:
        print("[kalshi] No API key configured. Skipping market fetch. Use --market-odds to provide manually.")
        return {}

    try:
        client = KalshiClient()
        show_config = config.load_show_config(show_slug)
        market_slug = show_config["show"].get("kalshi_market_slug", "survivor")

        # Search for winner markets
        markets = client.search_markets(f"survivor {season} winner")
        if not markets:
            markets = client.search_markets(market_slug)

        winner_odds: dict[str, float] = {}
        for market in markets:
            ticker = market.get("ticker", "")
            title = market.get("title", "")
            matched_name = _fuzzy_name_match(title, contestant_names)
            if matched_name:
                # Yes price in cents → implied probability
                yes_price = market.get("yes_ask") or market.get("yes_bid") or market.get("last_price")
                if yes_price is not None:
                    winner_odds[matched_name] = yes_price / 100.0

        snapshot = {
            "episode_number": episode_number,
            "season": season,
            "show": show_slug,
            "source": "kalshi_api",
            "winner_odds": winner_odds,
            "raw_markets": markets[:20],  # store for debugging
            "timestamp": datetime.utcnow().isoformat(),
        }
        if cache:
            with open(cache_path, "w") as f:
                json.dump(snapshot, f, indent=2)

        print(f"[kalshi] Fetched odds for {len(winner_odds)} contestants.")
        return winner_odds

    except Exception as e:
        print(f"[kalshi] API error: {e}. Continuing without market odds.")
        return {}


def poll_live_odds(
    show_slug: str,
    season: int,
    episode_number: int,
    contestant_names: list[str],
    interval_seconds: int = 60,
    duration_seconds: int = 3600,
) -> None:
    """
    Poll Kalshi odds every `interval_seconds` for `duration_seconds` during a live episode.
    Saves a timeline file to data/live/.
    """
    live_dir = config.live_dir(show_slug, season)
    live_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = live_dir / f"episode_{episode_number:02d}_kalshi_timeline.json"

    timeline: list[dict] = []
    prev_odds: dict[str, float] = {}
    start_ts = time.time()

    print(f"[kalshi-live] Polling every {interval_seconds}s for {duration_seconds}s...")

    while time.time() - start_ts < duration_seconds:
        ts = datetime.utcnow().isoformat()
        try:
            odds = fetch_survivor_odds(
                show_slug, season, episode_number, contestant_names, cache=False
            )
        except Exception as e:
            print(f"[kalshi-live] Poll error: {e}")
            odds = {}

        snapshot: dict = {"timestamp": ts, "odds": {}}
        for name, p in odds.items():
            delta = round(p - prev_odds.get(name, p), 4)
            snapshot["odds"][name] = {"price": p, "delta_from_last": delta}
            if abs(delta) >= 0.05:
                print(f"[kalshi-live] ⚡ {name}: {p:.0%} (Δ{delta:+.1%})")

        timeline.append(snapshot)
        prev_odds = {name: d["price"] for name, d in snapshot["odds"].items()}

        # Write incrementally
        with open(timeline_path, "w") as f:
            json.dump({"episode": episode_number, "timeline": timeline}, f, indent=2)

        time.sleep(interval_seconds)

    print(f"[kalshi-live] Polling complete. Timeline saved to {timeline_path}")
