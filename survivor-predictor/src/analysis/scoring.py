"""Convert LLM output to normalized scores; maintain cumulative history."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from src import config


def normalize_score(raw: float, scale: float = 10.0) -> float:
    """Convert a 1-10 score to 0-1 float, clamped."""
    return max(0.0, min(1.0, (raw - 1) / (scale - 1)))


def normalize_contestant_scores(contestant_data: dict, show_slug: str) -> dict:
    """Normalize all numeric scores for a single contestant to 0-1."""
    keys = config.get_scoring_keys(show_slug)
    result = {}
    for k, v in contestant_data.items():
        if k in keys and isinstance(v, (int, float)):
            result[k] = normalize_score(v)
        else:
            result[k] = v
    return result


def load_all_episode_scores(show_slug: str, season: int) -> list[dict]:
    """
    Load all saved episode analysis files for this season, sorted by episode.
    Returns a list of episode analysis dicts.
    """
    ep_dir = config.episodes_dir(show_slug, season)
    if not ep_dir.exists():
        return []
    files = sorted(ep_dir.glob("episode_*.json"))
    episodes = []
    for f in files:
        with open(f) as fh:
            episodes.append(json.load(fh))
    return episodes


def compute_cumulative_scores(
    show_slug: str,
    season: int,
    decay_half_life: int = config.SCORE_DECAY_HALF_LIFE_EPISODES,
) -> dict[str, dict[str, float]]:
    """
    Compute exponentially-decayed cumulative scores per contestant across all episodes.

    More recent episodes are weighted more heavily:
        weight(ep) = 2^((ep - latest_ep) / half_life)

    Returns: { contestant_name: { score_key: weighted_avg, ... } }
    """
    episodes = load_all_episode_scores(show_slug, season)
    if not episodes:
        return {}

    scoring_keys = config.get_scoring_keys(show_slug)
    latest_ep = max(ep["episode_number"] for ep in episodes)

    # Accumulate weighted scores
    weighted_sums: dict[str, dict[str, float]] = {}
    weight_totals: dict[str, dict[str, float]] = {}

    for ep in episodes:
        ep_num = ep["episode_number"]
        weight = math.pow(2, (ep_num - latest_ep) / decay_half_life)
        for name, data in ep.get("contestants", {}).items():
            if name not in weighted_sums:
                weighted_sums[name] = {k: 0.0 for k in scoring_keys}
                weight_totals[name] = {k: 0.0 for k in scoring_keys}
            for key in scoring_keys:
                raw = data.get(key)
                if isinstance(raw, (int, float)):
                    normalized = normalize_score(raw)
                    weighted_sums[name][key] += normalized * weight
                    weight_totals[name][key] += weight

    result: dict[str, dict[str, float]] = {}
    for name in weighted_sums:
        result[name] = {}
        for key in scoring_keys:
            total_w = weight_totals[name][key]
            result[name][key] = weighted_sums[name][key] / total_w if total_w > 0 else 0.5

    return result


def compute_score_trends(show_slug: str, season: int, last_n: int = 3) -> dict[str, dict[str, str]]:
    """
    Compute directional trends for each contestant over the last N episodes.
    Returns { contestant: { score_key: "up" | "down" | "flat" } }
    """
    episodes = load_all_episode_scores(show_slug, season)
    if len(episodes) < 2:
        return {}

    scoring_keys = config.get_scoring_keys(show_slug)
    recent = sorted(episodes, key=lambda e: e["episode_number"])[-last_n:]
    trends: dict[str, dict[str, str]] = {}

    # Only look at episodes where the contestant appeared
    for name in {n for ep in recent for n in ep.get("contestants", {})}:
        appearing = [ep for ep in recent if name in ep.get("contestants", {})]
        if len(appearing) < 2:
            continue
        trends[name] = {}
        first_ep = appearing[0]["contestants"][name]
        last_ep = appearing[-1]["contestants"][name]
        for key in scoring_keys:
            v_first = first_ep.get(key, 5)
            v_last = last_ep.get(key, 5)
            diff = v_last - v_first if isinstance(v_last, (int, float)) and isinstance(v_first, (int, float)) else 0
            if diff > 0.5:
                trends[name][key] = "up"
            elif diff < -0.5:
                trends[name][key] = "down"
            else:
                trends[name][key] = "flat"

    return trends
