"""Export predictions to JSON for dashboard consumption."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from src import config


def save_episode_prediction(
    show_slug: str,
    season: int,
    episode_number: int,
    predictions: dict,
    market_odds: dict | None = None,
) -> Path:
    """Write per-episode prediction JSON."""
    pred_dir = config.predictions_dir(show_slug, season)
    pred_dir.mkdir(parents=True, exist_ok=True)

    output = {
        **predictions,
        "market_odds": market_odds or {},
        "generated_at": datetime.utcnow().isoformat(),
    }

    out_path = pred_dir / f"episode_{episode_number:02d}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Also write latest.json
    latest_path = pred_dir / "latest.json"
    shutil.copy(out_path, latest_path)

    return out_path


def save_episode_analysis(
    show_slug: str,
    season: int,
    episode_number: int,
    analysis: dict,
) -> Path:
    """Write per-episode LLM analysis JSON."""
    ep_dir = config.episodes_dir(show_slug, season)
    ep_dir.mkdir(parents=True, exist_ok=True)
    out_path = ep_dir / f"episode_{episode_number:02d}.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    return out_path


def build_history_json(show_slug: str, season: int) -> Path:
    """Aggregate all episode predictions into a single history.json for the dashboard."""
    pred_dir = config.predictions_dir(show_slug, season)
    if not pred_dir.exists():
        return None

    episodes = []
    for path in sorted(pred_dir.glob("episode_*.json")):
        with open(path) as f:
            episodes.append(json.load(f))

    history = {
        "show": show_slug,
        "season": season,
        "episodes": episodes,
        "generated_at": datetime.utcnow().isoformat(),
    }

    out_path = pred_dir / "history.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    return out_path
