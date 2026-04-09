"""Manual recap ingestion: accept pasted text or a file path."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src import config


def ingest_manual(
    show_slug: str,
    season: int,
    episode: int,
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    cache: bool = True,
) -> dict:
    """
    Ingest a manually provided recap text blob.

    Supply either `text` (string) or `file_path` (path to a .txt file).
    """
    if text is None and file_path is None:
        raise ValueError("Provide either `text` or `file_path`.")
    if file_path:
        with open(file_path) as f:
            text = f.read()

    result = {
        "episode_number": episode,
        "season": season,
        "show": show_slug,
        "source": "manual",
        "content": text,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if cache:
        cache_path = config.episodes_dir(show_slug, season) / f"manual_ep{episode:02d}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)

    return result
