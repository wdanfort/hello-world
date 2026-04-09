"""YouTube transcript ingestion using youtube-transcript-api."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

import requests
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

from src import config


# RHAP (Rob Has a Podcast) and similar channels post Survivor recaps
RECAP_CHANNEL_SEARCH_TERMS = [
    "RHAP Survivor recap",
    "Rob Has a Podcast Survivor",
    "Survivor recap episode",
]


def _search_youtube_video_id(season: int, episode: int) -> Optional[str]:
    """
    Search YouTube Data API (no key needed for basic search via RSS/scraping).
    Falls back to a simple requests-based search on the public YouTube search page.
    Returns a video ID or None.
    """
    query = f"Survivor {season} Episode {episode} recap RHAP"
    url = "https://www.youtube.com/results"
    try:
        resp = requests.get(url, params={"search_query": query}, timeout=10)
        resp.raise_for_status()
        # Extract video IDs from the response (YouTube embeds them in JSON-like blocks)
        ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
        return ids[0] if ids else None
    except Exception as e:
        print(f"[youtube] Search failed: {e}")
        return None


def fetch_youtube_transcript(
    show_slug: str,
    season: int,
    episode: int,
    video_id: Optional[str] = None,
    cache: bool = True,
) -> Optional[dict]:
    """
    Fetch a YouTube transcript for an episode recap.

    Parameters
    ----------
    video_id: if provided, skip search and use this ID directly.
    Returns None if no transcript is available (graceful degradation).
    """
    cache_path = config.episodes_dir(show_slug, season) / f"youtube_ep{episode:02d}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache and cache_path.exists():
        print(f"[youtube] Cache hit: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    if video_id is None:
        video_id = _search_youtube_video_id(season, episode)
    if video_id is None:
        print(f"[youtube] Could not find a video for S{season}E{episode}")
        return None

    try:
        transcript_parts = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join(part["text"] for part in transcript_parts)
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        print(f"[youtube] No transcript available for video {video_id}: {e}")
        return None
    except Exception as e:
        print(f"[youtube] Error fetching transcript: {e}")
        return None

    result = {
        "episode_number": episode,
        "season": season,
        "show": show_slug,
        "source": "youtube",
        "video_id": video_id,
        "content": full_text,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if cache:
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)

    return result
