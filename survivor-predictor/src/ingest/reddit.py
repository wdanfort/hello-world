"""Reddit ingestion: scrapes post-episode discussion threads via PRAW."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import praw

from src import config


def _build_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        ratelimit_seconds=60,
    )


def _find_episode_thread(
    reddit: praw.Reddit,
    subreddit_name: str,
    season: int,
    episode: int,
    show_name: str = "Survivor",
) -> Optional[praw.models.Submission]:
    """Search for the post-episode discussion thread."""
    subreddit = reddit.subreddit(subreddit_name)
    search_queries = [
        f"Post-Episode Discussion Season {season} Episode {episode}",
        f"{show_name} {season} Episode {episode} Discussion",
        f"S{season}E{episode}",
        f"Season {season} Episode {episode}",
    ]
    for query in search_queries:
        results = list(subreddit.search(query, sort="relevance", limit=5, time_filter="year"))
        for post in results:
            title_lower = post.title.lower()
            ep_str = str(episode)
            if ep_str in title_lower and ("episode" in title_lower or "discussion" in title_lower):
                return post
    return None


def fetch_episode_comments(
    show_slug: str,
    season: int,
    episode: int,
    subreddits: list[str],
    air_date: Optional[str] = None,
    comment_window_hours: int = 24,
    top_n: int = 50,
    cache: bool = True,
) -> dict:
    """
    Fetch top comments from the post-episode discussion thread.

    Parameters
    ----------
    show_slug:            e.g. "survivor"
    season:               season number
    episode:              episode number
    subreddits:           list of subreddit names to search
    air_date:             ISO date string "YYYY-MM-DD"; used for 24h comment window
    comment_window_hours: only include comments posted within this many hours of air
    top_n:                max comments to return
    cache:                if True, write result to data dir and return cached on re-run
    """
    cache_path = config.episodes_dir(show_slug, season) / f"reddit_ep{episode:02d}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache and cache_path.exists():
        print(f"[reddit] Cache hit: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    reddit = _build_reddit_client()
    show_config = config.load_show_config(show_slug)
    show_name = show_config["show"]["name"]

    # Determine air date cutoff for comment window
    cutoff_ts: Optional[float] = None
    if air_date:
        air_dt = datetime.fromisoformat(air_date).replace(tzinfo=timezone.utc)
        cutoff_ts = air_dt.timestamp() + comment_window_hours * 3600

    combined_comments: list[str] = []
    used_subreddits: list[str] = []

    for subreddit_name in subreddits:
        thread = _find_episode_thread(reddit, subreddit_name, season, episode, show_name)
        if thread is None:
            print(f"[reddit] No thread found in r/{subreddit_name} for S{season}E{episode}")
            continue

        print(f"[reddit] Found thread: '{thread.title}' (r/{subreddit_name})")
        thread.comments.replace_more(limit=0)
        used_subreddits.append(subreddit_name)

        for comment in thread.comments.list():
            if len(combined_comments) >= top_n:
                break
            if cutoff_ts and comment.created_utc > cutoff_ts:
                continue  # comment posted after the window — skip
            if comment.score < 1:
                continue
            combined_comments.append(comment.body)

    result = {
        "episode_number": episode,
        "season": season,
        "show": show_slug,
        "source": "reddit",
        "subreddits": used_subreddits,
        "air_date": air_date,
        "comment_window_hours": comment_window_hours,
        "comment_count": len(combined_comments),
        "content": "\n\n---\n\n".join(combined_comments),
        "timestamp": datetime.utcnow().isoformat(),
    }

    if cache:
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)

    return result
