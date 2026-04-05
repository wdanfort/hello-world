"""Format predictions into shareable social media posts."""

from __future__ import annotations

from typing import Optional


def format_post(
    predictions: dict,
    market_odds: Optional[dict[str, float]] = None,
    top_n: int = 5,
) -> str:
    """
    Format a prediction update as a social media post (Twitter/Bluesky-friendly).
    Prints to stdout; use --post to actually publish.
    """
    show = predictions.get("show", "survivor").title()
    season = predictions.get("season", "?")
    episode = predictions.get("episode_number", "?")
    contestants = predictions.get("contestants", {})

    # Sort by win probability
    sorted_c = sorted(contestants.items(), key=lambda x: x[1]["win_prob"], reverse=True)

    lines = [f"🏝️ {show} {season} — Episode {episode} Model Update", ""]

    # Winner odds
    lines.append("🏆 Winner Odds:")
    for i, (name, data) in enumerate(sorted_c[:top_n], 1):
        win_m = data["win_prob"]
        market_p = market_odds.get(name) if market_odds else None

        if market_p is not None:
            delta = win_m - market_p
            arrow = "📈" if delta > 0.05 else "📉" if delta < -0.05 else "➡️"
            line = f"{i}. {name} — {win_m * 100:.0f}% (Kalshi: {market_p * 100:.0f}%) {arrow}"
        else:
            line = f"{i}. {name} — {win_m * 100:.0f}%"
        lines.append(line)

    lines.append("")

    # Most likely eliminated
    top_elim = max(contestants.items(), key=lambda x: x[1]["elim_prob"], default=None)
    if top_elim:
        name, data = top_elim
        lines.append(f"🔥 Most likely eliminated next: {name} ({data['elim_prob'] * 100:.0f}%)")

    lines.append("")

    # Biggest market edge
    if market_odds:
        edges = []
        for name, data in contestants.items():
            market_p = market_odds.get(name)
            if market_p is not None:
                delta = data["win_prob"] - market_p
                edges.append((name, data["win_prob"], market_p, delta))

        if edges:
            best_edge = max(edges, key=lambda x: abs(x[3]))
            name, model_p, market_p, delta = best_edge
            if abs(delta) >= 0.08:
                direction = "underpriced" if delta > 0 else "overpriced"
                lines.append(
                    f"💡 Biggest edge: {name} is {direction} — "
                    f"model says {model_p * 100:.0f}%, market says {market_p * 100:.0f}%"
                )

    return "\n".join(lines)


def post_to_twitter(text: str, bearer_token: str) -> bool:
    """Post text to Twitter/X. Returns True on success."""
    try:
        import tweepy

        # Tweepy v4 uses OAuth 1.0a or 2.0 depending on access level
        # This requires full API access (not just bearer token for reads)
        # Placeholder — configure keys in .env
        client = tweepy.Client(bearer_token=bearer_token)
        # For posting, need consumer_key + access_token OAuth 1.0a:
        # client = tweepy.Client(
        #     consumer_key=..., consumer_secret=...,
        #     access_token=..., access_token_secret=...
        # )
        client.create_tweet(text=text[:280])
        return True
    except Exception as e:
        print(f"[twitter] Error posting: {e}")
        return False


def post_to_bluesky(text: str, handle: str, app_password: str) -> bool:
    """Post text to Bluesky via atproto. Returns True on success."""
    try:
        from atproto import Client as BskyClient

        client = BskyClient()
        client.login(handle, app_password)
        client.send_post(text=text[:300])
        return True
    except Exception as e:
        print(f"[bluesky] Error posting: {e}")
        return False


def publish(
    predictions: dict,
    market_odds: Optional[dict[str, float]] = None,
    post_twitter: bool = False,
    post_bluesky: bool = False,
    twitter_bearer_token: str = "",
    bluesky_handle: str = "",
    bluesky_password: str = "",
) -> str:
    """Format and optionally publish the post. Always returns the text."""
    text = format_post(predictions, market_odds)
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")

    if post_twitter:
        if not twitter_bearer_token:
            print("[social] No Twitter bearer token configured. Skipping.")
        else:
            ok = post_to_twitter(text, twitter_bearer_token)
            print(f"[social] Twitter post {'succeeded' if ok else 'failed'}.")

    if post_bluesky:
        if not bluesky_handle or not bluesky_password:
            print("[social] Bluesky credentials not configured. Skipping.")
        else:
            ok = post_to_bluesky(text, bluesky_handle, bluesky_password)
            print(f"[social] Bluesky post {'succeeded' if ok else 'failed'}.")

    return text
