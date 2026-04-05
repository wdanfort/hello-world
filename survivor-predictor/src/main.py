"""
Main orchestrator: ingest → analyze → predict → compare → output

Usage:
  python -m src.main --show survivor --season 50 --episode auto --sources reddit,youtube
  python -m src.main --episode 5 --paste "Episode recap text here..."
  python -m src.main --episode 5 --paste-file recap.txt
  python -m src.main --episode 5 --market-odds '{"Name": 0.35, ...}'
  python -m src.main --episode 5 --post-twitter --post-bluesky
  python -m src.main --episode auto --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import config
from src.ingest import manual as manual_ingest
from src.ingest import reddit as reddit_module
from src.ingest import youtube as youtube_module
from src.analysis import llm_analyzer, scoring
from src.prediction import model as prediction_model
from src.market import kalshi
from src.output import cli as cli_output
from src.output import export, social


def _detect_episode(show_slug: str, season: int) -> int:
    """Auto-detect the next unprocessed episode number."""
    ep_dir = config.episodes_dir(show_slug, season)
    if not ep_dir.exists():
        return 1
    existing = sorted(ep_dir.glob("episode_*.json"))
    if not existing:
        return 1
    latest = int(existing[-1].stem.split("_")[1])
    return latest + 1


def _merge_episode_content(sources: list[dict]) -> str:
    """Combine multiple source content blobs into one string."""
    parts = []
    for s in sources:
        label = s.get("source", "unknown").upper()
        content = s.get("content", "").strip()
        if content:
            parts.append(f"[SOURCE: {label}]\n{content}")
    return "\n\n".join(parts)


def run(
    show_slug: str,
    season: int,
    episode: int | str,
    sources: list[str],
    paste_text: str | None = None,
    paste_file: str | None = None,
    market_odds_json: str | None = None,
    force: bool = False,
    post_twitter: bool = False,
    post_bluesky: bool = False,
) -> None:
    # --- 1. Determine episode number ---
    if episode == "auto":
        episode_number = _detect_episode(show_slug, season)
        print(f"[main] Auto-detected episode: {episode_number}")
    else:
        episode_number = int(episode)

    # --- 2. Check idempotency ---
    pred_path = config.predictions_dir(show_slug, season) / f"episode_{episode_number:02d}.json"
    if pred_path.exists() and not force:
        print(f"[main] Episode {episode_number} already processed. Use --force to redo.")
        # Still display results
        with open(pred_path) as f:
            predictions = json.load(f)
        market_data = predictions.get("market_odds", {})
        cli_output.print_odds_table(predictions, market_data)
        cli_output.print_elimination_forecast(predictions)
        return

    # --- 3. Ingest content ---
    episode_sources: list[dict] = []

    if paste_text:
        result = manual_ingest.ingest_manual(show_slug, season, episode_number, text=paste_text)
        episode_sources.append(result)

    elif paste_file:
        result = manual_ingest.ingest_manual(show_slug, season, episode_number, file_path=paste_file)
        episode_sources.append(result)

    else:
        show_config = config.load_show_config(show_slug)
        subreddits = show_config["show"]["subreddits"]

        if "reddit" in sources:
            reddit_result = reddit_module.fetch_episode_comments(
                show_slug=show_slug,
                season=season,
                episode=episode_number,
                subreddits=subreddits,
                comment_window_hours=24,
                cache=True,
            )
            if reddit_result.get("content"):
                episode_sources.append(reddit_result)
            else:
                print("[main] Reddit returned no content.")

        if "youtube" in sources:
            yt_result = youtube_module.fetch_youtube_transcript(
                show_slug=show_slug,
                season=season,
                episode=episode_number,
                cache=True,
            )
            if yt_result and yt_result.get("content"):
                episode_sources.append(yt_result)
            else:
                print("[main] YouTube returned no content. Skipping.")

    if not episode_sources:
        print("[main] No episode content available. Exiting.")
        sys.exit(1)

    combined_content = _merge_episode_content(episode_sources)
    sources_used = [s["source"] for s in episode_sources]
    print(f"[main] Content gathered from: {sources_used}. Length: {len(combined_content):,} chars")

    # --- 4. Get active contestants ---
    active = prediction_model.get_active_contestants(show_slug, season)
    contestant_names = [c["name"] for c in active]
    eliminated = prediction_model.get_eliminated_names(show_slug, season)
    print(f"[main] Active contestants ({len(contestant_names)}): {', '.join(contestant_names)}")

    # --- 5. LLM Analysis ---
    print("[main] Running episode analysis...")
    analysis = llm_analyzer.analyze_episode(
        show_slug=show_slug,
        season=season,
        episode_number=episode_number,
        episode_content=combined_content,
        contestant_list=contestant_names,
    )
    analysis["episode_number"] = episode_number
    analysis["sources_used"] = sources_used
    export.save_episode_analysis(show_slug, season, episode_number, analysis)
    print(f"[main] Analysis saved. Cost: ${analysis.get('llm_cost', {}).get('cost_usd', 0):.4f}")

    # --- 6. Compute cumulative scores ---
    all_scores = scoring.compute_cumulative_scores(show_slug, season)
    all_scores_str = json.dumps(all_scores, indent=2)

    # --- 7. Predict winner ---
    print("[main] Predicting winner probabilities...")
    winner_raw = llm_analyzer.predict_winner(
        show_slug=show_slug,
        season=season,
        all_episode_scores=all_scores_str,
        eliminated=eliminated,
    )
    winner_probs = winner_raw.get("winner_probabilities", {})

    # --- 8. Predict elimination ---
    print("[main] Predicting next elimination...")
    scores_for_elim = {name: analysis.get("contestants", {}).get(name, {}) for name in contestant_names}
    game_state = (
        f"Episode {episode_number}. Season {season}. "
        f"Eliminated: {', '.join(eliminated) or 'none'}"
    )
    elim_raw = llm_analyzer.predict_elimination(
        json.dumps(scores_for_elim, indent=2), game_state
    )
    elim_probs = elim_raw.get("elimination_probabilities", {})

    # --- 9. Build predictions ---
    predictions = prediction_model.build_predictions(
        show_slug=show_slug,
        season=season,
        episode_number=episode_number,
        llm_winner_probs=winner_probs,
        llm_elim_probs=elim_probs,
    )

    # --- 10. Fetch market odds ---
    manual_odds: dict | None = None
    if market_odds_json:
        try:
            manual_odds = json.loads(market_odds_json)
        except json.JSONDecodeError as e:
            print(f"[main] Could not parse --market-odds JSON: {e}")

    market_data = kalshi.fetch_survivor_odds(
        show_slug=show_slug,
        season=season,
        episode_number=episode_number,
        contestant_names=contestant_names,
        manual_odds=manual_odds,
        cache=True,
    )

    # --- 11. Save predictions ---
    saved_path = export.save_episode_prediction(
        show_slug, season, episode_number, predictions, market_data
    )
    export.build_history_json(show_slug, season)
    print(f"[main] Predictions saved to {saved_path}")

    # --- 12. CLI output ---
    cli_output.print_odds_table(predictions, market_data or None)
    cli_output.print_elimination_forecast(predictions)
    cli_output.print_cost_summary()

    # --- 13. Social post ---
    social.publish(
        predictions=predictions,
        market_odds=market_data or None,
        post_twitter=post_twitter,
        post_bluesky=post_bluesky,
        twitter_bearer_token=config.TWITTER_BEARER_TOKEN,
        bluesky_handle=config.BLUESKY_HANDLE,
        bluesky_password=config.BLUESKY_APP_PASSWORD,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Survivor prediction model — run full pipeline for an episode."
    )
    parser.add_argument("--show", default="survivor", help="Show slug (survivor, traitors, ...)")
    parser.add_argument("--season", type=int, default=50, help="Season number")
    parser.add_argument(
        "--episode", default="auto",
        help="Episode number, or 'auto' to detect next unprocessed episode"
    )
    parser.add_argument(
        "--sources", default="reddit,youtube",
        help="Comma-separated list of sources: reddit, youtube"
    )
    parser.add_argument("--paste", dest="paste_text", default=None, help="Manually paste recap text")
    parser.add_argument("--paste-file", dest="paste_file", default=None, help="Path to a recap .txt file")
    parser.add_argument(
        "--market-odds", dest="market_odds_json", default=None,
        help='JSON string of market odds: \'{"Name": 0.35, ...}\''
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if episode already processed")
    parser.add_argument("--post-twitter", action="store_true", help="Publish to Twitter/X")
    parser.add_argument("--post-bluesky", action="store_true", help="Publish to Bluesky")

    args = parser.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    run(
        show_slug=args.show,
        season=args.season,
        episode=args.episode,
        sources=sources,
        paste_text=args.paste_text,
        paste_file=args.paste_file,
        market_odds_json=args.market_odds_json,
        force=args.force,
        post_twitter=args.post_twitter,
        post_bluesky=args.post_bluesky,
    )


if __name__ == "__main__":
    main()
