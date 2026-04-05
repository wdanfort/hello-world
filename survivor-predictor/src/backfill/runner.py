"""Backfill engine: retroactively run the model on completed seasons."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from src import config
from src.ingest import reddit as reddit_module
from src.analysis import llm_analyzer, scoring
from src.prediction import model as prediction_model
from src.output import export


def load_outcomes(show_slug: str, season: int) -> dict:
    path = config.season_dir(show_slug, season) / "outcomes.json"
    if not path.exists():
        raise FileNotFoundError(f"outcomes.json not found at {path}. Create it from the wiki first.")
    with open(path) as f:
        return json.load(f)


def _get_active_at_episode(outcomes: dict, episode: int) -> list[str]:
    """Return contestants still active at the START of the given episode."""
    eliminated_before = {
        e["eliminated"]
        for e in outcomes["eliminations"]
        if e["episode"] < episode
    }
    # We don't know full cast from outcomes alone; caller must supply from contestants.json
    return eliminated_before


def run_backfill(
    show_slug: str,
    season: int,
    source: str = "reddit",
    comment_window_hours: int = 24,
    force: bool = False,
    start_episode: int = 1,
    end_episode: Optional[int] = None,
) -> dict:
    """
    Run the full analysis pipeline on a completed season for calibration.

    Parameters
    ----------
    source:                "reddit" | "manual"
    comment_window_hours:  Only use comments within N hours of air date
    force:                 Rerun even if output already exists
    start_episode:         First episode to process
    end_episode:           Last episode to process (None = process all)
    """
    outcomes = load_outcomes(show_slug, season)
    air_dates: dict[str, str] = outcomes.get("air_dates", {})

    contestants_path = config.season_dir(show_slug, season) / "contestants.json"
    with open(contestants_path) as f:
        all_contestants = json.load(f)["contestants"]

    show_config = config.load_show_config(show_slug)
    subreddits = show_config["show"]["subreddits"]

    all_episode_nums = sorted(int(k) for k in air_dates.keys())
    if end_episode:
        all_episode_nums = [e for e in all_episode_nums if start_episode <= e <= end_episode]
    else:
        all_episode_nums = [e for e in all_episode_nums if e >= start_episode]

    calibration_results: list[dict] = []
    eliminated_so_far: list[str] = []

    # Set up output dir
    backfill_ep_dir = config.backfill_dir(show_slug, season) / "episodes"
    backfill_ep_dir.mkdir(parents=True, exist_ok=True)

    for ep_num in all_episode_nums:
        out_path = backfill_ep_dir / f"episode_{ep_num:02d}.json"
        if out_path.exists() and not force:
            print(f"[backfill] Skipping episode {ep_num} (already processed). Use --force to redo.")
            # Still need to track elimination for downstream episodes
            ep_elim = next(
                (e["eliminated"] for e in outcomes["eliminations"] if e["episode"] == ep_num), None
            )
            if ep_elim:
                eliminated_so_far.append(ep_elim)
            continue

        print(f"\n[backfill] Processing S{season}E{ep_num}...")

        air_date = air_dates.get(str(ep_num))

        # Active contestants BEFORE this episode
        active_at_ep = [
            c["name"] for c in all_contestants
            if c["name"] not in eliminated_so_far
        ]

        if not active_at_ep:
            print(f"[backfill] No active contestants left at episode {ep_num}. Stopping.")
            break

        # Ingest
        content_blob = ""
        if source == "reddit":
            ingest_result = reddit_module.fetch_episode_comments(
                show_slug=show_slug,
                season=season,
                episode=ep_num,
                subreddits=subreddits,
                air_date=air_date,
                comment_window_hours=comment_window_hours,
                cache=True,
            )
            content_blob = ingest_result.get("content", "")
        else:
            raise ValueError(f"Unsupported source for backfill: {source}. Use 'reddit'.")

        if not content_blob.strip():
            print(f"[backfill] No content found for S{season}E{ep_num}. Skipping.")
            continue

        # Analyze
        analysis = llm_analyzer.analyze_episode(
            show_slug=show_slug,
            season=season,
            episode_number=ep_num,
            episode_content=content_blob,
            contestant_list=active_at_ep,
        )
        analysis["episode_number"] = ep_num
        analysis["air_date"] = air_date
        analysis["sources_used"] = [source]
        analysis["active_contestants"] = active_at_ep

        # Save to episodes dir (for cumulative scoring) AND backfill dir
        export.save_episode_analysis(show_slug, season, ep_num, analysis)
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2)

        # Build cumulative scores for winner prediction
        all_scores = scoring.compute_cumulative_scores(show_slug, season)
        all_scores_str = json.dumps(all_scores, indent=2)

        # Predict winner
        winner_raw = llm_analyzer.predict_winner(
            show_slug=show_slug,
            season=season,
            all_episode_scores=all_scores_str,
            eliminated=eliminated_so_far,
        )
        winner_probs = winner_raw.get("winner_probabilities", {})

        # Predict elimination
        scores_str = json.dumps(
            {name: analysis["contestants"].get(name, {}) for name in active_at_ep},
            indent=2,
        )
        game_state = (
            f"Episode {ep_num}. Eliminated so far: {', '.join(eliminated_so_far) or 'none'}"
        )
        elim_raw = llm_analyzer.predict_elimination(scores_str, game_state)
        elim_probs = elim_raw.get("elimination_probabilities", {})

        # Build final predictions
        predictions = {
            "show": show_slug,
            "season": season,
            "episode_number": ep_num,
            "n_remaining": len(active_at_ep),
            "contestants": {
                name: {
                    "win_prob": round(winner_probs.get(name, 0.0), 4),
                    "elim_prob": round(elim_probs.get(name, 0.0), 4),
                }
                for name in active_at_ep
            },
        }
        export.save_episode_prediction(show_slug, season, ep_num, predictions)

        # Evaluate accuracy
        actual_eliminated = next(
            (e["eliminated"] for e in outcomes["eliminations"] if e["episode"] == ep_num), None
        )

        calibration_entry: dict = {
            "episode": ep_num,
            "actual_eliminated": actual_eliminated,
            "predicted_elim_probs": {
                k: round(v, 4) for k, v in elim_probs.items()
            },
            "actual_winner": outcomes.get("winner"),
            "predicted_winner_probs": {
                k: round(v, 4) for k, v in winner_probs.items()
            },
        }
        if actual_eliminated:
            rank = sorted(elim_probs, key=lambda x: elim_probs.get(x, 0), reverse=True)
            calibration_entry["correct_elim"] = actual_eliminated == rank[0] if rank else False
            calibration_entry["elim_in_top3"] = actual_eliminated in rank[:3] if len(rank) >= 3 else False
            calibration_entry["assigned_elim_prob"] = elim_probs.get(actual_eliminated, 0.0)

        calibration_results.append(calibration_entry)

        # Update eliminated list for next iteration
        if actual_eliminated:
            eliminated_so_far.append(actual_eliminated)

        print(f"[backfill] Episode {ep_num} done. Actual boot: {actual_eliminated or 'unknown'}")

    # Save calibration summary
    correct_elim = sum(1 for r in calibration_results if r.get("correct_elim"))
    top3_elim = sum(1 for r in calibration_results if r.get("elim_in_top3"))
    total = len([r for r in calibration_results if r.get("actual_eliminated")])

    summary = {
        "show": show_slug,
        "season": season,
        "episodes_processed": len(calibration_results),
        "elimination_accuracy_top1": correct_elim / total if total > 0 else None,
        "elimination_accuracy_top3": top3_elim / total if total > 0 else None,
        "calibration": calibration_results,
        "generated_at": datetime.utcnow().isoformat(),
    }

    summary_path = config.backfill_dir(show_slug, season) / "calibration_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[backfill] Done. Top-1 accuracy: {correct_elim}/{total}. Top-3: {top3_elim}/{total}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill season analysis")
    parser.add_argument("--show", default="survivor")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--source", default="reddit", choices=["reddit"])
    parser.add_argument("--comment-window-hours", type=int, default=24)
    parser.add_argument("--start-episode", type=int, default=1)
    parser.add_argument("--end-episode", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_backfill(
        show_slug=args.show,
        season=args.season,
        source=args.source,
        comment_window_hours=args.comment_window_hours,
        force=args.force,
        start_episode=args.start_episode,
        end_episode=args.end_episode,
    )
