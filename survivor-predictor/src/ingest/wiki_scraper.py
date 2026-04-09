"""
Scrape the Survivor wiki (survivor.fandom.com) to auto-update
contestants.json and outcomes.json after each episode airs.

Usage:
    python -m src.ingest.wiki_scraper --show survivor --season 50
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src import config

# Map season number → Fandom wiki page slug
WIKI_SEASON_SLUGS = {
    47: "Survivor_47",
    48: "Survivor_48",
    49: "Survivor_49",
    50: "Survivor_50:_In_the_Hands_of_the_Fans",
}

WIKI_BASE = "https://survivor.fandom.com/wiki"
HEADERS = {"User-Agent": "survivor-predictor/1.0 (educational project)"}


# ---------------------------------------------------------------------------
# Fetch + parse wiki page
# ---------------------------------------------------------------------------

def _fetch_wiki_page(season: int) -> BeautifulSoup:
    slug = WIKI_SEASON_SLUGS.get(season)
    if not slug:
        raise ValueError(f"No wiki slug configured for season {season}. Add it to WIKI_SEASON_SLUGS.")
    url = f"{WIKI_BASE}/{slug}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _parse_eliminations(soup: BeautifulSoup) -> list[dict]:
    """
    Extract elimination order from the wiki voting history table.
    Returns list of { episode, eliminated, method } dicts, sorted by episode.
    """
    eliminations: list[dict] = []

    # The wiki uses a table with class "wikitable" for the voting history.
    # The "Eliminated" row shows who left each episode.
    # Strategy: find all tables, look for one with episode numbers + eliminated row.

    tables = soup.find_all("table", class_=lambda c: c and "wikitable" in c)
    for table in tables:
        rows = table.find_all("tr")
        header_row = rows[0] if rows else None
        if not header_row:
            continue

        # Look for a row labelled "Eliminated" or "Voted Out"
        elim_row = None
        ep_row = None
        for row in rows:
            first_cell = row.find(["th", "td"])
            if not first_cell:
                continue
            text = first_cell.get_text(strip=True).lower()
            if text in ("eliminated", "voted out", "left the game"):
                elim_row = row
            if text in ("episode", "ep."):
                ep_row = row

        if elim_row is None:
            continue

        # Get episode numbers from the header or ep_row
        episode_numbers: list[Optional[int]] = []
        header_cells = (ep_row or rows[0]).find_all(["th", "td"])
        for cell in header_cells[1:]:  # skip label column
            txt = cell.get_text(strip=True)
            try:
                episode_numbers.append(int(txt))
            except ValueError:
                episode_numbers.append(None)

        # Get eliminated names per episode
        elim_cells = elim_row.find_all(["th", "td"])
        for i, cell in enumerate(elim_cells[1:]):
            ep_num = episode_numbers[i] if i < len(episode_numbers) else None
            if ep_num is None:
                continue
            name = cell.get_text(strip=True)
            if not name or name in ("—", "-", ""):
                continue
            method = "voted_out"
            title = cell.get("title", "").lower()
            aria = cell.get("data-sort-value", "").lower()
            cell_lower = name.lower()
            if any(k in title + aria + cell_lower for k in ("quit", "quit")):
                method = "quit"
            elif any(k in title + aria + cell_lower for k in ("med", "medevac", "evacuated", "evacuation")):
                method = "medical_evacuation"
            eliminations.append({
                "episode": ep_num,
                "eliminated": name,
                "method": method,
            })

        if eliminations:
            break  # found the right table

    return sorted(eliminations, key=lambda x: x["episode"])


def _parse_winner(soup: BeautifulSoup) -> Optional[str]:
    """Try to extract the winner's name from the wiki infobox."""
    # The infobox typically has "Winner" as a row label
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    if not infobox:
        return None
    for row in infobox.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2 and "winner" in cells[0].get_text(strip=True).lower():
            return cells[1].get_text(strip=True) or None
    return None


def _parse_finalist_names(soup: BeautifulSoup) -> list[str]:
    """Try to extract final tribal council participants."""
    # Look for "Final Tribal Council" section or "Finalists" infobox row
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    if not infobox:
        return []
    for row in infobox.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower()
            if "finalist" in label or "final tribal" in label or "runner" in label:
                names = [a.get_text(strip=True) for a in cells[1].find_all("a")]
                if names:
                    return names
    return []


# ---------------------------------------------------------------------------
# Reconcile wiki data against existing JSON files
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[wiki] Saved {path}")


def sync_season(show_slug: str, season: int, dry_run: bool = False) -> dict:
    """
    Fetch the wiki page, parse eliminations, and update contestants.json
    and outcomes.json if there are new eliminations.

    Returns a summary dict: { new_eliminations: [...], changes_made: bool }
    """
    print(f"[wiki] Fetching Survivor {season} wiki page...")
    soup = _fetch_wiki_page(season)

    wiki_elims = _parse_eliminations(soup)
    wiki_winner = _parse_winner(soup)
    wiki_finalists = _parse_finalist_names(soup)

    print(f"[wiki] Found {len(wiki_elims)} eliminations on wiki. Winner: {wiki_winner or 'TBD'}")

    season_dir = config.season_dir(show_slug, season)
    contestants_path = season_dir / "contestants.json"
    outcomes_path = season_dir / "outcomes.json"

    contestants_data = _load_json(contestants_path)
    outcomes_data = _load_json(outcomes_path) if outcomes_path.exists() else {
        "season": season, "air_dates": {}, "eliminations": [], "winner": None, "final_tribal": []
    }

    # Build lookup: name → contestant record
    contestant_map = {c["name"]: c for c in contestants_data["contestants"]}

    # Existing eliminations in outcomes.json (as a set of names)
    existing_elim_names = {e["eliminated"] for e in outcomes_data.get("eliminations", [])}

    new_eliminations: list[dict] = []
    changes_made = False

    for wiki_elim in wiki_elims:
        name = wiki_elim["eliminated"]
        ep = wiki_elim["episode"]
        method = wiki_elim["method"]

        # Fuzzy match: wiki names sometimes differ slightly (e.g. quotes, nicknames)
        matched_name = _fuzzy_match_name(name, list(contestant_map.keys()))
        if matched_name is None:
            print(f"[wiki] WARNING: Could not match wiki name '{name}' to any contestant. Skipping.")
            continue

        if matched_name in existing_elim_names:
            continue  # already recorded

        print(f"[wiki] New elimination: {matched_name} (episode {ep}, {method})")
        new_eliminations.append({"episode": ep, "eliminated": matched_name, "method": method})

        if not dry_run:
            # Update contestants.json
            c = contestant_map[matched_name]
            c["status"] = "eliminated"
            c["eliminated_episode"] = ep
            if method != "voted_out":
                c["eliminated_method"] = method
            changes_made = True

            # Update outcomes.json
            outcomes_data["eliminations"].append({
                "episode": ep,
                "eliminated": matched_name,
                "method": method,
            })
            # Keep sorted by episode
            outcomes_data["eliminations"].sort(key=lambda x: x["episode"])

    # Update winner if known and not already set
    if wiki_winner and not outcomes_data.get("winner"):
        matched_winner = _fuzzy_match_name(wiki_winner, list(contestant_map.keys()))
        if matched_winner:
            print(f"[wiki] Winner found: {matched_winner}")
            if not dry_run:
                outcomes_data["winner"] = matched_winner
                contestant_map[matched_winner]["status"] = "winner"
                changes_made = True

    # Update finalists
    if wiki_finalists and not outcomes_data.get("final_tribal"):
        matched_finalists = [
            _fuzzy_match_name(n, list(contestant_map.keys()))
            for n in wiki_finalists
        ]
        matched_finalists = [n for n in matched_finalists if n]
        if matched_finalists:
            print(f"[wiki] Finalists: {matched_finalists}")
            if not dry_run:
                outcomes_data["final_tribal"] = matched_finalists
                changes_made = True

    if changes_made and not dry_run:
        _save_json(contestants_path, contestants_data)
        _save_json(outcomes_path, outcomes_data)
        print(f"[wiki] Updated {len(new_eliminations)} new elimination(s).")
    elif not new_eliminations:
        print("[wiki] No new eliminations found. Files unchanged.")

    return {
        "show": show_slug,
        "season": season,
        "new_eliminations": new_eliminations,
        "winner": wiki_winner,
        "changes_made": changes_made,
    }


def _fuzzy_match_name(wiki_name: str, contestant_names: list[str]) -> Optional[str]:
    """
    Match a wiki name to a contestant name.
    Tries exact match first, then last-name match, then first-name match.
    """
    # Normalize: strip quotes, extra whitespace
    clean = wiki_name.strip().strip('"').strip("'")

    # Exact match
    if clean in contestant_names:
        return clean

    # Case-insensitive exact
    clean_lower = clean.lower()
    for name in contestant_names:
        if name.lower() == clean_lower:
            return name

    # Last name match
    wiki_last = clean.split()[-1].lower() if clean.split() else ""
    for name in contestant_names:
        parts = name.split()
        if parts and parts[-1].lower() == wiki_last:
            return name

    # First name or nickname match (handle "Coach" → "Benjamin \"Coach\" Wade")
    wiki_first = clean.split()[0].lower() if clean.split() else ""
    for name in contestant_names:
        name_lower = name.lower()
        if wiki_first and wiki_first in name_lower:
            return name

    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync wiki elimination data to JSON files")
    parser.add_argument("--show", default="survivor")
    parser.add_argument("--season", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing files")
    args = parser.parse_args()

    result = sync_season(args.show, args.season, dry_run=args.dry_run)
    if result["new_eliminations"]:
        print("\nNew eliminations recorded:")
        for e in result["new_eliminations"]:
            print(f"  Episode {e['episode']}: {e['eliminated']} ({e['method']})")
    sys.exit(0)
