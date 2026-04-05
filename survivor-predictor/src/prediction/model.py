"""Bayesian prediction engine: combines LLM scores + historical priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src import config
from src.prediction import history
from src.analysis import scoring


def _normalize(probs: dict[str, float]) -> dict[str, float]:
    """Normalize a probability dict so values sum to 1.0."""
    total = sum(probs.values())
    if total <= 0:
        n = len(probs)
        return {k: 1.0 / n for k in probs}
    return {k: v / total for k, v in probs.items()}


def load_contestants(show_slug: str, season: int) -> list[dict]:
    path = config.season_dir(show_slug, season) / "contestants.json"
    with open(path) as f:
        data = json.load(f)
    return data["contestants"]


def get_active_contestants(show_slug: str, season: int) -> list[dict]:
    return [c for c in load_contestants(show_slug, season) if c.get("status") == "active"]


def get_eliminated_names(show_slug: str, season: int) -> list[str]:
    return [
        c["name"]
        for c in load_contestants(show_slug, season)
        if c.get("status") != "active"
    ]


def compute_winner_probabilities(
    show_slug: str,
    season: int,
    llm_winner_probs: Optional[dict[str, float]] = None,
    archetype_weight: float = 0.2,
    llm_weight: float = 0.8,
) -> dict[str, float]:
    """
    Blend LLM winner predictions with archetype-based historical priors.

    If llm_winner_probs is provided (from predict_winner()), blend it with
    the archetype prior. Otherwise fall back to cumulative scores.

    Parameters
    ----------
    archetype_weight: weight for the historical prior (0-1)
    llm_weight:       weight for the LLM prediction (0-1)
    """
    active = get_active_contestants(show_slug, season)
    n = len(active)

    # Build archetype prior
    archetype_prior: dict[str, float] = {}
    for c in active:
        arch = c.get("archetype", "social")
        base = history.archetype_win_prior(arch)
        uniform = history.get_uniform_prior(n)
        # Blend archetype rate with uniform so all archetypes get a floor
        archetype_prior[c["name"]] = 0.6 * base + 0.4 * uniform

    archetype_prior = _normalize(archetype_prior)

    if llm_winner_probs:
        # Only keep active contestants
        filtered = {k: v for k, v in llm_winner_probs.items() if k in {c["name"] for c in active}}
        if not filtered:
            return archetype_prior
        filtered = _normalize(filtered)
        blended = {
            name: archetype_weight * archetype_prior.get(name, 1.0 / n)
                  + llm_weight * filtered.get(name, 0.0)
            for name in {c["name"] for c in active}
        }
        return _normalize(blended)

    # Fall back to cumulative LLM scores if no direct LLM winner probs
    cum_scores = scoring.compute_cumulative_scores(show_slug, season)
    if not cum_scores:
        return archetype_prior

    score_prior: dict[str, float] = {}
    for c in active:
        name = c["name"]
        scores = cum_scores.get(name, {})
        # Weight winner_signals most heavily, then edit_sentiment and strategic_positioning
        win_sig = scores.get("winner_signals", 0.5)
        edit_sent = scores.get("edit_sentiment", 0.5)
        strategic = scores.get("strategic_positioning", 0.5)
        score_prior[name] = 0.5 * win_sig + 0.3 * edit_sent + 0.2 * strategic

    score_prior = _normalize(score_prior)
    blended = {
        name: archetype_weight * archetype_prior.get(name, 1.0 / n)
              + llm_weight * score_prior.get(name, 0.0)
        for name in {c["name"] for c in active}
    }
    return _normalize(blended)


def compute_elimination_probabilities(
    show_slug: str,
    season: int,
    llm_elim_probs: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Compute next-elimination probabilities.
    Primarily driven by LLM output, calibrated with historical base rate.
    """
    active = get_active_contestants(show_slug, season)
    n = len(active)
    active_names = {c["name"] for c in active}

    if llm_elim_probs:
        filtered = {k: v for k, v in llm_elim_probs.items() if k in active_names}
        if filtered:
            return _normalize(filtered)

    # Fall back: use elimination_risk score from cumulative analysis
    cum_scores = scoring.compute_cumulative_scores(show_slug, season)
    if not cum_scores:
        return _normalize({c["name"]: 1.0 for c in active})

    elim_scores = {
        c["name"]: cum_scores.get(c["name"], {}).get("elimination_risk", 0.5)
        for c in active
    }
    return _normalize(elim_scores)


def build_predictions(
    show_slug: str,
    season: int,
    episode_number: int,
    llm_winner_probs: Optional[dict[str, float]] = None,
    llm_elim_probs: Optional[dict[str, float]] = None,
) -> dict:
    """
    Combine win and elimination probabilities into a single output dict.
    """
    win_probs = compute_winner_probabilities(show_slug, season, llm_winner_probs)
    elim_probs = compute_elimination_probabilities(show_slug, season, llm_elim_probs)

    active = get_active_contestants(show_slug, season)
    contestants_out: dict[str, dict] = {}
    for c in active:
        name = c["name"]
        contestants_out[name] = {
            "win_prob": round(win_probs.get(name, 0.0), 4),
            "elim_prob": round(elim_probs.get(name, 0.0), 4),
        }

    return {
        "show": show_slug,
        "season": season,
        "episode_number": episode_number,
        "n_remaining": len(active),
        "contestants": contestants_out,
    }
