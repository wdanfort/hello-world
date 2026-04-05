"""Tests for LLM analyzer components."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.analysis import scoring
from src.analysis.prompts import (
    build_scoring_dimensions_block,
    format_episode_analysis_prompt,
    format_elimination_prompt,
    format_winner_prompt,
)


def test_build_scoring_dimensions_block():
    block = build_scoring_dimensions_block("survivor")
    assert "edit_visibility" in block
    assert "winner_signals" in block
    assert "1." in block


def test_format_episode_analysis_prompt():
    prompt = format_episode_analysis_prompt(
        show_slug="survivor",
        season=50,
        episode_number=3,
        episode_content="Test recap content.",
        contestant_list=["Alice", "Bob", "Carol"],
    )
    assert "Season 50" in prompt
    assert "Episode 3" in prompt
    assert "Alice" in prompt
    assert "edit_visibility" in prompt


def test_format_elimination_prompt():
    prompt = format_elimination_prompt(
        scores_json='{"Alice": {"elimination_risk": 8}}',
        game_state="Post-merge, Alice on the outs.",
    )
    assert "elimination_probabilities" in prompt
    assert "Alice" in prompt


def test_format_winner_prompt():
    prompt = format_winner_prompt(
        show_slug="survivor",
        season=50,
        all_episode_scores='{"ep1": {"Alice": {"winner_signals": 8}}}',
        eliminated=["Bob"],
    )
    assert "winner_probabilities" in prompt
    assert "Bob" in prompt


def test_normalize_score():
    assert scoring.normalize_score(1.0) == pytest.approx(0.0)
    assert scoring.normalize_score(10.0) == pytest.approx(1.0)
    assert scoring.normalize_score(5.5) == pytest.approx(0.5)
    # Clamping
    assert scoring.normalize_score(0.0) == pytest.approx(0.0)
    assert scoring.normalize_score(11.0) == pytest.approx(1.0)


def test_normalize_contestant_scores():
    data = {
        "edit_visibility": 8,
        "winner_signals": 9,
        "narrative_summary": "Dominated the episode.",
    }
    result = scoring.normalize_contestant_scores(data, "survivor")
    assert result["edit_visibility"] == pytest.approx(scoring.normalize_score(8))
    assert result["winner_signals"] == pytest.approx(scoring.normalize_score(9))
    # Non-numeric fields preserved as-is
    assert result["narrative_summary"] == "Dominated the episode."
