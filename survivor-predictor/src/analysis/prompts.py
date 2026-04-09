"""All LLM prompt templates. Injected with show config at runtime."""

from __future__ import annotations

from src import config


EPISODE_ANALYSIS_PROMPT = """\
You are an expert {show_name} analyst. Given the following episode recap and fan commentary \
for {show_name} Season {season}, Episode {episode_number}, analyze each remaining contestant \
on these dimensions. Return ONLY valid JSON, no preamble, no markdown fences.

For each remaining contestant, score on a 1-10 scale:

{scoring_dimensions_block}

Also provide:
- "narrative_summary": 1-2 sentence summary of their episode arc
- "key_relationships": List of contestant names they are aligned with or in conflict with

Episode content:
{episode_content}

Remaining contestants: {contestant_list}

Respond with a JSON object:
{{
  "contestants": {{
    "Name": {{
      <scores>,
      "narrative_summary": "...",
      "key_relationships": ["Name (ally)", "Name (conflict)"]
    }},
    ...
  }},
  "episode_summary": "...",
  "key_events": ["...", ...]
}}
"""

ELIMINATION_PREDICTION_PROMPT = """\
Based on the analysis scores below and the current game state, predict the probability of each \
contestant being eliminated next episode. Consider: who is on the wrong side of the numbers, \
who is being set up by the edit, physical threats post-merge, and tribal dynamics.

Return ONLY valid JSON, no preamble, no markdown fences.
Format: {{ "elimination_probabilities": {{ "Name": <float 0-1>, ... }} }}
Probabilities must sum to 1.0.

Current scores:
{scores_json}

Game state:
{game_state}
"""

WINNER_PREDICTION_PROMPT = """\
Based on the cumulative analysis scores across all episodes so far, predict the probability of \
each remaining contestant winning {show_name} Season {season}. Weight the following factors:
- Cumulative winner edit signals (most important)
- Strategic positioning trend (improving vs declining)
- Threat level vs ability to manage it
- Alliance durability
- Historical {show_name} patterns (e.g., UTR women often win modern Survivor, challenge beasts rarely win)

Return ONLY valid JSON, no preamble, no markdown fences.
Format: {{ "winner_probabilities": {{ "Name": <float 0-1>, ... }} }}
Probabilities must sum to 1.0.

Cumulative scores by episode:
{all_episode_scores}

Eliminated contestants: {eliminated}
"""

LIVE_QUICK_ANALYSIS_PROMPT = """\
You are a fast {show_name} analyst during a live episode. Based on the act-break summary \
below, briefly update your read on each contestant. Be concise — this is real-time analysis.

Return ONLY valid JSON:
{{
  "quick_updates": {{
    "Name": {{
      "elimination_risk_delta": <int -3 to +3, change from before>,
      "winner_signals_delta": <int -3 to +3>,
      "note": "one sentence"
    }}
  }},
  "act_summary": "one sentence"
}}

Act break summary: {act_summary}
Contestants still in: {contestant_list}
Previous scores: {previous_scores}
"""


def build_scoring_dimensions_block(show_slug: str) -> str:
    """Format scoring dimensions from YAML config into prompt text."""
    dims = config.load_show_config(show_slug)["scoring_dimensions"]
    lines = []
    for i, dim in enumerate(dims, 1):
        lines.append(f'{i}. "{dim["key"]}": {dim["description"]} (1-10)')
    return "\n".join(lines)


def format_episode_analysis_prompt(
    show_slug: str,
    season: int,
    episode_number: int,
    episode_content: str,
    contestant_list: list[str],
) -> str:
    show_config = config.load_show_config(show_slug)
    show_name = show_config["show"]["name"]
    dims_block = build_scoring_dimensions_block(show_slug)
    return EPISODE_ANALYSIS_PROMPT.format(
        show_name=show_name,
        season=season,
        episode_number=episode_number,
        scoring_dimensions_block=dims_block,
        episode_content=episode_content,
        contestant_list=", ".join(contestant_list),
    )


def format_elimination_prompt(scores_json: str, game_state: str) -> str:
    return ELIMINATION_PREDICTION_PROMPT.format(
        scores_json=scores_json,
        game_state=game_state,
    )


def format_winner_prompt(
    show_slug: str,
    season: int,
    all_episode_scores: str,
    eliminated: list[str],
) -> str:
    show_config = config.load_show_config(show_slug)
    show_name = show_config["show"]["name"]
    return WINNER_PREDICTION_PROMPT.format(
        show_name=show_name,
        season=season,
        all_episode_scores=all_episode_scores,
        eliminated=", ".join(eliminated) if eliminated else "none yet",
    )


def format_live_prompt(
    show_slug: str,
    act_summary: str,
    contestant_list: list[str],
    previous_scores: str,
) -> str:
    show_config = config.load_show_config(show_slug)
    show_name = show_config["show"]["name"]
    return LIVE_QUICK_ANALYSIS_PROMPT.format(
        show_name=show_name,
        act_summary=act_summary,
        contestant_list=", ".join(contestant_list),
        previous_scores=previous_scores,
    )
