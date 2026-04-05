"""Claude API wrapper for episode analysis and predictions."""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from src import config
from src.analysis import prompts


# ---------------------------------------------------------------------------
# Cost tracking (cumulative within a process lifetime)
# ---------------------------------------------------------------------------

_cost_tracker: dict[str, Any] = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cost_usd": 0.0,
    "calls": 0,
}

# Approximate pricing for claude-sonnet-4 (update as needed)
_COST_PER_1K_INPUT = 0.003
_COST_PER_1K_OUTPUT = 0.015


def get_cost_summary() -> dict:
    return dict(_cost_tracker)


def reset_cost_tracker() -> None:
    for k in _cost_tracker:
        _cost_tracker[k] = 0 if isinstance(_cost_tracker[k], (int, float)) else []
    _cost_tracker["calls"] = 0


def _track_usage(usage: anthropic.types.Usage) -> dict:
    inp = usage.input_tokens
    out = usage.output_tokens
    cost = (inp / 1000) * _COST_PER_1K_INPUT + (out / 1000) * _COST_PER_1K_OUTPUT
    _cost_tracker["total_input_tokens"] += inp
    _cost_tracker["total_output_tokens"] += out
    _cost_tracker["total_cost_usd"] += cost
    _cost_tracker["calls"] += 1
    return {"input_tokens": inp, "output_tokens": out, "cost_usd": round(cost, 4)}


# ---------------------------------------------------------------------------
# Core LLM call with retry
# ---------------------------------------------------------------------------

def _call_llm(
    prompt: str,
    model: str = config.LLM_MODEL_ANALYSIS,
    temperature: float = config.LLM_TEMP_ANALYSIS,
    max_tokens: int = 4096,
) -> tuple[str, dict]:
    """
    Call Claude and return (raw_text, usage_dict).
    Retries up to LLM_MAX_RETRIES times on failure.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    last_error = None

    for attempt in range(config.LLM_MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            usage = _track_usage(response.usage)
            return text, usage
        except Exception as e:
            last_error = e
            if attempt < config.LLM_MAX_RETRIES:
                wait = 2 ** attempt
                print(f"[llm] Error on attempt {attempt + 1}: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {config.LLM_MAX_RETRIES + 1} attempts: {last_error}")


def _parse_json_response(text: str, attempt: int = 0) -> Any:
    """Extract and parse JSON from LLM response, with one retry on malformed output."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().rstrip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {e}\nRaw: {text[:500]}")


# ---------------------------------------------------------------------------
# High-level analysis functions
# ---------------------------------------------------------------------------

def analyze_episode(
    show_slug: str,
    season: int,
    episode_number: int,
    episode_content: str,
    contestant_list: list[str],
) -> dict:
    """
    Run episode analysis. Returns parsed JSON dict with contestant scores.
    Retries if JSON is malformed.
    """
    prompt = prompts.format_episode_analysis_prompt(
        show_slug, season, episode_number, episode_content, contestant_list
    )

    for attempt in range(config.LLM_MAX_RETRIES + 1):
        raw, usage = _call_llm(prompt, temperature=config.LLM_TEMP_ANALYSIS)
        try:
            result = _parse_json_response(raw)
            result["llm_cost"] = usage
            return result
        except ValueError as e:
            if attempt < config.LLM_MAX_RETRIES:
                print(f"[llm] Malformed JSON on attempt {attempt + 1}: {e}. Retrying...")
            else:
                raise

    raise RuntimeError("analyze_episode: unreachable")


def predict_elimination(scores_json: str, game_state: str) -> dict:
    """Return elimination probability dict from LLM."""
    prompt = prompts.format_elimination_prompt(scores_json, game_state)
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        raw, usage = _call_llm(prompt, temperature=config.LLM_TEMP_PREDICTION)
        try:
            result = _parse_json_response(raw)
            result["llm_cost"] = usage
            return result
        except ValueError as e:
            if attempt < config.LLM_MAX_RETRIES:
                print(f"[llm] Malformed JSON attempt {attempt + 1}: {e}. Retrying...")
            else:
                raise

    raise RuntimeError("predict_elimination: unreachable")


def predict_winner(
    show_slug: str,
    season: int,
    all_episode_scores: str,
    eliminated: list[str],
) -> dict:
    """Return winner probability dict from LLM."""
    prompt = prompts.format_winner_prompt(show_slug, season, all_episode_scores, eliminated)
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        raw, usage = _call_llm(prompt, temperature=config.LLM_TEMP_PREDICTION)
        try:
            result = _parse_json_response(raw)
            result["llm_cost"] = usage
            return result
        except ValueError as e:
            if attempt < config.LLM_MAX_RETRIES:
                print(f"[llm] Malformed JSON attempt {attempt + 1}: {e}. Retrying...")
            else:
                raise

    raise RuntimeError("predict_winner: unreachable")


def quick_live_analysis(
    show_slug: str,
    act_summary: str,
    contestant_list: list[str],
    previous_scores: str,
) -> dict:
    """Fast in-episode analysis using the faster model."""
    prompt = prompts.format_live_prompt(show_slug, act_summary, contestant_list, previous_scores)
    raw, usage = _call_llm(
        prompt,
        model=config.LLM_MODEL_LIVE,
        temperature=config.LLM_TEMP_ANALYSIS,
        max_tokens=1024,
    )
    result = _parse_json_response(raw)
    result["llm_cost"] = usage
    return result
