"""Historical Survivor base rates used as Bayesian priors."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Winner base rates by archetype
# Based on modern Survivor (Seasons 28+) historical outcomes
# ---------------------------------------------------------------------------

WINNER_RATE_BY_ARCHETYPE: dict[str, float] = {
    "utr": 0.32,        # Under-the-radar players win most often in modern era
    "social": 0.28,     # Strong social games frequently rewarded at FTC
    "strategic": 0.25,  # Strategic players win but are often targeted pre-FTC
    "wildcard": 0.10,   # Flashy players entertain but rarely win
    "physical": 0.05,   # Challenge beasts rarely win (jury perception)
}

# Base rate of winning for a random contestant (uniform prior adjusted by archetype)
UNIFORM_WIN_PRIOR = 1.0  # Will be normalized by N remaining contestants

# ---------------------------------------------------------------------------
# Edit type → implied win probability signal
# These are used as likelihood multipliers on top of the Bayesian prior.
# Higher = more predictive of winning.
# ---------------------------------------------------------------------------

EDIT_WIN_LIKELIHOOD: dict[str, float] = {
    "winner_edit_strong": 3.5,    # Classic winner arc: emotional, strategic, consistent
    "winner_edit_moderate": 2.0,  # Some winner signals but gaps in edit
    "invisible": 0.4,             # Invisible edits are risky — may be boot soon or UTR winner
    "villain_edit": 0.5,          # Villain edits rarely win; jury bitterness
    "positive_consistent": 1.8,   # Consistently positive but no clear winner narrative
    "neutral": 1.0,               # Neutral baseline
}

# ---------------------------------------------------------------------------
# Pre-merge vs post-merge elimination base rates
# Approximate: pre-merge ~ first 1/3 of episodes, post-merge follows swap dynamics
# ---------------------------------------------------------------------------

PRE_MERGE_ELIM_RATE: float = 0.125     # ~1 in 8 pre-merge players go per episode
POST_MERGE_ELIM_RATE: float = 0.111   # ~1 in 9 post-merge players go per episode (harder to predict)

# ---------------------------------------------------------------------------
# Position-based modifiers for elimination likelihood
# Applied multiplicatively to the LLM elimination risk score
# ---------------------------------------------------------------------------

DAYS_SINCE_VOTES_MODIFIER: dict[int, float] = {
    0: 1.2,   # Got votes last tribal — still in danger
    1: 1.0,   # Normal
    2: 0.9,   # Managed to avoid tribal last time
    3: 0.85,  # Multiple tribals survived — established enough to avoid
}


def archetype_win_prior(archetype: str) -> float:
    """Return the base win rate for a given archetype (0-1 scale)."""
    return WINNER_RATE_BY_ARCHETYPE.get(archetype, 0.15)


def get_uniform_prior(n_remaining: int) -> float:
    """Uniform win prior: 1/N."""
    return 1.0 / n_remaining if n_remaining > 0 else 0.0
