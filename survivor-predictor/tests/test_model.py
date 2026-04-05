"""Tests for the prediction model."""

import pytest
from src.prediction.model import _normalize
from src.prediction.history import archetype_win_prior, get_uniform_prior
from src.simulation.simulator import kelly_fraction


def test_normalize_sums_to_one():
    probs = {"Alice": 0.3, "Bob": 0.5, "Carol": 0.2}
    result = _normalize(probs)
    assert sum(result.values()) == pytest.approx(1.0)


def test_normalize_handles_zero_total():
    probs = {"Alice": 0.0, "Bob": 0.0}
    result = _normalize(probs)
    assert sum(result.values()) == pytest.approx(1.0)
    assert result["Alice"] == pytest.approx(0.5)


def test_archetype_prior_known():
    assert archetype_win_prior("utr") > archetype_win_prior("physical")
    assert archetype_win_prior("social") > archetype_win_prior("wildcard")


def test_archetype_prior_unknown():
    p = archetype_win_prior("unknown_archetype")
    assert 0 < p < 1


def test_uniform_prior():
    assert get_uniform_prior(4) == pytest.approx(0.25)
    assert get_uniform_prior(10) == pytest.approx(0.1)
    assert get_uniform_prior(0) == 0.0


def test_kelly_fraction_positive_edge():
    # Model says 60%, market says 30% — clear edge
    frac = kelly_fraction(0.60, 0.30, kelly_frac=1.0)
    assert frac > 0


def test_kelly_fraction_no_edge():
    # Model equals market — no Kelly bet (floating-point near-zero)
    frac = kelly_fraction(0.30, 0.30, kelly_frac=1.0)
    assert frac == pytest.approx(0.0, abs=1e-10)


def test_kelly_fraction_negative_edge():
    # Model below market — no bet (don't short)
    frac = kelly_fraction(0.10, 0.40, kelly_frac=1.0)
    assert frac == 0.0


def test_kelly_fraction_scaled():
    full = kelly_fraction(0.60, 0.30, kelly_frac=1.0)
    quarter = kelly_fraction(0.60, 0.30, kelly_frac=0.25)
    assert quarter == pytest.approx(full * 0.25)
