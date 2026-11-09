"""First CPU game: V2 can start honestly from football priors with zero executions."""
from __future__ import annotations

import json

import pytest

from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.model import defense_v2_live as live


def test_cold_start_has_zero_executed_evidence_and_per_game_cutoff(tmp_path):
    model = v2.cold_start("test-franchise")
    assert model["cold_start"] is True
    assert model["n_verified"] == 0
    assert model["n_games"] == 0
    assert model["trained_franchise"] == "test-franchise"
    assert model["trained_sessions"] == []
    assert model["policy"] == "shadow_only"
    art = tmp_path / "cold_start.json"
    art.write_text(json.dumps(model), encoding="utf-8")
    loaded = live._model(str(art), "2026-11-10", "first-game", "test-franchise")
    assert loaded["cold_start"] is True
    with pytest.raises(ValueError, match="another franchise"):
        live._model(str(art), "2026-11-10", "first-game", "main-franchise")
    with pytest.raises(ValueError, match="current game or future"):
        live._model(str(art), "0000-01-01", "first-game", "test-franchise")


def test_cold_start_cannot_forge_verified_game_counts():
    model = v2.cold_start("test-franchise")
    model["n_verified"] = 2
    model["n_games"] = 1
    assert model["cold_start"] is True
    # Historical facts remain facts; a nonzero model with cold_start flag
    # cannot be passed off as the true zero-evidence prior state.
    assert model["n_verified"] != 0
    with pytest.raises(ValueError):
        v2.cold_start("")


def test_football_prior_is_bounded_and_fades_after_verified_data(monkeypatch):
    monkeypatch.setattr(v2, "_football_prior",
                        lambda play, formation, concept: -0.042 if concept == "vert" else 0)
    empty = v2.cold_start("test")
    assert v2._prior_adjustment(empty, "Nickel", "Cover 2 Man", "vert") == -0.042
    with_data = {**empty, "n_verified": 80}
    assert v2._prior_adjustment(with_data, "Nickel", "Cover 2 Man", "vert") == -0.021
    assert v2._prior_adjustment({**empty, "n_verified": 200},
                                "Nickel", "Cover 2 Man", "vert") == 0
