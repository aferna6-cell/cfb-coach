"""Postgame V2 comparisons keep the incumbent artifact untouched and avoid leakage."""
from __future__ import annotations

import pytest

from cfb_coach.madden.model import defense_v2_pilot as v2


def sample(game: int, seq: int, success: bool) -> v2.Snap:
    return v2.Snap(
        f"g{game}", "cpu", f"2026-11-{game+1:02}", seq,
        "Nickel Over", "Cover 3 Sky", "single_high", "long", "cross", success
    )


def test_saved_champion_is_evaluated_only_on_newer_games():
    history = [sample(0, i, False) for i in range(1, 11)]
    champ = v2.fit(history)
    champ["trained_sessions"] = ["g0"]
    champ["trained_through_started_ts"] = "2026-11-01"
    future = [sample(1, i, True) for i in range(1, 12)]
    report = v2.compare_champion_challenger(history + future, champ, bootstraps=20)
    assert report["prospective_test_games"] == 1
    assert report["prospective_test_snaps"] == 11
    assert report["candidate_predictive_gate"] == "NO_GO"
    assert report["automatic_promotion"] is False
    assert report["live_policy_improvement_proven"] is False


def test_missing_champion_cutoff_refuses_comparison():
    model = v2.fit([sample(0, 1, True)])
    with pytest.raises(ValueError, match="chronological cutoff"):
        v2.compare_champion_challenger([], model)
