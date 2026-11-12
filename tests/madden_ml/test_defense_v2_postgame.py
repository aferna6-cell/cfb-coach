"""Postgame challenger evaluation must preserve the frozen champion."""
from __future__ import annotations

import pytest

from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.model import defense_v2_postgame as post


def record(game: str, start: str, seq: int, success: bool) -> v2.Snap:
    return v2.Snap(game, "cpu", start, seq,
                   "Nickel Over", "Cover 3 Sky", "single_high",
                   "long", "cross", success)


def champion(rows):
    model = v2.fit(rows)
    model["trained_sessions"] = sorted({x.session for x in rows})
    model["trained_through_started_ts"] = max(x.started_ts for x in rows)
    return model


def test_same_game_not_evaluated_and_underpowered_report_never_promotes():
    earlier = [record("g1", "2026-11-01", 1, True)]
    later = [record("g2", "2026-11-02", 1, False),
             record("g2", "2026-11-02", 2, True),
             record("g3", "2026-11-03", 1, False)]
    report = post.compare_champion(champion(earlier), earlier + later, bootstrap=30)
    assert report["new_games"] == 2
    assert report["new_snaps"] == 3
    assert {r["game"] for r in report["per_game"]} == {"g2", "g3"}
    assert report["predictive_promotion_gate"] == "NO_GO"
    assert report["model_promotion"] == "MANUAL_REVIEW_REQUIRED"
    assert not report["actual_policy_value_proven"]


def test_invalid_champion_fails_closed():
    with pytest.raises(ValueError):
        post.compare_champion({"schema": "untrusted"}, [])


def test_no_leak_of_future_same_game_concepts(monkeypatch):
    earlier = [record("g1", "2026-11-01", 1, True)]
    same = [record("g2", "2026-11-02", 1, False),
            record("g2", "2026-11-02", 2, True)]
    original = post.compare_champion(champion(earlier), earlier + same, bootstrap=25)
    # Change only the outcome of the *later* snap: the earlier-snap prediction
    # must be identical; the test isolates prediction rather than aggregate loss.
    model = v2.fit(earlier)
    first = post._predict(model, same[0], same)
    modified = [same[0], record("g2", "2026-11-02", 2, False)]
    assert post._predict(model, modified[0], modified) == first
    assert original["new_games"] == 1
