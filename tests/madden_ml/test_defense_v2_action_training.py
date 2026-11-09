"""Only observed/confirmed executed defensive changes can train the adjustment learner."""
from __future__ import annotations
from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.model import defense_v2_adjustments as adj


def row(game: int, seq: int, *, action: str | None, success: bool) -> v2.Snap:
    return v2.Snap(
        f"g{game}", "cpu", f"2026-11-{game+1:02}", seq,
        "Nickel Over", "Cover 3 Sky", "single_high", "long", "vert",
        success, adjustment_id=action,
    )


def test_verified_action_comparison_is_fit_from_executed_labels_only():
    action_id = "def.coverage_hub/Back Off"
    controls = [row(g, i, action=None, success=True)
                for g in range(3) for i in range(1, 16)]
    # 3 independent games with a recorded adjustment and fewer offensive wins.
    adjusted = [row(g, i, action=action_id, success=False)
                for g in range(3, 6) for i in range(1, 16)]
    model = v2.fit(controls + adjusted)
    gain, sample = adj._learned_delta(model, "long", "single_high", action_id)
    assert sample == 45
    assert gain is not None and gain > 0.0
    assert model["adjustment_effects"]


def test_no_comparison_games_never_promote_made_up_adjustment_lift():
    action_id = "def.coverage_hub/Back Off"
    model = v2.fit([row(1, i, action=action_id, success=False)
                    for i in range(1, 30)])
    gain, sample = adj._learned_delta(model, "long", "single_high", action_id)
    assert gain is None
    assert sample == 0
