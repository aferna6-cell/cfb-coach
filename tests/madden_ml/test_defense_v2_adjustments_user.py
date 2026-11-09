"""V2 defensive individual adjustments: real game named options, abstention and user jobs."""
from __future__ import annotations

from types import SimpleNamespace

from cfb_coach.madden.model import defense_v2_adjustments as a
from cfb_coach.madden.model import defense_v2_knowledge as k
from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.playcaller import MaddenCall


def belief(concept: str, strength: float) -> dict[str, float]:
    other = (1.0 - strength) / (len(v2.CONCEPTS) - 1)
    return {c: strength if c == concept else other for c in v2.CONCEPTS}


def ranked(play="Cover 3 Sky", risk=0.4, family="single_high"):
    return [{"formation": "Nickel Over", "play": play,
             "family": family, "defensive_utility_risk": risk,
             "predicted_offense_success": risk}]


def sit(distance=7, extras=None):
    return SimpleNamespace(distance=distance, red_zone=False, goal_line=False,
                           extras=extras or {})


def test_uncertain_offense_and_run_first_do_not_force_adjustment():
    model = v2.fit([])
    equal = {c: 1 / len(v2.CONCEPTS) for c in v2.CONCEPTS}
    pick = a.choose(model=model, sit=sit(), ranked=ranked(), belief=equal, context="long")
    assert pick["action_id"] == "NO_ADJUSTMENT"
    assert pick["adjustment_needed"] is False
    assert pick["compared_no_adjustment"]


def test_confident_vertical_threat_can_choose_existing_individual_adjustment():
    high = belief("vert", 0.89)
    pick = a.choose(model=v2.fit([]), sit=sit(), ranked=ranked(),
                    belief=high, context="long")
    assert pick["adjustment_needed"]
    assert pick["adjusted_risk"] < pick["stock_risk"]
    assert 1 <= len(pick["actions"]) <= 2
    assert all(x["kind"] == "individual" and x["game_exists"] for x in pick["actions"])
    assert all(x["source_ids"] and "buttons" not in x for x in pick["actions"])
    assert all(x["id"].startswith("def.") for x in pick["actions"])
    assert "macro" not in pick["source"]


def test_man_coverage_cannot_get_smart_zone_defender_change():
    actions = a._action_options(model={}, sit=sit(), play="Cover 2 Man",
                                family="man", belief=belief("vert", 0.88))
    assert not any("Smart Zone" in x["id"] for x in actions)


def test_individual_defender_man_match_only_if_both_target_names_known():
    none = a._action_options(model={}, sit=sit(), play="Cover 1 Robber",
                              family="man", belief=belief("cross", 0.88))
    assert not any(x["id"].startswith("def.hot_route") for x in none)
    with_target = a._action_options(
        model={}, sit=sit(extras={"defender_target": "MLB", "receiver_target": "TE1"}),
        play="Cover 1 Robber", family="man", belief=belief("cross", 0.88))
    assert any(x["id"] == "def.hot_route/match@MLB>TE1" for x in with_target)
    # No supported performance effect => should NOT be automatically executed.
    pick = a.choose(model=v2.fit([]), sit=sit(extras={
        "defender_target": "MLB", "receiver_target": "TE1"}),
        ranked=ranked(play="Cover 1 Robber", family="man"),
        belief=belief("cross", 0.88), context="long")
    assert all("def.hot_route" not in o["id"] for o in pick["actions"])


def test_user_guidance_updates_with_predicted_concept_and_scheme():
    ahead = k.user_guidance(
        play="Cover 3 Sky", formation="Nickel Over", family="single_high",
        belief=belief("cross", 0.85), sit=sit(distance=2))
    assert "run" in ahead["first_read"].lower()
    assert "crosser" in ahead["second_read"].lower()
    assert ahead["uncertain"] is False
    assert ahead["coverage_structure"]["named_defender_assignments_verified"] is False

    man = k.user_guidance(
        play="Cover 2 Man", formation="Dime", family="man",
        belief=belief("cross", 0.83), sit=sit())
    assert "assigned receiver" in man["second_read"]
    assert "man" in man["coverage_structure"]["concept_family"]


def test_defensive_adjustment_card_does_not_need_xbox_button_sequence():
    call = MaddenCall("defense", "Nickel Over", "Cover 3 Sky", "Back Off",
                      "User hook-curl defender; watch crossers", "test",
                      adjustment={"label": "Back Off", "id": "def.coverage_hub/Back Off",
                                  "why": "Expected vertical threats; prior based"})
    assert "ADJ: Back Off" in call.format()
    assert "press" not in call.adjustment_line()
    assert "watch crossers" in call.format()
