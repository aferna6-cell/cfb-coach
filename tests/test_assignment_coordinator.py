"""Research bridge integration tests, no live coaching changes."""
from types import SimpleNamespace

import pytest

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import assignment_coordinator as bridge
from cfb_coach.madden.model.play_knowledge import (
    KnowledgeError, PlayKnowledge, PlayerJob, play_key,
)


def _play(side="offense"):
    name = "Mesh" if side == "offense" else "Cover 3 Sky"
    jobs = (PlayerJob("WR1", "route", "curl", "source_documented", "diagram-1"),
            PlayerJob("WR2", "route", "streak", "source_documented", "diagram-1")) if side == "offense" else (
            PlayerJob("MLB", "zone", "hook_curl", "source_documented", "diagram-2"),)
    return PlayKnowledge(play_key(side, "Sample", "Gun Trips" if side == "offense" else "Nickel Over", name),
                         side, "Sample", "Gun Trips" if side == "offense" else "Nickel Over",
                         name, "catalog", jobs)


def _stub(monkeypatch, kind, name, action_id, params=None):
    spec = SimpleNamespace(kind=kind, name=name, id=action_id,
                           params={"route": {}} if action_id == "off.hot_route" else {})
    item = SimpleNamespace(spec=spec, params=params or {})
    monkeypatch.setattr(bridge.ex, "recommend", lambda *args, **kwargs:
                        SimpleNamespace(chosen=item))
    monkeypatch.setattr(bridge.ex, "coaching_card", lambda *args, **kwargs: "CARD")


def test_defaults_to_no_adjustment_and_advice_not_fabricated(monkeypatch):
    play = _play("defense")
    ctx = ex.PlayContext(side="defense", formation="Nickel Over", play="Cover 3 Sky", play_type="coverage")
    _stub(monkeypatch, "stock", "No adjustment", "def.no_adjustment")
    decision = bridge.evaluate_assignment_decision(ctx, play, {"cross": .8}, user_defender="MLB")
    assert decision.adjustment == "NO ADJUSTMENT"
    assert decision.user_advice == "USER MLB: zone hook_curl"
    assert decision.status == "stock"


def test_exact_hot_route_transformation_does_not_invent_progression(monkeypatch):
    play = _play()
    ctx = ex.PlayContext(side="offense", formation="Gun Trips", play="Mesh", play_type="pass",
                         eligible_receivers=("WR1", "WR2"))
    _stub(monkeypatch, "individual_presnap", "Hot Route", "off.hot_route",
          {"target": "WR1", "route": "Slant"})
    d = bridge.evaluate_assignment_decision(ctx, play, {"man": .9})
    assert d.adjustment == "Hot Route: WR1 -> Slant"
    assert d.status == "action_confirmed; assignment_change_known"
    assert d.read_advice == "READ UNKNOWN"
    assert play.jobs_for("WR1")[0].task == "curl"  # no mutation


def test_unknown_route_effect_stays_unknown_and_mismatched_play_fails(monkeypatch):
    play = _play()
    ctx = ex.PlayContext(side="offense", formation="Gun Trips", play="Mesh", play_type="pass",
                         eligible_receivers=("WR1", "WR2"))
    _stub(monkeypatch, "individual_presnap", "Hot Route", "off.hot_route",
          {"target": "WR1", "route": "Unrecognized Route"})
    assert bridge.evaluate_assignment_decision(ctx, play, {"man": .9}).status.endswith("detailed_effect_unknown")
    bad = ex.PlayContext(side="offense", formation="Gun Trips", play="Mesh Switch", play_type="pass")
    with pytest.raises(KnowledgeError):
        bridge.evaluate_assignment_decision(bad, play, {"man": .5})



def test_scored_plans_choose_no_adjustment_without_executable_gain(monkeypatch):
    ctx = ex.PlayContext(side="offense", formation="Gun Trips", play="Mesh", play_type="pass",
                         eligible_receivers=("WR1", "WR2"))
    monkeypatch.setattr(bridge.ex, "validate_plan", lambda p, *args, **kwargs:
                        ["not installed"] if p[0].action_id.endswith("bad") else [])
    bad = bridge.ScoredPlan((SimpleNamespace(action_id="off.saved.bad", params={}),), .95, "hypothesis")
    weak = bridge.ScoredPlan((SimpleNamespace(action_id="off.hot_route", params={"target":"WR1", "route":"Drag"}),), .605, "prior")
    chosen = bridge.select_scored_plan(ctx, .60, [bad, weak])
    assert chosen.label == "NO ADJUSTMENT"
    assert not chosen.actions


def test_scored_plans_support_validated_macro_plus_individual(monkeypatch):
    ctx = ex.PlayContext(side="offense", formation="Gun Trips", play="Mesh", play_type="pass",
                         eligible_receivers=("WR1", "WR2"))
    monkeypatch.setattr(bridge.ex, "validate_plan", lambda plan, *args, **kwargs: [])
    combo = bridge.ScoredPlan((SimpleNamespace(action_id="off.saved.Beat Cover 2", params={}),
                               SimpleNamespace(action_id="off.hot_route", params={"target":"WR2", "route":"Drag"})),
                              .74, "research_model")
    chosen = bridge.select_scored_plan(ctx, .65, [combo])
    assert len(chosen.actions) == 2
    assert "off.saved.Beat Cover 2" in chosen.label
    assert "WR2" in chosen.label