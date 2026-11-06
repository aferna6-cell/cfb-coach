import pytest

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.adjustments.executable import (
    AdjustmentError,
    Loadout,
    PlannedAction,
    PlayContext,
    validate_inputs,
    validate_plan,
)

CONFIRM_ALL_OFFENSE = {"off.hot_route": {"confirmed": True}, "off.hot_route|Slant": {"confirmed": True},
                       "off.hot_route|Flat": {"confirmed": True}, "off.audible": {"confirmed": True}}


def _loadout(**over):
    d = {
        "game_label": "example",
        "offense": [{"name": "Slants", "slot": "LB menu, 2nd item", "installed": True, "active": True,
                     "confirmed_by_user": True, "preset_id": "off.preset.slants"},
                    {"name": "O-RUN", "slot": "LB menu, 1st item", "installed": True, "active": True,
                     "confirmed_by_user": True}],
        "defense": [{"name": "Play Short Routes", "slot": "LB menu, 1st item", "installed": True, "active": True,
                     "confirmed_by_user": True, "preset_id": "def.preset.play_short_routes"},
                    {"name": "No Deep Passes", "slot": "LB menu, 2nd item", "installed": True, "active": True,
                     "confirmed_by_user": True, "preset_id": "def.preset.no_deep_passes"},
                    {"name": "TAMPA MABLE", "slot": "LB menu, 3rd item", "installed": True, "active": False,
                     "confirmed_by_user": True}],
    }
    d.update(over)
    return Loadout.from_dict(d)


def _off(**kw):
    base = dict(side="offense", formation="Gun Trips TE", play="Mesh", play_type="pass",
                eligible_receivers=("WR1", "WR2", "WR3", "TE1", "HB"),
                formation_audibles={"Mesh": "pass", "Inside Zone": "run"}, play_clock_remaining=20)
    base.update(kw)
    return PlayContext(**base)


def _def(**kw):
    base = dict(side="defense", formation="Nickel 2-4-5", play="Cover 3 Sky", play_type="coverage",
                eligible_receivers=("WR1", "WR2", "WR3", "TE1", "HB"), defenders=("CB1", "CB2", "SS", "FS", "MLB"),
                play_clock_remaining=20)
    base.update(kw)
    return PlayContext(**base)


# ---------------------------------------------------------------- registry integrity


def test_every_registered_input_sequence_is_valid_for_its_side():
    for spec in ex.registry().values():
        if spec.inputs:
            assert validate_inputs(spec.side, spec.category, spec.inputs) == [], spec.id
        for s in spec.sources:
            assert s in ex.load_registry_data()["sources"], (spec.id, s)


def test_unsupported_actions_have_no_inputs_and_a_reason():
    for spec in ex.registry().values():
        if spec.inputs_grade in ("undocumented", "conflict"):
            assert not spec.inputs and spec.blocked_reason, spec.id


def test_old_control_table_conflicts_are_flagged():
    assert ex.RESEARCH_CONTROL_AUDIT[("defense", "individual")][0] == "conflict"
    assert ex.RESEARCH_CONTROL_AUDIT[("defense", "qb_contain")][0] == "conflict"
    assert ex.RESEARCH_CONTROL_AUDIT[("offense", "hot_route")][0] == "confirmed"


# ---------------------------------------------------------------- unknown names and invalid inputs


def test_unknown_names_are_rejected():
    with pytest.raises(AdjustmentError):
        ex.find_by_name("defense", "Defend Underneath Passes")
    assert ex.find_by_name("defense", "Double Team").id == "def.double_team"
    errs = validate_plan([PlannedAction("def.bring_more_defenders")], _def())
    assert any("unknown adjustment" in e for e in errs)


@pytest.mark.parametrize("side,category,steps,fragment", [
    ("offense", "hot_route", ["tap:Z"], "not an Xbox button"),
    ("offense", "hot_route", ["hold:LT", "icon:receiver"], "Pass Protection on EA's offense map, not Hot Route"),
    ("offense", "hot_route", ["tap:Y", "tap:A"], "A snaps the ball"),
    ("defense", "individual_assignment", ["tap:Y", "icon:defender"], "Coverage Audible on EA's defense map"),
    ("defense", "coverage_hub", ["press LT"], "not a valid input"),
    ("defense", "front", ["stick:RS_UP"], "not a stick direction"),
])
def test_invalid_button_sequences_are_rejected(side, category, steps, fragment):
    assert any(fragment in e for e in validate_inputs(side, category, steps))


# ---------------------------------------------------------------- installed and active


def test_presets_and_saved_adjustments_need_a_confirmed_active_slot():
    errs = validate_plan([PlannedAction("off.preset.go_vertical")], _off(), loadout=_loadout())
    assert any("not confirmed installed and active" in e for e in errs)
    assert validate_plan([PlannedAction("off.preset.slants")], _off(), loadout=_loadout()) == []
    assert any("not confirmed" in e for e in validate_plan([PlannedAction("def.saved.TAMPA MABLE")], _def(), loadout=_loadout()))
    assert any("unknown adjustment" in e for e in validate_plan([PlannedAction("def.saved.NEW MACRO")], _def(), loadout=_loadout()))


def test_a_generated_design_is_never_installed():
    unconfirmed = Loadout.from_dict({"defense": [{"name": "QTRS OVERTOP", "slot": "LB menu, 1st item",
                                                  "installed": True, "active": True, "confirmed_by_user": False}]})
    assert any("not confirmed" in e for e in validate_plan([PlannedAction("def.saved.QTRS OVERTOP")], _def(), loadout=unconfirmed))


@pytest.mark.parametrize("bad, fragment", [
    ({"defense": [{"name": "A", "slot": "1", "installed": True, "active": True, "confirmed_by_user": True},
                  {"name": "B", "slot": "1", "installed": True, "active": True, "confirmed_by_user": True}]}, "same slot"),
    ({"defense": [{"name": "A", "slot": "1", "installed": False, "active": True, "confirmed_by_user": True}]}, "active but not installed"),
    ({"offense": [{"name": f"O{i}", "slot": str(i), "installed": True, "active": True, "confirmed_by_user": True} for i in range(6)],
      "defense": [{"name": f"D{i}", "slot": str(i), "installed": True, "active": True, "confirmed_by_user": True} for i in range(6)]},
     "may be per game"),
])
def test_invalid_loadouts_are_rejected(bad, fragment):
    with pytest.raises(AdjustmentError, match=fragment):
        Loadout.from_dict(bad)


# ---------------------------------------------------------------- legality and compatibility


def test_incompatible_combinations_are_rejected():
    two_routes = [PlannedAction("off.hot_route", {"target": "WR1", "route": "Slant"}),
                  PlannedAction("off.hot_route", {"target": "WR1", "route": "Flat"})]
    assert any("cannot both" in e for e in validate_plan(two_routes, _off()))
    preset_and_route = [PlannedAction("off.preset.slants"), PlannedAction("off.hot_route", {"target": "WR2", "route": "Flat"})]
    assert any("cannot both" in e for e in validate_plan(preset_and_route, _off(), loadout=_loadout()))
    zones = [PlannedAction("def.coverage_hub", {"option": "Smart Zone Aggressive"}),
             PlannedAction("def.coverage_hub", {"option": "Smart Zone Conservative"})]
    assert any("cannot both" in e for e in validate_plan(zones, _def()))
    ok = [PlannedAction("def.coverage_hub", {"option": "Inside Leverage"}), PlannedAction("def.double_team", {"target": "WR1"})]
    assert not any("cannot both" in e for e in validate_plan(ok, _def()))


def test_unsupported_receiver_or_defender_assignments_are_rejected():
    assert any("not an eligible receiver" in e
               for e in validate_plan([PlannedAction("off.hot_route", {"target": "WR4", "route": "Slant"})], _off()))
    assert any("not a defender" in e
               for e in validate_plan([PlannedAction("def.hot_route", {"target": "NB", "assignment": "x"})], _def()))
    assert any("unknown" in e
               for e in validate_plan([PlannedAction("def.double_team", {"target": "WR1"})], _def(eligible_receivers=())))
    assert any("UNSUPPORTED" in e.upper() or "not its Xbox input" in e
               for e in validate_plan([PlannedAction("off.chip_block", {"target": "TE1"})], _off()))


def test_actions_that_cannot_finish_before_the_snap_are_rejected():
    one = [PlannedAction("off.hot_route", {"target": "WR1", "route": "Slant"})]
    assert any("before the snap" in e for e in validate_plan(one, _off(play_clock_remaining=4)))
    two = one + [PlannedAction("off.pass_protection", {"protection": "Half Slide"})]
    assert validate_plan(two, _off(play_clock_remaining=20)) == []
    assert any("before the snap" in e for e in validate_plan(two, _off(play_clock_remaining=20, coach_mode=True)))


def test_actions_that_conflict_with_existing_settings_are_rejected():
    ctx = _def(current_settings={"coverage:smart_zone": "Smart Zone Conservative (applied this snap)"})
    errs = validate_plan([PlannedAction("def.coverage_hub", {"option": "Smart Zone Aggressive"})], ctx)
    assert any("already in effect" in e for e in errs)
    errs = validate_plan([PlannedAction("def.preset.play_short_routes")], ctx, loadout=_loadout())
    assert any("already in effect" in e for e in errs)


def test_actions_incompatible_with_formation_or_play_are_rejected():
    run = _off(play="Inside Zone", play_type="run")
    assert any("run play" in e for e in validate_plan([PlannedAction("off.hot_route", {"target": "WR1", "route": "Slant"})], run))
    assert any("run play" in e for e in validate_plan([PlannedAction("off.pass_protection", {"protection": "Half Slide"})], run))
    assert any("not an audible" in e for e in validate_plan([PlannedAction("off.audible", {"play": "Four Verts"})], _off()))
    assert any("Block" in e or "not a route" in e or "only" in e
               for e in validate_plan([PlannedAction("off.hot_route", {"target": "WR1", "route": "Wheel"})], _off()))


# ---------------------------------------------------------------- decisions


def test_the_unadjusted_play_is_always_a_candidate_and_the_live_default():
    opts = ex.candidate_options(_off(), {"man": 0.9, "two_high": 0.1})
    assert opts[0].spec.kind == "stock" and not opts[0].errors
    rec = ex.recommend(_off(), {"man": 0.9, "two_high": 0.1})
    assert rec.chosen.spec.kind == "stock"
    assert "best in theory" in rec.note


def test_live_mode_uses_only_confirmed_actions():
    rec = ex.recommend(_off(), {"man": 0.9, "two_high": 0.1}, confirmations=CONFIRM_ALL_OFFENSE)
    assert rec.chosen.status == "live_ready" and rec.chosen.spec.kind != "stock"
    research = ex.recommend(_off(), {"man": 0.9, "two_high": 0.1}, mode="research")
    assert research.chosen.status in ("live_ready", "needs_user_test")


def test_no_hardcoded_response_to_short_passing():
    lo = _loadout()
    cross_heavy = {"cross": 0.55, "flood": 0.25, "run": 0.1, "vert": 0.05, "rpo": 0.05}
    vert_heavy = {"vert": 0.6, "flood": 0.15, "cross": 0.1, "run": 0.15}
    even = {k: 1 / 7 for k in ("run", "rpo", "vert", "flood", "cross", "stack", "scram")}
    pick = lambda d: ex.recommend(_def(), d, loadout=lo).chosen.spec.name  # noqa: E731
    assert pick(cross_heavy) == "Play Short Routes"
    assert pick(vert_heavy) != "Play Short Routes"
    assert pick(even) == "No adjustment (run the called play)"


def test_coaching_card_has_every_field_and_uses_the_confirmed_slot():
    rec = ex.recommend(_def(), {"cross": 0.6, "flood": 0.3, "vert": 0.1}, loadout=_loadout())
    card = ex.coaching_card(rec, _def(), loadout=_loadout())
    for field in ("FORMATION:", "PLAY:", "MADDEN ADJUSTMENT:", "XBOX INPUTS:", "REQUIRED PLAYER OR SLOT:",
                  "WHY THIS HELPS:", "WHAT IT EXPOSES:", "CONFIDENCE:", "EXECUTION STATUS:"):
        assert field in card
    assert "Press LB" in card and "LB menu, 1st item" in card
    assert "MEASURED" not in card


def test_loadout_planner_never_claims_installation_and_respects_the_limit():
    rows = ex.plan_loadout("defense", [(0.6, {"cross": 0.6, "vert": 0.4}), (0.4, {"scram": 0.5, "run": 0.5})], limit=2)
    assert len(rows) <= 2
    assert all("not installed until you confirm" in r["status"] for r in rows)


def test_catalog_report_without_confirmations_has_nothing_fully_verified():
    rep = ex.catalog_report()
    assert rep["fully_verified"] == []
    assert "defense: Double Team" in rep["ea_documented_end_to_end_needs_test"]
    assert any("QB Contain" in r for r in rep["unsupported"])
    assert any("Go Vertical" in r for r in rep["not_installed"])
