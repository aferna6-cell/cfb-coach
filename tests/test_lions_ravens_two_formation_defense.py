"""Safety checks for the inert Ravens two-formation defensive scheme dataset.

No model weights, live caller, or game DB are modified by these tests.
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET = (
    Path(__file__).resolve().parents[1]
    / "datasets/madden27/lions_ravens_two_formation_defense_v1.json"
)


def catalog() -> dict:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def _by_id() -> dict:
    rows = catalog()["stock_calls"]
    return {x["call_id"]: x for x in rows}


def test_exactly_two_ravens_formations():
    d = catalog()
    assert d["source_defensive_playbook"] == "Baltimore Ravens"
    assert [x["name"] for x in d["formations"]] == [
        "Nickel 2-4 Wide", "Dime 2-3 Odd"
    ]
    assert len(d["stock_calls"]) == 12
    assert len(_by_id()) == 12
    assert all(x["formation"] in {"Nickel 2-4 Wide", "Dime 2-3 Odd"} for x in d["stock_calls"])
    assert all(x["formation"] == "Nickel 2-4 Wide"
               for x in d["stock_calls"] if x["package"] != "dime_blitz")


def test_nickel_stock_macro_required_including_red_zone():
    d = catalog()
    rows = [x for x in d["stock_calls"] if x["package"] == "nickel_stock"]
    assert len(rows) == 4
    assert all(x["required_macro_id"] == "24_BASE" for x in rows)
    assert d["macro_catalog"]["24_BASE"]["display_name"] == "24 Base"
    assert _by_id()["nickel_tampa2"]["required_macro_id"] == "24_BASE"
    assert _by_id()["nickel_tampa2"]["redzone_priority_context"] == (
        "inside_opponent_10_not_forced"
    )


def test_blitz_only_dime_and_mandatory_db_fire():
    d = catalog()
    rows = [x for x in d["stock_calls"] if x["formation"] == "Dime 2-3 Odd"]
    assert len(rows) == 1
    assert rows[0]["base_play"] == "DB Fire 2"
    assert rows[0]["package"] == "dime_blitz"
    assert rows[0]["required_macro_id"] == "DB_FIRE"
    assert rows[0]["eligible_only_if_model_selects_blitz"] is True
    assert d["macro_catalog"]["DB_FIRE"]["display_name"] == "DB Fire"


def test_duplicate_cover2_match_has_distinct_macro_requirements():
    calls = _by_id()
    assert calls["match_cover2"]["base_play"] == calls["nickel_cover2"]["base_play"] == "Cover 2 Match"
    assert calls["match_cover2"]["required_macro_id"] is None
    assert calls["nickel_cover2"]["required_macro_id"] == "24_BASE"
    assert calls["match_cover2"]["call_id"] != calls["nickel_cover2"]["call_id"]


def test_preserve_all_match_guide_users_without_unsafe_transfer():
    d = catalog()
    rows = d["user_assignment_rules"]
    assert len(rows) == 18
    assert len({x["assignment_id"] for x in rows}) == 18
    assert all(x["guide_user"] for x in rows)
    assignments = {x["assignment_id"]: x for x in rows}
    assert assignments["quarters_vs_trips"]["guide_user"].lower() == "backside safety"
    assert assignments["quarters_vs_bunch"]["guide_user"].lower() == "weak-side safety"
    assert assignments["cover3_match_vs_trips"]["guide_user"].lower() == "hook curl"
    assert assignments["cover2_match_vs_5wr_tight"]["guide_user"].lower() == "short-side deep-half safety"
    assert assignments["cover6_vs_trips"]["guide_user"].lower() == "backside safety"
    assert all(x["transfer_to_nickel_formation"] == "requires_owner_in_game_confirmation" for x in rows)
    assert len(d["nickel_guide_examples_reference_only"]) == 10
    assert all(x["transfer_to_stock_without_setup"] == "NOT_VERIFIED"
               for x in d["nickel_guide_examples_reference_only"])


def test_zone_it_optional_and_stock_fallback_distinct():
    d = catalog()
    calls = _by_id()
    assert d["macro_catalog"]["ZONE_IT"]["display_name"] == "Zone It"
    assert "ZONE_IT" in calls["match_quarters"]["optional_macros"]
    assert "ZONE_IT" in calls["match_palms"]["optional_macros"]
    assert "ZONE_IT" not in calls["stock_cover4_zone"]["optional_macros"]
    assert calls["stock_cover4_zone"]["base_play"] == "Cover 4 Zone"
    assert all(x["manual_setup_changes_allowed"] is False for x in calls.values())


def test_cover9_match_guide_alias_does_not_pretend_verification():
    d = catalog()
    calls = _by_id()
    assert calls["match_cover9_show2"]["coverage_alias_status"] == "guide_cover9_to_show2_unverified"
    assert calls["match_cover9_show2"]["live_eligible"] is False
    assert all(x["intended_guide_play"] == "Cover 9"
               for x in d["user_assignment_rules"] if x["call_id"] == "match_cover9_show2")
