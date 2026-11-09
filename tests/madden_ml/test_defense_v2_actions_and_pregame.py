"""Individual defensive V2 action legality, real-execution support and pregame model scoring."""
from __future__ import annotations

import json
import sqlite3

from cfb_coach.madden.model import defense_v2_actions as actions
from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.model import playbook_architect as architect


def _confirmed(i="def.coverage_hub", instance="press"):
    return {
        "id": instance, "catalog_action_id": i, "label": "Physically verified press",
        "buttons": "Hold LT, choose Press", "params": {"option": "Press"},
        "target": None, "exclusive_groups": ["hub:coverage"], "touches": ["coverage"],
        "estimated_seconds": 3, "verified_on": "2026-11-09",
    }


def test_no_adjustment_wins_without_verified_effect():
    model = {"executed_adjustments": {"counts": {}}}
    result = actions.choose_adjustments(
        model, context="long", formation="Nickel Over", play="Cover 3 Sky",
        stock_offense_success=0.7, confirmed=[_confirmed()])
    assert result["id"] == "NO_ADJUSTMENT"
    assert not result["actions"]


def test_only_exact_verified_action_support_can_improve_on_stock():
    from cfb_coach.madden.model.defense_v2_pilot import _key
    key = _key("long", "Nickel Over", "Cover 3 Sky", "press")
    model = {"executed_adjustments": {"counts": {key: [20, 200]}}}
    result = actions.choose_adjustments(
        model, context="long", formation="Nickel Over", play="Cover 3 Sky",
        stock_offense_success=0.7, confirmed=[_confirmed()])
    assert result["id"] == "press"
    assert result["actions"][0]["buttons"] == "Hold LT, choose Press"
    assert result["no_adjustment_compared"]
    assert not result["effect_is_causal"]
    # A candidate against a different call must not transfer its outcome.
    other = actions.choose_adjustments(
        model, context="long", formation="Dime", play="Cover 3 Sky",
        stock_offense_success=0.7, confirmed=[_confirmed()])
    assert other["id"] == "NO_ADJUSTMENT"


def test_conflicts_block_multi_adjustment():
    a = _confirmed(instance="press")
    b = _confirmed(instance="backoff")
    assert not actions.compatible([a, b])
    b["exclusive_groups"] = ["front"]
    b["touches"] = ["line"]
    assert actions.compatible([a, b])


def test_unverified_or_unsupported_xbox_action_is_not_enabled(tmp_path):
    confirmation_file = tmp_path / "confirmations.json"
    confirmation_file.write_text(json.dumps({
        "schema": actions.CONFIRM_SCHEMA,
        "confirmed_actions": [
            {"action_id": "def.qb_contain", "instance_id": "contain",
             "physically_verified": True, "console": "Xbox",
             "verified_on": "2026-11-09", "verified_buttons": "RB LB"},
            {"action_id": "def.coverage_hub", "instance_id": "press",
             "physically_verified": True, "console": "Xbox",
             "verified_on": "2026-11-09", "verified_buttons": "LT press",
             "params": {"option": "Press"}},
            {"action_id": "def.coverage_hub", "instance_id": "not_verified",
             "physically_verified": False, "console": "Xbox",
             "verified_on": "2026-11-09", "verified_buttons": "LT", "params": {"option": "Press"}},
        ],
    }))
    opts = actions.read_confirmed_options(confirmation_file)
    assert [o["id"] for o in opts] == ["press"]


def test_adjustment_trainer_requires_verified_execution_and_exact_franchise(monkeypatch):
    from cfb_coach.madden.model import defense_v1
    monkeypatch.setattr(defense_v1, "offense_won", lambda r, *_: {"W": True, "L": False}.get(r))
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
       CREATE TABLE game_sessions(session_id TEXT, dynasty TEXT);
       CREATE TABLE snaps(session_id TEXT, side TEXT, executed_status TEXT,
         executed_verification TEXT, executed_formation TEXT, executed_play TEXT,
         executed_adjustment_id TEXT, result TEXT, down INTEGER, distance INTEGER);
       INSERT INTO game_sessions VALUES ('a','test'),('b','main');
       INSERT INTO snaps VALUES ('a','defense','identified','verified',
          'Nickel Over','Cover 3 Sky','press','L',3,9);
       INSERT INTO snaps VALUES ('a','defense','unknown','unknown',
          'Nickel Over','Cover 3 Sky','press','W',3,9);
       INSERT INTO snaps VALUES ('b','defense','identified','verified',
          'Nickel Over','Cover 3 Sky','press','W',3,9);
    """)
    model = actions.fit_verified_adjustment_effects(conn, franchise="test")
    assert model["n"] == 1
    assert list(model["counts"].values()) == [[0, 1]]


def test_pregame_learned_scores_replace_legacy_counter_table():
    rows = [
        v2.Snap("g1", "cpu", "2026-11-01", i, "Nickel Over", "Cover 3 Sky",
                "single_high", "long", "cross", False)
        for i in range(1, 31)
    ]
    model = v2.fit(rows)
    recs = [
        {"family": "Nickel", "formation": "Nickel Over",
         "source_book": "Test", "plays": ["Cover 3 Sky"],
         "package": "nickel", "profile": {"coverage_families": {"single_high": 1}}},
    ]
    usual = architect.defense_candidates(None, "cpu", {}, recs)[0]
    learned = architect.defense_candidates(None, "cpu", {}, recs, v2_model=model)[0]
    assert usual["values"] != learned["values"]
    assert set(usual["values"]) == set(learned["values"])
