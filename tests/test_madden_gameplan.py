"""Madden prep game plan: trimmed custom book, 8+8 call packages, opponent tailoring."""

from __future__ import annotations

import json

from cfb_coach.madden.gameplan_macros import build_gameplan
from cfb_coach.madden.macros import load_gameplan, load_selection
from cfb_coach.madden.prep import build_prep_plan, format_delta_text
from tests.test_madden27 import _Isolated


def _pairs(book: dict) -> set[tuple[str, str]]:
    return {(f, p) for f, plays in (book or {}).items() for p in plays}


class TestCustomGameplan(_Isolated):
    def test_prep_defaults_to_trimmed_custom_book(self) -> None:
        plan = build_prep_plan("james", offline=True, persist=False)
        off = plan["playbook"]["offense"]["record"]
        de = plan["playbook"]["defense"]["record"]
        self.assertEqual(off["mode"], "custom")
        self.assertEqual(off["source_book"], "Buccaneers")
        self.assertTrue(off["formations"])
        self.assertEqual(de["mode"], "custom")
        self.assertEqual(de["source_book"], "49ers")
        self.assertTrue(de["formations"])
        self.assertEqual(plan["playbook"]["offense"]["status"], "applied")
        self.assertEqual(plan["playbook"]["defense"]["status"], "applied")
        self.assertFalse(plan["playbook_warnings"])

    def test_eight_and_eight_only_use_custom_book_plays(self) -> None:
        plan = build_prep_plan("james", offline=True, persist=False)
        gp = plan["gameplan"]
        self.assertEqual(len(gp["offense"]), 8)
        self.assertEqual(len(gp["defense"]), 8)
        self.assertEqual(len({m["id"] for m in gp["offense"]}), 8)
        self.assertEqual(len({m["id"] for m in gp["defense"]}), 8)
        situations = {m["situation"] for m in gp["offense"]}
        self.assertTrue({"opener", "3rd_short", "3rd_long", "red_zone", "two_minute", "vs_blitz", "vs_run", "vs_pass"} <= situations)
        o_book = plan["playbook"]["offense"]["record"]["formations"]
        d_book = plan["playbook"]["defense"]["record"]["formations"]
        o_ok, d_ok = _pairs(o_book), _pairs(d_book)
        for m in gp["offense"]:
            self.assertIn((m["formation"], m["play"]), o_ok, m)
            self.assertTrue(m["when"] and m["read"] and m["counter"] and m["adjustments"])
            self.assertIn("score", m["score_situation"].lower())
        for m in gp["defense"]:
            self.assertIn((m["formation"], m["play"]), d_ok, m)
        text = format_delta_text(plan)
        self.assertIn("## Game plan — 8 offense + 8 defense", text)
        self.assertIn("CALL:", text)
        self.assertNotIn("WARNING:", text)

    def test_stock_pin_warns_and_still_stays_inside_that_book(self) -> None:
        plan = build_prep_plan("james", offline=True, persist=False, o_book="stock:Buccaneers", d_book="stock:49ers")
        self.assertEqual(plan["playbook"]["offense"]["record"]["mode"], "stock")
        self.assertTrue(any("no custom offense playbook" in w for w in plan["playbook_warnings"]))
        self.assertTrue(any("no custom defense playbook" in w for w in plan["playbook_warnings"]))
        text = format_delta_text(plan)
        self.assertIn("WARNING: no custom offense playbook", text)
        o_ok = _pairs(plan["playbook"]["offense"]["record"]["formations"])
        for m in plan["gameplan"]["offense"]:
            self.assertIn((m["formation"], m["play"]), o_ok)

    def test_pending_editor_custom_uses_latest_applied_book(self) -> None:
        db = self.madden_db()
        try:
            first = build_prep_plan("james", db=db, offline=True)
            applied = _pairs(first["playbook"]["offense"]["record"]["formations"])
            self.assertEqual(first["playbook"]["offense"]["record"]["mode"], "custom")
            second = build_prep_plan("james", db=db, offline=True, o_book="custom")
            self.assertEqual(second["playbook"]["offense"]["status"], "pending")
            self.assertTrue(any("latest applied" in w for w in second["playbook_warnings"]))
            for m in second["gameplan"]["offense"]:
                self.assertIn((m["formation"], m["play"]), applied)
        finally:
            db.close()

    def test_stored_on_active_macros_schema(self) -> None:
        db = self.madden_db()
        try:
            plan = build_prep_plan("james", db=db, offline=True)
            raw = json.loads(db.get_meta("active_macros:james"))
            self.assertEqual(raw["schema"], 2)
            self.assertEqual(len(raw["offense"]), 8)
            self.assertEqual(len(raw["defense"]), 8)
            self.assertEqual(len(raw["research_defense"]), 10)
            stored = load_gameplan(db, "james")
            self.assertEqual([m["id"] for m in stored["offense"]], raw["offense"])
            self.assertEqual(load_selection(db, "james")["defense"], plan["macro_selection"]["defense"])
            for m in stored["offense"] + stored["defense"]:
                book = plan["playbook"][m["side"]]["record"]["formations"]
                self.assertIn(m["play"], book[m["formation"]])
        finally:
            db.close()

    def test_vikings_tailoring_uses_roster_book_and_logs(self) -> None:
        db = self.madden_db()
        try:
            for _ in range(4):
                db.log_snap(
                    opponent_id="james", side="defense", situation_raw="d 1&10", our_call="x",
                    formation="Nickel Over", play="Cover 4 Quarters", result="+18",
                    concept_seen="Four Verticals",
                )
                db.log_snap(
                    opponent_id="james", side="offense", situation_raw="1&10", our_call="x",
                    formation="Gun Doubles Clamp Stack", play="Inside Zone", result="+6",
                    coverage_seen="Cover 3",
                )
            plan = build_prep_plan("james", db=db, offline=True, opp_team="Minnesota Vikings")
            self.assertEqual(plan["team"], "Minnesota Vikings")
            note = plan["gameplan"]["opponent_note"]
            self.assertIn("Vikings", note)
            self.assertIn("Four Verticals", note)
            self.assertIn("Cover 3", note)
            blob = json.dumps(plan["gameplan"])
            self.assertIn("Four Verticals", blob)
            self.assertIn("Vikings", blob)
            prof = db.get_opponent("james")
            self.assertEqual(prof["nfl_team"], "Minnesota Vikings")
        finally:
            db.close()

    def test_tiny_book_never_invents_a_play(self) -> None:
        book = {"Gun Doubles Clamp Stack": ["Inside Zone", "Texas Y-Stutter Wheel"]}
        dbook = {"Nickel Over": ["Cover 4 Quarters"]}
        gp = build_gameplan(
            offense_book=book, defense_book=dbook, n=8,
            opp={"nfl_team": "Minnesota Vikings", "display_name": "James"},
            opponent_id="james",
        )
        self.assertLessEqual(len(gp["offense"]), 8)
        self.assertTrue(gp["offense"])
        for m in gp["offense"]:
            self.assertEqual(m["formation"], "Gun Doubles Clamp Stack")
            self.assertIn(m["play"], book["Gun Doubles Clamp Stack"])
        for m in gp["defense"]:
            self.assertEqual((m["formation"], m["play"]), ("Nickel Over", "Cover 4 Quarters"))
        self.assertNotIn("Mesh Post", json.dumps(gp["offense"]))
        self.assertIn("Vikings", gp["opponent_note"])
