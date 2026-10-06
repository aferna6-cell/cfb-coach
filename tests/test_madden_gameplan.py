"""Madden user-game prep: 8+8 Custom Adjustments on the trimmed custom book."""

from __future__ import annotations

import json

from cfb_coach.madden.gameplan_macros import build_gameplan
from cfb_coach.madden.macros import load_selection
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

    def test_eight_and_eight_custom_adjustments_pair_with_the_book(self) -> None:
        plan = build_prep_plan("james", offline=True, persist=False)
        sel = plan["macro_selection"]
        self.assertEqual(len(sel["offense"]), 8)
        self.assertEqual(len(sel["defense"]), 8)
        o_book = plan["playbook"]["offense"]["record"]["formations"]
        d_book = plan["playbook"]["defense"]["record"]["formations"]
        o_plays = {p for plays in o_book.values() for p in plays}
        d_plays = {p for plays in d_book.values() for p in plays}
        man = next(c for c in plan["macro_cards"] if c["id"] == "MAN")
        settings = {(r["setting"], r["value"]) for r in man["ingame"]["settings"]}
        self.assertIn(("WR1", "deep cross"), settings)
        self.assertIn(("WR2", "zig"), settings)
        self.assertIn(("WR3", "short cross"), settings)
        self.assertIn(("TE", "wheel"), settings)
        self.assertIn(("HB", "Texas"), settings)
        self.assertIn("Everything else: Default", man["copy_block"])
        self.assertNotIn("Motion", man["copy_block"])
        self.assertIn("LB → MAN", man["ingame"]["buttons"])
        from cfb_coach.madden.research_db import editor_fields

        known_fields = {(sec, name) for sec, names in (editor_fields().get("defense") or {}).items() for name in names}
        for card in plan["macro_cards"]:
            ing = card["ingame"]
            self.assertTrue(ing["buttons"].startswith("LB"))
            if card["side"] == "defense":
                for row in ing["settings"]:
                    if row.get("new_field"):
                        self.assertNotEqual(row["source"], "default")
                    else:
                        self.assertIn((row["section"], row["setting"]), known_fields)
                    if row["source"] == "default":
                        self.assertEqual(row["value"], "Default")
            book = o_book if card["side"] == "offense" else d_book
            plays = o_plays if card["side"] == "offense" else d_plays
            for pair in ing.get("pairs_with") or []:
                if pair.endswith("(any call in this formation)"):
                    self.assertIn(pair.split(" (", 1)[0], book)
                    continue
                play, _, rest = pair.partition(" (")
                self.assertIn(play, plays, card["id"])
                self.assertTrue(rest.endswith(")"))
        text = format_delta_text(plan)
        self.assertTrue(text.split("## Research", 1)[0].count("## Custom Adjustments — 8 offense + 8 defense") == 1)
        self.assertNotIn("## Game plan", text)
        self.assertNotIn("CALL:", text)
        self.assertIn("WR1: deep cross", text)
        self.assertNotIn("WARNING:", text)
        from cfb_coach.madden.offense_macros import suggest_for_snap

        armed = suggest_for_snap(
            zone="open", play="Mesh", coverage="Cover 1", coverage_source="live",
            active=sel["offense"], book=o_book,
        )
        self.assertIsNotNone(armed)
        self.assertEqual(armed["id"], "MAN")
        self.assertIn("LB → MAN", armed["buttons"])
        self.assertIsNone(suggest_for_snap(
            zone="open", play="Mesh", coverage="Cover 1", coverage_source="last",
            active=sel["offense"], book=o_book,
        ))
        # the call sheet still exists for the details page, and still stays inside the book
        for m in plan["gameplan"]["offense"]:
            self.assertIn((m["formation"], m["play"]), _pairs(o_book))
        for m in plan["gameplan"]["defense"]:
            self.assertIn((m["formation"], m["play"]), _pairs(d_book))

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
            self.assertEqual(raw["offense"], plan["macro_selection"]["offense"])
            self.assertEqual(raw["defense"], plan["macro_selection"]["defense"])
            self.assertNotIn("packages", raw)
            self.assertEqual(len(raw["offense"]), 8)
            self.assertEqual(len(raw["defense"]), 8)
            self.assertIn("MAN", raw["offense"])
            self.assertTrue(all(" " in mid or mid.isupper() for mid in raw["defense"]))
            stored = load_selection(db, "james")
            self.assertEqual(stored["offense"], raw["offense"])
            self.assertEqual(stored["defense"], raw["defense"])
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
            self.assertIn("QTRS OVERTOP", plan["macro_selection"]["defense"])
            c3 = next(c for c in plan["macro_cards"] if c["id"] == "C3")
            self.assertIn("logged coverage", c3["why"])
            bare = build_prep_plan("ryan", offline=True, persist=False)
            self.assertLess(
                next(c["rank"] for c in plan["macro_cards"] if c["id"] == "C3"),
                next(c["rank"] for c in bare["macro_cards"] if c["id"] == "C3"),
            )
            quen = build_prep_plan("quen", offline=True, persist=False)
            self.assertIn("O-HEAT", quen["macro_selection"]["offense"])
            self.assertNotIn("O-HEAT", plan["macro_selection"]["offense"])
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
