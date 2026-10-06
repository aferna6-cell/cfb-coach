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

    def test_live_call_fires_a_stored_macro_only_when_its_trigger_matches(self) -> None:
        import random

        from cfb_coach.madden.playcaller import make_call
        from cfb_coach.madden.situation import parse_madden_situation

        db = self.madden_db()
        try:
            plan = build_prep_plan("james", db=db, offline=True)
            self.assertIn("MAN", plan["macro_selection"]["offense"])
            self.assertIn("RZ COVER 2", plan["macro_selection"]["defense"])
            self.assertIn("SAFE DEEP", plan["macro_selection"]["defense"])
            book = plan["playbook"]["offense"]["record"]["formations"]

            once = make_call(parse_madden_situation("1&10 showing cover 1"), "james", db, rng=random.Random(1))
            self.assertIsNone(once.macro)
            self.assertNotIn("MACRO:", once.headline())
            for _ in range(3):
                db.log_snap(
                    opponent_id="james", side="offense", situation_raw="1&10", our_call="x",
                    formation="Gun", play="Mesh", result="+6", coverage_seen="Cover 1", macro="none",
                )
            stale = make_call(parse_madden_situation("1&10 cover 1"), "james", db, rng=random.Random(1))
            self.assertIsNone(stale.macro)
            self.assertNotIn("MACRO:", stale.format())
            self.assertNotIn("SUGGEST", stale.format())
            off = make_call(parse_madden_situation("1&10 showing cover 1"), "james", db, rng=random.Random(1))
            self.assertEqual(off.macro, "MAN")
            self.assertIn("MACRO: MAN", off.headline())
            self.assertIn("LB → MAN", off.format())
            self.assertIn(off.play, book[off.formation])
            db.log_snap(
                opponent_id="james", side="offense", situation_raw="1&10 showing cover 1", our_call=off.format(),
                formation=off.formation, play=off.play, result="+5", coverage_seen="Cover 1", macro=off.macro,
            )
            again = make_call(parse_madden_situation("2&7 showing cover 1"), "james", db, rng=random.Random(1))
            self.assertIsNone(again.macro)
            self.assertNotIn("MACRO:", again.format())

            quiet_o = make_call(parse_madden_situation("1&10"), "james", db, rng=random.Random(1))
            self.assertIsNone(quiet_o.macro)
            self.assertNotIn("MACRO:", quiet_o.headline())
            quiet_d = make_call(parse_madden_situation("d 1&10"), "james", db, rng=random.Random(1))
            self.assertIsNone(quiet_d.macro)
            self.assertNotIn("MACRO:", quiet_d.headline())

            de = make_call(parse_madden_situation("d 1&10 opp 15"), "james", db, rng=random.Random(1))
            self.assertEqual(de.macro, "RZ COVER 2")
            self.assertIn("MACRO: RZ COVER 2", de.headline())
            self.assertIn("LB → RZ COVER 2", de.format())
            dbook = plan["playbook"]["defense"]["record"]["formations"]
            self.assertIn(de.play, dbook[de.formation])

            lead = make_call(
                parse_madden_situation("d 1&10 score 24-10 q4"), "james", db, rng=random.Random(1),
            )
            self.assertEqual(lead.macro, "SAFE DEEP")
            self.assertIn("LB → SAFE DEEP", lead.format())
            self.assertIn(lead.play, dbook[lead.formation])
            db.log_snap(
                opponent_id="james", side="defense", situation_raw="d 1&10 opp 15", our_call=de.format(),
                formation=de.formation, play=de.play, macro=de.macro, result="+4",
            )
            held = make_call(parse_madden_situation("d 2&6 opp 8"), "james", db, rng=random.Random(1))
            self.assertIsNone(held.macro)
            self.assertNotIn("MACRO:", held.format())
            self.assertNotIn("SUGGEST", held.format())
        finally:
            db.close()

    def test_logged_snap_stores_the_macro_on_screen(self) -> None:
        import random

        from cfb_coach.last_snap import LastSnapBook
        from cfb_coach.madden.playcaller import make_call
        from cfb_coach.madden.situation import parse_madden_situation

        db = self.madden_db()
        try:
            build_prep_plan("james", db=db, offline=True)
            book = LastSnapBook(db, "james", parse_madden_situation, session_id="log1")
            plain_sit = parse_madden_situation("d 1&10")
            plain = make_call(plain_sit, "james", db, rng=random.Random(1))
            self.assertIsNone(plain.macro)
            self.assertIsNone(plain.suggest_macro)
            self.assertNotIn("MACRO:", plain.format())
            self.assertNotIn("SUGGEST", plain.format())
            book.remember_call(plain, plain_sit)
            book.handle("+4")
            self.assertEqual(db.get_session_snaps("log1")[0]["macro"], "none")

            lead_sit = parse_madden_situation("d 1&10 score 24-10 q4")
            lead = make_call(lead_sit, "james", db, rng=random.Random(1))
            self.assertEqual(lead.macro, "SAFE DEEP")
            self.assertIn("MACRO: SAFE DEEP", lead.format())
            book.remember_call(lead, lead_sit)
            book.handle("+9")
            self.assertEqual(db.get_session_snaps("log1")[-1]["macro"], "SAFE DEEP")
        finally:
            db.close()

    def test_no_macros_silences_even_a_red_zone(self) -> None:
        import random

        from cfb_coach.madden.franchise import save_config
        from cfb_coach.madden.playcaller import make_call
        from cfb_coach.madden.situation import parse_madden_situation

        db = self.madden_db()
        try:
            build_prep_plan("james", db=db, offline=True)
            sit = parse_madden_situation("d 1&10 opp 15")
            flagged = make_call(sit, "james", db, rng=random.Random(1), live_macros=False)
            self.assertIsNone(flagged.macro)
            self.assertIsNone(flagged.suggest_macro)
            self.assertNotIn("MACRO:", flagged.format())
            self.assertTrue(flagged.play)
            on = make_call(sit, "james", db, rng=random.Random(1), live_macros=True)
            self.assertEqual(on.macro, "RZ COVER 2")
            self.assertEqual((flagged.formation, flagged.play), (on.formation, on.play))
            save_config(live_macros=False)
            stored = make_call(sit, "james", db, rng=random.Random(1))
            self.assertIsNone(stored.macro)
            save_config(live_macros=True)
            back = make_call(sit, "james", db, rng=random.Random(1))
            self.assertEqual(back.macro, "RZ COVER 2")
        finally:
            db.close()

    def test_policy_confidence_and_game_cap(self) -> None:
        from cfb_coach.madden.macro_policy import PER_GAME, allow_macro

        db = self.madden_db()
        try:
            for _ in range(3):
                db.log_snap(
                    opponent_id="james", side="offense", situation_raw="1&10", our_call="x",
                    coverage_seen="Cover 3 Sky", macro="none",
                )
            for _ in range(9):
                db.log_snap(
                    opponent_id="james", side="offense", situation_raw="1&10", our_call="x",
                    coverage_seen="Cover 2", macro="none",
                )
            self.assertFalse(allow_macro(
                db, opponent_id="james", side="offense", macro_id="C3", kind="look",
            ))
            for _ in range(PER_GAME):
                db.log_snap(
                    opponent_id="james", side="defense", situation_raw="d 1&10", our_call="x",
                    macro="RZ COVER 2",
                )
            self.assertFalse(allow_macro(
                db, opponent_id="james", side="defense", macro_id="SAFE DEEP", kind="situation",
            ))
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
