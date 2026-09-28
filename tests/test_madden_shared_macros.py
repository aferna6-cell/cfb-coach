"""v1.17: Madden reuses the CFB macro settings (one shared source, by name) and runs
10 offense + 10 defense macros per opponent instead of the Active 8."""

from __future__ import annotations

import json
import random

from cfb_coach import macro_settings as ms
from cfb_coach import macros as cfb_macros
from cfb_coach.madden import playbook as pb
from cfb_coach.madden.macro_pool import families, macro_side, pool_ids, pool_macro
from cfb_coach.madden.macro_select import rank_side, select_loadout
from cfb_coach.madden.macros import (
    PER_SIDE,
    aidan_settings,
    copy_block,
    copy_checklist,
    load_selection,
    macro_detail,
    migrate_selection,
    store_selection,
)
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.prep import build_prep_plan
from cfb_coach.madden.situation import parse_madden_situation
from tests.test_madden27 import _Isolated


def _cfb_confirmed_values(cfb_id: str) -> list[str]:
    fs = (cfb_macros.get_macro(cfb_id) or {}).get("full_settings") or {}
    return [str(v.get("value")) for v in fs.values() if isinstance(v, dict) and v.get("status") == "confirmed"]


def _books():
    return {"offense": pb.make_record("offense", "stock", "Buccaneers")["formations"],
            "defense": pb.make_record("defense", "stock", "49ers")["formations"]}


class TestSharedSettings(_Isolated):
    def test_same_name_copies_cfb_settings_word_for_word(self) -> None:
        # HEAT is a Madden catalog macro AND a CFB macro (defense): one macro, CFB's rows verbatim
        rows = aidan_settings("HEAT")
        cfb = (cfb_macros.get_macro("HEAT") or {})["full_settings"]
        expect = [(v["section"], k.partition(" / ")[2] or v["section"], v["value"]) for k, v in cfb.items()
                  if v.get("status") == "confirmed" and v.get("section") != "Notes"]
        self.assertEqual([(r["section"], r["setting"], r["value"]) for r in rows], expect)
        self.assertIn(("DL/LB", "LB Assignments", "Default — do not Blitz All"),
                      [(r["section"], r["setting"], r["value"]) for r in rows])
        # offense: identical to what CFB's own drill-down shows
        for mid in ("O-RPO", "MAN", "O-RUN", "ZERO"):
            got = [(r["section"], r["setting"], r["value"]) for r in aidan_settings(mid)]
            want = [(r["section"], r["setting"], r["value"]) for r in cfb_macros.aidan_offense_settings(mid)]
            self.assertEqual(got, want, mid)
            self.assertTrue(got, mid)
        det = macro_detail("VERT")
        self.assertEqual(det["cfb_match"], "VERT")
        blk = copy_block(det)
        self.assertIn("  [ ] Quarters Bunch: Bingo", blk)  # Aidan's exact wording
        self.assertIn("[ ] Everything else: Default", blk)
        self.assertNotIn("NEEDS SETTINGS", blk)

    def test_no_match_is_flagged_and_names_are_not_changed(self) -> None:
        self.assertIsNone(ms.cfb_match("O-MAN", "offense"))  # MAN is a different macro name
        self.assertIsNone(ms.cfb_match("O-PROT", "offense"))
        self.assertIsNone(ms.cfb_match("MATCH-4", "defense"))
        self.assertEqual(ms.cfb_match("HEAT", "defense"), "HEAT")  # same name, same side only
        self.assertEqual(ms.cfb_match("HEAT", "offense"), "O-HEAT")  # CFB's offense macro saved as HEAT
        self.assertIn("O-MAN", pool_ids("offense"))
        self.assertIn("MAN", pool_ids("offense"))
        for mid in ("O-MAN", "O-PROT", "MATCH-4", "FLAT-CAP", "MESH-RAT", "STACK", "SPY", "RUN-FIT"):
            det = macro_detail(mid)
            self.assertTrue(det["needs_settings"], mid)
            self.assertEqual(det["settings"], [], mid)
            blk = copy_block(det)
            self.assertIn("NEEDS SETTINGS", blk)
            for g in det["gap_rows"]:  # research guesses never appear as settings
                self.assertNotIn(g["research_value"], blk.split("[ ] Everything else")[0])

    def test_nothing_invented_every_row_comes_from_a_source(self) -> None:
        for mid in pool_ids():
            m = pool_macro(mid) or {}
            rows = aidan_settings(mid)
            if not m.get("cfb_match"):
                self.assertEqual(rows, [], mid)
                continue
            allowed = set(_cfb_confirmed_values(m["cfb_match"]))
            for r in rows:
                if r["value"] in allowed:
                    continue
                # route assignments are split per slot ("WR1 deep cross" → WR1 / deep cross)
                self.assertTrue(any(f"{r['setting']} {r['value']}" in v for v in allowed), (mid, r))

    def test_edit_from_either_game_updates_both_and_override_is_per_game(self) -> None:
        rc, _ = self.run_cli(["macro-settings", "--game", "madden27", "MAN", "--set", "Route assignments: WR1 = corner"])
        self.assertEqual(rc, 0)
        cfb_rows = cfb_macros.aidan_offense_settings("MAN")
        self.assertIn(("WR1", "corner"), [(r["setting"], r["value"]) for r in cfb_rows])
        self.assertEqual(sum(1 for r in cfb_rows if r["setting"] == "WR1"), 1)  # replaced, not duplicated
        self.assertIn(("WR1", "corner"), [(r["setting"], r["value"]) for r in aidan_settings("MAN")])
        # edit from CFB → Madden sees it (defense sheet too)
        rc, _ = self.run_cli(["macro-settings", "--game", "cfb27", "CROSS", "--set", "Secondary: CB Depth = 6"])
        self.assertEqual(rc, 0)
        self.assertIn(("CB Depth", "6"), [(r["setting"], r["value"]) for r in aidan_settings("CROSS")])
        cards, _ = cfb_macros.active_loadout_cards(None)
        cross = next(c for c in cards if c["id"] == "CROSS")
        self.assertIn("[ ] CB Depth: 6", cross["copy_block"])
        self.assertIn("WHEN: After 2+ live Cross Wheels", cross["copy_block"])  # his notes kept
        # a setting that only applies in one game
        rc, _ = self.run_cli(["macro-settings", "--game", "madden27", "VERT", "--this-game-only",
                              "--set", "Secondary: Safety Depth = 14"])
        self.assertEqual(rc, 0)
        self.assertIn(("Safety Depth", "14"), [(r["setting"], r["value"]) for r in aidan_settings("VERT")])
        cfb_vert = ms.settings_for("VERT", "defense", "cfb27")
        self.assertIn(("Safety Depth", "16"), [(r["setting"], r["value"]) for r in cfb_vert])
        # listing from Madden shows the shared settings
        rc, out = self.run_cli(["macro-settings", "--game", "madden27", "HEAT"])
        self.assertIn("Show Blitz: Both", out)
        self.assertIn("shared with CFB 27 macro HEAT", out)

    def test_cfb_unchanged_without_edits(self) -> None:
        cards, _ = cfb_macros.active_loadout_cards(None)
        for c in cards:
            self.assertEqual(c["copy_block"], cfb_macros.get_macro(c["id"])["copy_block"])
        self.assertEqual(cfb_macros.aidan_offense_settings("MAN"), cfb_macros.catalog_offense_rows("MAN"))
        self.assertFalse(ms.settings_path().exists())

    def test_legacy_madden_settings_file_is_migrated_not_lost(self) -> None:
        legacy = self.dir / ms.LEGACY_MADDEN_FILENAME
        legacy.write_text(json.dumps({"macros": {
            "MATCH-4": {"settings": [{"section": "Coverage", "setting": "Shading", "value": "Over Top"}]},
            "HEAT": {"settings": [{"section": "Pass rush", "setting": "Stunt", "value": "Twist"}]},
        }}), encoding="utf-8")
        self.assertEqual([(r["setting"], r["value"]) for r in aidan_settings("MATCH-4")], [("Shading", "Over Top")])
        self.assertFalse(legacy.exists())
        self.assertTrue(legacy.with_name(ms.LEGACY_MADDEN_FILENAME + ".migrated").exists())
        # HEAT already has CFB settings: Aidan's Madden row stays Madden-only, CFB is untouched
        self.assertIn(("Stunt", "Twist"), [(r["setting"], r["value"]) for r in aidan_settings("HEAT")])
        self.assertNotIn(("Stunt", "Twist"), [(r["setting"], r["value"]) for r in ms.settings_for("HEAT", "defense", "cfb27")])


class TestTenPlusTenSelection(_Isolated):
    def test_user_prep_splits_ten_offense_ten_defense(self) -> None:
        plan = build_prep_plan("gavin", offline=True, persist=False)
        sel = plan["macro_selection"]
        self.assertEqual((len(sel["offense"]), len(sel["defense"])), (PER_SIDE, PER_SIDE))
        self.assertTrue(all(macro_side(m) == "offense" for m in sel["offense"]))
        self.assertTrue(all(macro_side(m) == "defense" for m in sel["defense"]))
        self.assertEqual(len(set(sel["offense"] + sel["defense"])), 2 * PER_SIDE)
        # cards: two groups in rank order, each with its exact settings drill-down
        for side in ("offense", "defense"):
            cards = [c for c in plan["macro_cards"] if c["side"] == side]
            self.assertEqual([c["id"] for c in cards], sel[side])
            self.assertEqual([c["rank"] for c in cards], list(range(1, PER_SIDE + 1)))
            for c in cards:
                self.assertEqual(c["ingame"]["settings"], aidan_settings(c["id"]))
        # copy checklist: all 20, needs-settings flagged
        chk = plan["copy_checklist"]
        for m in sel["offense"] + sel["defense"]:
            self.assertIn(f"({m})", chk)
        need = [m for m in sel["offense"] + sel["defense"] if not aidan_settings(m)]
        self.assertEqual(chk.count("[!] NEEDS SETTINGS"), len(need))
        self.assertEqual(len(plan["missing_settings"]), len(need))

    def test_cpu_gets_ten_offense_only(self) -> None:
        plan = build_prep_plan("cpu", offline=True, persist=False)
        self.assertEqual(len(plan["macro_selection"]["offense"]), PER_SIDE)
        self.assertEqual(plan["macro_selection"]["defense"], [])
        self.assertIn("OFFENSE (10)", plan["copy_checklist"])
        self.assertNotIn("DEFENSE", plan["copy_checklist"])

    def test_ranking_uses_research_tendencies_and_learned_weights(self) -> None:
        db = self.madden_db()
        try:
            base = select_loadout("gavin", db=db, archetype="split_field_zone")
            # opponent tendency: repeated RPO bubbles vs us lifts the RPO macro into the defense 10
            if "RPO" not in base["defense"]:
                for _ in range(5):
                    db.bump_tendency("gavin", "offense_concept", "RPO bubble")
                self.assertIn("RPO", select_loadout("gavin", db=db, archetype="split_field_zone")["defense"])
            # learned weight at/below the suppress line drops a macro out of its 10
            top = base["defense"][0]
            db.bump_macro_weight("gavin", top, -0.5)
            db.conn.commit()
            self.assertNotIn(top, select_loadout("gavin", db=db, archetype="split_field_zone")["defense"])
            # fresh (live) research naming a macro lifts it
            fake = {"mode": "live", "suggestions": [{"macro_hint": "HEAT", "side": "defense"}] * 2}
            rows = rank_side("defense", hints=["HEAT", "HEAT"], baseline=[])
            self.assertEqual(rows[0]["id"], "HEAT")
            self.assertIn("live research names it", rows[0]["why"])
            self.assertIn("HEAT", select_loadout("gavin", scout=fake)["defense"])
            cached = dict(fake, mode="cache")  # a cache fallback is not fresh research
            self.assertNotIn("live research", " ".join(
                r["why"] for r in select_loadout("gavin", scout=cached)["ranked"]["defense"]))
        finally:
            db.close()

    def test_prep_selection_is_stored_per_opponent(self) -> None:
        db = self.madden_db()
        try:
            build_prep_plan("gavin", db=db, offline=True)
            build_prep_plan("cpu", db=db, offline=True)
            g = load_selection(db, "gavin")
            c = load_selection(db, "cpu")
        finally:
            db.close()
        self.assertEqual((len(g["offense"]), len(g["defense"])), (10, 10))
        self.assertEqual((len(c["offense"]), c["defense"]), (10, []))


class TestPlayModeSides(_Isolated):
    SEL = {"offense": ["MAN", "O-MAN", "PROT", "C2", "C3", "MATCH", "O-RUN", "O-RPO", "RZ", "ZERO"],
           "defense": ["VERT", "MATCH-4", "CROSS", "FLOOD", "SCRAM", "RUN-IN", "RPO", "BUNCH", "SPY", "HEAT"]}

    def test_neutral_snaps_suggest_no_macro(self) -> None:
        db = self.madden_db()
        try:
            for seed in range(15):
                c = make_call(parse_madden_situation("1&10 my 30"), "gavin", db, rng=random.Random(seed),
                              playbook=_books(), active_macros=self.SEL)
                self.assertIsNone(c.macro)
                self.assertEqual(c.headline(), f"PLAY: {c.play} ({c.formation})")
        finally:
            db.close()

    def test_offense_call_picks_from_offense_ten(self) -> None:
        db = self.madden_db()
        try:
            for _ in range(2):
                db.log_snap(opponent_id="gavin", side="offense", situation_raw="3&8", our_call="x",
                            formation="x", play="x", result="+3", coverage_seen="Cover 1", down=3, distance=8)
            hits = []
            for seed in range(20):
                c = make_call(parse_madden_situation("3&8 showing cover 1"), "gavin", db, rng=random.Random(seed),
                              playbook=_books(), active_macros=self.SEL)
                if c.macro:
                    hits.append(c)
            self.assertTrue(hits)
            for c in hits:
                self.assertIn(c.macro, self.SEL["offense"])
                self.assertEqual(c.macro, "MAN")  # highest-ranked man answer of the offense 10
                self.assertIn(f"+ MACRO: {c.macro_info['name']}", c.headline())
                self.assertIn("WR1 deep cross", c.macro_line())  # the shared CFB settings
            only_def = {"offense": [], "defense": self.SEL["defense"]}
            c = make_call(parse_madden_situation("3&8 showing cover 1"), "gavin", db, rng=random.Random(0),
                          playbook=_books(), active_macros=only_def)
            self.assertIsNone(c.macro)  # a defense macro is never used on offense
        finally:
            db.close()

    def test_defense_call_picks_from_defense_ten(self) -> None:
        db = self.madden_db()
        try:
            for _ in range(3):
                db.log_snap(opponent_id="gavin", side="defense", situation_raw="x", our_call="x", formation="x",
                            play="x", result="+20", concept_seen="Four Verticals")
            sit = lambda: parse_madden_situation("d 2&6 showing 4 verts", default_side="defense")  # noqa: E731
            c = make_call(sit(), "gavin", db, rng=random.Random(0), playbook=_books(), active_macros=self.SEL)
            self.assertEqual(c.macro, "VERT")  # ranked above MATCH-4 in this defense 10
            self.assertIn(c.macro, self.SEL["defense"])
            self.assertEqual(c.headline(), f"PLAY: {c.play} ({c.formation}) + MACRO: VERT")
            flipped = dict(self.SEL, defense=["MATCH-4", "VERT"] + self.SEL["defense"][2:])
            c = make_call(sit(), "gavin", db, rng=random.Random(0), playbook=_books(), active_macros=flipped)
            self.assertEqual(c.macro, "MATCH-4")
            no_d = dict(self.SEL, defense=["CROSS", "FLOOD"])
            c = make_call(sit(), "gavin", db, rng=random.Random(0), playbook=_books(), active_macros=no_d)
            self.assertIsNone(c.macro)  # an offense list / other families never arm on D
        finally:
            db.close()

    def test_offense_macro_only_when_it_helps(self) -> None:
        from cfb_coach.madden.macros import suggest_offense_macro

        kw = dict(coverage_source="none", repeated=False, active=self.SEL)
        self.assertIsNone(suggest_offense_macro(play="RPO Alert Out", coverage_class=None, **kw))
        got = suggest_offense_macro(play="RPO Alert Out", coverage_class="two_high", coverage_source="live",
                                    repeated=False, active=self.SEL)
        self.assertEqual(got["id"], "O-RPO")
        self.assertIsNone(suggest_offense_macro(play="Mesh Post", coverage_class=None, zone="rz", down=1, **kw))
        self.assertEqual(suggest_offense_macro(play="Mesh Post", coverage_class=None, zone="rz", down=3, **kw)["id"], "RZ")
        # a live look alone never fires a coverage macro (needs the look repeated this game)
        self.assertIsNone(suggest_offense_macro(play="Mesh Post", coverage_class="man", coverage_source="live",
                                                repeated=False, active=self.SEL))
        self.assertIsNone(suggest_offense_macro(play="Mesh Post", coverage_class="man", coverage_source="live",
                                                repeated=True, active={"offense": [], "defense": ["MAN"]}))

    def test_family_answers_exist_on_each_side(self) -> None:
        for fam in ("vert", "flood", "cross", "stack", "scram", "run", "rpo"):
            self.assertTrue([m for m in pool_ids("defense") if fam in families(m)], fam)
        for fam in ("pressure", "man", "cover2", "single_high", "two_high", "two_high_run", "rpo", "red_zone"):
            self.assertTrue([m for m in pool_ids("offense") if fam in families(m)], fam)


class TestActiveEightMigration(_Isolated):
    OLD = ["MATCH-4", "FLAT-CAP", "MESH-RAT", "STACK", "SPY", "HEAT", "O-PROT", "O-MAN"]

    def _old_record(self, db, opp: str) -> None:
        db.set_meta(f"active_macros:{opp}", json.dumps({"active": self.OLD, "ts": "2026-09-20T00:00:00+00:00"}))

    def test_stored_active8_migrates_without_loss(self) -> None:
        db = self.madden_db()
        try:
            self._old_record(db, "gavin")
            self.assertTrue(migrate_selection(db, "gavin"))
            sel = load_selection(db, "gavin")
            self.assertEqual(sel["defense"], ["MATCH-4", "FLAT-CAP", "MESH-RAT", "STACK", "SPY", "HEAT"])
            self.assertEqual(sel["offense"], ["O-PROT", "O-MAN"])
            self.assertEqual(json.loads(db.get_meta("active_macros_legacy8:gavin"))["active"], self.OLD)
            self.assertFalse(migrate_selection(db, "gavin"))  # idempotent
            # next prep keeps every old pick in the 10 + 10 (carry-over), then re-stores schema 2
            plan = build_prep_plan("gavin", db=db, offline=True, profile="lab")
            kept = plan["macro_selection"]["offense"] + plan["macro_selection"]["defense"]
            self.assertTrue(set(self.OLD) <= set(kept), set(self.OLD) - set(kept))
            self.assertEqual(json.loads(db.get_meta("active_macros:gavin"))["schema"], 2)
        finally:
            db.close()

    def test_jaxon_learning_survives_migration_prep_and_postgame(self) -> None:
        from cfb_coach.madden.cli import open_db
        from cfb_coach.madden.postgame import summary

        db = self.madden_db()
        try:
            self._old_record(db, "jaxon")
            db.bump_macro_weight("jaxon", "STACK", -0.4)
            for res in ("stop", "stop", "+2", "td"):
                db.log_snap(opponent_id="jaxon", side="defense", situation_raw="x", our_call="x", formation="x",
                            play="x", result=res, concept_seen="Clamp Stack / Bunch", macro="MATCH-4")
            db.conn.commit()
        finally:
            db.close()
        db = open_db()  # migrates every stored Active 8 on open
        try:
            sel = load_selection(db, "jaxon")
            self.assertEqual(sel["offense"] + sel["defense"], ["O-PROT", "O-MAN"] + self.OLD[:6])
            self.assertEqual(json.loads(db.get_meta("active_macros:jaxon"))["schema"], 2)
            w = {r["macro"]: r["weight"] for r in db.get_macro_weights("jaxon")}
            self.assertAlmostEqual(w["STACK"], -0.4)
            plan = build_prep_plan("jaxon", db=db, offline=True)  # learned weight still steers the pick
            self.assertNotIn("STACK", plan["macro_selection"]["defense"])
            out = summary(db, "jaxon")
            self.assertIn("Retrain", out)
            self.assertIn("MATCH-4", out)
            self.assertEqual(db.count_snaps("jaxon"), 4)
        finally:
            db.close()

    def test_postgame_status_counts_exact_macro_names(self) -> None:
        from cfb_coach.madden.macros import macro_status
        from cfb_coach.madden.postgame import learn

        db = self.madden_db()
        try:
            for _ in range(3):  # MATCH (offense) wins must not promote MATCH-4 (defense)
                db.log_snap(opponent_id="gavin", side="offense", situation_raw="x", our_call="x", formation="x",
                            play="x", result="td", macro="MATCH")
            learn(db, "gavin")
            self.assertEqual(macro_status("MATCH", db), "proven")
            self.assertEqual(macro_status("MATCH-4", db), "meta_grounded")
        finally:
            db.close()


class TestChecklistText(_Isolated):
    def test_copy_checklist_lists_both_groups(self) -> None:
        txt = copy_checklist({"offense": ["MAN", "O-MAN"], "defense": ["VERT", "SPY"]})
        self.assertIn("OFFENSE (2)", txt)
        self.assertIn("DEFENSE (2)", txt)
        self.assertIn("O-MAN (O-MAN) — [!] NEEDS SETTINGS", txt)
        self.assertIn("MAN (MAN) — 5 setting(s) — shared", txt)
        store = self.madden_db()
        try:
            store_selection(store, "ryan", {"offense": ["MAN", "VERT"], "defense": ["VERT", "MAN"]})
            self.assertEqual(load_selection(store, "ryan"), {"offense": ["MAN"], "defense": ["VERT"]})
        finally:
            store.close()
