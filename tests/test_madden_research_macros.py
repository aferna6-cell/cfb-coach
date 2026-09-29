"""v1.17: Madden defense macros are research-built (research DB from the daily routine, pulled
every prep), offense calls carry pre-snap adjustments instead of macros, and every macro /
adjustment comes with its Xbox buttons. CFB keeps Aidan's own sheets (shared settings store)."""

from __future__ import annotations

import copy
import json
import random
from unittest import mock

from cfb_coach import macro_settings as ms
from cfb_coach import macros as cfb_macros
from cfb_coach.madden import playbook as pb
from cfb_coach.madden import research_db as rdb
from cfb_coach.madden.adjustments import defense_adjustment, offense_adjustment
from cfb_coach.madden.macro_pool import pool_ids
from cfb_coach.madden.macro_select import rank_defense, select_loadout
from cfb_coach.madden.macros import (
    PER_SIDE,
    copy_checklist,
    legacy_picks,
    load_selection,
    macro_detail,
    migrate_selection,
)
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.prep import build_prep_plan
from cfb_coach.madden.situation import parse_madden_situation
from tests.test_madden27 import _Isolated


def _books():
    return {"offense": pb.make_record("offense", "stock", "Buccaneers")["formations"],
            "defense": pb.make_record("defense", "stock", "49ers")["formations"]}


class _DB(_Isolated):
    def setUp(self) -> None:
        super().setUp()
        rdb.reset()

    def tearDown(self) -> None:
        rdb.reset()
        super().tearDown()


class TestResearchDB(_DB):
    def test_packaged_seed_is_valid_and_cited(self) -> None:
        db = rdb.load()
        self.assertEqual(rdb.validate(db), [])
        self.assertGreaterEqual(len(rdb.defense_macros()), PER_SIDE)
        srcs = rdb.sources()
        for m in rdb.defense_macros():
            for r in m["settings"]:
                self.assertIn(r["source"], srcs, (m["id"], r))
                self.assertTrue(srcs[r["source"]]["url"])

    def test_validator_rejects_uncited_or_malformed(self) -> None:
        good = rdb.load()
        bad = copy.deepcopy(good)
        bad["defense_macros"][0]["settings"][0]["source"] = "made-up"
        self.assertTrue(any("unknown source" in e for e in rdb.validate(bad)))
        bad = copy.deepcopy(good)
        bad["defense_macros"] = bad["defense_macros"][:5]
        self.assertTrue(any("at least 10" in e for e in rdb.validate(bad)))
        bad = copy.deepcopy(good)
        bad["controls_xbox"]["defense"]["qb_contain"]["sources"] = []
        self.assertTrue(any("qb_contain" in e for e in rdb.validate(bad)))
        bad = copy.deepcopy(good)
        bad["defense_macros"][0]["answers"] = ["everything"]
        self.assertTrue(any("answers" in e for e in rdb.validate(bad)))

    def test_every_editor_field_researched_or_default(self) -> None:
        fields = rdb.editor_fields()["defense"]
        n_editor = sum(len(v) for v in fields.values())
        for mid in pool_ids("defense"):
            rows = rdb.full_settings(mid)
            listed = {(r["section"], r["setting"]) for r in rows}
            for sec, names in fields.items():
                for name in names:
                    self.assertIn((sec, name), listed, (mid, sec, name))
            self.assertGreaterEqual(len(rows), n_editor)
            for r in rows:
                self.assertTrue(r["value"])
                self.assertTrue(r["source"] == "default" or r["source"] in rdb.sources())
                if r["source"] == "default":
                    self.assertEqual(r["value"], "Default")
        rows = {(r["section"], r["setting"]): r for r in rdb.full_settings("RZ COVER 2")}
        self.assertEqual(rows[("General", "QB Contain")]["value"], "On")  # alias "Contain" → QB Contain
        self.assertEqual(rows[("Individuals", "FS")]["value"], "Inside Quarters")

    def test_prep_pulls_latest_db_and_falls_back(self) -> None:
        newer = copy.deepcopy(rdb.load())
        newer["updated"] = "2026-09-30T09:00:00+00:00"
        newer["defense_macros"][0]["name"] = newer["defense_macros"][0]["name"]  # same shape
        rdb.reset()
        with mock.patch.object(rdb, "_git_pull_db", return_value=newer):
            rdb.load(pull=True)
        self.assertEqual(rdb.load()["updated"], "2026-09-30T09:00:00+00:00")
        self.assertIn("pulled origin/madden-research-db", rdb.status()["line"])
        rdb.reset()  # next prep offline: the last pulled copy, not the older packaged seed
        with mock.patch.object(rdb, "_git_pull_db", return_value=None):
            rdb.load(pull=True)
        self.assertEqual(rdb.load()["updated"], "2026-09-30T09:00:00+00:00")
        self.assertIn("last pulled copy", rdb.status()["line"])
        rdb.reset()  # a broken pull is refused
        broken = dict(newer, defense_macros=[])
        with mock.patch.object(rdb, "_git_pull_db", return_value=broken):
            rdb.load(pull=True)
        self.assertGreaterEqual(len(rdb.defense_macros()), PER_SIDE)

    def test_stale_db_is_flagged(self) -> None:
        old = copy.deepcopy(rdb.load())
        old["updated"] = "2026-09-01T00:00:00+00:00"
        rdb._STATE.update(db=old, origin="test")
        st = rdb.status()
        self.assertTrue(st["stale"])
        self.assertIn("STALE", st["line"])

    def test_validator_script(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        r = subprocess.run([sys.executable, str(root / "scripts" / "validate_madden_research_db.py")],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("OK", r.stdout)


class TestDefenseTen(_DB):
    def test_prep_defense_ten_no_offense_macros(self) -> None:
        plan = build_prep_plan("gavin", offline=True, persist=False)
        sel = plan["macro_selection"]
        self.assertEqual(sel["offense"], [])
        self.assertEqual(len(sel["defense"]), PER_SIDE)
        self.assertEqual(len(set(sel["defense"])), PER_SIDE)
        for c in plan["macro_cards"]:
            det = c["ingame"]
            self.assertEqual(c["side"], "defense")
            self.assertEqual(det["buttons"], f"LB → {c['id']}")
            self.assertEqual(len(det["settings"]), det["n_fields"])
            self.assertIn(f"In game: LB → {c['id']}", c["copy_block"])
        self.assertIn("fire: LB →", plan["copy_checklist"])
        self.assertTrue(plan["offense_adjustments"])
        for a in plan["offense_adjustments"]:
            self.assertTrue(a["buttons"])

    def test_cpu_gets_offense_adjustments_only(self) -> None:
        plan = build_prep_plan("cpu", offline=True, persist=False)
        self.assertEqual(plan["macro_selection"], {"offense": [], "defense": []})
        self.assertTrue(plan["offense_adjustments"])
        self.assertIn("offense only", copy_checklist(plan["macro_selection"]))

    def test_ranking_uses_research_rank_tendencies_and_learned_weights(self) -> None:
        db = self.madden_db()
        try:
            base = select_loadout("gavin", db=db)
            self.assertEqual(base["defense"][0], "TAMPA MABLE")  # research meta #1
            out = [m for m in pool_ids("defense") if m not in base["defense"]]
            self.assertEqual(len(out), len(pool_ids("defense")) - PER_SIDE)
            # opponent tendency: repeated scrambles lift the scramble answers
            for _ in range(5):
                db.bump_tendency("gavin", "offense_concept", "QB scramble")
            rows = rank_defense(db=db, tendencies=__import__("collections").Counter({"scram": 5}))
            self.assertIn("opponent tendency", next(r for r in rows if r["id"] == "SKY CONTAIN")["why"])
            self.assertIn("SKY CONTAIN", select_loadout("gavin", db=db)["defense"])
            # a learned weight at/below the suppress line drops a macro out of the 10
            db.bump_macro_weight("gavin", "TAMPA MABLE", -0.5)
            db.conn.commit()
            self.assertNotIn("TAMPA MABLE", select_loadout("gavin", db=db)["defense"])
            # live scout families count; cache fallbacks never do
            live = {"mode": "live", "suggestions": [{"macro_hint": "MATCH-4"}, {"macro_hint": "MATCH-4"}]}
            why = next(r for r in select_loadout("gavin", scout=live)["ranked"] if r["id"] == "QTRS OVERTOP")["why"]
            self.assertIn("live research family", why)
            cached = dict(live, mode="cache")
            self.assertNotIn("live research", " ".join(r["why"] for r in select_loadout("gavin", scout=cached)["ranked"]))
        finally:
            db.close()


class TestAdjustmentsAndButtons(_DB):
    def test_offense_adjustment_rules(self) -> None:
        kw = dict(formation="Gun Doubles Clamp Stack",
                  audibles={"Gun Doubles Clamp Stack": ["Texas Y-Stutter Wheel", "Inside Zone"]})
        man = offense_adjustment(play="Mesh Post", coverage_class="man", coverage_source="live", repeated=False, **kw)
        self.assertEqual(man["label"], "Hot route WR1 → Slant")
        self.assertIn("Y → tap WR1's icon button → pick Slant", man["buttons"])
        pres = offense_adjustment(play="Mesh Post", coverage_class="pressure", coverage_source="live", repeated=False, **kw)
        self.assertEqual(pres["label"], "Hot route HB → Flat")
        aud = offense_adjustment(play="Mesh Post", coverage_class="two_high", coverage_source="live", repeated=False, **kw)
        self.assertEqual(aud["label"], "Audible → Inside Zone")
        self.assertIn("X → the button shown for Inside Zone", aud["buttons"])
        # only when it helps: previous-snap look (not repeated), a run play vs man, no look at all
        self.assertIsNone(offense_adjustment(play="Mesh Post", coverage_class="man", coverage_source="last",
                                             repeated=False, **kw))
        self.assertIsNotNone(offense_adjustment(play="Mesh Post", coverage_class="man", coverage_source="last",
                                                repeated=True, **kw))
        self.assertIsNone(offense_adjustment(play="Inside Zone", coverage_class="man", coverage_source="live",
                                             repeated=False, **kw))
        self.assertIsNone(offense_adjustment(play="Mesh Post", coverage_class=None, coverage_source="none",
                                             repeated=False, **kw))

    def test_neutral_snaps_no_macro_no_adjustment(self) -> None:
        db = self.madden_db()
        try:
            for seed in range(12):
                for raw in ("1&10 my 30", "d 1&10"):
                    c = make_call(parse_madden_situation(raw, default_side="defense" if raw.startswith("d") else "offense"),
                                  "gavin", db, rng=random.Random(seed), playbook=_books())
                    self.assertIsNone(c.macro)
                    self.assertIsNone(c.adjustment)
                    self.assertEqual(c.headline(), f"PLAY: {c.play} ({c.formation})")
        finally:
            db.close()

    def test_defense_macro_from_the_ten_else_adjustment_with_buttons(self) -> None:
        db = self.madden_db()
        try:
            for _ in range(3):
                db.log_snap(opponent_id="gavin", side="defense", situation_raw="x", our_call="x", formation="x",
                            play="x", result="stop", concept_seen="QB scramble")  # no PIVOT (it resets to base)
            sit = lambda: parse_madden_situation("d 3&6 showing qb scramble", default_side="defense")  # noqa: E731
            ten = {"defense": ["TAMPA MABLE", "SKY CONTAIN", "TEX2 L CONT"]}
            c = make_call(sit(), "gavin", db, rng=random.Random(0), playbook=_books(), active_macros=ten)
            self.assertEqual(c.macro, "SKY CONTAIN")  # primary answer to scrambles, in the 10
            self.assertIn("MACRO: SKY CONTAIN — press LB → SKY CONTAIN", c.format())
            no_scram = {"defense": ["TAMPA MABLE", "PRESS SHADE IN"]}
            c = make_call(sit(), "gavin", db, rng=random.Random(0), playbook=_books(), active_macros=no_scram)
            self.assertIsNone(c.macro)
            self.assertEqual(c.adjustment["label"], "QB contain")
            self.assertIn("ADJ: QB contain — press RB then LB", c.format())
            self.assertEqual(c.headline(), f"PLAY: {c.play} ({c.formation}) + ADJ: QB contain")
        finally:
            db.close()

    def test_unconfirmed_buttons_say_so(self) -> None:
        self.assertIn("VERIFY", rdb.buttons("offense", "motion"))
        self.assertIn("single source", rdb.buttons("offense", "audible"))
        self.assertEqual(rdb.buttons("defense", "qb_contain"), "RB then LB")
        self.assertIn("VERIFY", rdb.buttons("defense", "no_such_control"))
        adj = defense_adjustment("vert")
        self.assertIn("hold LT", adj["buttons"])


class TestMigrationAndLearning(_DB):
    OLD = ["MATCH-4", "FLAT-CAP", "MESH-RAT", "STACK", "SPY", "RUN-FIT", "O-PROT", "O-MAN"]

    def _old(self, db, opp):
        db.set_meta(f"active_macros:{opp}", json.dumps({"active": self.OLD, "ts": "2026-09-20T00:00:00+00:00"}))

    def test_active8_migrates_with_backup(self) -> None:
        db = self.madden_db()
        try:
            self._old(db, "gavin")
            self.assertTrue(migrate_selection(db, "gavin"))
            self.assertFalse(migrate_selection(db, "gavin"))  # idempotent
            self.assertEqual(json.loads(db.get_meta("active_macros_legacy8:gavin"))["active"], self.OLD)
            self.assertEqual(legacy_picks(db, "gavin"), self.OLD)
            plan = build_prep_plan("gavin", db=db, offline=True)
            self.assertEqual(len(plan["macro_selection"]["defense"]), PER_SIDE)
            self.assertEqual(json.loads(db.get_meta("active_macros:gavin"))["schema"], 2)
            self.assertEqual(legacy_picks(db, "gavin"), self.OLD)  # the old picks are never deleted
        finally:
            db.close()

    def test_jaxon_learning_survives(self) -> None:
        from cfb_coach.madden.cli import open_db
        from cfb_coach.madden.postgame import summary

        db = self.madden_db()
        try:
            self._old(db, "jaxon")
            db.bump_macro_weight("jaxon", "TAMPA MABLE", -0.4)
            for res in ("stop", "stop", "+2", "td"):
                db.log_snap(opponent_id="jaxon", side="defense", situation_raw="x", our_call="x", formation="x",
                            play="x", result=res, concept_seen="Mesh", macro="TAMPA MABLE")
            db.conn.commit()
        finally:
            db.close()
        db = open_db()  # migrates every stored Active 8 on open
        try:
            self.assertEqual(json.loads(db.get_meta("active_macros:jaxon"))["schema"], 2)
            self.assertEqual(legacy_picks(db, "jaxon"), self.OLD)
            self.assertAlmostEqual({r["macro"]: r["weight"] for r in db.get_macro_weights("jaxon")}["TAMPA MABLE"], -0.4)
            plan = build_prep_plan("jaxon", db=db, offline=True)  # learned weight still steers the pick
            self.assertNotIn("TAMPA MABLE", plan["macro_selection"]["defense"])
            self.assertEqual(load_selection(db, "jaxon")["defense"], plan["macro_selection"]["defense"])
            out = summary(db, "jaxon")
            self.assertIn("Retrain", out)
            self.assertIn("TAMPA MABLE", out)
            self.assertEqual(db.count_snaps("jaxon"), 4)
        finally:
            db.close()

    def test_postgame_status_counts_exact_macro_names(self) -> None:
        from cfb_coach.madden.macros import macro_status
        from cfb_coach.madden.postgame import learn

        db = self.madden_db()
        try:
            for _ in range(3):  # "TEX2 L CONT" results must not count toward "TEX2 L"-prefixed names
                db.log_snap(opponent_id="gavin", side="defense", situation_raw="x", our_call="x", formation="x",
                            play="x", result="stop", macro="TEX 4 MAN")
            learn(db, "gavin")
            self.assertEqual(macro_status("TEX 4 MAN", db), "proven")
            self.assertEqual(macro_status("TEX2 L CONT", db), "meta_grounded")
        finally:
            db.close()


class TestCfbKeepsAidansSheets(_DB):
    """CFB is unchanged: its own exact sheets, editable from `macro-settings --game cfb27`."""

    def test_cfb_unchanged_without_edits(self) -> None:
        cards, _ = cfb_macros.active_loadout_cards(None)
        for c in cards:
            self.assertEqual(c["copy_block"], cfb_macros.get_macro(c["id"])["copy_block"])
        self.assertEqual(cfb_macros.aidan_offense_settings("MAN"), cfb_macros.catalog_offense_rows("MAN"))
        self.assertFalse(ms.settings_path().exists())

    def test_cfb_macro_settings_edit(self) -> None:
        rc, _ = self.run_cli(["macro-settings", "--game", "cfb27", "CROSS", "--set", "Secondary: CB Depth = 6"])
        self.assertEqual(rc, 0)
        cards, _ = cfb_macros.active_loadout_cards(None)
        cross = next(c for c in cards if c["id"] == "CROSS")
        self.assertIn("[ ] CB Depth: 6", cross["copy_block"])
        self.assertIn("WHEN: After 2+ live Cross Wheels", cross["copy_block"])  # his notes kept
        rc, _ = self.run_cli(["macro-settings", "--game", "cfb27", "MAN", "--set", "Route assignments: WR1 = corner"])
        rows = cfb_macros.aidan_offense_settings("MAN")
        self.assertIn(("WR1", "corner"), [(r["setting"], r["value"]) for r in rows])
        self.assertEqual(sum(1 for r in rows if r["setting"] == "WR1"), 1)
        # Madden defense macros never read Aidan's CFB sheets — research-built only
        self.assertEqual(macro_detail("TAMPA MABLE")["settings_source"].split(" ")[0], "research-built")
