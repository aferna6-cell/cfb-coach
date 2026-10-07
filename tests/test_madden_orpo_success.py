"""The 19-0 CPU game: O-RPO must not take the call list, and a bad weight is not proven.

Fixture: tests/fixtures/cpu_19-0_offense_snaps.csv (session 07540a2031384493).
"""

from __future__ import annotations

import csv
from pathlib import Path

from cfb_coach.madden.catalog import is_run
from cfb_coach.madden.macros import (
    LEARNED_SUPPRESS,
    macro_status,
    macros_cooled_this_half,
    migrate_proven_below_suppress,
    set_status,
)
from cfb_coach.madden.offense_macros import select_offense, situation_macro
from cfb_coach.madden.playcaller import make_call, narrow_pool
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.zones import field_zone
from tests.test_madden27 import _Isolated

FIXTURE = Path(__file__).parent / "fixtures" / "cpu_19-0_offense_snaps.csv"
ACTIVE = ["O-RUN", "C3", "MAN", "PROT", "SHOT", "RZ", "O-RPO", "MATCH"]
# O-RPO is still above the suppress line, so the old priority would arm it first.
DURING = {
    "PROT": 2.19, "O-RUN": 1.42, "C3": 0.58, "MAN": -0.03,
    "O-HEAT": -1.28, "MATCH": -1.61, "O-RPO": 0.05,
}
# Weights after the game. O-RPO is suppressed.
AFTER = {
    "PROT": 2.19, "O-RUN": 1.42, "C3": 0.58, "MAN": -0.03,
    "O-HEAT": -1.28, "MATCH": -1.61, "O-RPO": -2.22,
}


def _fixture_rows() -> list[dict[str, str]]:
    with FIXTURE.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _zone(down: int, dist: int, yl: int) -> str:
    z = field_zone(yl, dist)
    return {"open": "open", "rz": "rz", "gl": "gl"}[z]


class TestLivePicker(_Isolated):
    def setUp(self) -> None:
        super().setUp()
        from cfb_coach.madden.prep import build_prep_plan

        plan = build_prep_plan("cpu", offline=True, persist=False)
        self.book = plan["playbook"]["offense"]["record"]["formations"]
        self.pool = [(f, p) for f, plays in self.book.items() for p in plays]
        self.arch = plan["archetype"]

    def _steer(self, raw: str, *, down: int, dist: int, yl: int, weights: dict[str, float]):
        sit = parse_madden_situation(raw)
        hint = sit.coverage_hint
        if not hint:
            return None
        return situation_macro(
            zone=_zone(down, dist, yl),
            coverage=hint,
            coverage_source="live",
            active=ACTIVE,
            down=down,
            book=self.book,
            weights=weights,
            pool=self.pool,
        )

    def test_fixture_replay_orun_fires_and_orpo_does_not_take_over(self) -> None:
        rows = _fixture_rows()
        self.assertEqual(len(rows), 35)
        orun = 0
        orpo = 0
        run_looks = 0
        for row in rows:
            down, dist, yl = int(row["down"]), int(row["distance"]), int(row["yardline"])
            steer = self._steer(row["situation_raw"], down=down, dist=dist, yl=yl, weights=DURING)
            if steer is None:
                continue
            self.assertNotEqual(steer["id"], "O-RPO", row["situation_raw"])
            if steer["id"] == "O-RPO":
                orpo += 1
            cls_raw = row["situation_raw"].lower()
            run_look = any(w in cls_raw for w in ("cover 2", "invert", "tampa", "cover 4", "cover 6"))
            if run_look and steer["id"] == "O-RUN":
                orun += 1
                run_looks += 1
                self.assertTrue(any(is_run(play) for _form, play in steer["pairs"]), steer["pairs"][:4])
        self.assertGreater(orun, 0)
        self.assertEqual(orpo, 0)
        # The recorded weights suppress O-RPO outright.
        suppressed = 0
        for row in rows:
            down, dist, yl = int(row["down"]), int(row["distance"]), int(row["yardline"])
            steer = self._steer(row["situation_raw"], down=down, dist=dist, yl=yl, weights=AFTER)
            if steer is not None and steer["id"] == "O-RPO":
                suppressed += 1
        self.assertEqual(suppressed, 0)

    def test_make_call_arms_orun_on_cover_2_and_keeps_a_run(self) -> None:
        db = self.madden_db()
        try:
            for mid, w in DURING.items():
                db.bump_macro_weight("cpu", mid, w)
            sit = parse_madden_situation("1&10 showing cover 2 invert")
            call = make_call(
                sit, "cpu", db, active_macros={"offense": ACTIVE, "defense": []},
                playbook={"offense": self.book},
            )
            self.assertEqual(call.macro, "O-RUN")
            self.assertTrue(is_run(call.play), call.play)
        finally:
            db.close()

    def test_rpo_steer_does_not_delete_designed_runs(self) -> None:
        steer = self._steer("1&10 showing cover 2 invert", down=1, dist=10, yl=40,
                            weights={"O-RPO": 2.0, "O-RUN": 0.2})
        self.assertIsNotNone(steer)
        self.assertEqual(steer["id"], "O-RPO")
        self.assertTrue(steer["rpo"])
        narrowed = narrow_pool(self.pool, steer)
        self.assertTrue(any(is_run(play) for _f, play in narrowed))
        self.assertTrue(any("rpo" in play.lower() for _f, play in narrowed))


class TestPrepSlots(_Isolated):
    def test_experimental_does_not_fill_a_hole_and_suppressed_stays_out(self) -> None:
        from cfb_coach.madden.prep import build_prep_plan

        plan = build_prep_plan("cpu", offline=True, persist=False)
        book = plan["playbook"]["offense"]["record"]["formations"]
        # These weights are what let a neutral O-RPO into the old top 8.
        weights = {"C2": -0.8, "O-HEAT": -1.28, "MATCH": -0.2}
        picked, _rows = select_offense(
            "cpu", book=book, weights=weights, archetype=plan["archetype"],
        )
        self.assertNotIn("O-RPO", picked)
        self.assertNotIn("O-HEAT", picked)
        self.assertNotIn("C2", picked)
        self.assertNotIn("MATCH", picked)
        self.assertLess(len(picked), 8)
        self.assertIn("O-RUN", picked)
        opted, _ = select_offense(
            "cpu", book=book, weights=weights, archetype=plan["archetype"],
            allow_experimental=True,
        )
        self.assertIn("O-RPO", opted)
        swapped, _ = select_offense(
            "cpu", book=book, weights=weights, archetype=plan["archetype"],
            swaps={"O-RPO"},
        )
        self.assertIn("O-RPO", swapped)
        # Opting in does not rescue a suppressed experimental macro.
        buried = dict(weights)
        buried["O-RPO"] = -2.22
        still, _ = select_offense(
            "cpu", book=book, weights=buried, archetype=plan["archetype"],
            allow_experimental=True,
        )
        self.assertNotIn("O-RPO", still)
        self.assertLessEqual(buried["O-RPO"], LEARNED_SUPPRESS)

    def test_config_flag_and_swap_are_the_opt_in(self) -> None:
        from cfb_coach.madden.macros import experimental_opt_in

        self.assertFalse(experimental_opt_in("O-RPO"))
        rc, out = self.run_cli(["config", "--game", "madden27", "--swap-macro", "O-RPO"])
        self.assertEqual(rc, 0, out)
        self.assertIn("O-RPO", out)
        self.assertTrue(experimental_opt_in("O-RPO"))
        self.assertFalse(experimental_opt_in("HEAT"))
        rc, out = self.run_cli(["config", "--game", "madden27", "--experimental-macros"])
        self.assertEqual(rc, 0, out)
        self.assertIn("experimental macros: on", out)
        self.assertTrue(experimental_opt_in("HEAT"))

    def test_suppressed_defense_macro_leaves_its_slot_empty(self) -> None:
        from cfb_coach.madden.macro_select import select_loadout

        db = self.madden_db()
        try:
            full = select_loadout("james", db=db)
            self.assertEqual(len(full["defense"]), 8)
            top = full["defense"][0]
            db.bump_macro_weight("james", top, -1.5)
            again = select_loadout("james", db=db)
            self.assertNotIn(top, again["defense"])
            self.assertEqual(len(again["defense"]), 7)
            self.assertFalse(set(again["defense"]) - set(full["defense"]))
        finally:
            db.close()


class TestSuccessAndCooldown(_Isolated):
    def test_situational_success_and_turnover_are_not_proven(self) -> None:
        from cfb_coach.madden.postgame import learn

        db = self.madden_db()
        try:
            for _ in range(3):
                db.log_snap(
                    opponent_id="cpu", side="offense", situation_raw="3&10", our_call="x",
                    formation="Gun", play="RPO Alert Out", macro="O-RPO", result="+1",
                    down=3, distance=10, yardline=40,
                )
            learn(db, "cpu")
            self.assertNotEqual(macro_status("O-RPO", db), "proven")
            for _ in range(3):
                db.log_snap(
                    opponent_id="cpu", side="offense", situation_raw="1&10", our_call="x",
                    formation="Gun", play="RPO Alert Out", macro="O-RPO", result="int",
                    down=1, distance=10, yardline=40,
                )
            learn(db, "cpu")
            self.assertNotEqual(macro_status("O-RPO", db), "proven")
            for _ in range(3):
                db.log_snap(
                    opponent_id="cpu", side="offense", situation_raw="1&10", our_call="x",
                    formation="Gun", play="Inside Zone", macro="O-RUN", result="+8",
                    down=1, distance=10, yardline=25,
                )
            learn(db, "cpu")
            self.assertEqual(macro_status("O-RUN", db), "proven")
        finally:
            db.close()

    def test_high_rate_below_suppress_is_not_proven(self) -> None:
        from cfb_coach.learning import merged_macro_weights
        from cfb_coach.madden.postgame import learn

        db = self.madden_db()
        try:
            for _ in range(9):
                db.log_snap(
                    opponent_id="cpu", side="offense", situation_raw="1&10", our_call="x",
                    formation="Gun", play="RPO Alert Out", macro="O-RPO", result="+8",
                    down=1, distance=10, yardline=25,
                )
            for _ in range(3):
                db.log_snap(
                    opponent_id="cpu", side="offense", situation_raw="1&10", our_call="x",
                    formation="Gun", play="RPO Alert Out", macro="O-RPO", result="int",
                    down=1, distance=10, yardline=40,
                )
            learn(db, "cpu")
            weight = merged_macro_weights(db, "cpu").get("O-RPO")
            self.assertIsNotNone(weight)
            self.assertLessEqual(weight, LEARNED_SUPPRESS)
            self.assertNotEqual(macro_status("O-RPO", db), "proven")
        finally:
            db.close()

    def test_turnover_cools_that_macro_for_the_rest_of_the_half(self) -> None:
        db = self.madden_db()
        try:
            db.log_snap(
                opponent_id="cpu", side="offense", situation_raw="1&10", our_call="x",
                formation="Gun", play="RPO Alert Out", macro="O-RPO [meta_grounded]",
                result="int", down=1, distance=10, yardline=91, quarter=1, session_id="s19",
            )
            self.assertIn("O-RPO", macros_cooled_this_half(db, "s19", 2))
            self.assertNotIn("O-RPO", macros_cooled_this_half(db, "s19", 3))
            cooled = {"O-RPO"}
            from cfb_coach.madden.prep import build_prep_plan

            plan = build_prep_plan("cpu", offline=True, persist=False)
            book = plan["playbook"]["offense"]["record"]["formations"]
            pool = [(f, p) for f, plays in book.items() for p in plays]
            quiet = situation_macro(
                zone="open", coverage="Cover 2 Invert", coverage_source="live",
                active=ACTIVE, down=1, book=book, weights={"O-RPO": 3.0, "O-RUN": 0.2},
                pool=pool, cooled=cooled,
            )
            self.assertNotEqual(quiet["id"] if quiet else None, "O-RPO")
            self.assertEqual(quiet["id"], "O-RUN")
            later = situation_macro(
                zone="open", coverage="Cover 2 Invert", coverage_source="live",
                active=ACTIVE, down=1, book=book, weights={"O-RPO": 3.0, "O-RUN": 0.2},
                pool=pool, cooled=set(),
            )
            self.assertEqual(later["id"], "O-RPO")
            # Defense: the red-zone macro stays off for the rest of the half after a pick.
            db.log_snap(
                opponent_id="james", side="defense", situation_raw="d 1&10", our_call="x",
                formation="Nickel", play="Tampa 2", macro="RZ COVER 2", result="fumble",
                down=1, distance=10, yardline=85, quarter=1, session_id="s19",
            )
            sit = parse_madden_situation("d 1&10 opp 12")
            sit.extras["session_id"] = "s19"
            sit.extras["quarter"] = 1
            dbook = {"Nickel Over": ["Cover 4 Quarters", "Tampa 2"]}
            cooled_call = make_call(
                sit, "james", db, active_macros={"offense": [], "defense": ["RZ COVER 2"]},
                playbook={"defense": dbook},
            )
            self.assertNotEqual(cooled_call.macro, "RZ COVER 2")
            sit.extras["quarter"] = 3
            next_half = make_call(
                sit, "james", db, active_macros={"offense": [], "defense": ["RZ COVER 2"]},
                playbook={"defense": dbook},
            )
            self.assertEqual(next_half.macro, "RZ COVER 2")
        finally:
            db.close()


class TestProvenMigration(_Isolated):
    def test_migration_clears_bogus_orpo_proven(self) -> None:
        from cfb_coach.madden.cli import open_db

        db = self.madden_db()
        try:
            set_status(db, "O-RPO", "proven")
            set_status(db, "O-RUN", "proven")
            set_status(db, "C3", "proven")
            db.bump_macro_weight("cpu", "O-RPO", -2.22)
            db.bump_macro_weight("global", "O-RPO", -0.78)
            db.bump_macro_weight("global", "O-RUN", 0.57)
            db.bump_macro_weight("global", "C3", 0.18)
            db.bump_macro_weight("global", "MAN", -0.01)
            cleared = migrate_proven_below_suppress(db)
            self.assertIn("O-RPO", cleared)
            self.assertNotEqual(macro_status("O-RPO", db), "proven")
            self.assertEqual(macro_status("O-RUN", db), "proven")
            self.assertEqual(macro_status("C3", db), "proven")
            set_status(db, "O-RPO", "proven")
            self.assertEqual(migrate_proven_below_suppress(db), [])
            self.assertEqual(macro_status("O-RPO", db), "proven")
        finally:
            db.close()

    def test_opening_the_madden_db_runs_the_migration(self) -> None:
        from cfb_coach.madden.cli import open_db

        db = self.madden_db()
        try:
            set_status(db, "O-RPO", "proven")
            db.bump_macro_weight("cpu", "O-RPO", -2.22)
            db.bump_macro_weight("global", "O-RPO", -0.78)
        finally:
            db.close()
        opened = open_db()
        try:
            self.assertNotEqual(macro_status("O-RPO", opened), "proven")
        finally:
            opened.close()


class TestSituationRaw(_Isolated):
    def test_logged_raw_matches_the_advanced_spot(self) -> None:
        from cfb_coach.db import CoachDB
        from cfb_coach.live_server import LivePlayController
        from cfb_coach.situation import parse_situation

        class Call:
            def __init__(self, sit):
                self.side = sit.side
                self.formation = "Gun"
                self.play = "Inside Cross"
                self.macro = "C3"

            def format(self):
                return f"{self.formation} — {self.play}"

        db = CoachDB(self.dir / "coach.db")
        self.addCleanup(db.close)
        ctrl = LivePlayController(
            db=db, opponent_id="cpu", make_call=lambda sit, **k: Call(sit),
            parse_situation=parse_situation, learn_summary=lambda: "", session_id="s19",
        )
        ctrl.call_only("1&10 my 22")
        ctrl.result_and_call(outcome="+3", sit_raw="1&10 my 22 cover 3 sky")
        ctrl.result_and_call(outcome="incomplete", sit_raw="2&7 my 25")
        rows = list(db.get_session_snaps("s19"))
        self.assertEqual(len(rows), 2)
        self.assertIn("1&10", rows[0]["situation_raw"])
        raw = rows[1]["situation_raw"]
        self.assertIn("2&7", raw)
        self.assertIn("my 25", raw)
        self.assertIn("cover 3", raw)
        self.assertNotIn("1&10 my 22", raw)
