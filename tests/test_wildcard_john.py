"""John is a Madden-only cold start. Synthetic rows only. No real game database."""

from __future__ import annotations

import io
import json
import os
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from cfb_coach.cli import main
from cfb_coach.db import CoachDB
from cfb_coach.games import madden_db_path
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden import data as mdata
from cfb_coach.madden.human_opponents import (
    ensure_madden_opponent,
    register_cold_start,
    resolve_madden_opponent,
)
from cfb_coach.madden.model import inference
from cfb_coach.madden.model.schema import CoachingMode, ML_LATENCY_BUDGET_MS
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.opponents import resolve_opponent
from cfb_coach.seed import load_seed as load_cfb_seed
from cfb_coach.vod_model.flags import use_cli_flags

_BOOK = {
    "offense": {"Gun Bunch": ["Inside Zone"]},
    "defense": {"Nickel Over": ["Cover 4 Quarters", "Cover 3 Sky"]},
}


def _rows(db: CoachDB, oid: str) -> list[tuple]:
    return [
        (row["bucket"], row["key"], int(row["count"]), int(row["success"]))
        for row in db.get_tendencies(oid)
    ]


class JohnProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self._td.name) / "madden.db", seed=mdata.load_seed())

    def tearDown(self) -> None:
        self.db.close()
        self._td.cleanup()

    def test_john_is_absent_from_the_shared_rosters(self) -> None:
        self.assertNotIn("john", load_cfb_seed()["opponents"])
        self.assertNotIn("john", mdata.load_seed()["opponents"])
        self.assertIsNone(resolve_opponent("john"))
        self.assertIsNone(resolve_opponent("jo"))
        self.assertEqual(resolve_madden_opponent("john"), "john")
        self.assertEqual(resolve_madden_opponent("John"), "john")
        self.assertIsNone(resolve_madden_opponent("jo"))
        self.assertIsNone(resolve_madden_opponent("johnny"))
        self.assertEqual(resolve_madden_opponent("gavin"), "gavin")
        self.assertEqual(resolve_madden_opponent("auburn"), "gavin")

    def test_priors_are_neutral_and_not_shared(self) -> None:
        gavin_before = _rows(self.db, "gavin")
        tiano_before = _rows(self.db, "tiano")
        cfb = load_cfb_seed()["opponents"]
        gavin_coverages = list((cfb["gavin"].get("defense_vs_us") or {}).get("coverages") or [])
        tiano_coverages = list((cfb["tiano"].get("defense_vs_us") or {}).get("coverages") or [])
        self.assertTrue(gavin_coverages)
        self.assertTrue(tiano_coverages)
        # Madden personas keep the archetype and drop the CFB play lists, so the
        # tendency table starts empty. Those lists must not land on John either.
        self.assertEqual(gavin_before, [])
        self.assertEqual(tiano_before, [])
        profile = ensure_madden_opponent(self.db, "john")
        self.assertIsNotNone(profile)
        assert profile is not None
        gavin = self.db.get_opponent("gavin") or {}
        tiano = self.db.get_opponent("tiano") or {}
        self.assertEqual(profile["archetype"], "unknown")
        self.assertEqual(gavin.get("archetype"), "split_field_zone")
        self.assertEqual(tiano.get("archetype"), "c2_c3_mixer")
        self.assertEqual(profile["confidence"], "none")
        self.assertEqual(profile["persona_confidence"], "none")
        self.assertIsNone(profile["team_now"])
        self.assertIsNone(profile["nfl_team"])
        self.assertEqual(profile["offense"], {})
        self.assertEqual(profile["defense_vs_us"]["coverages"], [])
        rendered = json.dumps(profile)
        for name in gavin_coverages + tiano_coverages:
            self.assertNotIn(name, rendered)
        self.assertNotIn("coverages", gavin.get("defense_vs_us") or {})
        self.assertEqual(profile["defense_vs_us"]["man_blitz"], "unknown")
        self.assertEqual(profile["sample_snaps"], [])
        self.assertTrue(profile["madden_only"])
        self.assertTrue(profile["cold_start"])
        self.assertNotEqual(profile["archetype"], gavin.get("archetype"))
        self.assertNotEqual(profile["defense_vs_us"], gavin.get("defense_vs_us"))
        self.assertNotEqual(profile["archetype"], tiano.get("archetype"))
        self.assertFalse(gavin.get("madden_only"))
        self.assertFalse(tiano.get("cold_start"))
        self.assertEqual(_rows(self.db, "john"), [])
        self.assertEqual(_rows(self.db, "gavin"), gavin_before)
        self.assertEqual(_rows(self.db, "tiano"), tiano_before)
        with self.assertRaises(ValueError):
            register_cold_start(self.db, "gavin")
        with self.assertRaises(ValueError):
            register_cold_start(self.db, "auburn")
        self.assertEqual(self.db.get_opponent("gavin"), gavin)

    def test_another_cold_start_does_not_copy_john_or_gavin(self) -> None:
        ensure_madden_opponent(self.db, "john")
        alex = register_cold_start(self.db, "alex", display_name="Alex")
        self.assertEqual(alex["id"], "alex")
        self.assertEqual(alex["archetype"], "unknown")
        self.assertEqual(alex["offense"], {})
        self.assertNotEqual(alex["id"], "john")
        self.assertFalse(alex.get("persona_source", "").startswith("shared"))
        self.assertEqual(_rows(self.db, "alex"), [])
        self.assertEqual(resolve_madden_opponent("alex", self.db), "alex")

    def test_in_game_learning_updates_only_john(self) -> None:
        ensure_madden_opponent(self.db, "john")
        register_cold_start(self.db, "alex")
        gavin_before = _rows(self.db, "gavin")
        tiano_before = _rows(self.db, "tiano")
        alex_before = _rows(self.db, "alex")

        def _make(sit, **kwargs):
            return make_call(
                sit, "john", self.db, playbook=_BOOK, rng=random.Random(0),
                allow_human_ml=False, last_coverage=kwargs.get("last_coverage"),
                last_concept=kwargs.get("last_concept"),
            )

        ctrl = LivePlayController(
            db=self.db,
            opponent_id="john",
            make_call=_make,
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "synthetic",
            brand="Madden 27 Franchise",
            enable_execution_verify=True,
        )
        ctrl.start()
        first = ctrl.call_only("1&10 my 25")
        self.assertTrue(first["ok"])
        logged = ctrl.result_and_call(
            outcome="+4", sit_raw="2&6 my 40", their="cover 3", verify_touched=False,
        )
        self.assertTrue(logged["ok"])
        ctrl.end_game(result_wl="win", score="7-0")
        john_rows = _rows(self.db, "john")
        self.assertTrue(any(bucket == "their_coverage" and "cover 3" in key.lower() for bucket, key, _count, _success in john_rows))
        self.assertFalse(any(bucket.endswith("_seed") for bucket, _key, _count, _success in john_rows))
        self.assertEqual(_rows(self.db, "gavin"), gavin_before)
        self.assertEqual(_rows(self.db, "tiano"), tiano_before)
        self.assertEqual(_rows(self.db, "alex"), alex_before)
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.HEURISTIC)


class JohnCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)
        env = {"CFB_COACH_DB": str(self.dir / "coach.db"), "HOME": str(self.dir)}
        self._env = mock.patch.dict(os.environ, env, clear=False)
        self._env.start()
        for key in ("CFB_COACH_MADDEN_DB", "CFB_COACH_MADDEN_PRIMARY_TEAM", "CFB_COACH_MADDEN_LAB_TEAM"):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        self._env.stop()
        self._td.cleanup()

    def run_cli(self, argv: list[str]) -> tuple[int, str]:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch("cfb_coach.prep_browser.open_prep_html"):
            rc = main(argv)
        return rc, out.getvalue() + err.getvalue()

    def test_prep_and_play_resolve_john(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_cli(["preflight", "--mode", "human", "-o", "john", "--no-vod-prior"])
        rc, out = self.run_cli(["opponents", "--game", "madden27"])
        self.assertEqual(rc, 0)
        self.assertIn("john", out)
        self.assertIn("Madden-only cold start", out)
        rc, cfb = self.run_cli(["opponents"])
        self.assertEqual(rc, 0)
        self.assertNotIn("Madden-only cold start", cfb)
        rc, blocked = self.run_cli(
            ["preflight", "--game", "madden27", "--mode", "human", "-o", "john"]
        )
        self.assertEqual(rc, 2)
        self.assertIn("FAIL", blocked)
        self.assertIn("no_vod_prior", blocked)
        rc, report = self.run_cli(
            ["preflight", "--game", "madden27", "--mode", "human", "-o", "john", "--no-vod-prior"]
        )
        self.assertEqual(rc, 0, report)
        self.assertIn("verdict: PASS", report)
        self.assertIn("heuristic only", report)
        self.assertIn("accepted stays unknown", report)
        self.assertIn("established caller", report)
        rc, prep = self.run_cli(
            ["prep", "--game", "madden27", "-o", "john", "--franchise", "primary",
             "--offline", "--text", "--mark-applied", "--no-vod-prior"]
        )
        self.assertEqual(rc, 0, prep)
        self.assertIn("John", prep)
        self.assertIn("unknown", prep.lower())
        rc, offense = self.run_cli(
            ["play", "--game", "madden27", "-o", "john", "--franchise", "primary",
             "--once", "1&10 my 25", "--no-overlay", "--terminal", "--no-vod-prior"]
        )
        self.assertEqual(rc, 0, offense)
        self.assertIn("HUMAN OFFENSE ML PILOT: OFF", offense)
        self.assertIn("vs john", offense)
        rc, defense = self.run_cli(
            ["play", "--game", "madden27", "-o", "john", "--once", "d 2&7",
             "--no-overlay", "--terminal", "--no-vod-prior"]
        )
        self.assertEqual(rc, 0, defense)
        self.assertRegex(defense, r"\| User ")
        rc, refused = self.run_cli(
            ["play", "--game", "madden27", "-o", "john", "--once", "1&10 my 25",
             "--no-overlay", "--terminal", "--ml-human-control", "--no-vod-prior"]
        )
        self.assertEqual(rc, 2, refused)
        self.assertIn("requires experimental mode", refused)

    def test_flag_on_requires_confirmation_and_defense_stays_established(self) -> None:
        self.run_cli(
            ["prep", "--game", "madden27", "-o", "john", "--offline", "--text",
             "--mark-applied", "--no-vod-prior"]
        )
        db = CoachDB(madden_db_path(), seed=mdata.load_seed())
        try:
            self.assertIs(inference.resolve_mode(db), CoachingMode.HEURISTIC)
            inference.set_mode(db, CoachingMode.EXPERIMENTAL)
            from cfb_coach.madden.model import defense_coordinator as defense

            defense.set_mode(db, "experimental")
        finally:
            db.close()
        rc, off = self.run_cli(
            ["play", "--game", "madden27", "-o", "john", "--once", "1&10 my 25",
             "--no-overlay", "--terminal", "--no-vod-prior"]
        )
        self.assertEqual(rc, 0, off)
        self.assertIn("HUMAN OFFENSE ML PILOT: OFF", off)
        self.assertNotIn("HUMAN OFFENSE ML PILOT: ON", off)
        rc, on = self.run_cli(
            ["play", "--game", "madden27", "-o", "john", "--once", "1&10 my 25",
             "--no-overlay", "--terminal", "--ml-human-control", "--no-vod-prior"]
        )
        self.assertEqual(rc, 0, on)
        self.assertIn("HUMAN OFFENSE ML PILOT: ON", on)
        self.assertIn("A human must confirm each call", on)
        rc, dcall = self.run_cli(
            ["play", "--game", "madden27", "-o", "john", "--once", "d 2&7",
             "--no-overlay", "--terminal", "--no-vod-prior"]
        )
        self.assertEqual(rc, 0, dcall)
        self.assertNotIn("DEFENSE ML", dcall)
        with mock.patch(
            "cfb_coach.madden.model.defense_coordinator.maybe_apply_defense"
        ) as apply:
            apply.side_effect = lambda call, *args, **kwargs: call
            db = CoachDB(madden_db_path(), seed=mdata.load_seed())
            try:
                sit = parse_madden_situation("d 2&7", default_side="defense")
                john = make_call(sit, "john", db, rng=random.Random(1), playbook=_BOOK)
                self.assertTrue(john.formation)
                apply.assert_not_called()
                gavin = make_call(
                    parse_madden_situation("d 2&7", default_side="defense"),
                    "gavin", db, rng=random.Random(1), playbook=_BOOK,
                )
                self.assertTrue(gavin.formation)
                apply.assert_called()
            finally:
                db.close()
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)

    def test_preflight_blocks_when_defense_experimental_is_on(self) -> None:
        self.run_cli(
            ["preflight", "--game", "madden27", "--mode", "human", "-o", "john", "--no-vod-prior"]
        )
        db = CoachDB(madden_db_path(), seed=mdata.load_seed())
        try:
            from cfb_coach.madden.model import defense_coordinator as defense
            from cfb_coach.madden.preflight import run_human_preflight

            defense.set_mode(db, "experimental")
            with use_cli_flags(no_vod_prior=True):
                report = run_human_preflight(db, "john")
            names = {item["name"]: item for item in report["checks"]}
            self.assertFalse(names["defense_mode_off"]["ok"])
            self.assertTrue(names["defense_established_only"]["ok"])
            self.assertIn("blocked", names["defense_established_only"]["detail"])
            self.assertFalse(report["ok"])
        finally:
            db.close()
