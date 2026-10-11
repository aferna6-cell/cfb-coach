"""Dedicated tests for the live snap line. One requirement, one test."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.madden.model.inference import (
    ML_LATENCY_BUDGET_MS,
    ManualClock,
    OppLookSession,
    clear_defense_model_cache,
    display_or_fallback,
    exact_defense_line,
    exact_offense_line,
    finish_live_snap,
    live_display_mode,
    present_snap,
    run_within_budget,
)
from cfb_coach.madden.model.schema import CoachingMode, MacroState, OpponentContext, OpponentContextKey
from cfb_coach.madden.playbook import make_record
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import parse_madden_situation

_OFFENSE_EXACT = r"^.+ — .+ \| .+ \| .+ → .+$"
_DEFENSE_EXACT = r"^.+ — .+ \| .+ \| User .+$"
_FALLBACK = "Gun Bunch — Mesh | No adj | Spot → Drag"
_SECRET = "TAMPA-MABLE-SECRET"


def _stock_books() -> dict:
    return {
        "offense": make_record("offense", "stock", "Buccaneers")["formations"],
        "defense": make_record("defense", "stock", "49ers")["formations"],
    }


class TestLiveExactLines(unittest.TestCase):
    def test_offense_and_defense_exact_line_regex(self) -> None:
        offense = exact_offense_line("Gun Doubles", "Mesh", "Hot route", "Spot", "Drag")
        defense = exact_defense_line("Nickel Over", "Cover 3", "TAMPA", "hook")
        self.assertRegex(offense, r"^Gun Doubles — Mesh \| Hot route \| Spot → Drag$")
        self.assertRegex(offense, _OFFENSE_EXACT)
        self.assertRegex(defense, r"^Nickel Over — Cover 3 \| TAMPA \| User hook$")
        self.assertRegex(defense, _DEFENSE_EXACT)
        self.assertNotIn("\n", offense)
        self.assertNotIn("\n", defense)
        self.assertNotRegex("Gun — Mesh | Hot | Spot -> Drag", _OFFENSE_EXACT)
        self.assertNotRegex("Nickel — Cover 3 | Tampa | hook", _DEFENSE_EXACT)

    def test_no_adj_when_adjustment_is_absent(self) -> None:
        missing = exact_offense_line("Gun Bunch", "Mesh", None, "Spot", "Drag")
        blank = exact_offense_line("Gun Bunch", "Mesh", "  ", "Spot", "Drag")
        self.assertIn("| No adj |", missing)
        self.assertIn("| No adj |", blank)
        self.assertEqual(missing, "Gun Bunch — Mesh | No adj | Spot → Drag")
        self.assertEqual(blank, missing)

    def test_single_read_uses_em_dash_not_checkdown(self) -> None:
        line = exact_offense_line("Gun Bunch", "Mesh", "Hot route", "Read 1", None)
        self.assertIn("Read 1 → —", line)
        self.assertNotIn("Checkdown", line)
        self.assertRegex(line, r"^Gun Bunch — Mesh \| Hot route \| Read 1 → —$")


class TestLiveMacros(unittest.TestCase):
    def test_macros_below_armed_are_never_shown(self) -> None:
        hidden = (
            MacroState.DRAFT,
            MacroState.EDITOR_VERIFIED,
            MacroState.INSTALLED,
            MacroState.RETIRED,
            MacroState.UNKNOWN,
            None,
            "verified",
            "draft",
        )
        for state in hidden:
            line = exact_defense_line(
                "Nickel Over", "Cover 3", None, "hook",
                macro_name=_SECRET, macro_state=state,
            )
            self.assertNotIn(_SECRET, line, msg=state)
            self.assertIn("| No adj |", line, msg=state)
            self.assertRegex(line, _DEFENSE_EXACT)
        shown = present_snap(
            side="defense",
            formation="Nickel Over",
            play="Cover 3",
            user_job="hook",
            macro_name=_SECRET,
            macro_state=MacroState.DRAFT,
        )
        self.assertNotIn(_SECRET, shown.line)
        for state in (MacroState.ARMED, MacroState.ELIGIBLE, MacroState.EVALUATED, "armed"):
            line = exact_defense_line(
                "Nickel Over", "Cover 3", None, "hook",
                macro_name=_SECRET, macro_state=state,
            )
            self.assertIn(f"| {_SECRET} |", line, msg=state)
            self.assertNotIn("No adj", line, msg=state)
            self.assertRegex(line, _DEFENSE_EXACT)


class TestLiveFallback(unittest.TestCase):
    def test_invalid_line_falls_back_and_is_not_displayed(self) -> None:
        candidate = "menu\nGun — Mesh | No adj | A → B"
        shown = display_or_fallback(candidate, _FALLBACK, "offense")
        self.assertEqual(shown, _FALLBACK)
        self.assertNotIn("menu", shown)
        self.assertNotIn(candidate, shown)
        self.assertRegex(shown, _OFFENSE_EXACT)
        shapeless = display_or_fallback("just a menu", _FALLBACK, "offense")
        self.assertEqual(shapeless, _FALLBACK)
        self.assertNotIn("just a menu", shapeless)


class TestLiveTimeout(unittest.TestCase):
    def test_slow_model_is_cut_off_and_fallback_lands_in_time(self) -> None:
        self.assertEqual(ML_LATENCY_BUDGET_MS, 150)
        deterministic_elapsed = 0.0
        for _ in range(20):
            clock = ManualClock()

            def slow(bound: ManualClock = clock) -> str:
                bound.advance(2.0)
                return "SLOW MODEL LINE THAT MUST NOT SHOW"

            line, elapsed_ms, timed_out = run_within_budget(slow, _FALLBACK, clock=clock)
            deterministic_elapsed = elapsed_ms
            self.assertTrue(timed_out)
            self.assertEqual(line, _FALLBACK)
            self.assertNotIn("SLOW", line)
            self.assertLessEqual(elapsed_ms, ML_LATENCY_BUDGET_MS)
            self.assertLess(elapsed_ms, 150)

        fast = ManualClock()

        def quick() -> str:
            fast.advance(0.01)
            return "Gun Trips — Inside Zone | No adj | Read → Cut"

        shown, fast_ms, fast_timed_out = run_within_budget(quick, _FALLBACK, clock=fast)
        self.assertFalse(fast_timed_out)
        self.assertEqual(shown, "Gun Trips — Inside Zone | No adj | Read → Cut")
        self.assertLess(fast_ms, 150)

        samples: list[float] = []
        for _ in range(20):
            def sleeper() -> str:
                time.sleep(0.4)
                return "SLOW MODEL LINE THAT MUST NOT SHOW"

            _line, elapsed_ms, _timed_out = run_within_budget(sleeper, _FALLBACK)
            samples.append(elapsed_ms)
        ordered = sorted(samples)
        max_ms = ordered[-1]
        p95_ms = ordered[18]
        print(
            f"MEASURED timeout_max_ms={max_ms:.2f} p95_ms={p95_ms:.2f} "
            f"deterministic_cutoff_ms={deterministic_elapsed:.2f}"
        )


class TestLiveSnap(unittest.TestCase):
    def test_no_vod_calls_at_snap_time(self) -> None:
        vod_calls: list[str] = []

        def refuse(*_args, **_kwargs):
            vod_calls.append("apply_madden_call")
            raise AssertionError("VOD was called from the live presentation path")

        sit = parse_madden_situation("1&10")
        sit.extras = {"opp_look": None}
        with mock.patch("cfb_coach.vod_model.live.apply_madden_call", side_effect=refuse):
            snap = present_snap(
                formation="Gun Bunch",
                play="Mesh",
                read1="Spot",
                read2="Drag",
                opp_look=None,
            )
            returned = finish_live_snap(sit, snap, None)
        self.assertEqual(vod_calls, [])
        self.assertIs(returned, snap)
        self.assertRegex(snap.line, _OFFENSE_EXACT)
        self.assertNotIn("vod", snap.line.lower())


class TestLiveDefenseModel(unittest.TestCase):
    def setUp(self) -> None:
        clear_defense_model_cache()
        self._td = tempfile.TemporaryDirectory()
        root = Path(self._td.name)
        self._env = mock.patch.dict(
            os.environ,
            {
                "HOME": str(root / "home"),
                "CFB_COACH_DB": str(root / "coach.db"),
                "CFB_COACH_MADDEN_DB": str(root / "madden27.db"),
            },
            clear=False,
        )
        (root / "home").mkdir()
        self._env.start()

    def tearDown(self) -> None:
        clear_defense_model_cache()
        self._env.stop()
        self._td.cleanup()

    def test_defense_model_loads_once_per_session(self) -> None:
        from cfb_coach.db import CoachDB
        from cfb_coach.madden.model import defense_coordinator as defense

        loads: list[str] = []
        sentinel = {"loaded": object()}

        def fake_load(path=None):
            loads.append("" if path is None else str(path))
            return sentinel

        book = {"Nickel Over": ["Cover 3 Sky", "Cover 4 Quarters"]}
        situation = parse_madden_situation("d 1&10", default_side="defense")
        db = CoachDB(Path(self._td.name) / "madden27.db")
        try:
            missing = Path(self._td.name) / "no-such-model.json"
            db.set_meta(defense.MODEL_META, str(missing))
            with mock.patch.object(defense, "load_model", side_effect=fake_load):
                defense.live_pick(
                    situation, db, "gavin", book, session_id="sess-a", snap_seq=1,
                )
                defense.live_pick(
                    situation, db, "gavin", book, session_id="sess-a", snap_seq=2,
                )
                self.assertEqual(len(loads), 1)
                defense.live_pick(
                    situation, db, "gavin", book, session_id="sess-b", snap_seq=1,
                )
                self.assertEqual(len(loads), 2)
        finally:
            db.close()
        self.assertTrue(all(item.endswith("no-such-model.json") for item in loads))
        self.assertTrue(all(".cfb-coach" not in item for item in loads))


class TestLiveMode(unittest.TestCase):
    def test_experimental_maps_to_shadow_never_hybrid(self) -> None:
        self.assertIs(live_display_mode(CoachingMode.EXPERIMENTAL), CoachingMode.SHADOW)
        self.assertIs(live_display_mode("experimental"), CoachingMode.SHADOW)
        self.assertIsNot(live_display_mode(CoachingMode.EXPERIMENTAL), CoachingMode.HYBRID)
        self.assertIs(live_display_mode(CoachingMode.HYBRID), CoachingMode.HYBRID)
        model_line = "Gun Trips — Four Verticals | No adj | Seam → Post"

        def refuse() -> str:
            raise AssertionError("experimental display called the model")

        snap = present_snap(
            formation="Gun Bunch",
            play="Mesh",
            read1="Spot",
            read2="Drag",
            mode=CoachingMode.EXPERIMENTAL,
            model_line=model_line,
            model_fn=refuse,
        )
        self.assertIs(snap.effective_mode, CoachingMode.SHADOW)
        self.assertIsNot(snap.effective_mode, CoachingMode.HYBRID)
        self.assertIn("Mesh", snap.line)
        self.assertNotIn("Four Verticals", snap.line)
        self.assertEqual(snap.line, "Gun Bunch — Mesh | No adj | Spot → Drag")


class TestOppLook(unittest.TestCase):
    def _isolated(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        (root / "home").mkdir()
        env = mock.patch.dict(
            os.environ,
            {
                "HOME": str(root / "home"),
                "CFB_COACH_DB": str(root / "coach.db"),
                "CFB_COACH_MADDEN_DB": str(root / "madden27.db"),
            },
            clear=False,
        )
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(td.cleanup)

    def test_null_opp_look_is_byte_identical(self) -> None:
        self._isolated()
        books = _stock_books()
        empty_macros = {"offense": [], "defense": []}

        def once(with_null_look: bool) -> bytes:
            sit = parse_madden_situation("1&10")
            sit.coverage_hint = None
            if with_null_look:
                sit.extras = {"opp_look": None}
            call = make_call(
                sit, "gavin", None,
                rng=random.Random(7),
                playbook=books,
                active_macros=empty_macros,
                live_macros=False,
            )
            return call.format().encode()

        omitted = once(False)
        explicit_null = once(True)
        self.assertEqual(explicit_null, omitted)
        self.assertTrue(omitted)

    def test_blank_opp_look_is_null_and_empty_stays_empty(self) -> None:
        """'' and whitespace become NULL. None stays NULL. 'empty' stays 'empty'."""
        self._isolated()
        from cfb_coach.db import CoachDB
        from cfb_coach.last_snap import LastSnapBook
        from cfb_coach.live_server import LivePlayController, render_live_html
        from cfb_coach.madden.model.inference import UNSET_OPP_LOOK, bind_opp_look

        madden_db = Path(os.environ["CFB_COACH_MADDEN_DB"])
        db = CoachDB(madden_db)
        self.addCleanup(db.close)

        class _Call:
            def __init__(self, sit) -> None:
                self.side = sit.side
                self.formation = "Gun"
                self.play = "Mesh"
                self.macro = None

            def format(self) -> str:
                return "Gun — Mesh"

        passed: list[object] = []

        def _make(sit, **_kwargs):
            passed.append((sit.extras or {}).get("opp_look"))
            return _Call(sit)

        ctrl = LivePlayController(
            db=db,
            opponent_id="gavin",
            make_call=_make,
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "",
            session_id="api-look",
        )
        ctrl.call_only("1&10", opp_look="")
        ctrl.result_and_call(outcome="+4", sit_raw="2&7", opp_look=None)
        ctrl.result_and_call(outcome="+5", sit_raw="1&10", opp_look="empty")
        ctrl.result_and_call(outcome="+3", sit_raw="1&10", opp_look="   ")
        ctrl.result_and_call(outcome="+1", sit_raw="1&10", opp_look=None)
        self.assertEqual(passed[:4], [None, None, "empty", None])

        api_rows = db.conn.execute(
            "SELECT opp_look FROM snaps WHERE session_id = ? ORDER BY id",
            ("api-look",),
        ).fetchall()
        self.assertEqual([row["opp_look"] for row in api_rows], [None, None, "empty", None])
        self.assertNotIn("", [row["opp_look"] for row in api_rows])

        def _cli(raw: str, look: object = UNSET_OPP_LOOK):
            sit = parse_madden_situation(raw, default_side="offense")
            return bind_opp_look(sit, look, raw=raw)

        blank = _cli("1&10", "")
        missing = _cli("1&10", None)
        empty = _cli("1&10", "empty")
        spaced = _cli("1&10", " \t ")
        inferred = _cli("1&10 empty", UNSET_OPP_LOOK)
        self.assertIsNone(blank.extras["opp_look"])
        self.assertIsNone(missing.extras["opp_look"])
        self.assertEqual(empty.extras["opp_look"], "empty")
        self.assertIsNone(spaced.extras["opp_look"])
        self.assertEqual(inferred.extras["opp_look"], "empty")

        book = LastSnapBook(db, "gavin", parse_madden_situation, session_id="cli-look")
        for sit in (blank, missing, empty):
            book.remember_call(_Call(sit), sit, opp_look=sit.extras["opp_look"])
            book.close_from_form("+6", their="")
        cli_rows = db.conn.execute(
            "SELECT opp_look FROM snaps WHERE session_id = ? ORDER BY id",
            ("cli-look",),
        ).fetchall()
        self.assertEqual([row["opp_look"] for row in cli_rows], [None, None, "empty"])
        self.assertTrue(all(row["opp_look"] != "" for row in db.conn.execute("SELECT opp_look FROM snaps")))

        html = render_live_html(ctrl)
        self.assertIn("function opponentLook()", html)
        self.assertIn("return raw ? raw : null;", html)
        self.assertIn("opp_look: opponentLook()", html)

    def test_opp_look_resets_after_each_snap(self) -> None:
        self._isolated()
        session = OppLookSession()
        first = present_snap(session=session, opp_look="cover 3", read1="Spot", read2="Drag")
        self.assertEqual(first.opp_look, "cover 3")
        self.assertIsNone(session.current)
        second = present_snap(session=session, read1="Spot", read2="Drag")
        self.assertIsNone(second.opp_look)
        self.assertIsNone(session.current)

        sit = parse_madden_situation("1&10")
        hooked = OppLookSession()
        sit.extras = {"opp_look": "cover 3", "opp_look_session": hooked, "live_macros": True}
        make_call(
            sit, "gavin", None,
            rng=random.Random(3),
            playbook=_stock_books(),
            active_macros={"offense": [], "defense": []},
            live_macros=False,
        )
        self.assertIsNone(hooked.current)
        self.assertIsNone(sit.extras.get("opp_look"))


class TestSeedFields(unittest.TestCase):
    def test_seed_fields_set_together_or_both_left_out(self) -> None:
        both_unset = OpponentContext(opponent_id="gavin")
        self.assertIsNone(both_unset.seeded_from_key)
        self.assertIsNone(both_unset.seed_shrink_weight)
        prior = OpponentContextKey(
            opponent_id="gavin", patch_id="1.006", roster_snapshot_id="r1", table_version="1",
        )
        row = dict(opponent_id="gavin", patch_id="1.007", roster_snapshot_id="r2", table_version="1")
        both_set = OpponentContext(**row, seeded_from_key=prior, seed_shrink_weight=0.25)
        self.assertEqual(both_set.seeded_from_key, prior)
        self.assertEqual(both_set.seed_shrink_weight, 0.25)
        with self.assertRaises(ValueError):
            OpponentContext(**row, seeded_from_key=prior)
        with self.assertRaises(ValueError):
            OpponentContext(**row, seed_shrink_weight=0.25)


class TestPostgameOverride(unittest.TestCase):
    def test_override_rate_denominator_is_eligible_decisions(self) -> None:
        from cfb_coach.db import CoachDB
        from cfb_coach.madden.model.offense_postgame import evaluate_saved_games

        payloads = [
            {
                "propensity_method": "override",
                "propensity_override": True,
                "propensity_override_reason": "unexpressible",
            },
            {"propensity_method": "exact", "propensity_override": False, "behavior_propensity": 0.4},
            {"propensity_method": "exact", "propensity_override": False, "behavior_propensity": 0.25},
            {"propensity_method": "unknown"},
            {"propensity_method": "deterministic", "behavior_propensity": 1.0, "propensity_override": False},
            {"propensity_method": "deterministic", "behavior_propensity": 1, "propensity_override": False},
        ]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "madden.db"
        db = CoachDB(path, seed={"opponents": {}})
        try:
            for index, payload in enumerate(payloads, start=1):
                db.conn.execute(
                    "INSERT INTO ml_decisions (decision_ts, game_id, snap_id, session_id, mode, "
                    "final_formation, final_play, shadow_status, decision_json, latency_ms_model, created_ts) "
                    "VALUES ('t', 'g-rate', ?, 'g-rate', 'experimental', "
                    "'Gun Bunch', 'Mesh', 'ok', ?, NULL, 't')",
                    (f"snap-{index}", json.dumps(payload)),
                )
            db.conn.commit()
        finally:
            db.close()
        report_db = CoachDB.open_read_only(path)
        try:
            report = evaluate_saved_games(report_db, game_id="g-rate")
        finally:
            report_db.close()
        game = report["games"][0]
        rate = game["override_rate"]
        self.assertEqual(game["verified_executed_offensive_plays"], 0)
        self.assertEqual(rate["overrides"], 1)
        self.assertEqual(rate["denominator"], 3)
        self.assertNotEqual(rate["denominator"], 6)
        self.assertNotEqual(rate["denominator"], 4)
        self.assertNotEqual(rate["denominator"], 1)
        self.assertEqual(rate["vod_overrides"], 2)
        self.assertEqual(rate["rate"], 1 / 3)
        rendered = json.dumps(game)
        self.assertEqual(game["override_rate_line"], "Off-policy override rate: 1/3 eligible decisions")
        self.assertEqual(game["vod_override_line"], "VOD overrides excluded: 2")
        self.assertIn(game["override_rate_line"], rendered)
        self.assertIn(game["vod_override_line"], rendered)
        self.assertNotEqual(game["override_rate_line"], game["vod_override_line"])
