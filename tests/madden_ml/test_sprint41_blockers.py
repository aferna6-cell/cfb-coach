"""Sprint 4.1: research priors, identity-aware logging, situational ML, model quality."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden.data import reads_for
from cfb_coach.madden.model import experimental_live, experimental_model, inference
from cfb_coach.madden.model.schema import (
    CoachingMode,
    ExecutedStatus,
    MLStatus,
    PolicySource,
    Verification,
)
from cfb_coach.madden.playcaller import MaddenCall
from cfb_coach.madden.situation import parse_madden_situation
from cfb_coach.situation import Situation


def _seed() -> dict:
    return {
        "opponents": {
            "cpu_lions": {
                "display_name": "Lions CPU",
                "team_now": "DET",
                "skill": "cpu",
                "confidence": "low",
                "profile_json": "{}",
            }
        }
    }


def _book() -> dict[str, list[str]]:
    return {
        "Gun Bunch": ["Mesh", "Stick", "Smash", "Four Verticals"],
        "Gun Doubles": ["Y Cross", "HB Dive", "Mesh Spot", "Flood"],
        "Singleback Ace": ["PA Boot", "Inside Zone", "Slant Flat"],
        "I Form Twins": ["Power O", "PA Cross", "QB Sneak"],
        "Pistol Trips": ["RPO Peek", "HB Zone", "Stick"],
    }


class ResearchPriorsTests(unittest.TestCase):
    def test_load_research_unpacks_tuple(self) -> None:
        fam, concept, version, origin = experimental_model._load_research_boosts()
        # Validated repo research/madden27.json should load offline.
        self.assertNotEqual(version, "none")
        self.assertTrue(origin in ("repo", "cache", "github") or origin == "")
        # Must produce nonempty attributed concept or family priors from real findings.
        self.assertTrue(
            fam or concept,
            f"expected nonempty research priors, got fam={fam} concept={concept}",
        )
        # Caps prevent fabricated precise matchup odds.
        for v in list(fam.values()) + list(concept.values()):
            self.assertLessEqual(abs(v), experimental_model.MAX_RESEARCH_BOOST + 1e-9)

    def test_tuple_mistaken_as_dict_would_fail(self) -> None:
        """Regression: load_research returns (doc, origin), not a bare dict."""
        from cfb_coach import ai_research

        loaded = ai_research.load_research("madden27", offline=True)
        self.assertIsInstance(loaded, tuple)
        self.assertEqual(len(loaded), 2)
        doc, origin = loaded
        self.assertIsInstance(doc, dict)
        self.assertIn("findings", doc)

    def test_train_carries_knowledge_origin(self) -> None:
        art = experimental_model.train_experimental([], side="offense")
        self.assertTrue(art.knowledge_version)
        self.assertTrue(art.research_concept_boost or art.research_family_boost)


class IdentityCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())
        art = experimental_model.train_experimental([], side="offense")
        # Force a concept boost so ML can diverge from a run-heavy heuristic.
        art.research_concept_boost["mesh"] = 0.08
        art.research_concept_boost["vert"] = 0.06
        art.research_family_boost["pass"] = 0.05
        art.research_family_boost["run"] = -0.05
        dest = Path(self.tmp.name) / "exp.json"
        experimental_model.save_artifact(art, dest)
        experimental_live.set_artifact_path(self.db, dest)
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        self.book = _book()

    def tearDown(self) -> None:
        inference.set_mode(self.db, CoachingMode.HEURISTIC)
        self.db.close()
        self.tmp.cleanup()

    def test_make_call_path_does_not_log_anonymous_row(self) -> None:
        heur = MaddenCall("offense", "Gun Doubles", "HB Dive", "No adj", reads_for("HB Dive"), "h")
        sit = Situation(raw="3&12", side="offense", down=3, distance=12, yardline=50)
        call, dec = experimental_live.apply_experimental_offense(
            heuristic_call=heur,
            sit=sit,
            opponent_id="cpu_lions",
            db=self.db,
            book=self.book,
            commit=False,
        )
        self.assertIsNotNone(getattr(call, "_pending_ml_decision", None))
        self.assertIsNone(dec.snap_id)
        n = self.db.conn.execute("SELECT COUNT(*) AS n FROM ml_decisions").fetchone()["n"]
        self.assertEqual(int(n), 0)

    def test_commit_after_seal_links_identity(self) -> None:
        heur = MaddenCall("offense", "Gun Doubles", "HB Dive", "No adj", reads_for("HB Dive"), "h")
        sit = Situation(raw="3&12", side="offense", down=3, distance=12, yardline=50)
        call, _dec = experimental_live.apply_experimental_offense(
            heuristic_call=heur,
            sit=sit,
            opponent_id="cpu_lions",
            db=self.db,
            book=self.book,
            commit=False,
        )
        row_id = experimental_live.commit_experimental_decision(
            self.db,
            call,
            game_id="game-abc",
            snap_id="game-abc-0001",
            snap_seq=1,
            session_id="game-abc",
        )
        self.assertIsNotNone(row_id)
        row = self.db.conn.execute(
            "SELECT * FROM ml_decisions WHERE id=?", (row_id,)
        ).fetchone()
        self.assertEqual(row["snap_id"], "game-abc-0001")
        self.assertEqual(row["game_id"], "game-abc")
        self.assertEqual(row["mode"], "experimental")
        self.assertEqual(row["heuristic_play"], "HB Dive")
        self.assertEqual(row["final_play"], call.play)
        # Idempotent recommit
        row_id2 = experimental_live.commit_experimental_decision(
            self.db,
            call,
            game_id="game-abc",
            snap_id="game-abc-0001",
            snap_seq=1,
            session_id="game-abc",
        )
        n = self.db.conn.execute("SELECT COUNT(*) AS n FROM ml_decisions").fetchone()["n"]
        self.assertEqual(int(n), 1)
        self.assertEqual(row_id2, row_id)


class HtmlJoinIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())
        art = experimental_model.train_experimental([], side="offense")
        art.research_concept_boost["mesh"] = 0.08
        art.research_family_boost["pass"] = 0.05
        art.research_family_boost["run"] = -0.06
        dest = Path(self.tmp.name) / "exp.json"
        experimental_model.save_artifact(art, dest)
        experimental_live.set_artifact_path(self.db, dest)
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        book = _book()

        def _make(sit, **kwargs):
            heur = MaddenCall(
                "offense",
                "Gun Doubles",
                "HB Dive",
                "No adj",
                reads_for("HB Dive"),
                "heuristic run",
            )
            call, _ = experimental_live.apply_experimental_offense(
                heuristic_call=heur,
                sit=sit,
                opponent_id="cpu_lions",
                db=self.db,
                book=book,
                commit=False,
            )
            return call

        self.ctrl = LivePlayController(
            db=self.db,
            opponent_id="cpu_lions",
            make_call=_make,
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "ok",
            brand="Madden 27 Franchise",
            cpu_only=True,
            enable_execution_verify=True,
        )
        self.ctrl.start()

    def tearDown(self) -> None:
        inference.set_mode(self.db, CoachingMode.HEURISTIC)
        self.db.close()
        self.tmp.cleanup()

    def test_live_ml_call_execution_outcome_join(self) -> None:
        out = self.ctrl.call_only("3rd and 11 midfield", side="offense")
        self.assertTrue(out.get("ok"), out)
        st = out["state"]
        self.assertIsNotNone(st.get("ml_experimental"))
        # Decision committed with snap identity
        rows = list(self.db.conn.execute("SELECT * FROM ml_decisions WHERE mode='experimental'"))
        self.assertEqual(len(rows), 1)
        dec = rows[0]
        self.assertTrue(dec["snap_id"])
        self.assertEqual(dec["game_id"], self.ctrl.session_id)
        self.assertEqual(dec["session_id"], self.ctrl.session_id)
        self.assertEqual(dec["heuristic_play"], "HB Dive")
        self.assertEqual(dec["final_play"], self.ctrl.last_call.play)
        self.assertEqual(self.ctrl.last_call.read_or_user, reads_for(self.ctrl.last_call.play))

        # Close the sealed snap with verified execution + outcome via the live path.
        # HTML form uses used_recommended → close_from_form maps to identified+verified.
        nxt = self.ctrl.result_and_call(
            outcome="+8",
            sit_raw="1st and 10 midfield",
            side="offense",
            executed_status="used_recommended",
        )
        self.assertTrue(nxt.get("ok"), nxt)
        snap = self.db.conn.execute(
            "SELECT * FROM snaps WHERE ml_snap_id=?", (dec["snap_id"],)
        ).fetchone()
        self.assertIsNotNone(snap)
        outc = self.db.conn.execute(
            "SELECT * FROM ml_outcomes WHERE snap_id=?", (dec["snap_id"],)
        ).fetchone()
        self.assertIsNotNone(outc)
        self.assertEqual(outc["executed_play"], dec["final_play"])
        self.assertEqual(outc["executed_status"], ExecutedStatus.IDENTIFIED.value)
        report = experimental_live.postgame_experimental_compare(
            self.db, game_id=self.ctrl.session_id
        )
        self.assertGreaterEqual(report["n_experimental_calls"], 1)
        self.assertGreaterEqual(report["verified_executions_linked"], 1)
        self.assertEqual(report["anonymous_unlinked_rows_ignored"], 0)

    def test_undo_retry_idempotency(self) -> None:
        key = "sprint41-idem"
        a = self.ctrl.call_only("2nd and 7 own 40", side="offense", request_key=key)
        b = self.ctrl.call_only("2nd and 7 own 40", side="offense", request_key=key)
        self.assertEqual(a.get("call_text"), b.get("call_text"))
        n = self.db.conn.execute(
            "SELECT COUNT(*) AS n FROM ml_decisions WHERE mode='experimental'"
        ).fetchone()["n"]
        self.assertEqual(int(n), 1)


class SituationalEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.book = _book()
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_third_and_long_excludes_pure_dive_bias_via_pool_bonus(self) -> None:
        sit = Situation(
            raw="3&12", side="offense", down=3, distance=12, yardline=50, long_yardage=True
        )
        pool, bonus = experimental_live.situational_offense_candidates(
            sit, self.book, db=self.db, opponent_id="cpu_lions"
        )
        self.assertTrue(pool)
        # HB Dive stays in book but should not get positive long-yardage bonus.
        dive = ("Gun Doubles", "HB Dive")
        if dive in pool:
            self.assertLessEqual(bonus.get(dive, 0.0), 0.0)

    def test_short_yardage_and_goal_line_pools(self) -> None:
        short = Situation(
            raw="3&1", side="offense", down=3, distance=1, yardline=50, short_yardage=True
        )
        pool_s, bonus_s = experimental_live.situational_offense_candidates(
            short, self.book, db=self.db, opponent_id="cpu_lions"
        )
        self.assertTrue(pool_s)
        dive = ("Gun Doubles", "HB Dive")
        if dive in pool_s:
            self.assertGreaterEqual(bonus_s.get(dive, 0.0), 0.0)

        gl = Situation(
            raw="1&G",
            side="offense",
            down=1,
            distance=1,
            yardline=99,
            goal_line=True,
            red_zone=True,
        )
        pool_g, _ = experimental_live.situational_offense_candidates(
            gl, self.book, db=self.db, opponent_id="cpu_lions"
        )
        self.assertTrue(pool_g)

    def test_ml_only_ranks_eligible(self) -> None:
        art = experimental_model.train_experimental([], side="offense")
        dest = Path(self.tmp.name) / "e.json"
        experimental_model.save_artifact(art, dest)
        experimental_live.set_artifact_path(self.db, dest)
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        sit = Situation(
            raw="3&15", side="offense", down=3, distance=15, yardline=40, long_yardage=True
        )
        heur = MaddenCall("offense", "Gun Doubles", "Y Cross", "No adj", "Primary → Checkdown", "h")
        call, dec = experimental_live.apply_experimental_offense(
            heuristic_call=heur,
            sit=sit,
            opponent_id="cpu_lions",
            db=self.db,
            book=self.book,
            commit=False,
        )
        self.assertFalse(dec.fell_back)
        info = call.ml_experimental
        self.assertGreaterEqual(info["n_eligible_candidates"], 1)
        # Final play must be in situational pool
        pool, _ = experimental_live.situational_offense_candidates(
            sit, self.book, db=self.db, opponent_id="cpu_lions"
        )
        self.assertIn((call.formation, call.play), set(pool) | {("Gun Doubles", "Y Cross")})

    def test_hurry_up_two_minute_pool(self) -> None:
        sit = Situation(
            raw="2&10 1:42",
            side="offense",
            down=2,
            distance=10,
            yardline=55,
            two_minute=True,
        )
        pool, bonus = experimental_live.situational_offense_candidates(
            sit, self.book, db=self.db, opponent_id="cpu_lions"
        )
        self.assertTrue(pool)
        # Clock situations prefer pass concepts over pure dive.
        pass_names = {"Mesh", "Flood", "Y Cross", "Four Verticals", "Stick", "Smash"}
        self.assertTrue(any(p in pass_names for _, p in pool))

    def test_obvious_passing_situation_pool(self) -> None:
        sit = Situation(
            raw="3&14", side="offense", down=3, distance=14, yardline=45, long_yardage=True
        )
        pool, bonus = experimental_live.situational_offense_candidates(
            sit, self.book, db=self.db, opponent_id="cpu_lions"
        )
        self.assertTrue(pool)
        for form, play in pool:
            if play == "HB Dive":
                self.assertLessEqual(bonus.get((form, play), 0.0), 0.0)


class CpuReadinessAcceptanceTests(unittest.TestCase):
    """Controlled real-call-path demo: ML diverges, seals, executes, joins outcome."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self.tmp.name) / "t.db", seed=_seed())
        rows = []
        for play, ok, n in (("Mesh", True, 20), ("HB Dive", False, 20), ("Flood", True, 8)):
            for _ in range(n):
                rows.append(
                    {
                        "side": "offense",
                        "eligibility": "verified_execution",
                        "supervised_eligible": True,
                        "label_available": True,
                        "success": "true" if ok else "false",
                        "action_play": play,
                        "executed_play": play,
                        "down": 3,
                        "distance": 11,
                        "yardline": 50,
                        "opponent_type": "cpu",
                        "opponent_id": "cpu_lions",
                        "coverage_seen": "Cover 3",
                    }
                )
        art = experimental_model.train_experimental(rows, side="offense")
        art.research_concept_boost["mesh"] = 0.08
        art.research_family_boost["pass"] = 0.04
        art.research_family_boost["run"] = -0.05
        dest = Path(self.tmp.name) / "exp.json"
        experimental_model.save_artifact(art, dest)
        experimental_live.set_artifact_path(self.db, dest)
        self.art_path = dest
        self.book = _book()

        def _make(sit, **kwargs):
            heur = MaddenCall(
                "offense",
                "Gun Doubles",
                "HB Dive",
                "No adj",
                reads_for("HB Dive"),
                "heuristic run",
            )
            call, _ = experimental_live.apply_experimental_offense(
                heuristic_call=heur,
                sit=sit,
                opponent_id="cpu_lions",
                db=self.db,
                book=self.book,
                commit=False,
            )
            return call

        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)
        self.ctrl = LivePlayController(
            db=self.db,
            opponent_id="cpu_lions",
            make_call=_make,
            parse_situation=parse_madden_situation,
            learn_summary=lambda: "ok",
            brand="Madden 27 Franchise",
            cpu_only=True,
            enable_execution_verify=True,
        )
        self.ctrl.start()

    def tearDown(self) -> None:
        inference.set_mode(self.db, CoachingMode.HEURISTIC)
        self.db.close()
        self.tmp.cleanup()

    def test_cpu_readiness_chain(self) -> None:
        # 7. Default heuristic mode still resolves when not experimental.
        inference.set_mode(self.db, CoachingMode.HEURISTIC)
        self.assertIs(inference.resolve_mode(self.db), CoachingMode.HEURISTIC)
        inference.set_mode(self.db, CoachingMode.EXPERIMENTAL)

        out = self.ctrl.call_only("3rd and 11 midfield", side="offense")
        self.assertTrue(out.get("ok"), out)
        st = out["state"]
        ml_ui = st.get("ml_experimental")
        self.assertIsNotNone(ml_ui)
        info = getattr(self.ctrl.last_call, "ml_experimental", None)
        self.assertIsInstance(info, dict)
        # 1. Experimental chooses a different eligible play.
        self.assertEqual(info["heuristic_play"], "HB Dive")
        self.assertNotEqual(info["final_play"], info["heuristic_play"])
        self.assertEqual(self.ctrl.last_call.play, info["final_play"])
        # 2. Final read/adjustment matches selected play.
        self.assertEqual(self.ctrl.last_call.read_or_user, reads_for(info["final_play"]))
        # 3. Heuristic alternative remains visible in HTML state.
        self.assertIn("heuristic", ml_ui)
        self.assertIn("HB Dive", ml_ui["heuristic"])
        self.assertNotEqual(ml_ui["heuristic"], ml_ui["final"])
        # 4. Snap identity on the decision.
        rows = list(self.db.conn.execute("SELECT * FROM ml_decisions WHERE mode='experimental'"))
        self.assertEqual(len(rows), 1)
        dec = rows[0]
        self.assertTrue(dec["snap_id"])
        self.assertEqual(dec["game_id"], self.ctrl.session_id)
        self.assertEqual(dec["final_play"], info["final_play"])
        self.assertEqual(dec["heuristic_play"], "HB Dive")
        # 8. Latency budget preserved on sealed decision payload.
        payload = json.loads(dec["decision_json"] or "{}")
        self.assertEqual(int(payload.get("inference_budget_ms") or 0), 150)
        self.assertLess(float(dec["latency_ms_model"] or 0.0), 150.0)
        # 5. Executed action + outcome link to that decision.
        nxt = self.ctrl.result_and_call(
            outcome="+12",
            sit_raw="1st and 10 midfield",
            side="offense",
            executed_status="used_recommended",
        )
        self.assertTrue(nxt.get("ok"), nxt)
        outc = self.db.conn.execute(
            "SELECT * FROM ml_outcomes WHERE snap_id=?", (dec["snap_id"],)
        ).fetchone()
        self.assertIsNotNone(outc)
        self.assertEqual(outc["executed_play"], dec["final_play"])
        self.assertEqual(outc["executed_status"], ExecutedStatus.IDENTIFIED.value)
        # 6. Low-confidence / invalid candidates fall back (timeout path).
        heur = MaddenCall(
            "offense", "Gun Doubles", "HB Dive", "No adj", reads_for("HB Dive"), "h"
        )

        def _slow(*_a, **_k):
            import time

            time.sleep(0.05)
            return []

        with mock.patch(
            "cfb_coach.madden.model.experimental_live.exp_mod.rank_candidates",
            side_effect=_slow,
        ):
            call, fdec = experimental_live.apply_experimental_offense(
                heuristic_call=heur,
                sit=Situation(raw="3&11", side="offense", down=3, distance=11),
                opponent_id="cpu_lions",
                db=self.db,
                book=self.book,
                budget_ms=10,
                commit=False,
            )
        self.assertTrue(fdec.fell_back)
        self.assertEqual(call.play, "HB Dive")
        self.assertEqual(fdec.inference_budget_ms, 150)

    def test_restart_attribution_survives(self) -> None:
        out = self.ctrl.call_only("3rd and 11 midfield", side="offense")
        self.assertTrue(out.get("ok"), out)
        dec = self.db.conn.execute(
            "SELECT * FROM ml_decisions WHERE mode='experimental'"
        ).fetchone()
        snap_id = dec["snap_id"]
        path = self.db.path
        self.db.close()
        # Reopen DB (process-restart stand-in).
        reopened = CoachDB(path)
        try:
            row = reopened.conn.execute(
                "SELECT * FROM ml_decisions WHERE snap_id=?", (snap_id,)
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["final_play"], dec["final_play"])
            self.assertEqual(row["game_id"], dec["game_id"])
        finally:
            reopened.close()
        # Restore for tearDown.
        self.db = CoachDB(path)


class ModelQualityTests(unittest.TestCase):
    def test_concepts_distinguish_pass_plays(self) -> None:
        self.assertEqual(experimental_model._play_concept("Mesh Spot"), "mesh")
        self.assertEqual(experimental_model._play_concept("Four Verticals"), "vert")
        self.assertEqual(experimental_model._play_concept("Flood"), "flood")
        self.assertNotEqual(
            experimental_model._play_concept("Mesh"),
            experimental_model._play_concept("Four Verticals"),
        )

    def test_last_snap_coverage_is_soft_not_confirmation(self) -> None:
        art = experimental_model.train_experimental([], side="offense")
        art.concept_vs_cov["mesh|cover2"] = [8.0, 10.0]
        art.play_concept["mesh"] = [5.0, 10.0]
        live = experimental_model.predict_success(
            art,
            formation="Gun Bunch",
            play="Mesh",
            coverage_hint="Cover 2",
            coverage_source="live",
        )
        last = experimental_model.predict_success(
            art,
            formation="Gun Bunch",
            play="Mesh",
            coverage_hint="Cover 2",
            coverage_source="last",
        )
        none = experimental_model.predict_success(
            art,
            formation="Gun Bunch",
            play="Mesh",
            coverage_hint="Cover 2",
            coverage_source="none",
        )
        self.assertEqual(live["coverage_weight"], 1.0)
        self.assertAlmostEqual(last["coverage_weight"], experimental_model.LAST_SNAP_COV_WEIGHT)
        self.assertEqual(none["coverage_bucket"], "unknown")
        self.assertEqual(none["coverage_weight"], 0.0)

    def test_not_alphabetical_generic_pass_family(self) -> None:
        art = experimental_model.train_experimental([], side="offense")
        # Research/historical boost for mesh should outrank alphabetically-first Smash
        # when both are generic passes with no other signal.
        art.research_concept_boost = {"mesh": 0.08, "smash": 0.0}
        ranked = experimental_model.rank_candidates(
            art,
            [("Gun Bunch", "Smash"), ("Gun Bunch", "Mesh")],
            down=1,
            distance=10,
            yardline=50,
            near_tie_margin=0.0,
        )
        self.assertEqual(ranked[0]["play"], "Mesh")
        self.assertEqual(ranked[0]["play_concept"], "mesh")

    def test_supervised_history_moves_rates(self) -> None:
        rows = [
            {
                "side": "offense",
                "eligibility": "verified_execution",
                "supervised_eligible": True,
                "label_available": True,
                "success": "true",
                "action_play": "Mesh",
                "executed_play": "Mesh",
                "down": 1,
                "distance": 10,
                "yardline": 50,
                "opponent_type": "cpu",
                "opponent_id": "cpu_lions",
                "coverage_seen": "Cover 3",
            }
            for _ in range(12)
        ] + [
            {
                "side": "offense",
                "eligibility": "verified_execution",
                "supervised_eligible": True,
                "label_available": True,
                "success": "false",
                "action_play": "HB Dive",
                "executed_play": "HB Dive",
                "down": 1,
                "distance": 10,
                "yardline": 50,
                "opponent_type": "cpu",
                "opponent_id": "cpu_lions",
                "coverage_seen": "Cover 3",
            }
            for _ in range(12)
        ]
        art = experimental_model.train_experimental(rows, side="offense")
        self.assertEqual(art.evidence_quality, "empirical_light")
        self.assertGreaterEqual(art.n_supervised, 10)
        mesh = experimental_model.predict_success(
            art, formation="Gun Bunch", play="Mesh", down=1, distance=10, yardline=50
        )
        dive = experimental_model.predict_success(
            art, formation="Gun Doubles", play="HB Dive", down=1, distance=10, yardline=50
        )
        self.assertGreater(mesh["probability"], dive["probability"])


class ResearchRefreshExtractTests(unittest.TestCase):
    def test_extract_and_diff(self) -> None:
        from cfb_coach.madden import research_refresh as rr

        fake = {
            "ok": True,
            "url": "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/title-update-1-008",
            "title_guess": "Madden NFL 27 Title Update 1.008",
            "snippet": "Title Update 1.008 improves pass lead and Cover 4 Quarters.",
            "body_text": (
                "Madden NFL 27 Title Update 1.008. "
                "Cover 4 Quarters tuning. Mesh Spot mentioned. Pass lead extended."
            ),
            "retrieved_at": "2026-10-08T00:00:00+00:00",
        }
        extracted = rr._extract_findings(fake, legal_plays={"Mesh Spot", "Clear Deep"})
        self.assertTrue(extracted)
        self.assertTrue(any(f.get("game_version") == "1.008" for f in extracted))
        active = {"patch": {"version": "1.007"}, "findings": [{"claim": "old claim"}]}
        diffs = rr._diff_findings(active, extracted)
        self.assertTrue(any(d.get("type") == "possible_patch_bump" for d in diffs))

    def test_gha_outputs_script(self) -> None:
        import importlib.util

        path = Path(__file__).resolve().parents[2] / "scripts" / "research_refresh_gha_outputs.py"
        spec = importlib.util.spec_from_file_location("research_refresh_gha_outputs", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            p.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "candidate_path": "/tmp/c.json",
                        "change_report_path": "/tmp/r.md",
                        "extracted_findings": [{}],
                        "contradictions_or_changes": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(mod.main([str(p)]), 0)


class FallbackBudgetTests(unittest.TestCase):
    def test_timeout_and_default_heuristic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db", seed=_seed())
            try:
                self.assertIs(inference.resolve_mode(db), CoachingMode.HEURISTIC)
                art = experimental_model.train_experimental([], side="offense")
                dest = Path(td) / "e.json"
                experimental_model.save_artifact(art, dest)
                experimental_live.set_artifact_path(db, dest)
                inference.set_mode(db, CoachingMode.EXPERIMENTAL)
                heur = MaddenCall("offense", "Gun Bunch", "Mesh", "No adj", "a → b", "h")

                def _slow(*_a, **_k):
                    import time

                    time.sleep(0.05)
                    return [
                        {
                            "formation": "Gun Bunch",
                            "play": "Mesh",
                            "probability": 0.5,
                            "play_family": "pass",
                            "play_concept": "mesh",
                            "evidence_quality": "prior_driven",
                            "uncertainty": 0.9,
                            "components": {},
                            "rank": 1,
                        }
                    ]

                with mock.patch(
                    "cfb_coach.madden.model.experimental_live.exp_mod.rank_candidates",
                    side_effect=_slow,
                ):
                    call, dec = experimental_live.apply_experimental_offense(
                        heuristic_call=heur,
                        sit=Situation(raw="1&10", side="offense", down=1, distance=10),
                        opponent_id="cpu_lions",
                        db=db,
                        book=_book(),
                        budget_ms=10,
                        commit=False,
                    )
                self.assertTrue(dec.fell_back)
                self.assertIs(dec.fallback_reason, MLStatus.TIMEOUT)
                self.assertEqual(call.play, "Mesh")
                self.assertEqual(dec.inference_budget_ms, 150)
                inference.set_mode(db, CoachingMode.HEURISTIC)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
