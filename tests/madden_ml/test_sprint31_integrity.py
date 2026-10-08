"""Sprint 3.1: action attribution, undo/idempotency, dedupe, safe DB, eval edges."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController, make_handler, pick_port
from cfb_coach.madden.model import dataset, evaluate, features, train
from cfb_coach.madden.model.schema import (
    CoachingMode,
    ExecutedStatus,
    GateResult,
    MLStatus,
    Verification,
)
from cfb_coach.madden.model.train import _row_vector
from cfb_coach.situation import parse_situation


def _seed(oid: str = "lions_franchise") -> dict:
    return {
        "opponents": {
            oid: {
                "display_name": "Lions",
                "team_now": "DET",
                "skill": "human",
                "confidence": "high",
                "profile_json": "{}",
            },
            "cpu_a": {
                "display_name": "CPU",
                "team_now": "X",
                "skill": "cpu",
                "confidence": "low",
                "profile_json": "{}",
            },
        }
    }


class _FakeCall:
    def __init__(self, side="offense", formation="Gun Bunch", play="Mesh"):
        self.side = side
        self.formation = formation
        self.play = play
        self.macro = None
        self.adj_or_macro = "base"
        self.rationale = "test"

    def format(self) -> str:
        return f"{self.formation} — {self.play}"


class ActionAttributionTests(unittest.TestCase):
    def test_verified_execution_trains_on_executed_not_recommendation(self) -> None:
        row = dataset._normalize_row(
            {
                "snap_id": "g-0001",
                "game_id": "g",
                "side": "offense",
                "down": 1,
                "distance": 10,
                "yardline": 25,
                "recommended_formation": "Gun Bunch",
                "recommended_play": "Mesh",
                "executed_status": "identified",
                "executed_verification": "verified",
                "executed_formation": "Gun Trips",
                "executed_play": "Inside Zone",
                "result": "+8",
            },
            provenance="test",
        )
        self.assertEqual(row["eligibility"], dataset.ELIGIBILITY_VERIFIED_EXECUTION)
        self.assertEqual(row["action_play"], "Inside Zone")
        self.assertEqual(row["action_formation"], "Gun Trips")
        self.assertEqual(row["recommended_play"], "Mesh")
        vec = _row_vector(row)
        names = features.pre_snap_feature_names()
        # Inside Zone → run family; Mesh would be pass-ish. Assert run family on.
        run_idx = names.index("cand_play_family_run")
        pass_idx = names.index("cand_play_family_pass")
        self.assertEqual(vec.items[run_idx].value, 1.0)
        self.assertEqual(vec.items[pass_idx].value, 0.0)
        # Formation family: Gun Trips is still gun.
        gun_idx = names.index("cand_form_gun")
        self.assertEqual(vec.items[gun_idx].value, 1.0)

    def test_unknown_execution_excluded_from_supervised_features(self) -> None:
        row = dataset._normalize_row(
            {
                "snap_id": "g-0002",
                "game_id": "g",
                "side": "offense",
                "down": 1,
                "distance": 10,
                "recommended_formation": "Gun Bunch",
                "recommended_play": "Mesh",
                "result": "+8",
            },
            provenance="test",
        )
        self.assertFalse(row["supervised_eligible"])
        self.assertIsNone(row["action_play"])
        with self.assertRaises(ValueError):
            _row_vector(row)

    def test_outside_book_executed_still_attributable(self) -> None:
        row = dataset._normalize_row(
            {
                "snap_id": "g-0003",
                "game_id": "g",
                "side": "offense",
                "down": 2,
                "distance": 7,
                "recommended_formation": "Gun Bunch",
                "recommended_play": "Mesh",
                "executed_status": "identified",
                "executed_verification": "verified",
                "executed_formation": "External Form",
                "executed_play": "Manual Observation Play",
                "result": "incomplete",
            },
            provenance="test",
        )
        self.assertTrue(row["supervised_eligible"])
        self.assertEqual(row["action_play"], "Manual Observation Play")
        vec = _row_vector(row)
        from cfb_coach.madden.model.schema import FEATURE_SCHEMA_VERSION

        self.assertEqual(vec.feature_schema_version, FEATURE_SCHEMA_VERSION)

    def test_trusted_vod_uses_observed_action(self) -> None:
        row = dataset._normalize_row(
            {
                "snap_id": "vod-1",
                "game_id": "vod_game",
                "side": "offense",
                "down": 1,
                "distance": 10,
                "formation": "I Form",
                "play": "Power O",
                "recommended_formation": "I Form",
                "recommended_play": "Power O",
                "result": "+5",
                "trusted_vod": True,
            },
            provenance="jsonl:trusted_vod.jsonl",
        )
        self.assertEqual(row["eligibility"], dataset.ELIGIBILITY_TRUSTED_VOD)
        self.assertEqual(row["action_play"], "Power O")
        vec = _row_vector(row)
        names = features.pre_snap_feature_names()
        self.assertEqual(vec.items[names.index("cand_play_family_run")].value, 1.0)


class CanonicalDedupeTests(unittest.TestCase):
    def test_one_row_per_logical_snap_across_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "m.db", seed=_seed())
            try:
                from cfb_coach.madden.model.schema import CandidatePlay, CoachingDecision

                did = db.log_ml_decision(
                    CoachingDecision(
                        mode=CoachingMode.SHADOW,
                        shadow_status=MLStatus.MODEL_MISSING,
                        game_id="sessA",
                        session_id="sessA",
                        snap_id="sessA-0001",
                        snap_seq=1,
                        heuristic_pick=CandidatePlay(formation="Gun Bunch", play="Mesh"),
                        final_pick=CandidatePlay(formation="Gun Bunch", play="Mesh"),
                    )
                )
                db.log_snap(
                    opponent_id="lions_franchise",
                    side="offense",
                    situation_raw="1&10",
                    our_call="Gun Bunch — Mesh",
                    formation="Gun Bunch",
                    play="Mesh",
                    down=1,
                    distance=10,
                    result="+4",
                    session_id="sessA",
                    ml_snap_id="sessA-0001",
                    snap_seq=1,
                    executed_status="identified",
                    executed_formation="Gun Bunch",
                    executed_play="Mesh",
                    executed_verification="verified",
                    ml_decision_id=did,
                )
                db.log_ml_outcome(
                    snap_id="sessA-0001",
                    game_id="sessA",
                    decision_id=did,
                    executed_status="identified",
                    executed_formation="Gun Bunch",
                    executed_play="Mesh",
                    executed_verification="verified",
                    outcome={"result": "+4", "yards": 4},
                )
                # Duplicate-looking play_records row for same session without ml id — skipped.
                db.conn.execute(
                    "INSERT INTO play_records "
                    "(session_id, play_id, ts, opponent_id, side, formation, result_type, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "sessA",
                        "legacy-1",
                        "2026-01-01T00:00:00Z",
                        "lions_franchise",
                        "offense",
                        "Gun Bunch",
                        "+4",
                        json.dumps({"play": "Mesh", "situation_raw": "1&10"}),
                    ),
                )
                db.conn.commit()
                rows = dataset.build_rows(db=db)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["snap_id"], "sessA-0001")
                # Correction: latest outcome replaces result.
                db.log_ml_outcome(
                    snap_id="sessA-0001",
                    game_id="sessA",
                    decision_id=did,
                    executed_status="identified",
                    executed_formation="Gun Trips",
                    executed_play="Inside Zone",
                    executed_verification="verified",
                    outcome={"result": "+8", "yards": 8},
                    replace=True,
                )
                rows2 = dataset.build_rows(db=db)
                self.assertEqual(len(rows2), 1)
                self.assertEqual(rows2[0]["result"], "+8")
                self.assertEqual(rows2[0]["action_play"], "Inside Zone")
            finally:
                db.close()


class SafeDbTests(unittest.TestCase):
    def test_backup_api_and_read_only_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "madden27.db"
            db = CoachDB(src, seed=_seed())
            db.conn.execute("PRAGMA journal_mode=WAL")
            db.log_snap(
                opponent_id="lions_franchise",
                side="offense",
                situation_raw="1&10",
                our_call="x",
                formation="Gun Bunch",
                play="Mesh",
                result="+1",
            )
            db.conn.commit()
            dest = Path(tmp) / "backup.db"
            db.backup_to(dest)
            self.assertTrue(dest.is_file())
            # Concurrent reader while source still open.
            ro = CoachDB.open_read_only(src)
            try:
                n = ro.conn.execute("SELECT count(*) FROM snaps").fetchone()[0]
                self.assertEqual(int(n), 1)
                # Read-only must not allow writes.
                with self.assertRaises(Exception):
                    ro.conn.execute(
                        "INSERT INTO snaps (ts, opponent_id, side) VALUES ('t','x','offense')"
                    )
                    ro.conn.commit()
            finally:
                ro.close()
            db.close()
            # Interrupted backup: destination from successful backup is intact.
            ro2 = CoachDB.open_read_only(dest)
            try:
                self.assertEqual(ro2.conn.execute("SELECT count(*) FROM snaps").fetchone()[0], 1)
            finally:
                ro2.close()


class EvalEdgeTests(unittest.TestCase):
    def test_zero_baseline_rate_and_mixed_rows(self) -> None:
        rows = []
        for i in range(100):
            rows.append(
                dataset._normalize_row(
                    {
                        "snap_id": f"g{i // 20}-{i:04d}",
                        "game_id": f"game_{i // 20}",
                        "opponent_id": "human_a",
                        "opponent_type": "human",
                        "side": "offense",
                        "down": 1,
                        "distance": 10,
                        "recommended_formation": "Gun Bunch",
                        "recommended_play": "Mesh",
                        "executed_status": "identified",
                        "executed_verification": "verified",
                        "executed_formation": "Gun Bunch",
                        "executed_play": "Mesh",
                        # All failures → baseline_rate 0.0 on train fit.
                        "result": "incomplete",
                    },
                    provenance="test",
                )
            )
        # Mix in recommendation-only labeled noise.
        for i in range(10):
            rows.append(
                dataset._normalize_row(
                    {
                        "snap_id": f"rec-{i}",
                        "game_id": f"rec_game_{i}",
                        "side": "offense",
                        "down": 1,
                        "distance": 10,
                        "recommended_formation": "Gun Bunch",
                        "recommended_play": "Mesh",
                        "result": "+9",
                    },
                    provenance="test",
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            entry = train.train(rows, seed=9, out_dir=tmp)
            art = train.load_artifact(entry.artifact_path)
            self.assertIn("baseline_rate", art)
            self.assertAlmostEqual(float(art["baseline_rate"]), 0.0, places=5)
            updated = evaluate.evaluate(entry, rows, seed=9)
            self.assertFalse(updated.gate_passed)
            report = json.loads(Path(updated.gate_report_path).read_text(encoding="utf-8"))
            self.assertEqual(report.get("hybrid_activation"), False)
            self.assertGreaterEqual(report.get("mixed_recommendation_only_skipped") or 0, 1)

    def test_corrupted_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "model_artifact.json"
            bad.write_text("{not-json", encoding="utf-8")
            from cfb_coach.madden.model.schema import ModelRegistryEntry, Possession

            entry = ModelRegistryEntry(
                model_version="bad",
                artifact_path=str(bad),
                side=Possession.OFFENSE,
            )
            updated = evaluate.evaluate(entry, [], seed=1)
            self.assertEqual(updated.gate, GateResult.FAILED)


class HttpIdempotencyAndUndoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self._td.name) / "live.db", seed=_seed())
        from cfb_coach.madden.model import inference

        inference.set_mode(self.db, CoachingMode.SHADOW)
        n = {"i": 0}

        def make(sit, **kwargs):
            n["i"] += 1
            side = getattr(sit, "side", "offense") or "offense"
            if str(side).startswith("d"):
                return _FakeCall(side="defense", formation="Nickel", play="Cover 3")
            return _FakeCall(play=["Mesh", "Flood", "Inside Zone"][n["i"] % 3])

        def shadow(sit, call, *, game_id, snap_id, snap_seq, session_id, is_new):
            return inference.evaluate_live_shadow(
                db=self.db,
                situation=sit,
                call=call,
                opponent_id="lions_franchise",
                game_id=game_id,
                snap_id=snap_id,
                snap_seq=snap_seq,
                session_id=session_id,
                formations={
                    "Gun Bunch": ["Mesh", "Flood", "Inside Zone"],
                    "Nickel": ["Cover 3"],
                },
                run=is_new,
            )[1]

        self.ctrl = LivePlayController(
            db=self.db,
            opponent_id="lions_franchise",
            make_call=make,
            parse_situation=parse_situation,
            learn_summary=lambda: "ok",
            brand="Madden 27 Franchise",
            play_cmd="x",
            cpu_only=False,
            shadow_evaluate=shadow,
            enable_execution_verify=True,
        )
        self.ctrl.start()
        handler = make_handler(self.ctrl)
        port = pick_port()
        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.port = port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.db.close()
        self._td.cleanup()

    def _json(self, method: str, path: str, body: dict | None = None) -> dict:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        raw = json.dumps(body or {}).encode()
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if method == "GET":
            conn.request("GET", path)
        else:
            conn.request(method, path, body=raw, headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()
        self.assertEqual(resp.status, 200, msg=data)
        return data

    def test_duplicate_result_call_does_not_double_advance(self) -> None:
        self._json("POST", "/api/call", {"sit": "1&10 my 25", "idempotency_key": "call-1"})
        body = {
            "outcome": "+7",
            "sit": "2&3 my 32",
            "executed_status": "used_recommended",
            "idempotency_key": "res-1",
        }
        a = self._json("POST", "/api/result_call", body)
        b = self._json("POST", "/api/result_call", body)
        self.assertTrue(a["ok"])
        self.assertTrue(b.get("idempotent_replay"))
        snaps = self.db.conn.execute("SELECT count(*) FROM snaps").fetchone()[0]
        self.assertEqual(int(snaps), 1)
        outcomes = self.db.conn.execute("SELECT count(*) FROM ml_outcomes").fetchone()[0]
        self.assertEqual(int(outcomes), 1)
        # Log length should not double from the replay.
        self.assertEqual(len(b["state"]["log"]), len(a["state"]["log"]))

    def test_undo_voids_outcome_and_allows_relog(self) -> None:
        self._json("POST", "/api/call", {"sit": "1&10 my 25"})
        r = self._json(
            "POST",
            "/api/result_call",
            {
                "outcome": "sack",
                "sit": "2&17 my 18",
                "executed_status": "used_different",
                "executed_formation": "Gun Trips",
                "executed_play": "Inside Zone",
            },
        )
        self.assertEqual(r["logged"]["executed_play"], "Inside Zone")
        ml_id = r["logged"]["ml_snap_id"]
        self.assertEqual(
            self.db.conn.execute(
                "SELECT count(*) FROM ml_outcomes WHERE snap_id=?", (ml_id,)
            ).fetchone()[0],
            1,
        )
        undo = self._json("POST", "/api/undo", {"idempotency_key": "undo-1"})
        self.assertTrue(undo["ok"])
        self.assertEqual(
            self.db.conn.execute(
                "SELECT count(*) FROM ml_outcomes WHERE snap_id=?", (ml_id,)
            ).fetchone()[0],
            0,
        )
        audit = self.db.conn.execute(
            "SELECT count(*) FROM ml_outcome_audit WHERE snap_id=?", (ml_id,)
        ).fetchone()[0]
        self.assertGreaterEqual(int(audit), 1)
        # Relog against restored pending identity.
        self._json(
            "POST",
            "/api/result_call",
            {
                "outcome": "+3",
                "sit": "2&7 my 28",
                "executed_status": "used_recommended",
                "idempotency_key": "res-after-undo",
            },
        )
        self.assertEqual(
            self.db.conn.execute("SELECT count(*) FROM snaps").fetchone()[0],
            1,
        )


class TwoGameReadinessAcceptance(unittest.TestCase):
    """Simulates two Franchise games through the HTML API + backup/inspect/train."""

    def test_two_games_local_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "madden27.db"
            db = CoachDB(db_path, seed=_seed())
            from cfb_coach.madden.model import inference

            inference.set_mode(db, CoachingMode.SHADOW)

            def run_game(ctrl: LivePlayController) -> None:
                ctrl.call_only("1&10 my 25", request_key=f"{ctrl.session_id}-c1")
                ctrl.result_and_call(
                    outcome="incomplete",
                    sit_raw="2&10 my 25",
                    executed_status="unknown",
                    request_key=f"{ctrl.session_id}-r1",
                )
                ctrl.result_and_call(
                    outcome="+8",
                    sit_raw="1&10 my 33",
                    side="offense",
                    executed_status="used_recommended",
                    request_key=f"{ctrl.session_id}-r2",
                )
                ctrl.result_and_call(
                    outcome="sack",
                    sit_raw="1&10 opp 40",
                    side="defense",
                    executed_status="used_different",
                    executed_formation="Gun Trips",
                    executed_play="Inside Zone",
                    request_key=f"{ctrl.session_id}-r3",
                )
                # Duplicate submission
                ctrl.result_and_call(
                    outcome="sack",
                    sit_raw="1&10 opp 40",
                    side="defense",
                    executed_status="used_different",
                    executed_formation="Gun Trips",
                    executed_play="Inside Zone",
                    request_key=f"{ctrl.session_id}-r3",
                )
                # Correction via undo + relog
                ctrl.undo_last(request_key=f"{ctrl.session_id}-u1")
                ctrl.result_and_call(
                    outcome="+2",
                    sit_raw="2&8 opp 38",
                    side="defense",
                    executed_status="used_different",
                    executed_formation="Gun Trips",
                    executed_play="Inside Zone",
                    request_key=f"{ctrl.session_id}-r3b",
                )
                # Model timeout simulation: shadow hook raises
                ctrl.end_game(result_wl="win", score="21-17", request_key=f"{ctrl.session_id}-end")

            def make(sit, **k):
                side = getattr(sit, "side", "offense") or "offense"
                if str(side).startswith("d"):
                    return _FakeCall(side="defense", formation="Nickel", play="Cover 3")
                return _FakeCall()

            def shadow_ok(sit, call, *, game_id, snap_id, snap_seq, session_id, is_new):
                return inference.evaluate_live_shadow(
                    db=db,
                    situation=sit,
                    call=call,
                    opponent_id="lions_franchise",
                    game_id=game_id,
                    snap_id=snap_id,
                    snap_seq=snap_seq,
                    session_id=session_id,
                    formations={
                        "Gun Bunch": ["Mesh", "Flood"],
                        "Gun Trips": ["Inside Zone"],
                        "Nickel": ["Cover 3"],
                    },
                    run=is_new,
                )[1]

            ctrl1 = LivePlayController(
                db=db,
                opponent_id="lions_franchise",
                make_call=make,
                parse_situation=parse_situation,
                learn_summary=lambda: "g1",
                brand="Madden 27 Franchise",
                play_cmd="x",
                cpu_only=False,
                shadow_evaluate=shadow_ok,
                enable_execution_verify=True,
            )
            ctrl1.start()
            run_game(ctrl1)
            sid1 = ctrl1.session_id

            # Session restart (process restart simulation): new controller, same DB.
            ctrl2 = LivePlayController(
                db=db,
                opponent_id="lions_franchise",
                make_call=make,
                parse_situation=parse_situation,
                learn_summary=lambda: "g2",
                brand="Madden 27 Franchise",
                play_cmd="x",
                cpu_only=False,
                shadow_evaluate=shadow_ok,
                enable_execution_verify=True,
            )
            ctrl2.start()
            self.assertNotEqual(ctrl2.session_id, sid1)
            # Intentional inference failure mid-game.
            def shadow_fail(*a, **k):
                raise TimeoutError("model timeout")

            ctrl2.shadow_evaluate = shadow_fail
            ctrl2.call_only("1&10 my 20")
            ctrl2.shadow_evaluate = shadow_ok
            ctrl2.result_and_call(
                outcome="+6",
                sit_raw="2&4 my 26",
                executed_status="used_recommended",
                request_key=f"{ctrl2.session_id}-r1",
            )
            ctrl2.end_game(result_wl="loss", score="14-21")

            # Backup + read-only inspect
            backup = Path(tmp) / "madden27.backup.db"
            db.backup_to(backup)
            ro = CoachDB.open_read_only(db_path)
            try:
                rows = dataset.build_rows(db=ro)
                report = dataset.quality_report(rows)
                self.assertGreaterEqual(report["unique_games"], 2)
                self.assertGreaterEqual(report["verified_executions"], 1)
                supervised = dataset.supervised_rows(rows)
                for row in supervised:
                    self.assertIsNotNone(row["action_play"])
                    self.assertNotEqual(row.get("eligibility"), dataset.ELIGIBILITY_RECOMMENDATION_ONLY)
                    # Attribution: if recommended != executed, action is executed.
                    if row.get("executed_play") and row.get("recommended_play"):
                        if row["executed_play"] != row["recommended_play"]:
                            self.assertEqual(row["action_play"], row["executed_play"])
                export = Path(tmp) / "supervised.jsonl"
                dataset.export_jsonl(supervised, str(export))
                entry = train.train(rows, seed=13, out_dir=tmp)
                updated = evaluate.evaluate(entry, rows, seed=13)
                self.assertFalse(updated.gate_passed)
            finally:
                ro.close()
            db.close()


if __name__ == "__main__":
    unittest.main()
