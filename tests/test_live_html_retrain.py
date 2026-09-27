"""HTML live play + outcome parser + smarter retrain (play vs coverage)."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

from cfb_coach.cli import build_parser
from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController, make_handler, pick_port
from cfb_coach.outcome import outcome_success, parse_outcome
from cfb_coach.retrain import grade_play_vs_look, retrain_from_snaps
from cfb_coach.situation import parse_situation


class TestOutcomeParser(unittest.TestCase):
    def test_gain_loss_plus_minus(self) -> None:
        g = parse_outcome("gain 13")
        self.assertEqual(g.kind, "gain")
        self.assertEqual(g.yards, 13)
        self.assertTrue(g.success)
        self.assertEqual(g.to_result_text(), "+13")

        p = parse_outcome("+7")
        self.assertEqual(p.kind, "gain")
        self.assertEqual(p.yards, 7)

        loss = parse_outcome("loss 2")
        self.assertEqual(loss.kind, "loss")
        self.assertEqual(loss.yards, -2)
        self.assertFalse(loss.success)
        self.assertEqual(loss.to_result_text(), "-2")

        m = parse_outcome("-3")
        self.assertEqual(m.kind, "loss")
        self.assertEqual(m.yards, -3)

    def test_incomplete_sack_td_int(self) -> None:
        self.assertEqual(parse_outcome("incomplete").kind, "incomplete")
        self.assertFalse(parse_outcome("incomplete").success)
        self.assertEqual(parse_outcome("sack").kind, "sack")
        self.assertFalse(parse_outcome("sack").success)
        td = parse_outcome("TD")
        self.assertEqual(td.kind, "td")
        self.assertTrue(td.success)
        self.assertEqual(parse_outcome("INT").kind, "int")
        self.assertFalse(parse_outcome("INT").success)

    def test_stop_convert_side_aware(self) -> None:
        stop = parse_outcome("stop")
        self.assertEqual(stop.kind, "stop")
        self.assertFalse(stop.success_for("offense"))
        self.assertTrue(stop.success_for("defense"))

        conv = parse_outcome("convert")
        self.assertTrue(conv.success_for("offense"))
        self.assertFalse(conv.success_for("defense"))

        self.assertTrue(outcome_success("+12", "offense"))
        self.assertFalse(outcome_success("+12", "defense"))
        self.assertTrue(outcome_success("sack", "defense"))
        self.assertFalse(outcome_success("sack", "offense"))


class TestRetrainGrades(unittest.TestCase):
    def test_grades_play_vs_coverage(self) -> None:
        snaps = [
            {
                "id": 1,
                "side": "offense",
                "formation": "Gun Trips",
                "play": "Mesh Spot",
                "result": "+13",
                "coverage_seen": "Cover 2",
                "concept_seen": None,
            },
            {
                "id": 2,
                "side": "offense",
                "formation": "Gun Trips",
                "play": "Mesh Spot",
                "result": "incomplete",
                "coverage_seen": "Cover 2",
                "concept_seen": None,
            },
            {
                "id": 3,
                "side": "offense",
                "formation": "Gun Trips",
                "play": "Mesh Spot",
                "result": "gain 8",
                "coverage_seen": "Cover 3",
                "concept_seen": None,
            },
        ]
        grades = grade_play_vs_look(snaps)
        self.assertGreaterEqual(len(grades), 2)
        c2 = next(g for g in grades if g["look"] == "Cover 2")
        self.assertEqual(c2["n"], 2)
        self.assertEqual(c2["successes"], 1)
        self.assertAlmostEqual(c2["success_rate"], 0.5)
        self.assertEqual(c2["formation"], "Gun Trips")
        self.assertEqual(c2["play"], "Mesh Spot")

    def test_retrain_bumps_weights(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = CoachDB(Path(td) / "t.db")
            try:
                snaps = [
                    {
                        "id": 1,
                        "side": "offense",
                        "formation": "Gun Mesh",
                        "play": "Mesh Spot",
                        "result": "+15",
                        "coverage_seen": "Cover 2",
                    },
                    {
                        "id": 2,
                        "side": "offense",
                        "formation": "Gun Mesh",
                        "play": "Mesh Spot",
                        "result": "td",
                        "coverage_seen": "Cover 2",
                    },
                ]
                # Seed fake snap ids in DB so learn_from_snaps can see them
                for s in snaps:
                    db.log_snap(
                        opponent_id="gavin",
                        side=s["side"],
                        situation_raw="1&10",
                        our_call="Gun Mesh — Mesh Spot",
                        formation=s["formation"],
                        play=s["play"],
                        result=s["result"],
                        coverage_seen=s["coverage_seen"],
                    )
                rows = db.get_recent_snaps("gavin", limit=10)
                out = retrain_from_snaps(db, "gavin", list(reversed(rows)))
                self.assertGreaterEqual(out["snaps"], 2)
                self.assertTrue(out["grades"])
                # Weight key should exist
                weights = {r["key"]: r["weight"] for r in db.get_gameplan_weights("gavin")}
                vs_keys = [k for k in weights if k.startswith("vs_look::")]
                self.assertTrue(vs_keys, msg=f"expected vs_look keys, got {weights}")
            finally:
                db.close()


class _FakeCall:
    def __init__(self, side="offense", formation="Gun Trips", play="Mesh Spot"):
        self.side = side
        self.formation = formation
        self.play = play
        self.adj_or_macro = "base"
        self.macro = None
        self.rationale = "test"

    def format(self) -> str:
        return f"{self.formation} — {self.play}"


class TestLiveHttpSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self._td.name) / "live.db")
        self.calls = []

        def make(sit, **kwargs):
            c = _FakeCall()
            self.calls.append((sit, kwargs))
            return c

        def learn():
            return "# POSTGAME stub\nNew snaps learned: 0"

        self.ctrl = LivePlayController(
            db=self.db,
            opponent_id="gavin",
            make_call=make,
            parse_situation=parse_situation,
            learn_summary=learn,
            brand="CFB Coach",
            play_cmd="cfb-coach play",
            cpu_only=True,
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

    def test_submit_sit_call_and_end_game(self) -> None:
        # First call only
        data = self._json("POST", "/api/call", {"sit": "1&10 my 35"})
        self.assertTrue(data["ok"])
        self.assertIn("Mesh Spot", data["call_text"])

        # Result + next sit
        data = self._json(
            "POST",
            "/api/result_call",
            {"outcome": "gain 13", "sit": "2&7 my 40 cover 2"},
        )
        self.assertTrue(data["ok"])
        self.assertIsNotNone(data.get("logged"))
        self.assertEqual(data["logged"]["result"], "+13")
        self.assertEqual(len(data["state"]["log"]), 1)

        # End game → learn summary
        data = self._json(
            "POST",
            "/api/end_game",
            {"result_wl": "win", "score": "24-17"},
        )
        self.assertTrue(data["ok"])
        self.assertIn("GAME OVER", data["retrain_summary"])
        self.assertIn("24-17", data["retrain_summary"])
        self.assertTrue(data["state"]["ended"])

        # Session persisted
        row = self.db.conn.execute(
            "SELECT result_wl, score, play_count FROM game_sessions WHERE session_id = ?",
            (self.ctrl.session_id,),
        ).fetchone()
        self.assertEqual(row["result_wl"], "win")
        self.assertEqual(row["score"], "24-17")
        self.assertEqual(int(row["play_count"]), 1)


class TestPlayCliFlags(unittest.TestCase):
    def test_terminal_and_html_port_flags(self) -> None:
        p = build_parser()
        ns = p.parse_args(["play", "-o", "gavin", "--terminal"])
        self.assertTrue(ns.terminal)
        ns2 = p.parse_args(["play", "-o", "gavin", "--html-port", "9001"])
        self.assertEqual(ns2.html_port, 9001)


if __name__ == "__main__":
    unittest.main()
