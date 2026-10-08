"""Browser HTML must send idempotency keys; retries must not duplicate snaps."""

from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController, make_handler, pick_port, render_live_html
from cfb_coach.situation import parse_situation


class _FakeCall:
    def __init__(self, side="offense", formation="Gun Bunch", play="Mesh"):
        self.side = side
        self.formation = formation
        self.play = play
        self.macro = None
        self.adj_or_macro = "base"
        self.rationale = "t"

    def format(self) -> str:
        return f"{self.formation} — {self.play}"


def _seed() -> dict:
    return {
        "opponents": {
            "lions_franchise": {
                "display_name": "Lions",
                "team_now": "DET",
                "skill": "human",
                "confidence": "high",
                "profile_json": "{}",
            }
        }
    }


class HtmlSourceIdempotencyTests(unittest.TestCase):
    def test_rendered_html_wires_idempotency_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "t.db", seed=_seed())
            try:
                ctrl = LivePlayController(
                    db=db,
                    opponent_id="lions_franchise",
                    make_call=lambda sit, **k: _FakeCall(),
                    parse_situation=parse_situation,
                    learn_summary=lambda: "x",
                    brand="Madden 27 Franchise",
                    play_cmd="x",
                    enable_execution_verify=True,
                )
                ctrl.start()
                html = render_live_html(ctrl)
            finally:
                db.close()
        self.assertIn("idempotency_key", html)
        self.assertIn("claimIdempotencyKey", html)
        self.assertIn("apiAction", html)
        self.assertIn("withUiLock", html)
        self.assertIn("cfb_coach_live_idempotency_v1", html)
        self.assertIn("sessionStorage", html)
        # Buttons go through apiAction, not a bare fetch without keys.
        self.assertIn('apiAction("/api/result_call"', html)
        self.assertIn('apiAction("/api/call"', html)
        self.assertIn('apiAction("/api/undo"', html)
        self.assertIn('apiAction("/api/end_game"', html)
        # Double-click guard
        self.assertIn("uiLocked", html)
        self.assertRegex(html, re.compile(r"if\s*\(\s*uiLocked\s*\)\s*return\s+null"))


class BrowserProtocolClient:
    """Mirrors the browser claimIdempotencyKey / retry semantics in Python."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.pending: dict | None = None
        self._seq = 0

    def _fp(self, path: str, body: dict) -> str:
        copy = {k: v for k, v in body.items() if k not in ("idempotency_key", "request_key")}
        return path + "|" + json.dumps(copy, sort_keys=True)

    def _claim(self, path: str, body: dict) -> str:
        fp = self._fp(path, body)
        if (
            self.pending
            and self.pending.get("path") == path
            and self.pending.get("fingerprint") == fp
            and self.pending.get("key")
        ):
            return self.pending["key"]
        self._seq += 1
        key = f"browser-key-{self._seq}"
        self.pending = {"path": path, "fingerprint": fp, "key": key, "status": "in_flight"}
        return key

    def post(self, path: str, body: dict, *, force_key: str | None = None) -> dict:
        key = force_key or self._claim(path, body)
        payload = dict(body)
        payload["idempotency_key"] = key
        conn = HTTPConnection(self.host, self.port, timeout=5)
        raw = json.dumps(payload).encode()
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()
        if resp.status >= 500:
            self.pending = {
                "path": path,
                "fingerprint": self._fp(path, body),
                "key": key,
                "status": "ambiguous",
            }
        elif data.get("ok"):
            self.pending = None
        else:
            # client error → new key next time (corrected form)
            self.pending = None
        data["_status"] = resp.status
        data["_key_used"] = key
        return data


class BrowserOriginatedRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = CoachDB(Path(self._td.name) / "live.db", seed=_seed())
        self.ctrl = LivePlayController(
            db=self.db,
            opponent_id="lions_franchise",
            make_call=lambda sit, **k: _FakeCall(
                side=getattr(sit, "side", "offense") or "offense",
                play="Mesh",
            ),
            parse_situation=parse_situation,
            learn_summary=lambda: "done",
            brand="Madden 27 Franchise",
            play_cmd="x",
            cpu_only=False,
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
        self.client = BrowserProtocolClient("127.0.0.1", port)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.db.close()
        self._td.cleanup()

    def _counts(self) -> tuple[int, int, int]:
        snaps = int(self.db.conn.execute("SELECT count(*) FROM snaps").fetchone()[0])
        outs = int(self.db.conn.execute("SELECT count(*) FROM ml_outcomes").fetchone()[0])
        # Recommendations ≈ sealed decisions when shadow off; count logs + pending call.
        log_n = len(self.ctrl.log)
        return snaps, outs, log_n

    def test_double_submit_same_key_no_duplicate_snap(self) -> None:
        c = self.client.post("/api/call", {"sit": "1&10 my 25", "side": "offense"})
        self.assertTrue(c["ok"])
        body = {
            "outcome": "+7",
            "sit": "2&3 my 32",
            "side": "offense",
            "executed_status": "used_recommended",
        }
        a = self.client.post("/api/result_call", body)
        self.assertTrue(a["ok"])
        key = a["_key_used"]
        # Simulate double-click / retry with the same key (browser reuses pending).
        self.client.pending = {
            "path": "/api/result_call",
            "fingerprint": self.client._fp("/api/result_call", body),
            "key": key,
            "status": "ambiguous",
        }
        b = self.client.post("/api/result_call", body)
        self.assertTrue(b["ok"])
        self.assertTrue(b.get("idempotent_replay") or b["_key_used"] == key)
        snaps, outs, log_n = self._counts()
        self.assertEqual(snaps, 1)
        self.assertEqual(outs, 1)
        self.assertEqual(log_n, 1)

    def test_new_intentional_action_gets_new_key(self) -> None:
        self.client.post("/api/call", {"sit": "1&10 my 25"})
        a = self.client.post(
            "/api/result_call",
            {"outcome": "incomplete", "sit": "2&10 my 25", "executed_status": "unknown"},
        )
        b = self.client.post(
            "/api/result_call",
            {
                "outcome": "sack",
                "sit": "3&10 my 25",
                "executed_status": "used_different",
                "executed_formation": "Gun Trips",
                "executed_play": "Inside Zone",
            },
        )
        self.assertNotEqual(a["_key_used"], b["_key_used"])
        snaps, outs, log_n = self._counts()
        self.assertEqual(snaps, 2)
        self.assertEqual(outs, 2)
        self.assertEqual(log_n, 2)

    def test_undo_then_relog_uses_new_key_no_extra_rows(self) -> None:
        self.client.post("/api/call", {"sit": "1&10 my 25"})
        self.client.post(
            "/api/result_call",
            {"outcome": "+5", "sit": "2&5 my 30", "executed_status": "used_recommended"},
        )
        u = self.client.post("/api/undo", {})
        self.assertTrue(u["ok"])
        # After undo, outcome voided; relog is a new intentional action.
        r = self.client.post(
            "/api/result_call",
            {"outcome": "+3", "sit": "2&7 my 28", "executed_status": "used_recommended"},
        )
        self.assertTrue(r["ok"])
        self.assertNotEqual(u["_key_used"], r["_key_used"])
        snaps, outs, _ = self._counts()
        self.assertEqual(snaps, 1)
        self.assertEqual(outs, 1)

    def test_ambiguous_retry_reuses_key_after_simulated_refresh(self) -> None:
        """sessionStorage-like pending survives 'refresh' (new client with restored pending)."""
        self.client.post("/api/call", {"sit": "1&10 my 25"})
        body = {"outcome": "td", "sit": "1&10 opp 20", "executed_status": "used_recommended"}
        first = self.client.post("/api/result_call", body)
        key = first["_key_used"]
        # Refresh: new BrowserProtocolClient, restore pending from sessionStorage.
        restored = BrowserProtocolClient("127.0.0.1", self.port)
        restored.pending = {
            "path": "/api/result_call",
            "fingerprint": restored._fp("/api/result_call", body),
            "key": key,
            "status": "ambiguous",
        }
        # Also restore controller state is same server process (refresh keeps server).
        again = restored.post("/api/result_call", body)
        self.assertEqual(again["_key_used"], key)
        self.assertTrue(again.get("idempotent_replay"))
        snaps, outs, log_n = self._counts()
        self.assertEqual(snaps, 1)
        self.assertEqual(outs, 1)
        self.assertEqual(log_n, 1)

    def test_session_restart_new_controller_new_keys(self) -> None:
        self.client.post("/api/call", {"sit": "1&10 my 25"})
        self.client.post(
            "/api/result_call",
            {"outcome": "+1", "sit": "2&9 my 26", "executed_status": "unknown"},
        )
        self.client.post("/api/end_game", {"result_wl": "win", "score": "7-0"})
        # Process restart: new controller on same DB.
        ctrl2 = LivePlayController(
            db=self.db,
            opponent_id="lions_franchise",
            make_call=lambda sit, **k: _FakeCall(),
            parse_situation=parse_situation,
            learn_summary=lambda: "g2",
            brand="Madden 27 Franchise",
            play_cmd="x",
            enable_execution_verify=True,
        )
        ctrl2.start()
        self.assertNotEqual(ctrl2.session_id, self.ctrl.session_id)
        # Swap handler would need new server; exercise controller methods with keys.
        r1 = ctrl2.call_only("1&10 my 25", request_key="restart-call-1")
        r2 = ctrl2.call_only("1&10 my 25", request_key="restart-call-1")
        self.assertTrue(r2.get("idempotent_replay"))
        self.assertEqual(r1["call_text"], r2["call_text"])


class ConcurrentDoubleClickTests(unittest.TestCase):
    def test_parallel_posts_same_key_single_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = CoachDB(Path(tmp) / "c.db", seed=_seed())
            ctrl = LivePlayController(
                db=db,
                opponent_id="lions_franchise",
                make_call=lambda sit, **k: _FakeCall(),
                parse_situation=parse_situation,
                learn_summary=lambda: "x",
                brand="Madden 27 Franchise",
                play_cmd="x",
                enable_execution_verify=True,
            )
            ctrl.start()
            handler = make_handler(ctrl)
            port = pick_port()
            from http.server import ThreadingHTTPServer

            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                # Prime a call
                def post(path, body):
                    conn = HTTPConnection("127.0.0.1", port, timeout=5)
                    conn.request(
                        "POST",
                        path,
                        body=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    resp = conn.getresponse()
                    data = json.loads(resp.read().decode())
                    conn.close()
                    return data

                post("/api/call", {"sit": "1&10 my 25", "idempotency_key": "c0"})
                body = {
                    "outcome": "+6",
                    "sit": "2&4 my 31",
                    "executed_status": "used_recommended",
                    "idempotency_key": "same-click-key",
                }
                results: list[dict] = []

                def worker():
                    results.append(post("/api/result_call", body))

                t1 = threading.Thread(target=worker)
                t2 = threading.Thread(target=worker)
                t1.start()
                t2.start()
                t1.join()
                t2.join()
                self.assertTrue(all(r.get("ok") for r in results))
                snaps = int(db.conn.execute("SELECT count(*) FROM snaps").fetchone()[0])
                outs = int(db.conn.execute("SELECT count(*) FROM ml_outcomes").fetchone()[0])
                self.assertEqual(snaps, 1)
                self.assertEqual(outs, 1)
                self.assertEqual(len(ctrl.log), 1)
            finally:
                server.shutdown()
                server.server_close()
                db.close()


if __name__ == "__main__":
    unittest.main()
