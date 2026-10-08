"""Real Chromium-driven test: browser must send idempotency_key and not duplicate."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController, make_handler, pick_port
from cfb_coach.situation import parse_situation

try:
    from playwright.sync_api import sync_playwright

    HAS_PW = True
except ImportError:
    HAS_PW = False


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


@unittest.skipUnless(HAS_PW, "playwright not installed")
class PlaywrightBrowserIdempotencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db = CoachDB(
            Path(self._td.name) / "live.db",
            seed={
                "opponents": {
                    "lions_franchise": {
                        "display_name": "Lions",
                        "team_now": "DET",
                        "skill": "human",
                        "confidence": "high",
                        "profile_json": "{}",
                    }
                }
            },
        )
        self.ctrl = LivePlayController(
            db=self.db,
            opponent_id="lions_franchise",
            make_call=lambda sit, **k: _FakeCall(),
            parse_situation=parse_situation,
            learn_summary=lambda: "ok",
            brand="Madden 27 Franchise",
            play_cmd="x",
            cpu_only=False,
            enable_execution_verify=True,
        )
        self.ctrl.start()
        port = pick_port()
        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(self.ctrl))
        # Wait for in-flight request handlers before closing their shared
        # SQLite connection. Otherwise teardown can segfault in CI.
        self.server.daemon_threads = False
        self.port = port
        self.url = f"http://127.0.0.1:{port}/"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.db.close()
        self._td.cleanup()

    def test_browser_sends_keys_double_click_no_duplicate(self) -> None:
        posted: list[dict] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def on_request(req):
                if req.method == "POST" and "/api/" in req.url:
                    try:
                        body = req.post_data_json or {}
                    except Exception:
                        body = {}
                    posted.append({"url": req.url, "body": body})

            page.on("request", on_request)
            page.goto(self.url, wait_until="domcontentloaded")
            page.wait_for_selector("#btn-call-only")

            # Helpers present in page JS
            self.assertTrue(page.evaluate("typeof claimIdempotencyKey === 'function'"))
            self.assertTrue(page.evaluate("typeof apiAction === 'function'"))

            # Call only — browser must attach idempotency_key
            page.click("#btn-call-only")
            page.wait_for_timeout(300)
            call_posts = [x for x in posted if x["url"].endswith("/api/call")]
            self.assertTrue(call_posts)
            self.assertIn("idempotency_key", call_posts[-1]["body"])
            first_call_key = call_posts[-1]["body"]["idempotency_key"]

            # Double-click call-only while first may still be finishing / second ignored by uiLock
            page.click("#btn-call-only", click_count=2, delay=20)
            page.wait_for_timeout(400)

            # Set outcome and double-click submit in one turn (uiLock + same key protect).
            page.fill("#outcome-text", "+7")
            page.evaluate(
                """() => {
                  const b = document.getElementById('btn-submit');
                  b.click();
                  b.click();
                  b.click();
                }"""
            )
            page.wait_for_timeout(800)

            result_posts = [x for x in posted if "/api/result_call" in x["url"]]
            self.assertTrue(result_posts)
            keys = [x["body"].get("idempotency_key") for x in result_posts]
            self.assertTrue(all(keys))
            # Double-click / multi-fire of the same action → one network key (or one request).
            self.assertEqual(
                len(set(keys)),
                1,
                msg=f"expected one key for duplicate submits, got {keys}",
            )
            # At most one result_call should leave the browser while uiLocked.
            self.assertEqual(len(result_posts), 1, msg=result_posts)

            # After success, pending cleared
            pending = page.evaluate(
                "sessionStorage.getItem('cfb_coach_live_idempotency_v1')"
            )
            self.assertIsNone(pending)

            # Undo via browser
            page.click("#btn-undo")
            page.wait_for_timeout(300)
            undo_posts = [x for x in posted if x["url"].endswith("/api/undo")]
            self.assertTrue(undo_posts)
            self.assertIn("idempotency_key", undo_posts[-1]["body"])
            self.assertNotEqual(undo_posts[-1]["body"]["idempotency_key"], keys[0])

            # Relog — new intentional action → new key
            page.fill("#outcome-text", "+3")
            page.click("#btn-submit")
            page.wait_for_timeout(400)
            result_posts2 = [x for x in posted if "/api/result_call" in x["url"]]
            last_key = result_posts2[-1]["body"]["idempotency_key"]
            self.assertNotEqual(last_key, keys[0])

            # Refresh — log persists; new call gets a new key
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector("#btn-call-only")
            page.click("#btn-call-only")
            page.wait_for_timeout(300)
            call_posts2 = [x for x in posted if x["url"].endswith("/api/call")]
            self.assertNotEqual(call_posts2[-1]["body"]["idempotency_key"], first_call_key)

            browser.close()

        snaps = int(self.db.conn.execute("SELECT count(*) FROM snaps").fetchone()[0])
        outs = int(self.db.conn.execute("SELECT count(*) FROM ml_outcomes").fetchone()[0])
        # One snap after undo+relog (not three from triple-click)
        self.assertEqual(snaps, 1)
        self.assertEqual(outs, 1)
        self.assertEqual(len(self.ctrl.log), 1)


if __name__ == "__main__":
    unittest.main()
