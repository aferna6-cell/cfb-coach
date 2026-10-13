"""HTML live workers see --no-vod-prior and --freeze-vod-book.

The server loop runs on a thread that did not inherit the command's
contextvars. Each request is handled on a further worker. Both flags have to
be visible there, and they are checked separately: turning the prior off also
freezes the book, so freeze is only observable while the prior stays on.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

from cfb_coach.db import CoachDB
from cfb_coach.live_server import ContextThreadingHTTPServer, LivePlayController, make_handler
from cfb_coach.situation import parse_situation
from cfb_coach.vod_model.flags import book_frozen, prior_enabled, use_cli_flags


class _FakeCall:
    side = "offense"
    formation = "Gun"
    play = "Mesh"
    adj_or_macro = "base"
    macro = None
    rationale = "test"

    def format(self) -> str:
        return f"{self.formation} — {self.play}"


class TestHtmlVodFlags(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        root = Path(self._td.name)
        env = {"HOME": str(root), "CFB_COACH_DB": str(root / "coach.db")}
        self._env = mock.patch.dict(os.environ, env, clear=False)
        self._env.start()
        for key in ("CFB_COACH_NO_VOD_PRIOR", "CFB_COACH_VOD_FREEZE_BOOK", "CFB_COACH_MADDEN_DB"):
            os.environ.pop(key, None)
        self.db = CoachDB(root / "live.db")

    def tearDown(self) -> None:
        self.db.close()
        self._env.stop()
        self._td.cleanup()

    def _post_call_on_worker(self, *, no_vod_prior: bool, freeze_vod_book: bool) -> dict[str, object]:
        seen: dict[str, object] = {}

        def make(sit, **kwargs):
            seen["prior"] = prior_enabled()
            seen["frozen"] = book_frozen()
            seen["thread"] = threading.get_ident()
            return _FakeCall()

        ctrl = LivePlayController(
            db=self.db,
            opponent_id="gavin",
            make_call=make,
            parse_situation=parse_situation,
            learn_summary=lambda: "",
            cpu_only=True,
        )
        ctrl.start()
        handler = make_handler(ctrl)
        with use_cli_flags(no_vod_prior=no_vod_prior, freeze_vod_book=freeze_vod_book):
            server = ContextThreadingHTTPServer(("127.0.0.1", 0), handler)
            port = int(server.server_address[1])
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                data = self._post(port)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
        self.assertTrue(data["ok"])
        self.assertNotEqual(seen["thread"], threading.get_ident())
        return seen

    def _post(self, port: int) -> dict:
        last: Exception | None = None
        body = json.dumps({"sit": "1&10 my 35"}).encode()
        for _ in range(50):
            try:
                conn = HTTPConnection("127.0.0.1", port, timeout=2)
                conn.request(
                    "POST",
                    "/api/call",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                data = json.loads(resp.read().decode())
                conn.close()
                self.assertEqual(resp.status, 200, msg=data)
                return data
            except (ConnectionRefusedError, OSError) as exc:
                last = exc
                time.sleep(0.02)
        raise AssertionError(f"HTML server did not accept: {last}")

    def test_no_vod_prior_reaches_the_html_worker(self) -> None:
        seen = self._post_call_on_worker(no_vod_prior=True, freeze_vod_book=False)
        self.assertIs(seen["prior"], False)

    def test_freeze_vod_book_reaches_the_html_worker_while_prior_stays_on(self) -> None:
        seen = self._post_call_on_worker(no_vod_prior=False, freeze_vod_book=True)
        self.assertIs(seen["prior"], True)
        self.assertIs(seen["frozen"], True)


if __name__ == "__main__":
    unittest.main()
