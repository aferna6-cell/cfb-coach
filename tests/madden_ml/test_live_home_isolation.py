"""Live server and CLI must not create or read ~/.cfb-coach when HOME is empty."""

from __future__ import annotations

import hashlib
import io
import os
import pwd
import sqlite3
import tempfile
import threading
import time
import traceback
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock


def _real_home() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _snapshot(path: Path) -> tuple:
    if not path.exists():
        return ()
    return tuple(sorted(
        (item.relative_to(path).as_posix(), item.stat().st_mtime_ns, item.stat().st_size)
        for item in path.rglob("*")
    ))


def _is_forbidden(path, roots: tuple[str, ...]) -> str | None:
    if path is None or isinstance(path, int):
        return None
    try:
        raw = os.fspath(path)
    except TypeError:
        return None
    text = os.path.abspath(os.path.expanduser(str(raw)))
    raw_text = str(raw)
    for root in roots:
        if text == root or text.startswith(root + os.sep):
            return text
        if root in raw_text:
            needle = root
            if needle in raw_text:
                return raw_text
    return None


@contextmanager
def _audit(roots: tuple[str, ...]):
    hits: list[str] = []

    def consider(path) -> None:
        found = _is_forbidden(path, roots)
        if found is None:
            return
        stack = traceback.extract_stack(limit=8)
        where = " | ".join(f"{frame.name}:{frame.lineno}" for frame in stack[-5:-1])
        hits.append(f"{found} via {where}")

    real_mkdir = Path.mkdir
    real_open = open
    real_connect = sqlite3.connect
    real_stat = os.stat
    real_lstat = os.lstat
    real_listdir = os.listdir
    real_scandir = os.scandir
    real_os_open = os.open
    real_os_mkdir = os.mkdir
    real_makedirs = os.makedirs

    def tracked_mkdir(self, *args, **kwargs):
        consider(self)
        return real_mkdir(self, *args, **kwargs)

    def tracked_open(file, *args, **kwargs):
        consider(file)
        return real_open(file, *args, **kwargs)

    def tracked_connect(database, *args, **kwargs):
        consider(database)
        return real_connect(database, *args, **kwargs)

    def tracked_stat(path, *args, **kwargs):
        consider(path)
        return real_stat(path, *args, **kwargs)

    def tracked_lstat(path, *args, **kwargs):
        consider(path)
        return real_lstat(path, *args, **kwargs)

    def tracked_listdir(path, *args, **kwargs):
        consider(path)
        return real_listdir(path, *args, **kwargs)

    def tracked_scandir(path, *args, **kwargs):
        consider(path)
        return real_scandir(path, *args, **kwargs)

    def tracked_os_open(path, flags, mode=0o777, *, dir_fd=None):
        consider(path)
        if dir_fd is None:
            return real_os_open(path, flags, mode)
        return real_os_open(path, flags, mode, dir_fd=dir_fd)

    def tracked_os_mkdir(path, mode=0o777, *, dir_fd=None):
        consider(path)
        if dir_fd is None:
            return real_os_mkdir(path, mode)
        return real_os_mkdir(path, mode, dir_fd=dir_fd)

    def tracked_makedirs(name, mode=0o777, exist_ok=False):
        consider(name)
        return real_makedirs(name, mode, exist_ok)

    with mock.patch.object(Path, "mkdir", tracked_mkdir), \
            mock.patch("builtins.open", tracked_open), \
            mock.patch("sqlite3.connect", tracked_connect), \
            mock.patch("os.stat", tracked_stat), \
            mock.patch("os.lstat", tracked_lstat), \
            mock.patch("os.listdir", tracked_listdir), \
            mock.patch("os.scandir", tracked_scandir), \
            mock.patch("os.open", tracked_os_open), \
            mock.patch("os.mkdir", tracked_os_mkdir), \
            mock.patch("os.makedirs", tracked_makedirs):
        yield hits


class TestEmptyHome(unittest.TestCase):
    def test_empty_home_live_server_and_cli_do_not_touch_cfb_coach(self) -> None:
        from cfb_coach.cli import main
        from cfb_coach.live_server import (
            ContextThreadingHTTPServer,
            LivePlayController,
            make_handler,
            pick_port,
        )
        from cfb_coach.madden.cli import open_db
        from cfb_coach.madden.model.inference import (
            load_opponent_context,
            resolve_live_db_path,
            save_opponent_context,
        )
        from cfb_coach.madden.model.schema import OpponentContext
        from cfb_coach.madden.playcaller import make_call
        from cfb_coach.madden.prep import build_prep_plan
        from cfb_coach.madden.research_db import _STATE
        from cfb_coach.madden.situation import parse_madden_situation

        real_home = _real_home()
        real_data = real_home / ".cfb-coach"
        before = _snapshot(real_data)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            empty_home = root / "empty-home"
            workspace = root / "workspace"
            empty_home.mkdir()
            workspace.mkdir()
            roots = (
                os.path.abspath(str(_real_home() / ".cfb-coach")),
                os.path.abspath(str(empty_home)),
            )
            coach_db = workspace / "coach.db"
            madden_db = workspace / "madden27.db"
            env = {
                "HOME": str(empty_home),
                "CFB_COACH_DB": str(coach_db),
                "CFB_COACH_MADDEN_DB": str(madden_db),
            }
            previous_madden = os.environ.pop("CFB_COACH_MADDEN_DB", None)
            try:
                self._run_isolated(roots, empty_home, workspace, coach_db, madden_db, env)
            finally:
                if previous_madden is not None:
                    os.environ["CFB_COACH_MADDEN_DB"] = previous_madden
            self.assertEqual(list(empty_home.rglob("*")), [])
        self.assertEqual(_snapshot(real_data), before)

    def _run_isolated(self, roots, empty_home, workspace, coach_db, madden_db, env) -> None:
        from cfb_coach.cli import main
        from cfb_coach.live_server import (
            ContextThreadingHTTPServer,
            LivePlayController,
            make_handler,
            pick_port,
        )
        from cfb_coach.madden.cli import open_db
        from cfb_coach.madden.model.inference import (
            load_opponent_context,
            resolve_live_db_path,
            save_opponent_context,
        )
        from cfb_coach.madden.model.schema import OpponentContext
        from cfb_coach.madden.playcaller import make_call
        from cfb_coach.madden.prep import build_prep_plan
        from cfb_coach.madden.research_db import _STATE
        from cfb_coach.madden.situation import parse_madden_situation

        with mock.patch.dict(os.environ, env, clear=False):
                _STATE.clear()
                with _audit(roots) as hits:
                    db = open_db()
                    try:
                        build_prep_plan("cpu", db=db, offline=True, apply_books=True)
                        build_prep_plan("gavin", db=db, offline=True, apply_books=True)
                    finally:
                        db.close()

                    def run_cli(argv: list[str]) -> tuple[int, str, float]:
                        out = io.StringIO()
                        err = io.StringIO()
                        started = time.perf_counter()
                        with redirect_stdout(out), redirect_stderr(err):
                            code = main(argv)
                        elapsed = (time.perf_counter() - started) * 1000.0
                        return code, out.getvalue() + err.getvalue(), elapsed

                    call_code, call_out, call_ms = run_cli(
                        ["call", "--game", "madden27", "--opponent", "cpu", "--situation", "1&10"]
                    )
                    play_code, play_out, play_ms = run_cli(
                        ["play", "--game", "madden27", "--opponent", "gavin", "--once", "1&10", "--terminal"]
                    )
                    bare_code, bare_out, bare_ms = run_cli(
                        ["play", "--game", "madden27", "--opponent", "cpu", "--once", "1&10", "--no-overlay"]
                    )
                    self.assertEqual(call_code, 0, call_out)
                    self.assertEqual(play_code, 0, play_out)
                    self.assertEqual(bare_code, 0, bare_out)
                    overlay = workspace / "madden27_overlay.html"
                    self.assertTrue(overlay.is_file(), play_out)
                    self.assertIn(str(overlay), play_out)

                    live_db = open_db()
                    server = None
                    try:
                        def _make(sit, **kwargs):
                            return make_call(sit, "gavin", live_db, **kwargs)

                        ctrl = LivePlayController(
                            db=live_db,
                            opponent_id="gavin",
                            make_call=_make,
                            parse_situation=parse_madden_situation,
                            learn_summary=lambda: "",
                            brand="Madden 27 Franchise",
                            play_cmd="cfb-coach play --game madden27",
                            cpu_only=False,
                        )
                        ctrl.start()
                        port = pick_port()
                        server = ContextThreadingHTTPServer(("127.0.0.1", port), make_handler(ctrl))
                        thread = threading.Thread(target=server.serve_forever, daemon=True)
                        thread.start()

                        def request(method: str, path: str, body: bytes | None = None) -> tuple[int, str, float]:
                            started = time.perf_counter()
                            conn = HTTPConnection("127.0.0.1", port, timeout=10)
                            headers = {"Content-Type": "application/json"} if body is not None else {}
                            conn.request(method, path, body=body, headers=headers)
                            resp = conn.getresponse()
                            payload = resp.read().decode()
                            status = resp.status
                            conn.close()
                            return status, payload, (time.perf_counter() - started) * 1000.0

                        state_status, state_body, state_ms = request("GET", "/api/state")
                        call_status, call_body, http_call_ms = request(
                            "POST", "/api/call", b'{"sit":"1&10"}',
                        )
                    finally:
                        if server is not None:
                            server.shutdown()
                            server.server_close()
                        live_db.close()

                    self.assertEqual(state_status, 200, state_body)
                    self.assertEqual(call_status, 200, call_body)
                    self.assertIn("call_text", state_body)
                    self.assertNotIn(roots[0], call_out + play_out + bare_out + state_body + call_body)
                    self.assertNotIn(roots[1], call_out + play_out + bare_out + state_body + call_body)
                    print(
                        "MEASURED cli_call_ms="
                        f"{call_ms:.2f} cli_play_once_ms={play_ms:.2f} "
                        f"cli_play_no_overlay_ms={bare_ms:.2f} "
                        f"http_state_ms={state_ms:.2f} http_call_ms={http_call_ms:.2f}"
                    )
                    self.assertEqual(resolve_live_db_path(), madden_db)
                    self.assertNotEqual(resolve_live_db_path(), coach_db)
                    explicit = workspace / "explicit.db"
                    self.assertEqual(resolve_live_db_path(str(explicit)), explicit)
                    saved = save_opponent_context(OpponentContext(opponent_id="gavin"))
                    self.assertEqual(saved, madden_db)
                    self.assertNotEqual(saved, coach_db)
                    loaded = load_opponent_context("gavin")
                    self.assertIsNotNone(loaded)
                    self.assertEqual(loaded.opponent_id, "gavin")
                    self.assertTrue(str(saved).startswith(str(workspace)))
                    self.assertEqual(hits, [])


def _table_counts(path: Path) -> dict[str, int]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        counts = {}
        for table in ("ml_decisions", "ml_outcomes", "snaps"):
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        counts["opponent_context"] = int(
            conn.execute(
                "SELECT COUNT(*) FROM meta WHERE key LIKE 'ml_opponent_context.v1:%'"
            ).fetchone()[0]
        )
        return counts
    finally:
        conn.close()


class TestMaddenDbSeparateFromCoach(unittest.TestCase):
    def test_live_server_and_cli_write_madden_rows_not_coach_db(self) -> None:
        from cfb_coach.cli import main
        from cfb_coach.live_server import (
            ContextThreadingHTTPServer,
            LivePlayController,
            make_handler,
            pick_port,
        )
        from cfb_coach.madden.cli import _html_shadow_hook, open_db
        from cfb_coach.madden.model.inference import (
            load_opponent_context,
            resolve_live_db_path,
            save_opponent_context,
            set_mode,
        )
        from cfb_coach.madden.model.offense_postgame import evaluate_saved_games
        from cfb_coach.madden.model.schema import CoachingMode, OpponentContext
        from cfb_coach.madden.playcaller import make_call
        from cfb_coach.madden.prep import build_prep_plan
        from cfb_coach.madden.situation import parse_madden_situation
        from cfb_coach.db import CoachDB

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            empty_home = root / "empty-home"
            empty_home.mkdir()
            coach_db = root / "coach.db"
            madden_db = root / "separate" / "madden27.db"
            coach_db.write_bytes(b"coach-sentinel-not-a-madden-database")
            coach_hash = hashlib.sha256(coach_db.read_bytes()).hexdigest()
            env = {
                "HOME": str(empty_home),
                "CFB_COACH_DB": str(coach_db),
                "CFB_COACH_MADDEN_DB": str(madden_db),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                self.assertEqual(resolve_live_db_path(), madden_db)
                self.assertNotEqual(resolve_live_db_path(), coach_db)
                db = open_db()
                try:
                    self.assertEqual(Path(db.path), madden_db)
                    build_prep_plan("cpu", db=db, offline=True, apply_books=True)
                    build_prep_plan("gavin", db=db, offline=True, apply_books=True)
                    set_mode(db, CoachingMode.SHADOW)
                finally:
                    db.close()

                def run_cli(argv: list[str], lines: list[str] | None = None) -> tuple[int, str]:
                    feed = iter(lines or [])

                    def _input(*_args, **_kwargs):
                        try:
                            return next(feed)
                        except StopIteration:
                            raise EOFError

                    out = io.StringIO()
                    err = io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err), mock.patch("builtins.input", _input):
                        code = main(argv)
                    return code, out.getvalue() + err.getvalue()

                call_code, call_out = run_cli(
                    ["call", "--game", "madden27", "--opponent", "cpu", "--situation", "1&10"]
                )
                self.assertEqual(call_code, 0, call_out)
                play_code, play_out = run_cli(
                    ["play", "--game", "madden27", "--opponent", "gavin", "--terminal", "--no-overlay"],
                    ["1&10", "+7", "quit"],
                )
                self.assertEqual(play_code, 0, play_out)
                self.assertIn(str(madden_db), play_out)
                self.assertNotIn(str(coach_db), play_out)

                live_db = open_db()
                server = None
                try:
                    def _make(sit, **kwargs):
                        return make_call(sit, "gavin", live_db, **kwargs)

                    ctrl = LivePlayController(
                        db=live_db,
                        opponent_id="gavin",
                        make_call=_make,
                        parse_situation=parse_madden_situation,
                        learn_summary=lambda: "",
                        brand="Madden 27 Franchise",
                        play_cmd="cfb-coach play --game madden27",
                        cpu_only=False,
                        shadow_evaluate=_html_shadow_hook(live_db, "gavin"),
                        enable_execution_verify=True,
                    )
                    ctrl.start()
                    port = pick_port()
                    server = ContextThreadingHTTPServer(("127.0.0.1", port), make_handler(ctrl))
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()

                    def post(path: str, body: bytes) -> tuple[int, str]:
                        conn = HTTPConnection("127.0.0.1", port, timeout=15)
                        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
                        resp = conn.getresponse()
                        payload = resp.read().decode()
                        status = resp.status
                        conn.close()
                        return status, payload

                    call_status, call_body = post("/api/call", b'{"sit":"1&10"}')
                    result_status, result_body = post(
                        "/api/result_call",
                        b'{"outcome":"+4","sit":"2&6"}',
                    )
                finally:
                    if server is not None:
                        server.shutdown()
                        server.server_close()
                    live_db.close()
                self.assertEqual(call_status, 200, call_body)
                self.assertEqual(result_status, 200, result_body)

                saved = save_opponent_context(OpponentContext(opponent_id="gavin"))
                self.assertEqual(saved, madden_db)
                loaded = load_opponent_context("gavin")
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.opponent_id, "gavin")

                post_code, post_out = run_cli(
                    ["postgame", "--game", "madden27", "--opponent", "gavin"]
                )
                self.assertEqual(post_code, 0, post_out)
                self.assertIn("New snaps learned:", post_out)
                self.assertNotIn("New snaps learned: 0", post_out)

                madden_counts = _table_counts(madden_db)
                self.assertGreaterEqual(madden_counts["ml_decisions"], 1)
                self.assertGreaterEqual(madden_counts["snaps"], 1)
                self.assertGreaterEqual(madden_counts["opponent_context"], 1)
                report_db = CoachDB.open_read_only(madden_db)
                try:
                    report = evaluate_saved_games(report_db)
                finally:
                    report_db.close()
                self.assertEqual(Path(report_db.path), madden_db)
                self.assertIsInstance(report["games"], list)

                self.assertEqual(hashlib.sha256(coach_db.read_bytes()).hexdigest(), coach_hash)
                self.assertEqual(coach_db.read_bytes(), b"coach-sentinel-not-a-madden-database")
                self.assertEqual(list(empty_home.rglob("*")), [])
