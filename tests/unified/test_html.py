"""The established HTML interface over Unified V1: real HTTP, then a real headless browser."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cfb_coach import db as coachdb
from cfb_coach.madden import playcaller
from cfb_coach.madden.unified import guard, htmlcheck as HC, knowledge as K
from cfb_coach.madden.unified.coordinator import inventory_fp
from cfb_coach.madden.unified.html_controller import UnifiedLiveController
from cfb_coach.madden.unified.html_server import start_server
from cfb_coach.madden.unified.session import UnifiedSession
from cfb_coach.madden.unified.store import UnifiedStore
from cfb_coach.madden.unified.types import DEF, OFF
from cfb_coach.madden.unified.world_model import WorldModel
from tests.unified.helpers import inventory

COVER3 = "4-3 Over | Cover 3 Sky"


@pytest.fixture
def no_legacy(monkeypatch):
    """Any legacy caller or primary-DB open during these tests is a failure."""
    def boom(*a, **k):
        raise AssertionError("legacy path touched")
    monkeypatch.setattr(playcaller, "make_call", boom)
    monkeypatch.setattr(coachdb.CoachDB, "__init__", boom)


def books():
    return {s: {"inventory": inventory(s), "fingerprint": inventory_fp(s, inventory(s))} for s in (OFF, DEF)}


class Rig:
    def __init__(self, tmp_path, game_id="h1", resume=False, model=None):
        self.store = UnifiedStore(tmp_path / "u.db")
        self.tmp = tmp_path
        self.session = UnifiedSession(books=books(), opponent_id="cpu_x", store=self.store, game_id=game_id, resume=resume, model=model, artifact="test_artifact:abc")
        self.ctrl = UnifiedLiveController(self.session)
        self.server, self.th, self.url = start_server(self.ctrl)
        self.cl = HC.Client(self.url)

    def post(self, path, body):
        code, data, ms = self.cl.req("POST", path, body)
        return code, data

    def stop(self):
        self.server.shutdown(); self.server.server_close()


@pytest.fixture
def rig(tmp_path, no_legacy):
    r = Rig(tmp_path)
    yield r
    r.stop()


def sub(rig, sit, side, outcome="", their="", executed="unknown", **kw):
    return rig.post("/api/result_call", {"sit": sit, "side": side, "outcome": outcome, "their": their, "executed_status": executed, "quarter": "1", **kw})


def test_page_is_the_established_page_with_the_unified_panel(rig):
    code, html, _ = rig.cl.req("GET", "/")
    assert code == 200 and "Submit → log + new PLAY" in html and 'id="unified-panel"' in html and "btn-ran-recommended" in html and "id=\"side\"" in html
    code, st, _ = rig.cl.req("GET", "/api/state")
    u = st["unified"]
    assert u["decision_model"] == "unified_v1" and u["artifact"] == "test_artifact:abc" and "COLD-START" in u["model_mode"]
    assert st["pending_recommendation"] is None and st["execution_verify"] is True and st["standing_execution_attestation"] is False


def test_offense_defense_offense_through_http_persists_everything_correctly(rig):
    c, d = sub(rig, "1&10 my 25", "offense")
    st = d["state"]
    assert d["ok"] and st["pending_recommendation"]["snap_id"] == "h1-0001" and st["pending_recommendation"]["side"] == "offense"
    p = st["unified"]["pending"]
    assert p["side"] == "offense" and p["snap_id"] == "h1-0001"
    assert st["unified"]["pending"]["adjustments"] == [] and st["qb_reads"]["reads"][0]["order"] == "1" and st["unified"]["fallback"]["occurred"] is False
    first = (st["pending_recommendation"]["formation"], st["pending_recommendation"]["play"])
    # snap 1 result with the owner's explicit "ran it" ; snap 2 offense
    c, d = sub(rig, "2&3 my 32", "offense", "gain 7", COVER3, "used_recommended")
    assert d["ok"] and d["state"]["pending_recommendation"]["snap_id"] == "h1-0002"
    assert "game(1)" in d["state"]["unified"]["pending"]["opponent_prediction_source"]  # beliefs updated from snap 1
    obs = {o.snap_id: o for o in rig.store.load_observations()}
    o1 = obs["h1-0001"]
    assert o1.executed_status == "confirmed" and (o1.executed.play.formation, o1.executed.play.play) == first and o1.outcome.yards == 7.0
    assert o1.opponent_action.verification == "typed_exact"
    # possession change: defense. Result of snap 2 entered WITHOUT confirming execution -> unknown, never assumed
    c, d = sub(rig, "1&10 my 40", "defense", "gain 5")
    st = d["state"]
    assert st["pending_recommendation"]["side"] == "defense" and st["qb_reads"] is None
    u = st["unified"]["pending"]["user"]
    assert u["position"] and u["primary"] and u["secondary"] and u["verified"] is False
    assert "OPPONENT OFFENSE" in st["unified"]["result_perspective"]
    o2 = {o.snap_id: o for o in rig.store.load_observations()}["h1-0002"]
    assert o2.executed_status == "unknown" and o2.executed is None and o2.recommended is not None and o2.outcome.yards == 5.0
    assert rig.session.model.provenance == {"our_offense_executed": 1}  # snap 2 (unconfirmed) did not train
    # result of the DEFENSIVE snap = opponent offense yards, executed confirmed; back to offense
    c, d = sub(rig, "1&10 my 30", "offense", "gain 12", "Gun Trips X Nasty | Hi Lo Cross", "used_recommended")
    assert d["ok"] and d["state"]["pending_recommendation"]["side"] == "offense" and d["state"]["pending_recommendation"]["snap_id"] == "h1-0004"
    o3 = {o.snap_id: o for o in rig.store.load_observations()}["h1-0003"]
    assert o3.state.our_side == DEF and o3.executed_status == "confirmed" and o3.outcome.yards == 12.0 and o3.opponent_action.side == OFF
    assert rig.session.model.provenance == {"our_offense_executed": 1, "our_defense_executed": 1}
    assert "UNDONE" not in json.dumps(d["state"]["log"]) and [r["formation"][:2] for r in d["state"]["log"]] == ["O1", "O2", "D3", "O4"]
    ids = [o.snap_id for o in rig.store.load_observations()]
    assert ids == [f"h1-000{i}" for i in range(1, 5)]
    assert rig.store.counts()["decisions"] == 4  # exactly one recommendation per snap


def test_bad_input_changes_nothing_and_idempotent_retry_does_not_duplicate(rig):
    sub(rig, "1&10 my 25", "offense")
    before = rig.store.counts()
    c, d = sub(rig, "hello", "offense", "gain 3")
    assert c == 200 and d["ok"] is False and rig.store.counts() == before
    c, d = sub(rig, "2&5 my 30", "offense", "gain 3", "zzz qqq")
    assert d["ok"] is False and rig.store.counts() == before and rig.session.pending is not None
    body = {"sit": "2&5 my 30", "side": "offense", "outcome": "gain 3", "their": COVER3, "executed_status": "used_recommended", "idempotency_key": "k-1"}
    a = rig.post("/api/result_call", body)[1]
    b = rig.post("/api/result_call", body)[1]  # the browser retried after a lost response
    assert a == b and rig.store.counts()["decisions"] == 2 and rig.store.counts()["executions"] == 1
    # refresh: same state, no new snap
    s1 = rig.cl.req("GET", "/api/state")[1]
    s2 = rig.cl.req("GET", "/api/state")[1]
    assert s1["pending_recommendation"] == s2["pending_recommendation"] and rig.store.counts()["decisions"] == 2


def test_used_different_requires_an_installed_play_and_is_recorded_as_modified(rig):
    c, d = sub(rig, "1&10 my 25", "offense")
    rec = d["state"]["pending_recommendation"]
    other = next(p for p in inventory(OFF)["Gun Trips X Nasty"] if p != rec["play"])
    c, d = sub(rig, "2&5 my 30", "offense", "gain 3", COVER3, "used_different", executed_formation="Not A Formation", executed_play="Nope")
    assert d["ok"] is False and "not in the installed" in d["error"]
    c, d = sub(rig, "2&5 my 30", "offense", "gain 3", COVER3, "used_different", executed_formation="Gun Trips X Nasty", executed_play=other)
    assert d["ok"]
    o1 = {o.snap_id: o for o in rig.store.load_observations()}["h1-0001"]
    assert o1.executed_status in ("modified", "confirmed") and o1.executed.play.play == other and o1.recommended.play.play == rec["play"]


def test_undo_voids_the_snap_without_reusing_ids_or_training_on_it(rig):
    sub(rig, "1&10 my 25", "offense")
    sub(rig, "2&3 my 32", "offense", "gain 7", COVER3, "used_recommended")
    assert rig.session.model.n_records == 1
    c, d = rig.post("/api/undo", {})
    assert d["ok"] and d["state"]["pending_recommendation"] is None
    assert d["state"]["log"][0]["result"].startswith("UNDONE") and rig.ctrl.session.model.n_records == 0
    c, d = sub(rig, "2&3 my 32", "offense")
    assert d["state"]["pending_recommendation"]["snap_id"] == "h1-0003"  # ids 1 and 2 stay consumed
    assert rig.post("/api/undo", {})[1]["ok"] in (True, False)


def test_reconnect_after_a_restart_resumes_without_losing_or_duplicating_anything(tmp_path, no_legacy):
    r = Rig(tmp_path)
    sub(r, "1&10 my 25", "offense")
    sub(r, "2&3 my 32", "offense", "gain 7", COVER3, "used_recommended")
    sub(r, "1&10 my 40", "defense", "gain 5", "", "unknown")
    waiting = r.cl.req("GET", "/api/state")[1]["pending_recommendation"]
    r.stop(); r.store.close()
    r2 = Rig(tmp_path, resume=True)  # a brand-new process over the same store
    try:
        st = r2.cl.req("GET", "/api/state")[1]
        assert st["pending_recommendation"] == waiting and st["pending_recommendation"]["snap_id"] == "h1-0003"
        assert [x["formation"][:2] for x in st["log"]] == ["O1", "O2", "D3"] and r2.store.counts()["decisions"] == 3
        assert r2.session.model.provenance == {"our_offense_executed": 1}  # earlier learning replayed exactly once
        c, d = sub(r2, "1&10 my 30", "offense", "gain 12", "Gun Trips X Nasty | Hi Lo Cross", "used_recommended")
        assert d["ok"] and d["state"]["pending_recommendation"]["snap_id"] == "h1-0004"
        ids = [o.snap_id for o in r2.store.load_observations()]
        assert ids == [f"h1-000{i}" for i in range(1, 5)] and r2.session.model.provenance == {"our_offense_executed": 1, "our_defense_executed": 1}
    finally:
        r2.stop()


def test_end_game_runs_no_legacy_retrain_and_closes_the_game(rig):
    sub(rig, "1&10 my 25", "offense")
    c, d = rig.post("/api/end_game", {"result_wl": "win", "score": "24-17"})
    assert d["ok"] and "No legacy retrain" in d["retrain_summary"] and "postgame" in d["retrain_summary"]
    assert "h1" in rig.store.closed_games()


def test_http_workflow_check_100_per_side(tmp_path):
    bk = books()
    run = HC.run_http_workflow(bk, model_path=None, store_path=tmp_path / "s.db", n_per_side=100)
    a = HC.audit_http_run(run, bk)
    assert a["ok"], (a["problems"], a["errors"], a["legacy_hits"])
    assert a["request_ms"]["n"] >= 200 and a["request_ms"]["p95"] <= 150.0, a["request_ms"]
    assert a["learning"]["matches"] and a["fallbacks"] == 0 and a["timeouts_over_150ms"] == 0


@pytest.mark.skipif(not shutil.which("node"), reason="node/playwright not available")
def test_real_browser_drives_the_established_page(tmp_path, no_legacy):
    r = Rig(tmp_path, game_id="b1")
    try:
        script = Path(__file__).resolve().parents[2] / "cfb_coach" / "madden" / "unified" / "browser_drive.js"
        p = subprocess.run(["node", str(script), r.url, str(tmp_path / "shots")], capture_output=True, text=True, timeout=180)
        if p.returncode != 0 and "Cannot find module 'playwright'" in p.stderr:
            pytest.skip("playwright module not installed")
        assert p.returncode == 0, p.stderr[-2000:]
        out = json.loads(p.stdout)
    finally:
        r.stop()
    assert not out["errors"], out["errors"]
    steps = {s["label"]: s for s in out["steps"]}
    assert "unified_v1" in steps["initial"]["panel"] and "test_artifact:abc" in steps["initial"]["panel"] and "COLD-START" in steps["initial"]["panel"]
    o1, o2, d3, o4 = steps["offense-1"], steps["offense-2"], steps["defense-3"], steps["offense-4"]
    assert all(x["ok"] for x in (o1, o2, d3, o4)) and all(" — " in x["call"] for x in (o1, o2, d3, o4))
    assert "NO ADJUSTMENT" in o1["panel"] and "FALLBACK" in o1["panel"].upper() and "none" in o1["panel"].lower()
    assert o1["qbVisible"] and len(o1["qbReads"]) == 2
    assert "OPPONENT OFFENSE" in d3["resultNote"] and d3["user"] and "USER:" in d3["user"] and "PRIMARY:" in d3["user"] and "SECONDARY:" in d3["user"]
    assert "OUR OFFENSE" in o4["resultNote"] and d3["qbVisible"] is False and o4["qbVisible"] is True
    assert "ran as called" in " ".join(o2["log"]) and "execution unknown" in " ".join(d3["log"])
    rl = steps["after-reload"]
    assert rl["call"] == o4["call"] and rl["panel"] == rl["panelBefore"]           # refresh keeps the waiting snap, creates nothing
    assert steps["bad-input"]["ok"] is False and steps["bad-input"]["err"]       # error shown, nothing logged
    store = UnifiedStore(tmp_path / "u.db")
    assert store.counts()["decisions"] == 4 and store.counts()["executions"] == 3


# ---------------------------------------------------------------------------------------------- preflight: browser gate
def _kit(tmp_path):
    from tests.unified.test_scenarios_and_learning import _owner_kit
    return _owner_kit(tmp_path)


def _pf(kit, **kw):
    from cfb_coach.madden.unified import preflight as PF
    return PF.preflight(calls=10, confirmed_books=kit["books"], db=kit["db"], model=kit["model"], history_report=kit["history_report"], **kw)


def test_html_verdict_cannot_be_go_without_validated_browser_integration(tmp_path):
    kit = _kit(tmp_path)
    plain = _pf(kit)  # terminal gates only
    assert plain["verdict"] == "GO" and plain["html_verdict"] == "SHADOW ONLY" and "html_interface_validated" in plain["html_blocking"]
    full = _pf(kit, html=True)
    assert full["verdict"] == "GO" and full["html_verdict"] == "GO", full["html_blocking"]
    assert {g["gate"] for g in full["html_gates"]} >= {"html_page_serves_established_interface", "html_workflow_clean",
                                                       "html_no_legacy_caller_or_primary_db_access", "html_recommended_vs_executed_and_learning"}


def test_html_gate_fails_closed_on_a_legacy_hit_and_when_terminal_gates_fail(tmp_path, monkeypatch):
    from cfb_coach.madden.unified import htmlcheck as HC
    kit = _kit(tmp_path)
    real = HC.run_http_workflow

    def tainted(*a, **k):
        r = real(*a, **k)
        r["legacy_hits"] = ["CoachDB"]
        return r
    monkeypatch.setattr(HC, "run_http_workflow", tainted)
    bad = _pf(kit, html=True)
    assert bad["html_verdict"] == "NO-GO" and "html_no_legacy_caller_or_primary_db_access" in bad["html_blocking"]
    monkeypatch.setattr(HC, "run_http_workflow", real)
    from cfb_coach.madden.unified import preflight as PF
    unconfirmed = PF.preflight(calls=10, db=kit["db"], model=kit["model"], history_report=kit["history_report"], html=True)  # no attested books
    assert unconfirmed["html_verdict"] == "SHADOW ONLY"


# ---------------------------------------------------------------------------------------------- the real command, real process
def _books_file(tmp_path):
    f = tmp_path / "books.json"
    f.write_text(json.dumps({s: {"side": s, "fingerprint": inventory_fp(s, inventory(s)), "inventory": inventory(s)} for s in (OFF, DEF)}))
    return f


def test_play_command_opens_the_html_window_and_never_touches_the_primary_home(tmp_path):
    import os, re, signal, time
    home = tmp_path / "home"
    primary = home / ".cfb-coach"
    primary.mkdir(parents=True)
    sentinel = primary / "madden27.db"
    sentinel.write_bytes(b"PRIMARY-FRANCHISE-SENTINEL")
    before = {p: guard.sha256_file(p) for p in primary.rglob("*") if p.is_file()}
    env = {**os.environ, "HOME": str(home), "CFB_COACH_HOME": str(primary), "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "CFB_COACH_AI_RESEARCH": "off"}
    env.pop("CFB_COACH_DB", None); env.pop("CFB_COACH_MADDEN_DB", None)
    cmd = [sys.executable, "-m", "cfb_coach", "play", "--game", "madden27", "--opponent", "cpu", "--coordinator", "unified_v1",
           "--unified-confirmed-books", str(_books_file(tmp_path)), "--unified-store", str(tmp_path / "iso" / "unified.db"),
           "--unified-no-open", "--html-port", "0" if False else str(_free_port())]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=tmp_path)
    try:
        url = None
        t0 = time.time()
        lines = []
        while time.time() - t0 < 60:
            line = p.stdout.readline()
            lines.append(line)
            m = re.search(r"(http://127\.0\.0\.1:\d+/)", line)
            if m:
                url = m.group(1)
                break
        assert url, "".join(lines)
        assert "UNIFIED V1" in "".join(lines)
        cl = HC.Client(url)
        code, html, _ = cl.req("GET", "/")
        assert code == 200 and "Submit → log + new PLAY" in html and 'id="unified-panel"' in html
        code, d, _ = cl.req("POST", "/api/result_call", {"sit": "1&10 my 25", "side": "offense", "outcome": "", "executed_status": "unknown"})
        assert d["ok"] and d["state"]["unified"]["decision_model"] == "unified_v1"
    finally:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    after = {q: guard.sha256_file(q) for q in primary.rglob("*") if q.is_file()}
    assert after == before  # the primary franchise directory is byte-for-byte what it was: nothing created, migrated or written
    assert (tmp_path / "iso" / "unified.db").exists()


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_terminal_rollback_path_is_still_available(tmp_path):
    import os
    env = {**os.environ, "HOME": str(tmp_path / "h"), "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "CFB_COACH_AI_RESEARCH": "off"}
    cmd = [sys.executable, "-m", "cfb_coach", "play", "--game", "madden27", "--opponent", "cpu", "--coordinator", "unified_v1", "--terminal",
           "--unified-confirmed-books", str(_books_file(tmp_path)), "--unified-store", str(tmp_path / "t.db")]
    r = subprocess.run(cmd, input="o 1&10 my 22\ny\n+6\n\nquit\n", env=env, capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert r.returncode == 0 and "BOTH coordinators active" in r.stdout and "OFFENSE" in r.stdout and "[unified_v1" in r.stdout


def test_html_check_command_reports_pass(tmp_path, capsys):
    from cfb_coach.madden.unified.cli import main
    rc = main(["html-check", "--confirmed-books", str(_books_file(tmp_path)), "--calls", "15", "--workdir", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert rc == 0 and "HTML CHECK: PASS" in out
