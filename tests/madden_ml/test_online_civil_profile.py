"""Online league (Chiefs, owner t351u/t352u): team-only start, civil offense + Ravens civil defense.

* `play --franchise online --opponent-team BUF` starts with no gamertag and no human-offense gate.
* Offense = civil_scheme_only, defense = Ravens two-formation civil caller (scheme dataset = legal book,
  UNTRAINED_PRIOR_ONLY), no silent fallback to the previous defensive caller.
* Every game / snap / defensive decision is keyed by the opponent team; Chiefs refused; Lions unchanged.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden import data as mdata
from cfb_coach.madden import online_league as ol
from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense import coordinator as rco
from cfb_coach.madden.ravens_defense import mode as rmode
from cfb_coach.madden.scheme import mode as smode
from cfb_coach.madden.situation import parse_madden_situation

from tests.madden_ml.test_online_league import KC, _apply_books, _cli

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tmp_db(tmp_path):
    CoachDB(tmp_path / "madden27.db", seed=mdata.load_seed()).close()
    return tmp_path


def _play(tmp: Path, *extra: str, stdin: str = "q\n"):
    return _cli(tmp, "play", "--game", "madden27", "--terminal", "--no-overlay", *extra, stdin=stdin)


@pytest.fixture
def online_env(monkeypatch):
    monkeypatch.setenv(ol.ENV_CIVIL_PROFILE, "1")
    monkeypatch.setenv(rmode.ENV_MODE, rmode.RAVENS)
    monkeypatch.setenv(smode.ENV_MODE, smode.CIVIL)
    monkeypatch.delenv(rmode.ENV_ARTIFACT, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent-home-for-tests")   # no trained artifacts: prior-only path


# ------------------------------------------------------------------ team-only identity
def test_team_opponent_id_is_stable_and_per_team():
    assert ol.team_opponent_id("Buffalo Bills") == "online_buffalo_bills"
    assert ol.team_opponent_id("San Francisco 49ers") == "online_san_francisco_49ers"
    ids = {ol.team_opponent_id(t) for t in ol.opponent_teams(KC)}
    assert len(ids) == 31 and all(len(i) <= 32 and i[0].isalpha() for i in ids)


def test_team_only_start_no_gamertag_civil_offense_and_defense(tmp_db):
    """No --opponent, no --heuristic-offense, no --civil-scheme-only, no --ravens-defense, no Lions D book."""
    r = _play(tmp_db, "--franchise", "online", "--opponent-team", "BUF",
              stdin="d 3&7 opp 40 trips\n+4 mesh\n1&10 my 25\n+3\nq\n")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "OFFENSE CONTROLLER: CIVIL_SCHEME_ONLY" in out
    assert "DEFENSE CONTROLLER: RAVENS_TWO_FORMATION_CIVIL" in out
    assert "UNTRAINED_PRIOR_ONLY" in out
    assert "HUMAN OFFENSE ML PILOT: OFF (heuristic offense" not in out
    assert "has no explicit offense mode" not in r.stderr
    assert "ONLINE LEAGUE: Kansas City Chiefs vs Buffalo Bills (logged as online_buffalo_bills" in out
    assert "LABEL: RAVENS_DEFENSE_ML (UNTRAINED_PRIOR_ONLY · scheme book)" in out
    assert "PLAY: Nickel 2-4 Wide — " in out or "PLAY: Dime 2-3 Odd — DB Fire 2" in out
    con = sqlite3.connect(tmp_db / "madden27.db")
    try:
        sess = con.execute("SELECT opponent_id, league, user_team, opponent_team, opponent_type "
                           "FROM game_sessions").fetchall()
        assert sess == [("online_buffalo_bills", "online", KC, "Buffalo Bills", "human")]
        snaps = con.execute("SELECT side, opponent_team, opponent_id, formation FROM snaps "
                            "WHERE data_source != 'session_close' OR data_source IS NULL").fetchall()
        assert {s[0] for s in snaps} >= {"defense"}
        assert all(s[1] == "Buffalo Bills" and s[2] == "online_buffalo_bills" for s in snaps)
        assert all(s[3] in rc.FORMATIONS for s in snaps if s[0] == "defense")
        dec = con.execute("SELECT status, league, opponent_team, formation FROM ravens_defense_decisions").fetchall()
        assert dec and all(d[0] == "CALL" and d[1] == "online" and d[2] == "Buffalo Bills" for d in dec)
        assert all(d[3] in rc.FORMATIONS for d in dec)
    finally:
        con.close()


def test_chiefs_rejected_and_team_required(tmp_db):
    r = _play(tmp_db, "--franchise", "online", "--opponent-team", "KC")
    assert r.returncode == 2 and "your team" in r.stderr
    r = _play(tmp_db, "--franchise", "online")
    assert r.returncode == 2 and "--opponent-team is required" in r.stderr


def test_two_teams_stay_separate_even_for_one_name(tmp_db):
    for team in ("BUF", "Miami Dolphins"):
        r = _play(tmp_db, "--franchise", "online", "-o", "same_guy", "--opponent-team", team,
                  stdin="1&10 my 25\n+3\nq\n")
        assert r.returncode == 0, r.stderr
    con = sqlite3.connect(tmp_db / "madden27.db")
    try:
        rows = con.execute("SELECT opponent_id, opponent_team FROM game_sessions ORDER BY started_ts").fetchall()
        assert rows == [("online_buffalo_bills", "Buffalo Bills"), ("online_miami_dolphins", "Miami Dolphins")]
        by_team = dict(con.execute("SELECT opponent_team, COUNT(*) FROM snaps GROUP BY opponent_team").fetchall())
        assert set(by_team) == {"Buffalo Bills", "Miami Dolphins"}
    finally:
        con.close()
    inv = _cli(tmp_db, "league", "--game", "madden27")
    assert "Opponent teams with data: 2/31" in inv.stdout
    assert not [ln for ln in inv.stdout.splitlines() if ln.startswith(KC)]


def test_lions_primary_unchanged(tmp_db):
    _apply_books(tmp_db / "madden27.db")
    r = _play(tmp_db, "-o", "cpu", stdin="1&10 my 25\nq\n")
    assert r.returncode == 0, r.stderr
    assert "RAVENS_TWO_FORMATION_CIVIL" not in r.stdout
    assert "ONLINE LEAGUE" not in r.stdout
    assert "team: Detroit Lions" in r.stdout
    r = _play(tmp_db)   # non-online still needs an opponent
    assert r.returncode == 2 and "--opponent" in (r.stderr + r.stdout)


def test_previous_defense_and_unrestricted_offense_opt_out(tmp_db):
    _apply_books(tmp_db / "madden27.db")
    r = _play(tmp_db, "--franchise", "online", "--opponent-team", "BUF", "--previous-defense",
              "--unrestricted-offense", "--heuristic-offense", stdin="q\n")
    assert r.returncode == 0, r.stderr
    assert "RAVENS_TWO_FORMATION_CIVIL" not in r.stdout
    assert "OFFENSE CONTROLLER: CIVIL_SCHEME_ONLY" not in r.stdout


# ------------------------------------------------------------------ defense caller
def _calls(n: int, session: str = "S1", db=None):
    from cfb_coach.madden.playcaller import make_call
    out = []
    for i in range(n):
        sit = parse_madden_situation("d 2&7 own 35", default_side="defense")
        sit.extras = {"session_id": session, "snap_seq": i + 1}
        out.append(make_call(sit, "online_buffalo_bills", db))
    return out


def test_defense_calls_need_no_installed_book_and_show_required_macros(tmp_db, online_env):
    db = CoachDB(tmp_db / "madden27.db", seed=mdata.load_seed())   # no applied books at all
    try:
        calls = _calls(160, db=db)
    finally:
        db.close()
    cat = rc.load()
    by = cat.by_id()
    seen = set()
    prev = None
    for c in calls:
        rd = c.ravens_defense
        assert rd["status"] == "CALL" and rd["scheme_book"] is True
        assert rd["outcome_model_status"] == "UNTRAINED_PRIOR_ONLY"
        cand = by[rd["candidate"]["candidate_id"]]
        seen.add(cand.candidate_id)
        text = c.format()
        assert text.startswith(f"PLAY: {cand.formation} — {cand.base_play}")
        assert "USER: UNVERIFIED" in text
        if cand.package == "nickel_stock":
            assert "MACRO: 24 Base" in text
        if cand.package == "dime_blitz":
            assert cand.formation == rc.DIME and "MACRO: DB Fire" in text
        if cand.optional_macro == rc.ZONE_IT:
            assert "Zone It" in text and cand.base_play in ("Cover 4 Quarters", "Cover 4 Palms")
        if not cand.macros:
            assert "MACRO:" not in text
        assert cand.candidate_id != "match_cover9_show2"        # dataset block still applies
        assert cand.candidate_id != prev                        # no back-to-back repeat
        prev = cand.candidate_id
        assert c.call_label == "RAVENS_DEFENSE_ML"
    assert "dime_dbfire2" in seen and {"nickel_cloud", "nickel_tampa2"} & seen and len(seen) >= 8


def test_no_silent_fallback_to_previous_defense(tmp_db, online_env, monkeypatch):
    from cfb_coach.madden import playcaller

    def boom(*a, **k):
        raise AssertionError("previous defensive caller must not run")
    monkeypatch.setattr(playcaller, "_pick_defense", boom)
    monkeypatch.setattr(rc, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dataset unreadable")))
    db = CoachDB(tmp_db / "madden27.db", seed=mdata.load_seed())
    try:
        call = _calls(1, db=db)[0]
    finally:
        db.close()
    assert call.play == "NO_LEGAL_RAVENS_CALL"
    assert "NO_LEGAL_RAVENS_CALL" in call.format() and "no fallback" in call.format()


def test_owner_reported_missing_macro_blocks_only_that_package(tmp_path, online_env, monkeypatch):
    import json
    p = tmp_path / "conf.json"
    p.write_text(json.dumps({"schema": "madden27.ravens_defense.owner_confirmations.v1",
                             "macros": {"DB_FIRE": {"exists": False}}}))
    monkeypatch.setenv("CFB_COACH_RAVENS_DEFENSE_CONFIRMATIONS", str(p))
    from cfb_coach.madden.ravens_defense import confirmations as cf, legality as lg
    legal, verdicts = lg.legal_set(rc.load(), installed=None, conf=cf.load(), scheme_book=True)
    ids = {c.candidate_id for c in legal}
    assert "dime_dbfire2" not in ids and "nickel_cloud" in ids
    assert "MACRO_OWNER_REPORTED_MISSING:DB_FIRE" in verdicts["dime_dbfire2"].reasons


def test_lions_path_still_needs_installed_book_and_confirmations(monkeypatch):
    """Without the online profile the #155 fail-closed rules are unchanged (0 legal on an empty book)."""
    monkeypatch.delenv(ol.ENV_CIVIL_PROFILE, raising=False)
    from cfb_coach.madden.ravens_defense import confirmations as cf, legality as lg
    legal, _ = lg.legal_set(rc.load(), installed={}, conf=cf.Confirmations())
    assert legal == []


def test_html_session_logs_defense_execution_by_team(tmp_db, online_env):
    from cfb_coach.madden.playcaller import make_call
    db = CoachDB(tmp_db / "madden27.db", seed=mdata.load_seed())
    try:
        ctx = ol.league_context(db, "online_green_bay_packers", "GB", cpu=False)
        shown = []

        def _make(sit, **kw):
            c = make_call(sit, "online_green_bay_packers", db)
            shown.append(c)
            return c
        ctrl = LivePlayController(
            db=db, opponent_id="online_green_bay_packers", make_call=_make,
            parse_situation=parse_madden_situation, learn_summary=lambda: "",
            brand="Madden 27 Franchise", dynasty="online", league_context=ctx,
            enable_execution_verify=True, standing_execution_attestation=True,
            standing_defense_attestation=True,
        )
        ctrl.start()
        assert ctrl.call_only("3&7 opp 40", side="defense")["ok"]
        assert ctrl.result_and_call(outcome="+4", sit_raw="4&3 opp 36", side="defense")["ok"]
        call = shown[0]
        snap = db.conn.execute("SELECT * FROM snaps WHERE side='defense' ORDER BY id LIMIT 1").fetchone()
        assert snap["opponent_team"] == "Green Bay Packers" and snap["league"] == "online"
        assert snap["executed_formation"] == call.formation and snap["executed_play"] == call.play
        assert snap["executed_verification"] == "verified"
        assert (snap["executed_macro"] or None) == (call.macro or None)
        dec = db.conn.execute("SELECT opponent_team, league, session_id FROM ravens_defense_decisions").fetchall()
        assert dec and all(tuple(d)[:2] == ("Green Bay Packers", "online") for d in dec)
        assert all(d[2] == ctrl.session_id for d in dec)
    finally:
        db.close()
