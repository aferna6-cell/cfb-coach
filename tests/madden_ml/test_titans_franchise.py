"""Titans franchise (t387u): team-keyed logging, civil defaults, isolated from Chiefs and Lions."""
from __future__ import annotations

import sqlite3

from cfb_coach.madden import franchise
from cfb_coach.madden import online_league as ol
from cfb_coach.madden.human_opponents import register_cold_start
from cfb_coach.session import start_session

from tests.madden_ml.test_online_civil_profile import _play, tmp_db  # noqa: F401
from tests.madden_ml.test_online_league import KC, _apply_books, _cli

TEN = "Tennessee Titans"
COLTS = "Indianapolis Colts"


def test_titans_profile_is_never_sticky_and_separate_from_chiefs(tmp_path):
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import data as mdata

    assert franchise.normalize_profile("tennessee") == franchise.TITANS
    assert franchise.team_for_profile("titans") == TEN
    assert franchise.is_team_keyed_profile("titans")
    spec = franchise.team_keyed_spec("titans")
    assert spec["league"] == "titans" and spec["id_prefix"] == "titans_"
    assert ol.team_opponent_id(COLTS, prefix=spec["id_prefix"]) == "titans_indianapolis_colts"
    assert ol.team_opponent_id(COLTS) == "online_indianapolis_colts"
    db = CoachDB(tmp_path / "m.db", seed=mdata.load_seed())
    try:
        franchise.set_session_profile(db, "primary")
        assert franchise.set_session_profile(db, "titans") == "titans"
        assert franchise.get_session_profile(db) == "primary"
        db.set_meta(franchise.META_KEY, "titans")
        assert franchise.get_session_profile(db) == "primary"
    finally:
        db.close()


def test_titans_rejects_titans_and_requires_opponent_team(tmp_db):
    r = _play(tmp_db, "--franchise", "titans", "--opponent-team", "TEN")
    assert r.returncode == 2 and "your team" in r.stderr
    r = _play(tmp_db, "--franchise", "titans")
    assert r.returncode == 2 and "--opponent-team is required" in r.stderr
    r = _play(tmp_db, "--franchise", "online", "--user-team", TEN, "--opponent-team", "IND")
    assert r.returncode == 2 and "does not match" in r.stderr


def test_titans_colts_start_civil_and_stays_separate_from_chiefs(tmp_db):
    r = _play(tmp_db, "--franchise", "titans", "--opponent-team", "Indianapolis Colts",
              stdin="d 3&7 opp 40 trips\n+4 mesh\n1&10 my 25\n+3\nq\n")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "OFFENSE CONTROLLER: CIVIL_SCHEME_ONLY" in out
    assert "DEFENSE CONTROLLER: RAVENS_TWO_FORMATION_CIVIL" in out
    assert "TITANS FRANCHISE: Tennessee Titans vs Indianapolis Colts (logged as titans_indianapolis_colts" in out
    assert "ONLINE LEAGUE:" not in out

    r2 = _play(tmp_db, "--franchise", "online", "--opponent-team", "Indianapolis Colts",
               stdin="1&10 my 25\n+3\nq\n")
    assert r2.returncode == 0, r2.stderr
    assert "ONLINE LEAGUE: Kansas City Chiefs vs Indianapolis Colts (logged as online_indianapolis_colts" in r2.stdout

    con = sqlite3.connect(tmp_db / "madden27.db")
    try:
        rows = con.execute(
            "SELECT dynasty, opponent_id, league, user_team, opponent_team FROM game_sessions "
            "ORDER BY started_ts"
        ).fetchall()
        assert rows == [
            ("titans", "titans_indianapolis_colts", "titans", TEN, COLTS),
            ("online", "online_indianapolis_colts", "online", KC, COLTS),
        ]
        sticky = con.execute("SELECT value FROM meta WHERE key = 'franchise_profile'").fetchall()
        assert ("titans",) not in sticky and ("online",) not in sticky
    finally:
        con.close()

    inv_on = _cli(tmp_db, "league", "--game", "madden27")
    assert inv_on.returncode == 0, inv_on.stderr
    assert "Online league — you: Kansas City Chiefs" in inv_on.stdout
    bills = [ln for ln in inv_on.stdout.splitlines() if ln.startswith("Indianapolis Colts")]
    assert len(bills) == 1 and bills[0][24:].split()[0] == "1"  # one Chiefs game only

    inv_t = _cli(tmp_db, "league", "--game", "madden27", "--franchise", "titans")
    assert inv_t.returncode == 0, inv_t.stderr
    assert "Titans franchise — you: Tennessee Titans" in inv_t.stdout
    tlines = [ln for ln in inv_t.stdout.splitlines() if ln.startswith("Indianapolis Colts")]
    assert len(tlines) == 1 and tlines[0][24:].split()[0] == "1"


def test_logged_summary_does_not_mix_chiefs_and_titans(tmp_path):
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import data as mdata

    db = CoachDB(tmp_path / "m.db", seed=mdata.load_seed())
    try:
        register_cold_start(db, "online_indianapolis_colts", display_name=COLTS)
        register_cold_start(db, "titans_indianapolis_colts", display_name=COLTS)
        ctx_c = ol.league_context(db, "online_indianapolis_colts", COLTS, cpu=False, user=KC, league="online")
        sid_c = start_session("online_indianapolis_colts", dynasty="online", db=db, notes="c", league=ctx_c)
        db.log_snap(opponent_id="online_indianapolis_colts", side="offense", situation_raw="1&10",
                    our_call="x", session_id=sid_c.session_id, down=1, distance=10, result="+6")
        ctx_t = ol.league_context(db, "titans_indianapolis_colts", COLTS, cpu=False, user=TEN, league="titans")
        sid_t = start_session("titans_indianapolis_colts", dynasty="titans", db=db, notes="t", league=ctx_t)
        db.log_snap(opponent_id="titans_indianapolis_colts", side="offense", situation_raw="1&10",
                    our_call="x", session_id=sid_t.session_id, down=1, distance=10, result="+3")
        chiefs = ol.logged_summary(db, COLTS, KC, league="online")
        titans = ol.logged_summary(db, COLTS, TEN, league="titans")
        assert chiefs["games"] == 1 and titans["games"] == 1
        assert chiefs["snaps"] == 1 and titans["snaps"] == 1
    finally:
        db.close()


def _insert_incomplete_refile_opponent(db_path, oid, display_name):
    """Reproduce the pre-fix refile insert: cold_start but not madden_only."""
    import json
    con = sqlite3.connect(db_path)
    try:
        profile = {
            "id": oid,
            "display_name": display_name,
            "skill": "human",
            "confidence": "cold",
            "cold_start": True,
        }
        con.execute(
            "INSERT INTO opponents (id, display_name, team_now, skill, confidence, profile_json) "
            "VALUES (?, ?, ?, 'human', 'cold', ?)",
            (oid, display_name, display_name, json.dumps(profile)),
        )
        con.commit()
    finally:
        con.close()


def test_refile_team_opponent_row_is_accepted_on_play(tmp_db):
    """A refile-created team opponent (missing madden_only) must not block play."""
    from cfb_coach.madden.human_opponents import is_cold_start_profile

    colts = "Indianapolis Colts"
    _insert_incomplete_refile_opponent(tmp_db / "madden27.db", "titans_indianapolis_colts", colts)
    _insert_incomplete_refile_opponent(tmp_db / "madden27.db", "online_indianapolis_colts", colts)

    r = _play(tmp_db, "--franchise", "titans", "--opponent-team", colts, stdin="q\n")
    assert r.returncode == 0, r.stderr
    assert "TITANS FRANCHISE: Tennessee Titans vs Indianapolis Colts" in r.stdout
    assert "Cannot register opponent team" not in r.stderr

    r2 = _play(tmp_db, "--franchise", "online", "--opponent-team", colts, stdin="q\n")
    assert r2.returncode == 0, r2.stderr
    assert "ONLINE LEAGUE: Kansas City Chiefs vs Indianapolis Colts" in r2.stdout
    assert "Cannot register opponent team" not in r2.stderr

    con = sqlite3.connect(tmp_db / "madden27.db")
    try:
        import json
        for oid in ("titans_indianapolis_colts", "online_indianapolis_colts"):
            raw = json.loads(con.execute("SELECT profile_json FROM opponents WHERE id=?", (oid,)).fetchone()[0])
            assert is_cold_start_profile(raw), (oid, raw)
            assert raw["madden_only"] is True and raw["cold_start"] is True
    finally:
        con.close()
