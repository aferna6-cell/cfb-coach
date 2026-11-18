"""Giants franchise (t455u/t456u): team-keyed logging, Lions/CPU book, isolated from other leagues."""
from __future__ import annotations

import sqlite3

from cfb_coach.madden import franchise
from cfb_coach.madden import online_league as ol
from cfb_coach.madden.human_opponents import register_cold_start
from cfb_coach.session import start_session

from tests.madden_ml.test_online_civil_profile import _play, tmp_db  # noqa: F401
from tests.madden_ml.test_online_league import KC, _cli
from tests.madden_ml.test_titans_franchise import COLTS, TEN

NYG = "New York Giants"


def test_giants_profile_is_never_sticky_and_separate_from_other_leagues(tmp_path):
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import data as mdata

    assert franchise.normalize_profile("new_york_giants") == franchise.GIANTS
    assert franchise.team_for_profile("giants") == NYG
    assert franchise.is_team_keyed_profile("giants")
    spec = franchise.team_keyed_spec("giants")
    assert spec["league"] == "giants" and spec["id_prefix"] == "giants_"
    assert ol.team_opponent_id(COLTS, prefix=spec["id_prefix"]) == "giants_indianapolis_colts"
    assert ol.team_opponent_id(COLTS) == "online_indianapolis_colts"
    assert ol.team_opponent_id(COLTS, prefix="titans_") == "titans_indianapolis_colts"
    assert ol.is_team_opponent_id("giants_indianapolis_colts")
    db = CoachDB(tmp_path / "m.db", seed=mdata.load_seed())
    try:
        franchise.set_session_profile(db, "primary")
        assert franchise.set_session_profile(db, "giants") == "giants"
        assert franchise.get_session_profile(db) == "primary"
        db.set_meta(franchise.META_KEY, "giants")
        assert franchise.get_session_profile(db) == "primary"
    finally:
        db.close()


def test_giants_rejects_giants_and_requires_opponent_team(tmp_db):
    r = _play(tmp_db, "--franchise", "giants", "--opponent-team", "NYG")
    assert r.returncode == 2 and "your team" in r.stderr
    r = _play(tmp_db, "--franchise", "giants")
    assert r.returncode == 2 and "--opponent-team is required" in r.stderr
    r = _play(tmp_db, "--franchise", "giants", "--user-team", TEN, "--opponent-team", "IND")
    assert r.returncode == 2 and "does not match" in r.stderr
    r = _play(tmp_db, "--franchise", "online", "--user-team", NYG, "--opponent-team", "IND")
    assert r.returncode == 2 and "does not match" in r.stderr


def test_giants_colts_start_civil_and_stay_separate(tmp_db):
    r = _play(tmp_db, "--franchise", "giants", "--opponent-team", "Indianapolis Colts",
              stdin="d 3&7 opp 40 trips\n+4 mesh\n1&10 my 25\n+3\nq\n")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "OFFENSE CONTROLLER: CIVIL_SCHEME_ONLY" in out
    assert "DEFENSE CONTROLLER: PATRIOTS_EVEN_61_CIVIL" in out
    assert "GIANTS FRANCHISE: New York Giants vs Indianapolis Colts (logged as giants_indianapolis_colts" in out
    assert "ONLINE LEAGUE:" not in out
    assert "TITANS FRANCHISE:" not in out

    r2 = _play(tmp_db, "--franchise", "online", "--opponent-team", "Indianapolis Colts",
               stdin="1&10 my 25\n+3\nq\n")
    assert r2.returncode == 0, r2.stderr
    assert "ONLINE LEAGUE: Kansas City Chiefs vs Indianapolis Colts (logged as online_indianapolis_colts" in r2.stdout

    r3 = _play(tmp_db, "--franchise", "titans", "--opponent-team", "Indianapolis Colts",
               stdin="1&10 my 25\n+3\nq\n")
    assert r3.returncode == 0, r3.stderr
    assert "TITANS FRANCHISE: Tennessee Titans vs Indianapolis Colts (logged as titans_indianapolis_colts" in r3.stdout

    con = sqlite3.connect(tmp_db / "madden27.db")
    try:
        rows = con.execute(
            "SELECT dynasty, opponent_id, league, user_team, opponent_team FROM game_sessions "
            "ORDER BY started_ts"
        ).fetchall()
        assert rows == [
            ("giants", "giants_indianapolis_colts", "giants", NYG, COLTS),
            ("online", "online_indianapolis_colts", "online", KC, COLTS),
            ("titans", "titans_indianapolis_colts", "titans", TEN, COLTS),
        ]
        sticky = con.execute("SELECT value FROM meta WHERE key = 'franchise_profile'").fetchall()
        assert ("giants",) not in sticky and ("titans",) not in sticky and ("online",) not in sticky
    finally:
        con.close()

    inv_on = _cli(tmp_db, "league", "--game", "madden27")
    assert inv_on.returncode == 0, inv_on.stderr
    assert "Online league — you: Kansas City Chiefs" in inv_on.stdout
    bills = [ln for ln in inv_on.stdout.splitlines() if ln.startswith("Indianapolis Colts")]
    assert len(bills) == 1 and bills[0][24:].split()[0] == "1"

    inv_t = _cli(tmp_db, "league", "--game", "madden27", "--franchise", "titans")
    assert inv_t.returncode == 0, inv_t.stderr
    assert "Titans franchise — you: Tennessee Titans" in inv_t.stdout
    tlines = [ln for ln in inv_t.stdout.splitlines() if ln.startswith("Indianapolis Colts")]
    assert len(tlines) == 1 and tlines[0][24:].split()[0] == "1"

    inv_g = _cli(tmp_db, "league", "--game", "madden27", "--franchise", "giants")
    assert inv_g.returncode == 0, inv_g.stderr
    assert "Giants franchise — you: New York Giants" in inv_g.stdout
    glines = [ln for ln in inv_g.stdout.splitlines() if ln.startswith("Indianapolis Colts")]
    assert len(glines) == 1 and glines[0][24:].split()[0] == "1"


def test_logged_summary_does_not_mix_giants_with_chiefs_or_titans(tmp_path):
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import data as mdata

    db = CoachDB(tmp_path / "m.db", seed=mdata.load_seed())
    try:
        register_cold_start(db, "online_indianapolis_colts", display_name=COLTS)
        register_cold_start(db, "titans_indianapolis_colts", display_name=COLTS)
        register_cold_start(db, "giants_indianapolis_colts", display_name=COLTS)
        ctx_c = ol.league_context(db, "online_indianapolis_colts", COLTS, cpu=False, user=KC, league="online")
        sid_c = start_session("online_indianapolis_colts", dynasty="online", db=db, notes="c", league=ctx_c)
        db.log_snap(opponent_id="online_indianapolis_colts", side="offense", situation_raw="1&10",
                    our_call="x", session_id=sid_c.session_id, down=1, distance=10, result="+6")
        ctx_t = ol.league_context(db, "titans_indianapolis_colts", COLTS, cpu=False, user=TEN, league="titans")
        sid_t = start_session("titans_indianapolis_colts", dynasty="titans", db=db, notes="t", league=ctx_t)
        db.log_snap(opponent_id="titans_indianapolis_colts", side="offense", situation_raw="1&10",
                    our_call="x", session_id=sid_t.session_id, down=1, distance=10, result="+3")
        ctx_g = ol.league_context(db, "giants_indianapolis_colts", COLTS, cpu=False, user=NYG, league="giants")
        sid_g = start_session("giants_indianapolis_colts", dynasty="giants", db=db, notes="g", league=ctx_g)
        db.log_snap(opponent_id="giants_indianapolis_colts", side="offense", situation_raw="1&10",
                    our_call="x", session_id=sid_g.session_id, down=1, distance=10, result="+4")
        chiefs = ol.logged_summary(db, COLTS, KC, league="online")
        titans = ol.logged_summary(db, COLTS, TEN, league="titans")
        giants = ol.logged_summary(db, COLTS, NYG, league="giants")
        assert chiefs["games"] == 1 and titans["games"] == 1 and giants["games"] == 1
        assert chiefs["snaps"] == 1 and titans["snaps"] == 1 and giants["snaps"] == 1
    finally:
        db.close()
