"""Bucs franchise (t473u/t474u): team-keyed logging, shared Bucs civil book, isolated from other leagues."""
from __future__ import annotations

import sqlite3

from cfb_coach.madden import franchise
from cfb_coach.madden import online_league as ol
from cfb_coach.madden.human_opponents import register_cold_start
from cfb_coach.session import start_session

from tests.madden_ml.test_giants_franchise import NYG
from tests.madden_ml.test_online_civil_profile import _play, tmp_db  # noqa: F401
from tests.madden_ml.test_online_league import KC, _cli
from tests.madden_ml.test_titans_franchise import COLTS, TEN

TB = "Tampa Bay Buccaneers"
DAL = "Dallas Cowboys"


def test_bucs_profile_is_never_sticky_and_separate_from_other_leagues(tmp_path):
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import data as mdata

    assert franchise.normalize_profile("tampa_bay_buccaneers") == franchise.BUCS
    assert franchise.normalize_profile("buccaneers") == franchise.BUCS
    assert franchise.team_for_profile("bucs") == TB
    assert franchise.is_team_keyed_profile("bucs")
    spec = franchise.team_keyed_spec("bucs")
    assert spec["league"] == "bucs" and spec["id_prefix"] == "bucs_"
    assert ol.team_opponent_id(DAL, prefix=spec["id_prefix"]) == "bucs_dallas_cowboys"
    assert ol.team_opponent_id(COLTS, prefix=spec["id_prefix"]) == "bucs_indianapolis_colts"
    assert ol.team_opponent_id(COLTS) == "online_indianapolis_colts"
    assert ol.team_opponent_id(COLTS, prefix="titans_") == "titans_indianapolis_colts"
    assert ol.team_opponent_id(COLTS, prefix="giants_") == "giants_indianapolis_colts"
    assert ol.is_team_opponent_id("bucs_dallas_cowboys")
    assert ol.league_id_for_user(TB) == "bucs"
    assert ol.id_prefix_for_user(TB) == "bucs_"
    db = CoachDB(tmp_path / "m.db", seed=mdata.load_seed())
    try:
        franchise.set_session_profile(db, "primary")
        assert franchise.set_session_profile(db, "bucs") == "bucs"
        assert franchise.get_session_profile(db) == "primary"
        db.set_meta(franchise.META_KEY, "bucs")
        assert franchise.get_session_profile(db) == "primary"
    finally:
        db.close()


def test_bucs_rejects_bucs_and_requires_opponent_team(tmp_db):
    r = _play(tmp_db, "--franchise", "bucs", "--opponent-team", "TB")
    assert r.returncode == 2 and "your team" in r.stderr
    r = _play(tmp_db, "--franchise", "bucs")
    assert r.returncode == 2 and "--opponent-team is required" in r.stderr
    r = _play(tmp_db, "--franchise", "bucs", "--user-team", TEN, "--opponent-team", "DAL")
    assert r.returncode == 2 and "does not match" in r.stderr
    r = _play(tmp_db, "--franchise", "online", "--user-team", TB, "--opponent-team", "IND")
    assert r.returncode == 2 and "does not match" in r.stderr


def test_bucs_cowboys_start_civil_and_stay_separate(tmp_db):
    r = _play(tmp_db, "--franchise", "bucs", "--opponent-team", "Dallas Cowboys",
              stdin="d 3&7 opp 40 trips\n+4 mesh\n1&10 my 25\n+3\nq\n")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "OFFENSE CONTROLLER: CIVIL_SCHEME_ONLY" in out
    assert "DEFENSE CONTROLLER: PATRIOTS_EVEN_61_CIVIL" in out
    assert "BUCS FRANCHISE: Tampa Bay Buccaneers vs Dallas Cowboys (logged as bucs_dallas_cowboys" in out
    assert "ONLINE LEAGUE:" not in out
    assert "TITANS FRANCHISE:" not in out
    assert "GIANTS FRANCHISE:" not in out

    r2 = _play(tmp_db, "--franchise", "online", "--opponent-team", "Dallas Cowboys",
               stdin="1&10 my 25\n+3\nq\n")
    assert r2.returncode == 0, r2.stderr
    assert "ONLINE LEAGUE: Kansas City Chiefs vs Dallas Cowboys (logged as online_dallas_cowboys" in r2.stdout

    r3 = _play(tmp_db, "--franchise", "titans", "--opponent-team", "Dallas Cowboys",
               stdin="1&10 my 25\n+3\nq\n")
    assert r3.returncode == 0, r3.stderr
    assert "TITANS FRANCHISE: Tennessee Titans vs Dallas Cowboys (logged as titans_dallas_cowboys" in r3.stdout

    r4 = _play(tmp_db, "--franchise", "giants", "--opponent-team", "Dallas Cowboys",
               stdin="1&10 my 25\n+3\nq\n")
    assert r4.returncode == 0, r4.stderr
    assert "GIANTS FRANCHISE: New York Giants vs Dallas Cowboys (logged as giants_dallas_cowboys" in r4.stdout

    con = sqlite3.connect(tmp_db / "madden27.db")
    try:
        rows = con.execute(
            "SELECT dynasty, opponent_id, league, user_team, opponent_team FROM game_sessions "
            "ORDER BY started_ts"
        ).fetchall()
        assert rows == [
            ("bucs", "bucs_dallas_cowboys", "bucs", TB, DAL),
            ("online", "online_dallas_cowboys", "online", KC, DAL),
            ("titans", "titans_dallas_cowboys", "titans", TEN, DAL),
            ("giants", "giants_dallas_cowboys", "giants", NYG, DAL),
        ]
        sticky = con.execute("SELECT value FROM meta WHERE key = 'franchise_profile'").fetchall()
        assert ("bucs",) not in sticky and ("giants",) not in sticky
        assert ("titans",) not in sticky and ("online",) not in sticky
    finally:
        con.close()

    inv_b = _cli(tmp_db, "league", "--game", "madden27", "--franchise", "bucs")
    assert inv_b.returncode == 0, inv_b.stderr
    assert "Bucs franchise — you: Tampa Bay Buccaneers" in inv_b.stdout
    blines = [ln for ln in inv_b.stdout.splitlines() if ln.startswith("Dallas Cowboys")]
    assert len(blines) == 1 and blines[0][24:].split()[0] == "1"

    inv_on = _cli(tmp_db, "league", "--game", "madden27")
    assert inv_on.returncode == 0, inv_on.stderr
    assert "Online league — you: Kansas City Chiefs" in inv_on.stdout
    olines = [ln for ln in inv_on.stdout.splitlines() if ln.startswith("Dallas Cowboys")]
    assert len(olines) == 1 and olines[0][24:].split()[0] == "1"


def test_logged_summary_does_not_mix_bucs_with_other_leagues(tmp_path):
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import data as mdata

    db = CoachDB(tmp_path / "m.db", seed=mdata.load_seed())
    try:
        register_cold_start(db, "online_dallas_cowboys", display_name=DAL)
        register_cold_start(db, "titans_dallas_cowboys", display_name=DAL)
        register_cold_start(db, "giants_dallas_cowboys", display_name=DAL)
        register_cold_start(db, "bucs_dallas_cowboys", display_name=DAL)
        ctx_c = ol.league_context(db, "online_dallas_cowboys", DAL, cpu=False, user=KC, league="online")
        sid_c = start_session("online_dallas_cowboys", dynasty="online", db=db, notes="c", league=ctx_c)
        db.log_snap(opponent_id="online_dallas_cowboys", side="offense", situation_raw="1&10",
                    our_call="x", session_id=sid_c.session_id, down=1, distance=10, result="+6")
        ctx_t = ol.league_context(db, "titans_dallas_cowboys", DAL, cpu=False, user=TEN, league="titans")
        sid_t = start_session("titans_dallas_cowboys", dynasty="titans", db=db, notes="t", league=ctx_t)
        db.log_snap(opponent_id="titans_dallas_cowboys", side="offense", situation_raw="1&10",
                    our_call="x", session_id=sid_t.session_id, down=1, distance=10, result="+3")
        ctx_g = ol.league_context(db, "giants_dallas_cowboys", DAL, cpu=False, user=NYG, league="giants")
        sid_g = start_session("giants_dallas_cowboys", dynasty="giants", db=db, notes="g", league=ctx_g)
        db.log_snap(opponent_id="giants_dallas_cowboys", side="offense", situation_raw="1&10",
                    our_call="x", session_id=sid_g.session_id, down=1, distance=10, result="+4")
        ctx_b = ol.league_context(db, "bucs_dallas_cowboys", DAL, cpu=False, user=TB, league="bucs")
        sid_b = start_session("bucs_dallas_cowboys", dynasty="bucs", db=db, notes="b", league=ctx_b)
        db.log_snap(opponent_id="bucs_dallas_cowboys", side="offense", situation_raw="1&10",
                    our_call="x", session_id=sid_b.session_id, down=1, distance=10, result="+5")
        chiefs = ol.logged_summary(db, DAL, KC, league="online")
        titans = ol.logged_summary(db, DAL, TEN, league="titans")
        giants = ol.logged_summary(db, DAL, NYG, league="giants")
        bucs = ol.logged_summary(db, DAL, TB, league="bucs")
        assert chiefs["games"] == 1 and titans["games"] == 1
        assert giants["games"] == 1 and bucs["games"] == 1
        assert chiefs["snaps"] == 1 and titans["snaps"] == 1
        assert giants["snaps"] == 1 and bucs["snaps"] == 1
    finally:
        db.close()
