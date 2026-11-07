"""Real previous-play form records feed the NEXT prediction, never current labels."""
import sqlite3

import pytest
from cfb_coach.madden.model import offense_live_coverage as oc


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE snaps(id INTEGER PRIMARY KEY,session_id TEXT,opponent_id TEXT,side TEXT,snap_seq INT,result TEXT,coverage_seen TEXT)")
    return conn


def add(c, seq, coverage, result="+5", session="game_a", opp="cpu", side="offense"):
    c.execute("INSERT INTO snaps(session_id,opponent_id,side,snap_seq,result,coverage_seen) VALUES(?,?,?,?,?,?)",
              (session, opp, side, seq, result, coverage))


def test_repeated_manual_cover2_man_affects_next_snap(db):
    prior = {"unknown": 1.0}
    add(db,1,"Cover 2 Man")
    first=oc.coverage_for_next_snap(db,session_id="game_a",opponent_id="cpu",next_snap_seq=2,prior_mix=prior)
    assert first.named_n == 1 and first.mix["cover_2_man"] == pytest.approx(1/9)
    for j in range(2,7):
        add(db,j,"Cover 2 Man")
    last=oc.coverage_for_next_snap(db,session_id="game_a",opponent_id="cpu",next_snap_seq=7,prior_mix=prior)
    assert last.mix["cover_2_man"] > first.mix["cover_2_man"]
    assert last.mix["unknown"] < first.mix["unknown"]


def test_current_and_future_results_are_never_used(db):
    add(db,1,"Cover 4 Palms")
    add(db,2,"Tampa 2")
    add(db,3,"Cover 2 Man")
    r=oc.coverage_for_next_snap(db,session_id="game_a",opponent_id="cpu",next_snap_seq=2)
    assert r.named_n == 1 and r.recent_named == ("cover_4_palms",)
    assert "tampa_2" not in r.mix and "cover_2_man" not in r.mix


def test_other_sessions_and_opponents_and_our_defense_are_excluded(db):
    add(db,1,"Cover 2 Man")
    add(db,2,"Cover 6",session="other")
    add(db,3,"Cover 6",opp="John")
    add(db,4,"Cover 3 Match",side="defense")
    add(db,5,"Cover 4 Quarters",result="")
    x=oc.coverage_for_next_snap(db,session_id="game_a",opponent_id="cpu",next_snap_seq=6)
    assert x.named_n == 1 and x.recent_named == ("cover_2_man",)


def test_pressure_and_ambiguous_cover2_are_not_invented_cover0_or_spot(db):
    add(db,1,"pressure")
    add(db,2,"Cover 2")
    x=oc.coverage_for_next_snap(db,session_id="game_a",opponent_id="cpu",next_snap_seq=3,
                                 prior_mix={"pressure":0.75,"cover3":0.25})
    assert x.named_n == 1 and x.pressure_n == 1
    assert "cover_0" not in x.mix and "cover_2_spot" not in x.mix
    assert x.mix["cover2"] > 0
    assert x.mix["unknown"] > 0


def test_empty_history_retains_historical_prior(db):
    x=oc.coverage_for_next_snap(db,session_id="game_a",opponent_id="cpu",next_snap_seq=1,
                                 prior_mix={"cover_3_match":0.8,"cover_4_palms":0.2})
    assert x.mix=={"cover_3_match":0.8,"cover_4_palms":0.2}
    assert x.named_n == 0


def test_no_session_or_bad_seq_fails_closed(db):
    for kw in ({"session_id":"", "opponent_id":"cpu", "next_snap_seq":2},
               {"session_id":"s", "opponent_id":"", "next_snap_seq":2},
               {"session_id":"s", "opponent_id":"cpu", "next_snap_seq":0}):
        with pytest.raises(ValueError):oc.coverage_for_next_snap(db,**kw)