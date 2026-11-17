"""t449u/t451u: every league uses the Steelers book with the Nickel Single Mug package.

The Ravens two-formation dataset bytes and the default 14-candidate ``rc.load()`` contract
stay unchanged. Live civil defense calls use ``load_lions_franchise()`` for Lions, Chiefs,
Titans, and CPU.
"""
from __future__ import annotations

import hashlib

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense import legality as lg
from cfb_coach.madden.ravens_defense.confirmations import Confirmations
from tests.test_civil_scheme_only import _db, civil_env  # noqa: F401
from tests.test_chiefs_play_cooldown import ravens_env  # noqa: F401
from tests.test_titans_play_cooldown import _lions, _online, _titans

DATASET_SHA256 = "5e0673b91112b272b19e715c1153bfb04bf9f109cc66e43cddfe1e7c2b960788"
NSM_IDS = set(rc.NSM_CALL_IDS)


def _conf():
    return Confirmations(path="test", installed_book_fingerprint=None, macros={}, coaching_profile={})


def test_steelers_is_the_only_stock_book_with_all_three_formations():
    assert rc.books_with_current_formations_and_nsm() == ["Steelers"]
    steel = rc.steelers_stock_book()
    assert set(steel) == set(rc.LIONS_FRANCHISE_FORMATIONS)
    # Every current 12-call base play exists in Steelers too.
    base = rc.load()
    for c in base.candidates:
        assert c.base_play in steel[c.formation], (c.call_id, c.formation, c.base_play)


def test_default_ravens_catalog_is_unchanged():
    raw = rc.DATASET.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATASET_SHA256
    cat = rc.load()
    assert rc.contract_errors(cat.raw) == []
    assert cat.sha256 == DATASET_SHA256
    assert len(cat.candidates) == 14 and set(cat.calls) == set(rc.APPROVED_CALL_IDS)
    assert cat.allowed_formations == rc.FORMATIONS and not cat.is_lions_franchise
    assert {c.formation for c in cat.candidates} == {rc.NICKEL, rc.DIME}
    assert NSM_IDS.isdisjoint(cat.by_id())


def test_lions_franchise_catalog_adds_pdf_stock_nsm_plays_only():
    cat = rc.load_lions_franchise()
    assert cat.is_lions_franchise and cat.stock_team == rc.STEELERS
    assert cat.allowed_formations == rc.LIONS_FRANCHISE_FORMATIONS
    assert len(cat.candidates) == 14 + 7
    assert NSM_IDS <= set(cat.by_id())
    assert all(rc.stock_catalog_check(cat).values())
    plays = {c.base_play for c in cat.candidates if c.formation == rc.NSM}
    assert plays == {"Cover 4 Quarters", "Tampa 2"}
    assert "Cover 9" not in plays and "Cover 9 Show 2" not in plays
    omitted = {p["pages"] for p in cat.raw["omitted_pdf_pages"]}
    assert omitted == {"p08_bot", "p09_bot", "p10_top"}
    legal, verdicts = lg.legal_set(cat, installed=rc.scheme_book(cat), conf=_conf(), scheme_book=True)
    assert {c.candidate_id for c in legal} == set(cat.by_id())
    for cid in NSM_IDS:
        notes = " ".join(verdicts[cid].notes)
        assert "UNCONFIRMED" in notes or "PROFILE_UNCONFIRMED" in notes


def test_session_routing_nsm_is_used_in_every_league(tmp_path, civil_env):
    db = _db(tmp_path, formations=["Gun Bunch TE"])
    _lions(db, "lions-cpu")
    _online(db, "online1")
    _titans(db, "titans1")
    db.conn.execute(
        "INSERT INTO game_sessions (session_id, opponent_id, dynasty, started_ts, "
        "league, user_team, opponent_team, opponent_type) "
        "VALUES ('lions-human', 'jaxon', 'primary', '2026-11-17T16:00:00+00:00', "
        "NULL, 'Detroit Lions', 'Dallas Cowboys', 'human')"
    )
    db.conn.commit()

    sessions = (
        ("lions-cpu", "cpu"),
        ("lions-human", "jaxon"),
        ("online1", "online_indianapolis_colts"),
        ("titans1", "titans_indianapolis_colts"),
        ("g1", "jaxon"),  # untagged
    )
    for sid, opp in sessions:
        cat = rc.catalog_for_defense_session(db, sid, {}, opp)
        assert cat.is_lions_franchise and cat.stock_team == rc.STEELERS
        assert cat.allowed_formations == rc.LIONS_FRANCHISE_FORMATIONS
        assert NSM_IDS <= set(cat.by_id())
        assert "Baltimore Ravens" not in (cat.raw.get("source_defensive_playbook") or "") or "Steelers" in (cat.raw.get("source_defensive_playbook") or "")
        assert "Steelers" in (cat.raw.get("source_defensive_playbook") or "")


def test_every_league_defense_calls_can_be_nsm_and_stay_legal(tmp_path, civil_env, ravens_env):
    from cfb_coach.madden import playcaller
    from cfb_coach.situation import Situation

    db = _db(tmp_path, formations=["Gun Bunch TE"])
    _lions(db, "lions-cpu")
    _online(db, "online1")
    _titans(db, "titans1")
    db.conn.execute(
        "INSERT INTO game_sessions (session_id, opponent_id, dynasty, started_ts, "
        "league, user_team, opponent_team, opponent_type) "
        "VALUES ('lions-human', 'jaxon', 'primary', '2026-11-17T16:00:00+00:00', "
        "NULL, 'Detroit Lions', 'Dallas Cowboys', 'human')"
    )
    db.conn.commit()

    sessions = (
        ("lions-human", "jaxon", False),
        ("online1", "online_indianapolis_colts", False),
        ("titans1", "titans_indianapolis_colts", False),
        ("lions-cpu", "cpu", True),
    )
    for sid, opp, cpu_defense in sessions:
        seen = set()
        forms = set()
        for i in range(80):
            s = Situation(raw=f"t{i}", down=1, distance=10, yardline=40)
            s.side = "defense"
            s.extras = {"session_id": sid, "snap_seq": i}
            call = playcaller.make_call(s, opp, db, cpu_defense=cpu_defense)
            rd = call.ravens_defense
            assert rd["status"] == "CALL", (sid, rd)
            assert call.formation in rc.LIONS_FRANCHISE_FORMATIONS
            seen.add(rd["candidate"]["call_id"])
            forms.add(call.formation)
            if call.formation == rc.NSM:
                text = "\n".join(rd["card"])
                assert "PACKAGE: Nickel Single Mug" in text
                assert "UNVERIFIED" in text
                assert "SETUP 1:" in text
                assert "Lions franchise only" not in text
        assert seen & NSM_IDS, (sid, seen)
        assert seen - NSM_IDS, (sid, seen)
        assert rc.NSM in forms, (sid, forms)
