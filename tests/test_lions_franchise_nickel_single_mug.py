"""t462u: live defense is Patriots 6-1 + Nickel 2-4 Wide + NSM; the Ravens file stays for rollback.

The Ravens two-formation dataset bytes and the default 14-candidate ``rc.load()`` contract
stay unchanged. The Steelers+NSM ``load_lions_franchise()`` loader also remains, unused.
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


def test_session_routing_uses_patriots_61_in_every_league(tmp_path, civil_env):
    db = _db(tmp_path, formations=["Gun Bunch TE"])
    _lions(db, "lions-cpu")
    _online(db, "online1")
    _titans(db, "titans1")
    from tests.test_titans_play_cooldown import _giants
    _giants(db, "giants1")
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
        ("giants1", "giants_indianapolis_colts"),
        ("g1", "jaxon"),  # untagged
    )
    base = rc.load()
    for sid, opp in sessions:
        cat = rc.catalog_for_defense_session(db, sid, {}, opp)
        assert cat.sha256 != base.sha256
        assert cat.stock_team == rc.PATRIOTS and cat.uses_prior_only
        assert cat.allowed_formations == rc.SIXTY_ONE_FORMATIONS
        assert NSM_IDS.isdisjoint(cat.by_id())
        assert set(rc.EVEN_61_CALL_IDS) <= set(cat.by_id())
        assert cat.raw.get("source_defensive_playbook") == "New England Patriots"
    # Unused Steelers+NSM catalog still loads and is distinct from the live book.
    unused = rc.load_lions_franchise()
    assert unused.stock_team == rc.STEELERS and NSM_IDS <= set(unused.by_id())
    assert unused.sha256 != cat.sha256


def test_every_league_defense_calls_stay_on_patriots_61_formations(tmp_path, civil_env, ravens_env):
    from cfb_coach.madden import playcaller
    from cfb_coach.situation import Situation
    from tests.test_titans_play_cooldown import _giants

    db = _db(tmp_path, formations=["Gun Bunch TE"])
    _lions(db, "lions-cpu")
    _online(db, "online1")
    _titans(db, "titans1")
    _giants(db, "giants1")
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
        ("giants1", "giants_indianapolis_colts", False),
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
            assert call.formation in rc.SIXTY_ONE_FORMATIONS
            assert call.formation != rc.DIME
            text = "\n".join(rd["card"])
            cid = rd["candidate"]["call_id"]
            assert cid != "dime_dbfire2" and cid not in NSM_IDS
            seen.add(cid)
            forms.add(call.formation)
        assert seen and "dime_dbfire2" not in seen and seen.isdisjoint(NSM_IDS)
        assert forms <= set(rc.SIXTY_ONE_FORMATIONS) and rc.DIME not in forms and rc.NSM not in forms
