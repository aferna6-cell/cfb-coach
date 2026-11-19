"""t473u/t474u: Bucs franchise uses the shared Bucs civil book and has no play cooldown."""
from __future__ import annotations

import hashlib

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import legality as lg
from cfb_coach.madden.scheme import play_cooldown as pcd
from cfb_coach.madden.scheme.coordinator import catalog_for_call
from tests.test_chiefs_play_cooldown import ravens_env  # noqa: F401
from tests.test_civil_scheme_only import (
    CAT,
    CAT_LIONS,
    _close_snap,
    _db,
    _sit,
    _verify_all,
    civil_env,  # noqa: F401
)
from tests.test_lions_cpu_book_t445u import SNAPS, _forms, _simulate
from tests.test_lions_cpu_book_t450u import SHARED_ADDITIONS_SHA256
from tests.test_titans_play_cooldown import MESH_IDS, _bucs, _giants, _lions, _online, _titans


def _verify_lions_book(db):
    _verify_all(db)
    for vid in set(CAT_LIONS.ids) - set(CAT.ids):
        lg.record_verified(db, CAT_LIONS, vid)


def test_shared_catalog_and_additions_hash_are_unchanged():
    raw = (sc.repo_root() / sc.ADDITIONS_RELPATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHARED_ADDITIONS_SHA256
    assert CAT.version() == CAT_LIONS.version() == "lions_civil_bucs_5wr_v4@15c120decc4b"
    assert CAT.book_id == sc.SHARED_BOOK
    assert sc.STOCK_BOOK == "Buccaneers"
    assert len(CAT.variants) == 49 and len(CAT_LIONS.variants) == sc.EXPECTED_LIONS_CPU_VARIANTS == 129
    assert sc.contract_errors(CAT) == [] and sc.contract_errors(CAT_LIONS) == []


def test_bucs_catalog_is_shared_book_and_has_no_cooldown(tmp_path):
    db = _db(tmp_path, formations=_forms())
    _bucs(db, "b-route")
    sit = _sit(1, 10, 40, session="b-route")
    assert catalog_for_call(db, sit).book_id == sc.SHARED_BOOK
    assert len(catalog_for_call(db, sit).variants) == 49
    assert pcd.is_bucs_session(db, "b-route")
    assert not pcd.is_titans_session(db, "b-route")
    assert not pcd.is_online_session(db, "b-route")
    assert not pcd.is_giants_session(db, "b-route")
    assert pcd.cooldown_spec(db, "b-route", side="offense") is None
    assert pcd.cooldown_spec(db, "b-route", side="defense") is None

    # Other leagues keep their existing books and cooldowns.
    _giants(db, "g-route")
    _online(db, "o-route")
    _titans(db, "t-route")
    _lions(db, "l-route")
    assert catalog_for_call(db, _sit(1, 10, 40, session="g-route")).book_id == sc.LIONS_CPU_BOOK
    assert catalog_for_call(db, _sit(1, 10, 40, session="l-route")).book_id == sc.LIONS_CPU_BOOK
    assert catalog_for_call(db, _sit(1, 10, 40, session="o-route")).book_id == sc.SHARED_BOOK
    assert catalog_for_call(db, _sit(1, 10, 40, session="t-route")).book_id == sc.SHARED_BOOK
    assert pcd.cooldown_spec(db, "g-route", side="offense")["window"] == 3
    assert pcd.cooldown_spec(db, "o-route", side="offense")["window"] == 3
    assert pcd.cooldown_spec(db, "t-route", side="offense")["window"] == 5
    assert pcd.cooldown_spec(db, "l-route", side="offense") is None


def test_bucs_mesh_family_is_not_blocked_after_a_call(tmp_path, civil_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path, formations=_forms())
    _verify_all(db)
    _bucs(db, "bmesh")
    first = playcaller.make_call(_sit(1, 10, 40, session="bmesh"), "bucs_indianapolis_colts", db)
    assert first.scheme_variant["status"] == "CALL"
    assert first.scheme_variant["catalog_book"] == sc.SHARED_BOOK
    _close_snap(db, "bmesh", "civil_bucs_mesh_2_provisional", 40)

    second = playcaller.make_call(_sit(1, 10, 35, session="bmesh"), "bucs_indianapolis_colts", db)
    assert second.scheme_variant["status"] == "CALL"
    cd = second.scheme_variant.get("play_cooldown") or {}
    assert not cd.get("active")
    assert any(v in MESH_IDS for v in second.scheme_variant["candidates"])


def test_bucs_600_snap_sim(tmp_path, civil_env, ravens_env):
    db = _db(tmp_path, formations=_forms())
    _verify_lions_book(db)
    _bucs(db, "bucs-sim")

    bucs = _simulate(db, "bucs-sim", "bucs_dallas_cowboys")
    lions_only = set(CAT_LIONS.ids) - set(CAT.ids)

    assert bucs["statuses"].count("CALL") == SNAPS and lg.NO_LEGAL not in bucs["statuses"]
    assert bucs["books"] == {sc.SHARED_BOOK}, bucs["books"]
    assert set(bucs["counts"]) <= set(CAT.ids)
    assert set(bucs["counts"]).isdisjoint(lions_only)
    assert len(bucs["counts"]) == 49, len(bucs["counts"])
    assert all(not (cd or {}).get("active") for cd in bucs["cooldowns"])

    summary = {
        "bucs_unique": len(bucs["counts"]),
        "bucs_no_legal": bucs["statuses"].count(lg.NO_LEGAL),
        "bucs_lions_only": len(set(bucs["counts"]) & lions_only),
        "book": sorted(bucs["books"]),
    }
    print("OFFENSE_SIM_SUMMARY", summary)

    from cfb_coach.madden import playcaller

    nsm_ids = set(rc.NSM_CALL_IDS)
    seen, forms, statuses = set(), set(), []
    for i in range(SNAPS):
        s = _sit(1 + i % 3, 5 + i % 8, 30 + i % 40, session="bucs-sim")
        s.side = "defense"
        s.extras = {"session_id": "bucs-sim", "snap_seq": i}
        call = playcaller.make_call(s, "bucs_dallas_cowboys", db, cpu_defense=False)
        rd = call.ravens_defense
        statuses.append(rd["status"])
        assert rd["status"] == "CALL", rd
        assert call.formation in rc.SIXTY_ONE_FORMATIONS
        assert call.formation != rc.DIME
        cid = rd["candidate"]["call_id"]
        assert cid != "dime_dbfire2" and cid not in nsm_ids
        assert not (rd.get("play_cooldown") or {}).get("active")
        assert pcd.cooldown_spec(db, "bucs-sim", side="defense") is None
        seen.add(cid)
        forms.add(call.formation)
    assert "NO_LEGAL_RAVENS_CALL" not in statuses and statuses.count("CALL") == SNAPS
    assert "dime_dbfire2" not in seen and seen.isdisjoint(nsm_ids), seen
    assert rc.DIME not in forms and rc.NSM not in forms, forms
    assert forms <= set(rc.SIXTY_ONE_FORMATIONS), forms
    defense_summary = {
        "unique_calls": len(seen), "forms": sorted(forms), "no_legal": 0,
        "nsm": 0, "even61": sum(1 for x in seen if x.startswith("even61_")),
    }
    print("DEFENSE_SIM_SUMMARY", defense_summary)
    summary["defense"] = defense_summary
    print("BUCS_SIM_SUMMARY", summary)
