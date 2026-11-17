"""t455u/t456u: Giants franchise uses the Lions/CPU book and a 3-snap offensive play cooldown."""
from __future__ import annotations

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
from tests.test_titans_play_cooldown import MESH_IDS, _giants, _lions, _online, _titans

import hashlib


def _verify_lions_book(db):
    _verify_all(db)
    for vid in set(CAT_LIONS.ids) - set(CAT.ids):
        lg.record_verified(db, CAT_LIONS, vid)


def test_shared_catalog_and_additions_hash_are_unchanged():
    raw = (sc.repo_root() / sc.ADDITIONS_RELPATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHARED_ADDITIONS_SHA256
    assert CAT.version() == CAT_LIONS.version() == "lions_civil_bucs_5wr_v4@15c120decc4b"
    assert len(CAT.variants) == 49 and len(CAT_LIONS.variants) == sc.EXPECTED_LIONS_CPU_VARIANTS == 128
    assert sc.contract_errors(CAT) == [] and sc.contract_errors(CAT_LIONS) == []


def test_giants_catalog_is_lions_cpu_and_cooldown_is_offense_3(tmp_path):
    db = _db(tmp_path, formations=_forms())
    _giants(db, "g-route")
    sit = _sit(1, 10, 40, session="g-route")
    assert catalog_for_call(db, sit).book_id == sc.LIONS_CPU_BOOK
    assert pcd.is_giants_session(db, "g-route")
    assert not pcd.is_titans_session(db, "g-route")
    assert not pcd.is_online_session(db, "g-route")
    assert pcd.cooldown_spec(db, "g-route", side="offense") == {
        "window": 3, "league": "Giants", "league_key": "giants", "side": "offense",
    }
    assert pcd.cooldown_spec(db, "g-route", side="defense") is None


def test_giants_mesh_call_blocks_family_for_three_snaps(tmp_path, civil_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path, formations=_forms())
    _verify_lions_book(db)
    _giants(db, "gmesh")
    first = playcaller.make_call(_sit(1, 10, 40, session="gmesh"), "giants_indianapolis_colts", db)
    assert first.scheme_variant["status"] == "CALL"
    _close_snap(db, "gmesh", "civil_bucs_mesh_2_provisional", 40)

    for i in range(3):
        call = playcaller.make_call(_sit(1, 10, 35 + i, session="gmesh"), "giants_indianapolis_colts", db)
        vid = call.scheme_variant["variant_id"]
        assert vid not in MESH_IDS, (i, vid, call.scheme_variant.get("play_cooldown"))
        cd = call.scheme_variant.get("play_cooldown") or {}
        assert cd.get("active") is True and cd.get("window") == 3 and cd.get("league") == "Giants"
        assert cd.get("fallback") is False
        _close_snap(db, "gmesh", vid, 35 + i)

    fourth = playcaller.make_call(_sit(1, 10, 40, session="gmesh"), "giants_indianapolis_colts", db)
    assert any(v in MESH_IDS for v in fourth.scheme_variant["candidates"])


def test_giants_600_snap_sim_and_other_leagues_unchanged(tmp_path, civil_env, ravens_env):
    db = _db(tmp_path, formations=_forms())
    _verify_lions_book(db)
    _giants(db, "giants-sim")
    _lions(db, "lions-sim")
    _online(db, "chiefs-sim")
    _titans(db, "titans-sim")

    giants = _simulate(db, "giants-sim", "giants_indianapolis_colts")
    lions = _simulate(db, "lions-sim", "jaxon")
    cpu = _simulate(db, "cpu-sim", "cpu")
    chiefs = _simulate(db, "chiefs-sim", "online_indianapolis_colts")
    titans = _simulate(db, "titans-sim", "titans_indianapolis_colts")

    lions_only = set(CAT_LIONS.ids) - set(CAT.ids)

    for label, rec in (("giants", giants), ("lions", lions), ("cpu", cpu)):
        assert rec["statuses"].count("CALL") == SNAPS and lg.NO_LEGAL not in rec["statuses"], label
        assert rec["books"] == {sc.LIONS_CPU_BOOK}, (label, rec["books"])
        assert set(rec["counts"]) <= set(CAT_LIONS.ids)
        assert set(rec["counts"]) & lions_only, (label, "expected Lions-only plays")

    # Giants unique-play count should be close to the 128-variant book.
    g_unique = len(giants["counts"])
    assert g_unique <= 128
    assert g_unique >= 90, g_unique
    # Lions keep no cooldown; Giants use the 3-snap offensive family cooldown.
    assert all(not (cd or {}).get("active") for cd in lions["cooldowns"])
    assert all(not (cd or {}).get("active") for cd in cpu["cooldowns"])
    active = [cd for cd in giants["cooldowns"] if cd.get("active")]
    assert active and all(cd.get("window") == 3 and cd.get("league") == "Giants" for cd in active)

    for label, rec, window, league in (
        ("chiefs", chiefs, 3, "Chiefs"),
        ("titans", titans, 5, "Titans"),
    ):
        assert rec["statuses"].count("CALL") == SNAPS and lg.NO_LEGAL not in rec["statuses"], label
        assert rec["books"] == {sc.SHARED_BOOK}, (label, rec["books"])
        assert set(rec["counts"]).isdisjoint(lions_only), (label, set(rec["counts"]) & lions_only)
        assert set(rec["counts"]) <= set(CAT.ids)
        assert len(rec["counts"]) <= 49
        active = [cd for cd in rec["cooldowns"] if cd.get("active")]
        assert active and all(cd.get("window") == window and cd.get("league") == league for cd in active), label

    # Publish offense sim counts before the defense loop so a later assert still leaves a record.
    summary = {
        "giants_unique": g_unique,
        "giants_no_legal": giants["statuses"].count(lg.NO_LEGAL),
        "lions_unique": len(lions["counts"]),
        "lions_no_legal": lions["statuses"].count(lg.NO_LEGAL),
        "cpu_unique": len(cpu["counts"]),
        "cpu_no_legal": cpu["statuses"].count(lg.NO_LEGAL),
        "chiefs_unique": len(chiefs["counts"]),
        "chiefs_no_legal": chiefs["statuses"].count(lg.NO_LEGAL),
        "titans_unique": len(titans["counts"]),
        "titans_no_legal": titans["statuses"].count(lg.NO_LEGAL),
        "giants_lions_only": sum(giants["counts"][vid] for vid in lions_only),
        "chiefs_lions_only": len(set(chiefs["counts"]) & lions_only),
        "titans_lions_only": len(set(titans["counts"]) & lions_only),
    }
    assert summary["chiefs_unique"] <= 49 and summary["titans_unique"] <= 49, summary
    print("GIANTS_SIM_SUMMARY", summary)

    # Giants defense calls NSM and has no play-family cooldown.
    from cfb_coach.madden import playcaller

    nsm_ids = set(rc.NSM_CALL_IDS)
    seen, forms = set(), set()
    for i in range(80):
        s = _sit(1, 10, 40, session="giants-sim")
        s.side = "defense"
        s.extras = {"session_id": "giants-sim", "snap_seq": i}
        call = playcaller.make_call(s, "giants_indianapolis_colts", db)
        rd = call.ravens_defense
        assert rd["status"] == "CALL", rd
        assert pcd.cooldown_spec(db, "giants-sim", side="defense") is None
        assert not (rd.get("play_cooldown") or {}).get("active")
        assert call.formation in rc.LIONS_FRANCHISE_FORMATIONS
        seen.add(rd["candidate"]["call_id"])
        forms.add(call.formation)
    assert seen & nsm_ids, seen
    assert rc.NSM in forms, forms
    summary["giants_nsm"] = True
    summary["giants_def_forms"] = sorted(forms)
    print("GIANTS_DEFENSE_SUMMARY", {"nsm": True, "forms": sorted(forms), "calls": len(seen)})
