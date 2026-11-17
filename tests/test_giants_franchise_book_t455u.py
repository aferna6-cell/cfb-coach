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

    # Unique-play counts stay on the current books.
    g_unique = len(giants["counts"])
    assert g_unique == 128, g_unique
    assert len(lions["counts"]) == 128, len(lions["counts"])
    assert len(cpu["counts"]) == 128, len(cpu["counts"])
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
    assert summary["chiefs_unique"] == 49 and summary["titans_unique"] == 49, summary
    print("OFFENSE_SIM_SUMMARY", summary)

    # 600-snap defense sims: Ravens two-formation only, 0 NSM, 0 NO_LEGAL.
    from cfb_coach.madden import playcaller

    nsm_ids = set(rc.NSM_CALL_IDS)
    defense_summary = {}
    defense_sessions = (
        ("giants-sim", "giants_indianapolis_colts", False, "Giants", False),
        ("lions-sim", "jaxon", False, "Lions", False),
        ("cpu-sim", "cpu", True, "CPU", False),
        ("chiefs-sim", "online_indianapolis_colts", False, "Chiefs", True),
        ("titans-sim", "titans_indianapolis_colts", False, "Titans", False),
    )
    for sid, opponent, cpu_defense, label, chiefs_cd in defense_sessions:
        seen, forms, statuses = set(), set(), []
        fams = []
        for i in range(SNAPS):
            s = _sit(1 + i % 3, 5 + i % 8, 30 + i % 40, session=sid)
            s.side = "defense"
            s.extras = {"session_id": sid, "snap_seq": i}
            call = playcaller.make_call(s, opponent, db, cpu_defense=cpu_defense)
            rd = call.ravens_defense
            statuses.append(rd["status"])
            assert rd["status"] == "CALL", (label, rd)
            assert call.formation in rc.FORMATIONS
            assert call.formation != rc.NSM
            cid = rd["candidate"]["call_id"]
            assert cid not in nsm_ids, (label, cid)
            text = "\n".join(rd["card"])
            assert "PACKAGE: Nickel Single Mug" not in text
            cd = rd.get("play_cooldown") or {}
            if chiefs_cd:
                if not cd.get("fallback"):
                    fam = pcd.defense_family_key(rd["candidate"]["base_play"])
                    assert fam not in set(fams[-3:]), (label, i, fam, fams[-3:], cd)
                    assert cd.get("active") is True and cd.get("window") == 3
                    fams.append(fam)
            else:
                assert pcd.cooldown_spec(db, sid, side="defense") is None
                assert not cd.get("active")
            seen.add(cid)
            forms.add(call.formation)
        assert "NO_LEGAL_RAVENS_CALL" not in statuses and statuses.count("CALL") == SNAPS, label
        assert seen.isdisjoint(nsm_ids) and rc.NSM not in forms, (label, seen, forms)
        assert forms <= {rc.NICKEL, rc.DIME}, (label, forms)
        defense_summary[label] = {"unique_calls": len(seen), "forms": sorted(forms), "no_legal": 0, "nsm": 0}
    print("DEFENSE_SIM_SUMMARY", defense_summary)
    summary["defense"] = defense_summary
    print("GIANTS_SIM_SUMMARY", summary)
