"""t461u: add Packers Gun Bunch X Nasty X Post Return to the Lions/CPU/Giants book only."""
from __future__ import annotations

import collections
import hashlib
import json

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
from tests.test_lions_cpu_book_t445u import SNAPS, _forms
from tests.test_lions_cpu_book_t450u import SHARED_ADDITIONS_SHA256
from tests.test_titans_play_cooldown import _giants, _lions, _online, _titans

NEW_ID = "civil_t461_gun_bunch_x_nasty_x_post_return_1"
SETUP = [
    ("TE", "Slant", False, "Slant TE"),
    ("Middle bunch WR", "Streak", False, "Streak Middle Bunch WR"),
    ("Outside bunch WR", "In route", False, "In Route Outside Bunch WR"),
]


def _verify_lions_book(db):
    _verify_all(db)
    for vid in set(CAT_LIONS.ids) - set(CAT.ids):
        lg.record_verified(db, CAT_LIONS, vid)


def test_shared_catalog_and_additions_hash_are_unchanged():
    raw = (sc.repo_root() / sc.ADDITIONS_RELPATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHARED_ADDITIONS_SHA256
    assert CAT.version() == CAT_LIONS.version() == "lions_civil_bucs_5wr_v4@15c120decc4b"
    assert len(CAT.variants) == 49 and sc.contract_errors(CAT) == []
    assert CAT.get(NEW_ID) is None
    assert NEW_ID not in CAT.ids


def test_lions_cpu_book_adds_x_post_return_on_its_own_family():
    assert CAT_LIONS.book_id == sc.LIONS_CPU_BOOK
    assert len(CAT_LIONS.variants) == sc.EXPECTED_LIONS_CPU_VARIANTS == 129
    assert sc.contract_errors(CAT_LIONS) == []
    v = CAT_LIONS.get(NEW_ID)
    assert v is not None
    assert v.formation == "Gun Bunch X Nasty"
    assert v.base_play == "X Post Return"
    assert v.variant_name == "X Post Return 1"
    assert v.group == sc.OWNER_PASS and v.field_zone == "any" and not v.is_run
    assert v.scheme_category == "Everything Beater"
    assert v.expert_coverage_tags == ("Cover 2", "Cover 3", "Cover 4", "Man Coverage")
    assert v.name_in_stock_data is False
    assert [(a.target, a.action, a.macro_marked, a.guide_text) for a in v.adjustments] == SETUP
    assert v.raw["guide_setup_text"] == [a[3] for a in SETUP]
    assert pcd.family_of_variant(v) == "x post return"
    fams = pcd.catalog_families(CAT_LIONS)
    assert "x post return" in fams
    assert sum(1 for key in fams if key == "x post return") == 1
    assert "x post return" not in pcd.catalog_families(CAT)

    owner = json.loads((sc.repo_root() / sc.LIONS_CPU_RELPATH).read_text())
    assert owner["owner_instruction"]["source"] == "t461u"
    assert {"formation": "Gun Bunch X Nasty", "base_play": "X Post Return", "closest_stock": None} in owner["name_not_in_stock_data"]
    assert "X Post Return" in owner["cooldown_families"]
    stock_all = sc.stock_offense_plays_all()
    assert "Gun Bunch X Nasty" in stock_all
    assert "X Post Return" not in stock_all["Gun Bunch X Nasty"]
    bucs = sc.stock_offense_plays()
    assert "Gun Bunch X Nasty" not in bucs

    mid = {x.variant_id for x in lg.legal_set(CAT_LIONS, db=None, yardline=40)[0]}
    rz = {x.variant_id for x in lg.legal_set(CAT_LIONS, db=None, yardline=95)[0]}
    chiefs_mid = {x.variant_id for x in lg.legal_set(CAT, db=None, yardline=40)[0]}
    assert NEW_ID in mid and NEW_ID not in rz and NEW_ID not in chiefs_mid


def test_x_post_return_is_called_in_lions_cpu_giants_sims_not_chiefs_titans(tmp_path, civil_env, ravens_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path, formations=_forms())
    _verify_lions_book(db)
    _lions(db, "lions-sim")
    _giants(db, "giants-sim")
    _online(db, "chiefs-sim")
    _titans(db, "titans-sim")

    sessions = (
        ("lions", "lions-sim", "jaxon", sc.LIONS_CPU_BOOK, True),
        ("cpu", "cpu-sim", "cpu", sc.LIONS_CPU_BOOK, True),
        ("giants", "giants-sim", "giants_indianapolis_colts", sc.LIONS_CPU_BOOK, True),
        ("chiefs", "chiefs-sim", "online_indianapolis_colts", sc.SHARED_BOOK, False),
        ("titans", "titans-sim", "titans_indianapolis_colts", sc.SHARED_BOOK, False),
    )
    report = {}
    for label, session, opponent, book, expect_new in sessions:
        counts = collections.Counter()
        sits = []
        statuses = []
        books = set()
        for i in range(SNAPS):
            down = 1 + i % 4
            dist = (10, 7, 3, 1)[i % 4]
            y = 15 + (i * 5) % 80
            distance = min(dist, 100 - y)
            call = playcaller.make_call(_sit(down, distance, y, session=session), opponent, db)
            sv = call.scheme_variant
            statuses.append(sv["status"])
            assert sv["status"] == "CALL" and sv.get("variant_id") and sv["variant_id"] != lg.NO_LEGAL, (label, i, sv)
            books.add(sv.get("catalog_book"))
            vid = sv["variant_id"]
            counts[vid] += 1
            if vid == NEW_ID:
                sits.append({"snap": i, "down": down, "distance": distance, "yardline": y,
                             "formation": sv.get("formation"), "base_play": sv.get("base_play")})
            _close_snap(db, session, vid, y)
        assert lg.NO_LEGAL not in statuses, label
        assert books == {book}, (label, books)
        if expect_new:
            assert counts[NEW_ID] > 0, (label, "X Post Return was never called")
            assert catalog_for_call(db, _sit(1, 10, 40, session=session)).book_id == sc.LIONS_CPU_BOOK
            assert len(counts) == 129, (label, len(counts))
        else:
            assert counts[NEW_ID] == 0
            assert NEW_ID not in counts
            assert len(counts) == 49, (label, len(counts))
        report[label] = {
            "unique": len(counts),
            "no_legal": statuses.count(lg.NO_LEGAL),
            "x_post_return": counts[NEW_ID],
            "situations": sits,
        }
    print("T461U_SIM_REPORT", json.dumps(report, indent=2))
