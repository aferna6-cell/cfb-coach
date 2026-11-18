"""t462u: live defense is Patriots 4-3 Even 6-1 + Nickel 2-4 Wide + Nickel Single Mug."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from cfb_coach.madden.cli import RAVENS_CIVIL_DEFENSE_BANNER
from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense import legality as lg
from cfb_coach.madden.ravens_defense.confirmations import Confirmations
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import legality as slg
from tests.test_civil_scheme_only import CAT, CAT_LIONS, _db, _sit, _verify_all, civil_env  # noqa: F401
from tests.test_chiefs_play_cooldown import ravens_env  # noqa: F401
from tests.test_lions_cpu_book_t445u import SNAPS, _simulate
from tests.test_lions_cpu_book_t450u import RAVENS_DATASET_SHA256, SHARED_ADDITIONS_SHA256
from tests.test_titans_play_cooldown import _giants, _lions, _online, _titans

PDF_PLAYS = ("Cover 4 Quarters", "Sam Will Blitz", "Tampa 2", "Cover 3 Buzz")
EVEN = "4-3 Even 6-1"
NICKEL = "Nickel 2-4 Wide"
DIME = "Dime 2-3 Odd"
NSM = "Nickel Single Mug"
NSM_DATASET_SHA256 = "unused-checked-is-file"
OFFENSE_VERSION = "lions_civil_bucs_5wr_v4@15c120decc4b"


def _stock_defense_books():
    data = json.loads(rc.STOCK_BOOKS.read_text(encoding="utf-8"))
    return data["books"]["defense"]


def _plays(forms, name):
    rec = forms.get(name) or {}
    return list(rec.get("plays") or [])


def test_stock_overlap_patriots_is_the_combined_book():
    books = _stock_defense_books()
    has_61 = [team for team, rec in books.items() if EVEN in (rec.get("formations") or {})]
    assert len(has_61) == 20
    assert "Vikings" in has_61  # owner belief confirmed
    assert "Patriots" in has_61

    nickel_civil = {c.base_play for c in rc.load().candidates if c.formation == rc.NICKEL}
    nsm_civil = {c.base_play for c in rc.load_lions_franchise().candidates if c.formation == rc.NSM}

    overlap = {}
    for team in has_61:
        forms = books[team]["formations"]
        old = tuple(f for f in (NICKEL, DIME, NSM) if f in forms)
        plays61 = set(_plays(forms, EVEN))
        overlap[team] = {
            "old": old,
            "pdf": tuple(p for p in PDF_PLAYS if p in plays61),
            "tampa2": "Tampa 2" in plays61,
            "nickel_civil": len(nickel_civil & set(_plays(forms, NICKEL))) if NICKEL in forms else 0,
            "nsm_civil": len(nsm_civil & set(_plays(forms, NSM))) if NSM in forms else 0,
            "db_fire": "DB Fire 2" in _plays(forms, DIME),
        }

    # No stock book has 6-1 plus all three old formations.
    assert all(len(v["old"]) < 3 for v in overlap.values())

    # Vikings: 6-1 + Nickel 2-4 Wide, all four PDF plays, no NSM, no Dime.
    assert overlap["Vikings"]["old"] == (NICKEL,)
    assert overlap["Vikings"]["pdf"] == PDF_PLAYS
    assert overlap["Vikings"]["nickel_civil"] == len(nickel_civil)

    # Raiders is the only 6-1 book with Dime 2-3 Odd (and DB Fire 2).
    dime_books = [t for t, v in overlap.items() if DIME in v["old"]]
    assert dime_books == ["Raiders"] and overlap["Raiders"]["db_fire"]
    assert overlap["Raiders"]["old"] == (DIME,)

    # Chargers has Nickel + NSM but lacks Tampa 2 in 6-1.
    assert set(overlap["Chargers"]["old"]) == {NICKEL, NSM}
    assert overlap["Chargers"]["tampa2"] is False
    assert overlap["Chargers"]["nickel_civil"] == len(nickel_civil)
    assert overlap["Chargers"]["nsm_civil"] == len(nsm_civil)

    # Patriots is the unique max-overlap book that also has every PDF 6-1 play.
    full_pdf_with_old = [t for t, v in overlap.items() if v["pdf"] == PDF_PLAYS and v["old"]]
    assert "Patriots" in full_pdf_with_old
    pat = overlap["Patriots"]
    assert set(pat["old"]) == {NICKEL, NSM}
    assert pat["pdf"] == PDF_PLAYS and pat["tampa2"]
    assert pat["nickel_civil"] == len(nickel_civil)
    assert pat["nsm_civil"] == len(nsm_civil)
    assert pat["db_fire"] is False
    # Score: # of old formations among books that contain all four PDF plays.
    scored = sorted(
        ((len(overlap[t]["old"]), overlap[t]["nickel_civil"] + overlap[t]["nsm_civil"], t)
         for t in full_pdf_with_old),
        reverse=True,
    )
    assert scored[0][2] == "Patriots"


def test_rollback_datasets_are_byte_identical_and_unused():
    ravens = rc.DATASET.read_bytes()
    assert hashlib.sha256(ravens).hexdigest() == RAVENS_DATASET_SHA256
    base = rc.load()
    assert base.sha256 == RAVENS_DATASET_SHA256
    assert rc.contract_errors(base.raw) == []
    assert tuple(f["name"] for f in base.raw["formations"]) == rc.FORMATIONS
    assert "dime_dbfire2" in base.by_id()
    assert rc.LIONS_NSM_DATASET.is_file()
    unused = rc.load_lions_franchise()
    assert unused.stock_team == rc.STEELERS
    assert set(rc.NSM_CALL_IDS) <= set(unused.by_id())
    live = rc.load_61_defense()
    assert live.sha256 != base.sha256 and live.sha256 != unused.sha256


def test_live_catalog_is_patriots_61_combined_scheme():
    cat = rc.catalog_for_defense_session(None, None, {}, None)
    assert cat is not rc.load()
    assert cat.raw.get("catalog_id") == rc.SIXTY_ONE_CATALOG_ID
    assert cat.stock_team == rc.PATRIOTS and cat.uses_prior_only
    assert cat.allowed_formations == rc.SIXTY_ONE_FORMATIONS
    assert {c.formation for c in cat.candidates} == {rc.EVEN_61, rc.NICKEL, rc.NSM}
    assert len(cat.candidates) == 34
    assert set(rc.EVEN_61_CALL_IDS) <= set(cat.by_id())
    assert set(rc.NSM_CALL_IDS) <= set(cat.by_id())
    assert set(rc.NICKEL_CIVIL_CALL_IDS) <= {c.call_id for c in cat.candidates}
    assert "dime_dbfire2" not in cat.by_id()
    assert cat.raw.get("source_defensive_playbook") == "New England Patriots"
    assert "4-3 Even 6-1" in (cat.raw.get("label") or "")
    stock = rc.stock_catalog_check(cat)
    assert all(stock.values()), {k: v for k, v in stock.items() if not v}
    # Every 6-1 / nickel / NSM base play is in the Patriots stock book.
    book = rc.patriots_stock_book()
    for c in cat.candidates:
        assert c.base_play in book[c.formation], (c.candidate_id, c.formation, c.base_play)


def test_inside_10_tampa2_is_gated_from_our_goal():
    cat = rc.load_61_defense()
    conf = Confirmations()
    mid, _ = lg.legal_set(cat, installed=None, conf=conf, scheme_book=True, yardline=35)
    gl, _ = lg.legal_set(cat, installed=None, conf=conf, scheme_book=True, yardline=8)
    ten, _ = lg.legal_set(cat, installed=None, conf=conf, scheme_book=True, yardline=10)
    eleven, _ = lg.legal_set(cat, installed=None, conf=conf, scheme_book=True, yardline=11)
    backed, _ = lg.legal_set(cat, installed=None, conf=conf, scheme_book=True, yardline=95)
    mid_ids = {c.candidate_id for c in mid}
    gl_ids = {c.candidate_id for c in gl}
    assert "even61_rz_tampa2_61_base" not in mid_ids
    assert "even61_rz_tampa2_61_base" not in {c.candidate_id for c in eleven}
    assert "even61_rz_tampa2_61_base" not in {c.candidate_id for c in backed}
    assert "even61_rz_tampa2_61_base" in gl_ids
    assert "even61_rz_tampa2_61_base" in {c.candidate_id for c in ten}
    assert len(mid) == 33 and len(gl) == 34 and len(ten) == 34
    # Midfield still has a legal call in every formation, including long-yardage nickel / 6-1 base.
    assert {c.formation for c in mid} == {rc.EVEN_61, rc.NICKEL, rc.NSM}
    assert any(c.candidate_id.startswith("even61_base_") for c in mid)


def test_offense_catalog_and_shared_additions_unchanged():
    raw = (sc.repo_root() / sc.ADDITIONS_RELPATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHARED_ADDITIONS_SHA256
    assert CAT.version() == CAT_LIONS.version() == OFFENSE_VERSION
    assert sc.contract_errors(CAT) == [] and sc.contract_errors(CAT_LIONS) == []


def test_fail_safe_stays_no_legal_ravens_call(tmp_path, civil_env, ravens_env, monkeypatch):
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.ravens_defense import legality as rlg

    def _empty(*a, **k):
        cat = a[0] if a else rc.load_61_defense()
        return [], {c.candidate_id: type("V", (), {"legal": False, "reasons": ["TEST"]})() for c in cat.candidates}

    monkeypatch.setattr(rlg, "legal_set", _empty)
    db = _db(tmp_path, formations=["Gun Bunch TE"])
    _lions(db, "fs1")
    s = _sit(1, 10, 40, session="fs1")
    s.side = "defense"
    s.extras = {"session_id": "fs1"}
    call = playcaller.make_call(s, "jaxon", db)
    assert call.ravens_defense["status"] == "NO_LEGAL_RAVENS_CALL"
    assert call.ravens_defense.get("fallback") in (None, False, "") or "NO_LEGAL" in call.ravens_defense["status"]


def test_launcher_banner_names_patriots_61():
    assert RAVENS_CIVIL_DEFENSE_BANNER.startswith("PATRIOTS_EVEN_61_CIVIL")
    assert "4-3 Even 6-1" in RAVENS_CIVIL_DEFENSE_BANNER
    assert "Nickel 2-4 Wide" in RAVENS_CIVIL_DEFENSE_BANNER
    assert "Nickel Single Mug" in RAVENS_CIVIL_DEFENSE_BANNER
    assert "NO_LEGAL_RAVENS_CALL" in RAVENS_CIVIL_DEFENSE_BANNER
    assert "Dime 2-3 Odd" not in RAVENS_CIVIL_DEFENSE_BANNER


def test_600_snap_offense_and_defense_sims_all_leagues(tmp_path, civil_env, ravens_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path, formations=["Gun Bunch TE"])
    _verify_all(db)
    _lions(db, "lions-sim")
    _online(db, "chiefs-sim")
    _titans(db, "titans-sim")
    _giants(db, "giants-sim")
    db.conn.execute(
        "INSERT INTO game_sessions (session_id, opponent_id, dynasty, started_ts, "
        "league, user_team, opponent_team, opponent_type) "
        "VALUES ('cpu-sim', 'cpu', 'primary', '2026-11-17T16:00:00+00:00', "
        "NULL, 'Detroit Lions', 'Dallas Cowboys', 'cpu')"
    )
    db.conn.commit()

    offense_sessions = (
        ("lions-sim", "jaxon", "Lions", sc.LIONS_CPU_BOOK),
        ("cpu-sim", "cpu", "CPU", sc.LIONS_CPU_BOOK),
        ("giants-sim", "giants_indianapolis_colts", "Giants", sc.LIONS_CPU_BOOK),
        ("chiefs-sim", "online_indianapolis_colts", "Chiefs", sc.SHARED_BOOK),
        ("titans-sim", "titans_indianapolis_colts", "Titans", sc.SHARED_BOOK),
    )
    offense_summary = {}
    for session, opponent, label, book in offense_sessions:
        rec = _simulate(db, session, opponent, n=SNAPS)
        assert rec["statuses"].count("CALL") == SNAPS and slg.NO_LEGAL not in rec["statuses"], label
        assert rec["books"] == {book}, (label, rec["books"])
        offense_summary[label] = {
            "unique_calls": len(rec["counts"]),
            "no_legal": rec["statuses"].count(slg.NO_LEGAL) if isinstance(slg.NO_LEGAL, str) else 0,
        }
    print("OFFENSE_SIM_SUMMARY", offense_summary)

    buckets = (
        ("base", 1, 10, 40, False, False),
        ("obvious_pass_long", 3, 12, 45, False, True),
        ("long_yardage", 3, 15, 35, False, True),
        ("inside_10", 1, 10, 8, False, False),
        ("goal_line", 1, 3, 3, False, False),
        ("two_minute", 2, 8, 42, True, False),
    )
    defense_sessions = (
        ("lions-sim", "jaxon", False, "Lions"),
        ("cpu-sim", "cpu", True, "CPU"),
        ("chiefs-sim", "online_indianapolis_colts", False, "Chiefs"),
        ("titans-sim", "titans_indianapolis_colts", False, "Titans"),
        ("giants-sim", "giants_indianapolis_colts", False, "Giants"),
    )
    defense_summary = {}
    for sid, opponent, cpu_defense, label in defense_sessions:
        seen = Counter()
        forms = Counter()
        cov = Counter()
        sit_mix = defaultdict(Counter)
        statuses = []
        for i in range(SNAPS):
            name, down, distance, yardline, two_min, long_y = buckets[i % len(buckets)]
            s = _sit(down, distance, yardline, session=sid)
            s.side = "defense"
            s.two_minute = two_min
            s.long_yardage = long_y
            s.goal_line = name == "goal_line"
            s.red_zone = name in ("inside_10", "goal_line")
            s.extras = {"session_id": sid, "snap_seq": i, "bucket": name}
            call = playcaller.make_call(s, opponent, db, cpu_defense=cpu_defense)
            rd = call.ravens_defense
            statuses.append(rd["status"])
            assert rd["status"] == "CALL", (label, name, rd)
            assert call.formation in rc.SIXTY_ONE_FORMATIONS
            cid = rd["candidate"]["call_id"]
            assert cid != "dime_dbfire2"
            if name in ("inside_10", "goal_line") and cid == "even61_rz_tampa2_61_base":
                assert s.yardline <= 10
            if cid == "even61_rz_tampa2_61_base":
                assert s.yardline <= 10
            seen[cid] += 1
            forms[call.formation] += 1
            cov[rd["candidate"]["base_play"]] += 1
            sit_mix[name][cid] += 1
        assert "NO_LEGAL_RAVENS_CALL" not in statuses and statuses.count("CALL") == SNAPS, label
        assert "dime_dbfire2" not in seen and rc.DIME not in forms
        assert set(forms) <= set(rc.SIXTY_ONE_FORMATIONS)
        assert any(cid.startswith("even61_") for cid in seen), (label, sorted(seen))
        defense_summary[label] = {
            "unique_calls": len(seen),
            "forms": dict(forms),
            "coverages": dict(cov),
            "by_call": dict(seen),
            "by_situation": {k: dict(v) for k, v in sit_mix.items()},
            "no_legal": 0,
            "nsm": sum(v for k, v in seen.items() if k in set(rc.NSM_CALL_IDS)),
            "even61": sum(v for k, v in seen.items() if k.startswith("even61_")),
            "nickel": sum(v for k, v in seen.items() if k not in set(rc.NSM_CALL_IDS) and not k.startswith("even61_")),
        }
    print("DEFENSE_61_SIM_SUMMARY", json.dumps(defense_summary, sort_keys=True))
    # Keep a compact per-league mix for the inventory.
    print("DEFENSE_61_SIM_COMPACT", {
        label: {
            "unique": rec["unique_calls"],
            "forms": rec["forms"],
            "even61": rec["even61"],
            "nickel": rec["nickel"],
            "nsm": rec["nsm"],
            "top_coverages": dict(Counter(rec["coverages"]).most_common(8)),
        }
        for label, rec in defense_summary.items()
    })
