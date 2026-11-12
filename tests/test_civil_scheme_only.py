"""civil_scheme_only: the 26-variant restricted offensive coordinator (tests 1-12 + coverage-beater behavior)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from cfb_coach.madden.scheme import beliefs as bl
from cfb_coach.madden.scheme import catalog as sc
from cfb_coach.madden.scheme import coordinator as co
from cfb_coach.madden.scheme import design as dz
from cfb_coach.madden.scheme import evidence as ev
from cfb_coach.madden.scheme import features as ft
from cfb_coach.madden.scheme import legality as lg
from cfb_coach.madden.scheme import mode as md
from cfb_coach.madden.scheme import model as sm
from cfb_coach.madden.scheme import priors as pr
from cfb_coach.situation import Situation

CAT = sc.load()
FORMS = ["Gun Doubles Clamp Stack", "Gun Bunch TE", "Gun Tight Doubles", "Gun 5WR Tight"]
RZ_FORMS = ["Pistol Bunch TE", "Gun Empty Quads", "I Form Close Flex"]
FORMS_ALL = FORMS + RZ_FORMS
Z_SPOT = {"civil_5wr_z_spot_1", "civil_5wr_z_spot_shake_1"}
CORE = {v.variant_id for v in CAT.core_variants}
RZ = {v.variant_id for v in CAT.redzone_variants}
RZ10 = {v.variant_id for v in CAT.redzone_variants if v.field_zone == sc.INSIDE_10}    # inside the 10 only
RZ20 = {v.variant_id for v in CAT.redzone_variants if v.field_zone == sc.RED_ZONE}     # inside the 10 AND 10-20
CLOSE_FLEX = {"civil_rz_i_form_close_flex_stretch", "civil_rz_i_form_close_flex_mesh"}
RUNS = {v.variant_id for v in CAT.run_variants}                                       # owner t349u designed runs
REG = CORE | RUNS                                                                      # regular: outside the 10


def _db(tmp_path, *, formations=FORMS_ALL):
    from cfb_coach.madden import playbook
    from cfb_coach.madden.model.cli import open_explicit_madden_db
    db = open_explicit_madden_db(tmp_path / "madden27.db", read_only=False)
    rec = playbook.make_record("offense", "stock", "Buccaneers", formations=list(formations))
    playbook._save_state(db, {"applied": {"offense": rec}, "pending": {}})
    return db


def _artifact(tmp_path) -> Path:
    from cfb_coach.madden.model import experimental_model as em
    m = sm.SchemeModel({t: sm.Head(t, "existing", 0.0 if t != "negative" else -2.0) for t in sm.TARGETS}, None,
                       [1e9] * len(ft.NAMES), [-1e9] * len(ft.NAMES),
                       {"catalog_version": CAT.version(), "architecture": "test"})
    m.existing = em.train_experimental([], side="offense")
    return sm.save(m, tmp_path / "art")


@pytest.fixture
def civil_env(tmp_path, monkeypatch):
    art = _artifact(tmp_path)
    monkeypatch.setenv(md.ENV_MODE, md.CIVIL)
    monkeypatch.setenv(md.ENV_ARTIFACT, str(art))
    co._SCORER.clear()
    return art


def _verify_all(db, ids=None):
    for v in CAT.variants:
        if (ids is None or v.variant_id in ids) and v.variant_id not in Z_SPOT:
            lg.record_verified(db, CAT, v.variant_id)


def _sit(down=1, distance=10, yardline=35, session="g1", look=None):
    s = Situation(raw="t", down=down, distance=distance, yardline=yardline,
                  coverage_hint=look, coverage_source="live" if look else "none")
    s.extras = {"session_id": session}
    return s


# 1 -------------------------------------------------------------- catalog contract
def test_01_catalog_has_exactly_26_variants_and_blocked_entries_are_never_candidates():
    assert len(CAT.core_variants) == 26 and len(CAT.redzone_variants) == 7 and len(set(CAT.ids)) == 45
    assert len(RUNS) == 12 and not (RUNS & CORE) and not (RUNS & RZ)
    assert RZ20 == CLOSE_FLEX and len(RZ10) == 5
    assert sc.contract_errors(CAT) == []
    blocked = CAT.blocked_ids
    # t335u: the red-zone packages are now mapped to callable plays; the unnamed 5WR setup stays blocked
    assert set(blocked) == {"civil_5wr_unidentified_setup_between_z_spot_shake_1_and_z_spot_1"}
    assert not (set(blocked) & set(CAT.ids))
    for b in blocked:
        assert not lg.check(None, CAT, variant_id=b, book={"formations": {}}, yardline=95).legal
    assert all(not v.unadjusted_allowed and v.adjustments for v in CAT.core_variants)


# 2 -------------------------------------------------------------- playbook verification
def test_02_stock_book_membership_exact_names_no_alias():
    for f in FORMS:
        assert lg.stock_has(f, "x")[0], f
    ok = {v.variant_id for v in CAT.variants if all(lg.stock_has(v.formation, v.base_play))}
    assert len(ok) == 43 and not (ok & Z_SPOT) and RZ <= ok and RUNS <= ok


# 3 -------------------------------------------------------------- live legality
def test_03_owner_confirmed_book_makes_every_dataset_play_live_legal_without_the_applied_book(tmp_path):
    """t337u: every formation/play in the scheme dataset is in Aidan's book -> all live-legal (field gate kept)."""
    import dataclasses
    for db in (_db(tmp_path / "none", formations=["Gun Doubles"]), _db(tmp_path / "all")):
        mid = {v.variant_id for v in lg.legal_set(CAT, db=db, yardline=50)[0]}
        assert mid == REG                                                    # all 26 incl. Z Spot, Bunch TE + 12 runs
        assert {v.variant_id for v in lg.legal_set(CAT, db=db, yardline=85)[0]} == REG | RZ20    # 10-20
        rz = {v.variant_id for v in lg.legal_set(CAT, db=db, yardline=95)[0]}
        assert rz == RZ                                                      # inside the 10: red-zone plays only
        _l, verdicts = lg.legal_set(CAT, db=db, yardline=lg.ANY_FIELD)
        assert all(vd.checks["owner_confirmed_installed"] and vd.checks["owner_confirmed_adjustments"]
                   for vd in verdicts.values())
        assert not lg.check(None, CAT, db=db, variant_id=next(iter(CAT.blocked_ids)), yardline=95).legal
    # with no DB / no applied book record at all
    assert {v.variant_id for v in lg.legal_set(CAT, db=None, yardline=50)[0]} == REG
    # the confirmation is pinned to this catalog version: a different catalog falls back to the book check
    other = dataclasses.replace(CAT, digest="0" * 64)
    assert not lg.owner_book_confirmation_applies(other)
    assert lg.legal_set(other, db=_db(tmp_path / "x", formations=["Gun Doubles"]), yardline=50)[0] == []


# 4 -------------------------------------------------------------- NO_LEGAL, no fallback
def test_04_no_legal_scheme_play_without_fallback(tmp_path, civil_env, monkeypatch):
    from cfb_coach.madden import catalog as mc, playcaller
    other = next(f for f in mc.book_formations("offense", "Buccaneers") if f not in FORMS_ALL)
    db = _db(tmp_path, formations=[other])
    # without the owner confirmations nothing passes (book + adjustments) -> NO_LEGAL, never a fallback play
    monkeypatch.setattr(lg, "owner_book_confirmation_applies", lambda cat: False)
    monkeypatch.setattr(lg, "owner_confirmation_applies", lambda cat: False)
    monkeypatch.setattr(playcaller, "_pick_offense", lambda *a, **k: (_ for _ in ()).throw(AssertionError("heuristic")))
    call = playcaller.make_call(_sit(), "cpu", db)
    assert call.play == lg.NO_LEGAL and call.formation == ""
    assert call.scheme_variant["status"] == lg.NO_LEGAL
    assert "not in the applied offense book" in call.scheme_variant["explanation"]
    from cfb_coach.madden.model.offense_adjustment_designer import validate_live_call
    validate_live_call(db, call, opponent_id="cpu")


# 5 -------------------------------------------------------------- restricted live caller
def test_05_civil_mode_only_returns_exact_catalog_variants(tmp_path, civil_env, monkeypatch):
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.model import experimental_live
    db = _db(tmp_path)
    _verify_all(db)
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run"))  # noqa: E731
    monkeypatch.setattr(playcaller, "_pick_offense", boom)
    monkeypatch.setattr(experimental_live, "maybe_apply_experimental", boom)
    first = None
    for dd in ((1, 10, 25), (3, 2, 60), (2, 7, 85), (4, 1, 99)):
        call = playcaller.make_call(_sit(*dd), "cpu", db)
        first = first or call
        sv = call.scheme_variant
        v = CAT.get(sv["variant_id"])
        assert v is not None
        from cfb_coach.madden.catalog import norm
        assert (norm(call.formation), norm(call.play)) == (norm(v.formation), norm(v.base_play))  # book spelling
        assert (sv["formation"], sv["base_play"]) == (v.formation, v.base_play)
        assert sv["adjustments"] == [[a.target, a.action] for a in v.adjustments]
        assert call.adjustment["id"] == f"scheme:{v.variant_id}"
        assert "uncertainty_logit_sd" in sv and sv["warnings"] and sv["explanation"]
        assert "READS: none shown" in call.format()
        from cfb_coach.madden.model.offense_adjustment_designer import validate_live_call
        validate_live_call(db, call, opponent_id="cpu")
    # a tampered call is rejected before display
    call = first
    call.scheme_variant["adjustments"] = call.scheme_variant["adjustments"][:-1]
    with pytest.raises(ValueError):
        lg.revalidate_call(db, call, CAT)


# 6 -------------------------------------------------------------- unrestricted mode preserved
def test_06_default_and_unrestricted_modes_use_existing_coordinator(tmp_path, monkeypatch):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path)
    monkeypatch.delenv(md.ENV_MODE, raising=False)
    assert md.offense_mode(db) == md.UNRESTRICTED
    import cfb_coach.madden.scheme.coordinator as coord
    monkeypatch.setattr(coord, "call_civil", lambda *a, **k: (_ for _ in ()).throw(AssertionError("civil")))
    call = playcaller.make_call(_sit(), "cpu", db)
    assert not getattr(call, "scheme_variant", None)
    md.set_state(db, enabled=True, artifact="x", model_sha256=None)
    assert md.offense_mode(db) == md.CIVIL
    monkeypatch.setenv(md.ENV_MODE, md.UNRESTRICTED)
    call = playcaller.make_call(_sit(), "cpu", db)
    assert not getattr(call, "scheme_variant", None)


# 7 -------------------------------------------------------------- adjusted designs differ
def test_07_variants_of_the_same_base_have_different_designs_and_can_rank_differently(tmp_path, monkeypatch):
    from cfb_coach.madden.scheme import learners as ln
    geo = dz.GeometrySource(enabled=False)                 # no provisional export: commanded routes only
    a, b = CAT.get("civil_bucs_mesh_2_provisional"), CAT.get("civil_bucs_mesh_4_provisional")
    da, db_ = dz.adjusted_design(a, geo), dz.adjusted_design(b, geo)
    assert da.variant_id != db_.variant_id and da.play == db_.play
    ra, rb = dz.route_features(da), dz.route_features(db_)
    assert ra != rb
    # a learned head that values man-beating routes vs man and zone-beating routes vs zone
    d = len(ft.NAMES)
    w = [0.0] * (d + 1)
    w[1 + ft.NAMES.index("x_man_manbeater")] = 4.0
    w[1 + ft.NAMES.index("x_zone_zonebeater")] = 4.0
    head = ln.OffsetLogistic(weights=w, cov=[[0.0] * (d + 1) for _ in range(d + 1)],
                             std=ln.Standardizer([0.0] * d, [1.0] * d))
    m = sm.load(_artifact(tmp_path))
    m.heads["success"] = sm.Head("success", "logistic", 0.0, head)
    monkeypatch.setattr(pr, "BEATER_MAX", 0.0)              # isolate the design effect from expert tags
    s = co.Scorer(m, CAT, geo)
    man = bl.combine({}, 0.1, 0, "none", [("cover_1_unspecified", "cover 1")] * 8)
    zone = bl.combine({}, 0.1, 0, "none", [("cover_3_sky", "cover 3 sky")] * 8)

    def order(bel):
        r = s.score([a, b], down=2, distance=6, yardline=40, opponent_id="cpu", opponent_type="cpu", belief=bel,
                    offset_override={a.variant_id: 0.45, b.variant_id: 0.45})
        return [x.variant.variant_id for x in r]
    assert order(man) != order(zone), (ra, rb)


# 8 -------------------------------------------------------------- no leakage in beliefs
def test_08_beliefs_use_only_earlier_snaps_and_games():
    rows = []
    for g, covs in (("g1", ["cover 3"] * 4), ("g2", ["cover 2 man", "cover 1", "cover 3", "cover 4"])):
        for c in covs:
            rows.append({"game_id": g, "opponent_id": "cpu", "opponent_type": "cpu", "side": "offense",
                         "result": "+4", "coverage_seen": c})
    order = {"g1": 0, "g2": 1}
    base = bl.annotate_rows(rows, order)
    rows2 = [dict(r) for r in rows]
    rows2[5]["coverage_seen"] = "cover 0"          # change a row and everything after it
    rows2[6]["coverage_seen"] = "cover 0"
    rows2[7]["coverage_seen"] = "cover 0"
    alt = bl.annotate_rows(rows2, order)
    for i in range(6):
        assert base[i].mix == alt[i].mix, i
    assert base[0].n_history == 0 and base[4].n_history > 0
    assert base[0].mix == {}


# 9 -------------------------------------------------------------- evidence rules
def test_09_unexecuted_recommendations_are_not_evidence_and_partial_execution_is_preserved(tmp_path):
    db = _db(tmp_path)
    c = db.conn
    v = CAT.get("civil_bucs_mesh_3")
    def snap(rec, exe, status="identified", play=v.base_play):
        cur = c.execute("INSERT INTO snaps (ts, opponent_id, side, down, distance, yardline, result, formation, play,"
                        " session_id, executed_status, executed_formation, executed_play, recommended_adjustment_id,"
                        " executed_adjustment_id, yards) VALUES (datetime('now'),'cpu','offense',1,10,30,'+6',?,?,"
                        "'g1',?,?,?,?,?,6)", (v.formation, v.base_play, status, v.formation, play, rec, exe))
        c.commit()
        return cur.lastrowid
    rid = "scheme:" + v.variant_id
    s_unconf = snap(rid, None)
    s_conf = snap(rid, rid)
    s_other = snap(rid, rid, play="Mesh Post")              # a different base play was run
    s_part = snap(rid, None)
    s_changed = snap(rid, None)
    ev.report_execution(c, CAT, snap_id=s_part, applied=[0], skipped=list(range(1, len(v.adjustments))))
    ev.report_execution(c, CAT, snap_id=s_changed, applied=[0], changed={1: "Drag instead"},
                        skipped=list(range(2, len(v.adjustments))))
    got = {e.snap_id: e for e in ev.variant_evidence(c, CAT)}
    assert s_unconf not in got and s_other not in got
    assert got[s_conf].complete and got[s_conf].usable
    assert got[s_part].applied == (0,) and not got[s_part].complete and got[s_part].usable
    assert not got[s_changed].usable
    assert str(s_unconf) in ev.excluded_snap_keys(c)
    d = dz.adjusted_design(v, dz.GeometrySource(enabled=False), applied=got[s_part].applied)
    assert len(d.adjusted_roles) <= 1


# 10 ------------------------------------------------------------- training / eval never mutate the DB
def test_10_evaluation_is_read_only(tmp_path):
    from cfb_coach.madden.scheme import evaluate as evm
    db = _db(tmp_path)
    db.close()
    p = tmp_path / "madden27.db"
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    rep = evm.run(p, n_members=1, coordinator_sim=False, log=lambda s: None)
    assert rep["db_integrity"]["unchanged"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before
    assert not ev._has_table(__import__("sqlite3").connect(p), "civil_scheme_decisions")


# 11 ------------------------------------------------------------- defense path untouched
def test_11_defense_calls_never_enter_the_civil_coordinator(tmp_path, civil_env, monkeypatch):
    from cfb_coach.madden import playbook, playcaller
    db = _db(tmp_path)
    rec = playbook.make_record("defense", "stock", "49ers")
    st = playbook._load_state(db)
    st["applied"]["defense"] = rec
    playbook._save_state(db, st)
    import cfb_coach.madden.scheme.coordinator as coord
    monkeypatch.setattr(coord, "call_civil", lambda *a, **k: (_ for _ in ()).throw(AssertionError("civil on D")))
    s = _sit()
    s.side = "defense"
    call = playcaller.make_call(s, "john", db)
    assert call.side == "defense" and not getattr(call, "scheme_variant", None)


# 12 ------------------------------------------------------------- variant-level logging + latency
def test_12_decisions_are_logged_with_variant_belief_and_joined_to_outcomes(tmp_path, civil_env):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path)
    _verify_all(db)
    playcaller.make_call(_sit(), "cpu", db)                       # warm (loads artifact)
    t0 = time.perf_counter()
    call = playcaller.make_call(_sit(2, 5, 50), "cpu", db)
    assert (time.perf_counter() - t0) * 1000 < 150 * 3            # generous CI bound; target 150 ms
    assert call.scheme_variant["latency_ms"] < co.LATENCY_TARGET_MS * 3
    vid = call.scheme_variant["variant_id"]
    v = CAT.get(vid)
    rows = db.conn.execute("SELECT variant_id, base_play, adjustments_json, belief_json, session_id, down"
                           " FROM civil_scheme_decisions").fetchall()
    assert rows[-1][0] == vid and rows[-1][1] == v.base_play and json.loads(rows[-1][2])
    assert "families" in json.loads(rows[-1][3]) and rows[-1][4] == "g1" and rows[-1][5] == 2
    db.conn.execute("INSERT INTO snaps (ts, opponent_id, side, down, distance, yardline, result, coverage_seen, session_id,"
                    " executed_status, executed_formation, executed_play, recommended_adjustment_id,"
                    " executed_adjustment_id, yards) VALUES (datetime('now', '+1 minute'),'cpu','offense',2,5,50,'+12',"
                    "'cover 3','g1','identified',?,?,?,?,12)", (v.formation, v.base_play, "scheme:" + vid, "scheme:" + vid))
    db.conn.commit()
    log = ev.variant_log(db.conn, CAT)
    hit = [r for r in log if r.get("snap_id")]
    assert hit and hit[-1]["followed"] and hit[-1]["actual_coverage"] == "cover 3" and hit[-1]["yards"] == 12
    assert hit[-1]["actual_adjustments"] == [[a.target, a.action] for a in v.adjustments]


# coverage-beater behavior (Aidan 2026-11-10) ----------------------------------------------
def _scorer(tmp_path):
    return co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))


LEGAL = [v for v in CAT.variants if v.variant_id not in Z_SPOT]


def _rank(s, belief, recent=()):
    return s.score(LEGAL, down=1, distance=10, yardline=40, opponent_id="cpu", opponent_type="cpu", belief=belief,
                   recent_calls=recent, offset_override={v.variant_id: 0.45 for v in LEGAL})


def _tagged(vid, tag):
    return tag in pr.coverage_tags(CAT.get(vid))


def test_beater_a_repeated_cover3_shifts_to_cover3_beaters(tmp_path):
    s = _scorer(tmp_path)
    neutral = _rank(s, bl.combine({}, 0.1, 0, "none", []))
    c3 = _rank(s, bl.combine({}, 0.1, 0, "none", [("cover_3_sky", "cover 3 sky")] * 8))
    assert _tagged(c3[0].variant.variant_id, "Cover 3")
    top5 = lambda r, t: sum(_tagged(x.variant.variant_id, t) for x in r[:5])  # noqa: E731
    assert top5(c3, "Cover 3") == 5
    assert c3[0].parts["beater"] > 0.5 and c3[0].parts["tagged_beater_for_top"]
    assert max(x.parts["beater"] for x in neutral) < 0.05
    # variety: after the same beater was called 3 times in a row, another tagged beater leads
    again = _rank(s, bl.combine({}, 0.1, 0, "none", [("cover_3_sky", "cover 3 sky")] * 8),
                  recent=[c3[0].variant.variant_id] * 3)
    assert again[0].variant.variant_id != c3[0].variant.variant_id
    assert _tagged(again[0].variant.variant_id, "Cover 3")


def test_beater_b_switching_to_repeated_man_shifts_to_man_beaters(tmp_path):
    s = _scorer(tmp_path)
    seq = [("cover_1_unspecified", "cover 1")] * 8 + [("cover_3_sky", "cover 3 sky")] * 3   # most recent first
    man = _rank(s, bl.combine({}, 0.1, 0, "none", seq))
    assert _tagged(man[0].variant.variant_id, "Man Coverage")
    assert all(_tagged(x.variant.variant_id, "Man Coverage") for x in man[:3])
    c3_only = [v.variant_id for v in LEGAL if _tagged(v.variant_id, "Cover 3") and not _tagged(v.variant_id, "Man Coverage")]
    assert not ({x.variant.variant_id for x in man[:3]} & set(c3_only))


def test_beater_c_mixed_or_thin_evidence_does_not_force_a_tag(tmp_path):
    s = _scorer(tmp_path)
    mixed = bl.combine({}, 0.1, 0, "none", [("cover_3_sky", "c3"), ("cover_1_unspecified", "c1"),
                                            ("cover_2_unspecified", "c2"), ("cover_4_quarters", "c4")] * 2)
    thin = bl.combine({}, 0.1, 0, "none", [("cover_3_sky", "cover 3 sky")])
    for b in (mixed, thin):
        r = _rank(s, b)
        assert max(x.parts["beater"] for x in r) < 0.2
    # 5WR Tight has no tags: it gets the average beater term (unknown), never the minimum
    r = _rank(s, bl.combine({}, 0.1, 0, "none", [("cover_3_sky", "cover 3 sky")] * 8))
    by = {x.variant.variant_id: x.parts["beater"] for x in r}
    wr5 = [by[v.variant_id] for v in LEGAL if v.formation == "Gun 5WR Tight"]
    tagged = [by[v.variant_id] for v in LEGAL if v.formation != "Gun 5WR Tight"]
    assert min(wr5) > min(tagged) and max(wr5) < max(tagged)


# deployment: enable / disable / revert with backups, live path loads the expected model
def test_13_enable_disable_revert_round_trip(tmp_path, monkeypatch, capsys):
    import argparse
    from cfb_coach.madden.scheme import cli as scli
    monkeypatch.delenv(md.ENV_MODE, raising=False)
    monkeypatch.delenv(md.ENV_ARTIFACT, raising=False)
    monkeypatch.setattr(scli, "backups_root", lambda: tmp_path / "backups")
    db = _db(tmp_path)
    db.close()
    p = str(tmp_path / "madden27.db")
    art = _artifact(tmp_path)
    assert scli.cmd_enable(argparse.Namespace(db=p, artifact=str(art))) == 0
    from cfb_coach.madden.model.cli import open_explicit_madden_db
    d = open_explicit_madden_db(Path(p), read_only=True)
    st = md.load_state(d)
    assert md.offense_mode(d) == md.CIVIL and st["artifact"] == str(art)
    assert st["model_sha256"] == sm.load(art).meta["model_sha256"]
    d.close()
    assert scli.cmd_disable(argparse.Namespace(db=p, revert_artifact=False)) == 0
    d = open_explicit_madden_db(Path(p), read_only=True)
    assert md.offense_mode(d) == md.UNRESTRICTED
    d.close()
    assert len(list((tmp_path / "backups").glob("*.db"))) == 2
    # a tampered artifact refuses to load
    body = json.loads((art / "model.json").read_text())
    body["heads"]["success"]["base_logit"] = 0.7
    (art / "model.json").write_text(json.dumps(body))
    with pytest.raises(ValueError):
        sm.load(art)


def test_14_z_spot_is_a_name_mismatch_in_repo_stock_data_not_book_membership(tmp_path):
    db = _db(tmp_path)
    v = CAT.get("civil_5wr_z_spot_1")
    vd = lg.check(v, CAT, db=db, require_verified=False)
    assert vd.checks["stock_formation"] and not vd.checks["stock_play"]       # repo spells it 'L Spot'
    assert vd.legal and vd.checks["owner_confirmed_installed"]               # t337u: installed under 'Z Spot'
    from cfb_coach.madden.scheme.coordinator import build_call  # noqa: F401  (display keeps the dataset name)
    with pytest.raises(ValueError):                                         # a bogus alias is still rejected
        lg.record_verified(db, CAT, v.variant_id, book_play="Not A Play")
    lg.record_verified(db, CAT, v.variant_id, book_play="L Spot")              # optional owner alias still works
    assert lg.check(v, CAT, db=db).legal


# owner instruction t333u (1): only the 26 listed variant IDs can ever be output -----------------
def test_15_only_the_listed_catalog_variants_can_be_output(tmp_path, civil_env, monkeypatch):
    import dataclasses
    import random
    from cfb_coach.madden import playcaller
    db = _db(tmp_path)
    allowed = set(CAT.ids)
    assert len(allowed) == 45 and allowed == CORE | RZ | RUNS
    rng = random.Random(3)
    seen = set()
    labels = ["cover_3_sky", "cover_1_unspecified", "cover_2_man", "cover_4_quarters", "tampa_2", "cover_6"]
    for i in range(60):
        lab = rng.choice(labels)
        look = lab.replace("_", " ") if i % 2 else None
        call = playcaller.make_call(_sit(rng.randint(1, 4), rng.randint(1, 20), rng.randint(1, 99),
                                         session=f"g{i % 5}", look=look), "cpu", db)
        sv = call.scheme_variant
        assert sv["status"] == "CALL" and sv["variant_id"] in allowed
        y = call_yard(call)
        assert sv["variant_id"] in (RZ if y >= 90 else REG | RZ20 if y >= 80 else REG), (y, sv["variant_id"])
        seen.add(sv["variant_id"])
    assert seen <= allowed
    # no other path: anything that is not an exact catalog variant is rejected before display
    real = CAT.get("civil_bucs_mesh_3")
    for fake in (dataclasses.replace(real, variant_id="invented_play"),
                 dataclasses.replace(real, adjustments=real.adjustments[:-1]),          # dropped adjustment
                 dataclasses.replace(real, base_play="Mesh Post")):                     # substituted play
        orig = co.Scorer.score

        def bad(self, variants, **kw):
            out = orig(self, variants, **kw)
            for o in out:          # whichever candidate the mixer picks
                o.variant = fake
            return out
        monkeypatch.setattr(co.Scorer, "score", bad)
        call = playcaller.make_call(_sit(), "cpu", db)
        assert call.play == lg.NO_LEGAL, fake
        monkeypatch.setattr(co.Scorer, "score", orig)
    # blocked entries are never candidates even if injected into the legal list
    for b in CAT.blocked_ids:
        assert not lg.check(None, CAT, db=db, variant_id=b).legal


# owner instruction t333u (2): every recommendation lists ALL adjustments verbatim, in order, stems explicit ------
def test_16_every_variant_lists_all_adjustments_and_stems_are_applied(tmp_path):
    from cfb_coach.madden.scheme import adjustments as adjm
    geo = dz.GeometrySource(enabled=False)
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, geo)
    bel = bl.combine({}, 0.1, 0, "none", [])
    n_stems = 0
    for v in CAT.variants:
        top = s.score([v], down=1, distance=10, yardline=40, opponent_id="cpu", opponent_type="cpu", belief=bel,
                      offset_override={v.variant_id: 0.5})
        call = co.build_call(top[0], top, bel, book=None, shadow=False)
        text = call.format()
        pos = -1
        for a in v.adjustments:
            line = f"{a.index + 1}) {a.target}: {a.action}"
            at = text.find(line)
            assert at > pos, (v.variant_id, line)          # present and in order
            pos = at
        assert call.scheme_variant["adjustments"] == [[a.target, a.action] for a in v.adjustments]
        assert all(f"{a.target}: {a.action}" in call.adjustment["buttons"] for a in v.adjustments)
        specs = adjm.parse_variant(v)
        assert all(sp.representable for sp in specs)
        d = dz.adjusted_design(v, geo)
        stem_idx = [sp.index for sp in specs if sp.stem_shift]
        assert sorted(i for i, _sh, _m in d.stems) == sorted(stem_idx), v.variant_id   # no stem dropped
        for sp in specs:
            if sp.stem_shift:
                n_stems += 1
                tag = "STEM UP" if sp.stem_shift > 0 else "STEM DOWN"
                assert f"{sp.index + 1}) {sp.target}: {sp.action} [{tag}" in text
                without = dz.adjusted_design(v, geo, applied=[i for i in range(len(v.adjustments)) if i != sp.index])
                assert dz.route_features(without) != dz.route_features(d), (v.variant_id, sp.index)
        assert "verify in game" not in text and "hot-route vocabulary" not in text
    assert n_stems == 11


def test_17_civil_call_is_labeled_civil_not_heuristic(tmp_path, civil_env):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path)
    call = playcaller.make_call(_sit(), "cpu", db)
    assert call.call_label == "CIVIL_SCHEME_ONLY"
    text = call.format()
    assert "LABEL: CIVIL_SCHEME_ONLY" in text and "LABEL: HEURISTIC" not in text


def call_yard(call):
    return int(call.scheme_variant["yardline"])


# owner instruction t338u: field bands, enforced before scoring and before display ----------------------------
def test_18_field_bands_inside_10_red_zone_only_10_20_mix_outside_never(tmp_path, civil_env, monkeypatch):
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.model.offense_adjustment_designer import validate_live_call
    db = _db(tmp_path)
    # the one legal filter, before scoring
    for y in range(0, 101):
        ids = {v.variant_id for v in lg.legal_set(CAT, db=db, yardline=y)[0]}
        want = RZ if y >= 90 else (REG | RZ20) if y >= 80 else REG
        assert ids == want, y
    assert {v.variant_id for v in lg.legal_set(CAT, db=db, yardline=None)[0]} == REG    # unknown spot: regular only
    # live calls across the field
    for y in list(range(1, 100, 3)):
        for d, dist in ((1, 10), (3, 2), (4, 1)):
            call = playcaller.make_call(_sit(d, min(dist, 100 - y), y, session=f"s{y}"), "cpu", db)
            sv = call.scheme_variant
            cands = set(sv["candidates"])
            if y >= 90:
                assert sv["variant_id"] in RZ and cands == RZ, (y, sv["variant_id"])
            elif y >= 80:
                assert cands == REG | RZ20 and sv["variant_id"] in cands
            else:
                assert sv["variant_id"] in REG and not (cands & RZ), (y, sv["variant_id"])
    # before display: a play injected outside its band is rejected (fail closed, no fallback)
    orig = co.Scorer.score
    for bad, y in (("civil_rz_pistol_bunch_te_hb_zone", 50), ("civil_rz_pistol_bunch_te_hb_zone", 85),
                   ("civil_rz_i_form_close_flex_mesh", 70), ("civil_bucs_mesh_post_1", 95)):
        rzv = CAT.get(bad)

        def inject(self, variants, _v=rzv, **kw):
            out = orig(self, variants, **kw)
            for o in out:
                o.variant = _v
            return out
        monkeypatch.setattr(co.Scorer, "score", inject)
        assert playcaller.make_call(_sit(1, min(10, 100 - y), y), "cpu", db).play == lg.NO_LEGAL, (bad, y)
    monkeypatch.setattr(co.Scorer, "score", orig)
    # display gate: the call's own yardline must match the snap's
    call = playcaller.make_call(_sit(1, 5, 95, session="zz"), "cpu", db)
    assert call.scheme_variant["variant_id"] in RZ
    validate_live_call(db, call, opponent_id="cpu", yardline=95)
    with pytest.raises(ValueError):
        validate_live_call(db, call, opponent_id="cpu", yardline=50)
    call.scheme_variant["yardline"] = 50
    with pytest.raises(ValueError):
        lg.revalidate_call(db, call, CAT)
    call.scheme_variant.pop("yardline")
    with pytest.raises(ValueError):
        lg.revalidate_call(db, call, CAT)
    # the unnamed 5WR setup stays blocked everywhere
    for b in CAT.blocked_ids:
        for y in (50, 85, 95):
            assert not lg.check(None, CAT, db=db, variant_id=b, yardline=y).legal


def test_18b_ten_to_twenty_red_zone_plays_actually_appear_in_the_mix(tmp_path, civil_env):
    """A game's worth of 10-20 snaps: the Close Flex red-zone plays show up, and so do regular plays."""
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.scheme import evidence as evm
    db = _db(tmp_path)
    picks = []
    for i in range(30):
        y = 80 + (i * 7) % 10
        call = playcaller.make_call(_sit(1 + i % 3, 10, y, session="band"), "cpu", db)
        vid = call.scheme_variant["variant_id"]
        picks.append(vid)
        _close_snap(db, "band", vid, y)
    rz = sum(1 for p in picks if p in RZ20)
    assert rz >= 4, picks                    # red-zone plays are not buried
    assert len(picks) - rz >= 8, picks       # still a mix with the regular scheme plays
    assert len(set(picks)) >= 8, picks


def _close_snap(db, session, vid, yardline):
    """Stand-in for a closed, logged snap of this call (what recent_calls reads)."""
    db.conn.execute("INSERT INTO snaps (ts, opponent_id, side, session_id, recommended_adjustment_id, yardline, result)"
                    " VALUES ('2026-11-10T00:00:00', 'cpu', 'offense', ?, ?, ?, ?)",
                    (session, f"scheme:{vid}", yardline, "+3"))
    db.conn.commit()


def test_18c_inside_10_uses_only_red_zone_plays_and_mixes_them(tmp_path, civil_env):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path)
    picks = []
    for i in range(14):
        y = 90 + i % 9
        call = playcaller.make_call(_sit(1 + i % 4, min(100 - y, 5), y, session="goal"), "cpu", db)
        vid = call.scheme_variant["variant_id"]
        assert vid in RZ, vid
        picks.append(vid)
        _close_snap(db, "goal", vid, y)
    assert len(set(picks)) >= 4, picks


# owner instruction t338u: variants are separate calls; repetition is per variant_id only -------------------
def test_18d_variant_repetition_is_independent_of_sibling_variants(tmp_path, civil_env):
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))
    bel = bl.combine({}, 0.1, 0, "none", [])
    sibs = [CAT.get("civil_bucs_texas_y_stutter_wheel_1"), CAT.get("civil_bucs_texas_y_stutter_wheel_2")]
    kw = dict(down=1, distance=10, yardline=40, opponent_id="cpu", opponent_type="cpu", belief=bel,
              offset_override={v.variant_id: 0.5 for v in sibs})
    fresh = {r.variant.variant_id: r for r in s.score(sibs, **kw)}
    hist = ["civil_bucs_texas_y_stutter_wheel_1"] * 5
    after = {r.variant.variant_id: r for r in s.score(sibs, recent_calls=hist, game_calls=hist, **kw)}
    v1, v2 = "civil_bucs_texas_y_stutter_wheel_1", "civil_bucs_texas_y_stutter_wheel_2"
    assert after[v2].utility == fresh[v2].utility and after[v2].parts["repeat_penalty"] == 0
    assert after[v1].utility < fresh[v1].utility and after[v1].parts["repeat_penalty"] > 0
    assert co.repeat_penalty(v2, hist, {v1: 5}) == 0.0
    # stronger than before: one recent call of the same variant costs more than the t337u 0.25
    assert co.repeat_penalty(v1, [v1], {v1: 1}) > 0.5


def test_18e_mixing_is_seeded_and_only_among_near_best(tmp_path, civil_env):
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))
    bel = bl.combine({}, 0.1, 0, "none", [])
    ranked = s.score(CAT.core_variants, down=1, distance=10, yardline=40, opponent_id="cpu", opponent_type="cpu",
                     belief=bel)
    a, mix = co.pick(ranked, "seed-1")
    assert co.pick(ranked, "seed-1")[0] is a                  # reproducible
    # the PLAY is drawn among the near-best; the variation may be a near-best sibling of it (owner t370u)
    def near(vid):
        r = next(r for r in ranked if r.variant.variant_id == vid)
        g = max(x.utility for x in ranked if x.variant.pair == r.variant.pair)
        return ranked[0].utility - g <= co.MIX_MARGIN and g - r.utility <= co.SIBLING_MARGIN
    assert near(a.variant.variant_id)
    chosen = {co.pick(ranked, f"seed-{i}")[0].variant.variant_id for i in range(200)}
    assert all(near(c) for c in chosen)
    assert len(chosen) >= 2 or len(mix["pool"]) == 1


# owner instruction t336u: every displayed/logged call names the exact variation and all its adjustments -------
def test_19_every_call_shows_and_logs_its_variation_and_all_adjustments(tmp_path, civil_env):
    import json as _json
    from cfb_coach.madden import playcaller
    geo = dz.GeometrySource(enabled=False)
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, geo)
    bel = bl.combine({}, 0.1, 0, "none", [])
    for v in CAT.variants:
        y = 95 if v.field_zone != "any" else 40
        top = s.score([v], down=1, distance=5, yardline=y, opponent_id="cpu", opponent_type="cpu", belief=bel,
                      offset_override={v.variant_id: 0.5})
        call = co.build_call(top[0], top, bel, book=None, shadow=False)
        first = call.format().splitlines()[0]        # what the snap log stores as our_call
        head = call.headline()
        for text in (first, head):
            assert f"VARIATION: {v.variant_name} [{v.variant_id}]" in text, (v.variant_id, text)
            pos = -1
            for a in v.adjustments:
                at = text.find(f"{a.index + 1}) {a.target}: {a.action}")
                assert at > pos, (v.variant_id, a)
                pos = at
            if not v.adjustments:
                assert "none (catalog lists no adjustments: stock play)" in text
        assert f"VARIATION: {v.variant_name}  [{v.variant_id}]" in call.format()
    # the decision log carries the same variation text
    db = _db(tmp_path)
    call = playcaller.make_call(_sit(1, 10, 40), "cpu", db)
    row = db.conn.execute("SELECT variant_id, prediction_json, adjustments_json FROM civil_scheme_decisions "
                          "ORDER BY id DESC LIMIT 1").fetchone()
    pred = _json.loads(row[1])
    v = CAT.get(row[0])
    assert pred["variation"].startswith(f"VARIATION: {v.variant_name} [{v.variant_id}]")
    assert all(f"{a.target}: {a.action}" in pred["variation"] for a in v.adjustments)
    assert _json.loads(row[2]) == [[a.target, a.action] for a in v.adjustments]


# owner instruction t338u: the html-live page shows the variation next to the play call name ---------------
def test_21_html_live_shows_variation_next_to_play_name(tmp_path, civil_env):
    from cfb_coach.live_server import LivePlayController, LogRow, _call_main, render_live_html
    from cfb_coach.live_server import _variation_from_call_text
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.situation import parse_madden_situation
    import html as _html
    db = _db(tmp_path)
    ctrl = LivePlayController(db=db, opponent_id="cpu", make_call=playcaller.make_call,
                              parse_situation=parse_madden_situation, learn_summary=lambda: "ok",
                              brand="Madden 27 Franchise", cpu_only=True, enable_execution_verify=True)
    for y in (40, 85, 95):
        call = playcaller.make_call(_sit(1, min(10, 100 - y), y, session="h"), "cpu", db)
        sv = call.scheme_variant
        ctrl.last_call, ctrl.call_text = call, call.format()
        label = f"{sv['variant_name']} [{sv['variant_id']}]"
        main = _call_main(ctrl)
        assert main == f"{call.formation} — {call.play} · {label}", main
        page = render_live_html(ctrl)
        assert f'id="call" aria-live="polite">{_html.escape(main)}</div>' in page
        assert f"VARIATION: {_html.escape(label)}" in page        # #call-variation block, with every adjustment
        for a in sv["adjustment_lines"]:
            assert _html.escape(a) in page, a
        rec = ctrl.state()["pending_recommendation"]
        assert rec["variation"] == label and all(a in rec["variation_block"] for a in sv["adjustment_lines"])
        assert 'rec.variation ? " · " + rec.variation' in page    # the JS refresh keeps it next to the play
        assert 'r.variation ? " · " + r.variation' in page        # and the snap history rows
        row = LogRow(call.formation, call.play, "+4", call_text=call.format())
        assert row.to_dict()["variation"] == label and label in row.to_dict()["label"]
        assert _variation_from_call_text(call.format()) == label


# owner instruction t349u: designed runs from the Buccaneers book, and a real run-pass mix ---------------------
OWNER_RUNS = {
    "Gun Doubles Clamp Stack": ["Inside Zone", "Counter Y", "HB Mid Zone", "Same Side Zone"],
    "Gun Bunch TE": ["Inside Zone", "Read Option", "HB Off Tackle", "Boomerang HB Off Tackle", "Same Side Zone"],
    "Gun Tight Doubles": ["Inside Zone", "HB Draw"],
    "Gun 5WR Tight": ["QB Draw"],
}


def _runs_owner():
    return json.loads((sc.repo_root() / sc.RUNS_RELPATH).read_text())


def test_22_designed_runs_are_exactly_the_owner_list_of_stock_plays():
    stock = json.loads((sc.repo_root() / sc.STOCK_PLAYBOOKS_RELPATH).read_text())["books"]["offense"]["Buccaneers"]["formations"]
    got = {}
    for v in CAT.run_variants:
        got.setdefault(v.formation, []).append(v.base_play)
        assert v.base_play in stock[v.formation]["plays"], v.variant_id          # exact stock name
        assert v.variant_name == v.base_play and not v.adjustments and v.unadjusted_allowed   # variation = stock play
        assert v.is_run and v.field_zone == "any" and v.group == sc.DESIGNED_RUN
        assert "screen" not in v.base_play.lower() and "rpo" not in v.base_play.lower()
    assert got == OWNER_RUNS
    assert len({v.pair for v in CAT.run_variants}) == 12                        # each formation+play its own id
    assert {v.play_family for v in CAT.core_variants} == {sc.PASS}             # the 26 stay passes
    # the goal-to-go RPO Alert QB Draw stays a red-zone (inside the 10) play; no other RPO or screen anywhere
    rpo = [v for v in CAT.variants if "rpo" in v.base_play.lower()]
    assert [(v.variant_id, v.field_zone) for v in rpo] == [("civil_rz_gun_empty_quads_rpo_alert_qb_draw", sc.INSIDE_10)]
    assert not [v for v in CAT.variants if "screen" in v.base_play.lower()]
    # the 26-variant catalog bytes (and every confirmation / artifact pinned to them) are unchanged
    assert CAT.version() == "lions_civil_bucs_5wr_v4@15c120decc4b" and lg.owner_confirmation_applies(CAT)


@pytest.mark.parametrize("edit", ["invented_name", "screen", "rpo", "adjustment", "wrong_formation", "not_run",
                                  "renamed_variation"])
def test_22b_designed_run_loader_fails_closed_on_anything_not_in_the_book(edit):
    base = sc.parse(json.loads((sc.repo_root() / sc.DATASET_RELPATH).read_bytes()), digest=CAT.digest)
    owner = _runs_owner()
    m = owner["variants"][0]
    if edit == "invented_name":
        m["base_play"] = m["variant_name"] = "Inside Zone Weak"
    elif edit == "screen":
        m["base_play"] = m["variant_name"] = "HB Slip Screen"
    elif edit == "rpo":
        m["base_play"] = m["variant_name"] = "Return RPO Zone Alert"
    elif edit == "adjustment":
        m["adjustments"] = [{"target": "HB", "action": "Wheel"}]
    elif edit == "wrong_formation":
        m["formation"] = "Pistol Bunch TE"
    elif edit == "not_run":
        m["play_family"] = "pass"
    elif edit == "renamed_variation":
        m["variant_name"] = "Inside Zone 1"
    with pytest.raises(sc.CatalogError):
        sc.with_designed_runs(base, owner, digest="x")


def test_23_designed_runs_follow_the_field_bands(tmp_path):
    db = _db(tmp_path)
    for y, want in ((95, False), (90, False), (89, True), (80, True), (50, True), (5, True), (None, True)):
        ids = {v.variant_id for v in lg.legal_set(CAT, db=db, yardline=y)[0]}
        assert (RUNS <= ids) is want and (not (RUNS & ids)) is (not want), y
    # injected inside the 10 -> rejected before display
    v = CAT.get("civil_run_gun_bunch_te_inside_zone")
    assert not lg.check(v, CAT, db=db, yardline=95).legal and lg.check(v, CAT, db=db, yardline=85).legal


def _first_and_ten_run_share(db, session, n=60):
    from cfb_coach.madden import playcaller
    picks = []
    for i in range(n):
        y = 20 + (i * 7) % 55                               # between the 20s
        call = playcaller.make_call(_sit(1, 10, y, session=session), "cpu", db)
        sv = call.scheme_variant
        assert sv["status"] == "CALL" and sv["variant_id"] in REG, sv
        picks.append(sv["variant_id"])
        _close_snap(db, session, sv["variant_id"], y)
    return picks


def test_24_first_and_ten_between_the_20s_calls_a_real_share_of_designed_runs(tmp_path, civil_env):
    db = _db(tmp_path)
    picks = _first_and_ten_run_share(db, "mix")
    runs = [p for p in picks if p in RUNS]
    share = len(runs) / len(picks)
    assert 0.25 <= share <= 0.75, (share, picks)            # not pass-pass-pass, not run-run-run
    assert len(set(runs)) >= 6, runs                        # mixed among the runs themselves
    assert len(set(p for p in picks if p in CORE)) >= 8     # and among the pass variants
    # situation matters: 3rd and long passes much more than 1st and 10
    from cfb_coach.madden import playcaller
    long_runs = 0
    for i in range(40):
        y = 25 + i
        vid = playcaller.make_call(_sit(3, 12, y, session="long"), "cpu", db).scheme_variant["variant_id"]
        long_runs += vid in RUNS
        _close_snap(db, "long", vid, y)
    assert long_runs / 40 < share - 0.15, (long_runs, share)
    # the mix record says why
    call = playcaller.make_call(_sit(1, 10, 40, session="why"), "cpu", db)
    mix = call.scheme_variant["mix"]
    assert mix["family_gate"] == "on" and mix["family"] in ("run", "pass") and 0 < mix["p_run"] < 1
    assert {"prior_logit", "learned_gap", "tendency_logit"} <= set(mix)


def _fake(vid, util):
    return co.Scored(CAT.get(vid), 0.5, 0.3, 0.5, 0.1, 0.1, None, util, {})


def test_24b_run_probability_is_learned_and_situational_not_a_quota():
    passes = [_fake("civil_bucs_mesh_3", 0.50), _fake("civil_bucs_texas_y_stutter_wheel_1", 0.49)]
    runs_bad = [_fake("civil_run_gun_bunch_te_inside_zone", 0.30)]
    runs_good = [_fake("civil_run_gun_bunch_te_inside_zone", 0.60)]
    p_even = co.run_probability(passes + [_fake("civil_run_gun_bunch_te_inside_zone", 0.50)], down=1, distance=10)[0]
    p_bad = co.run_probability(passes + runs_bad, down=1, distance=10)[0]
    p_good = co.run_probability(passes + runs_good, down=1, distance=10)[0]
    assert p_bad < p_even < p_good                           # learned outcomes move the family odds
    assert p_bad >= co.RUN_P_MIN and p_good <= co.RUN_P_MAX
    even = passes + [_fake("civil_run_gun_bunch_te_inside_zone", 0.50)]
    assert co.run_probability(even, down=3, distance=12)[0] < p_even < co.run_probability(even, down=2, distance=2)[0]
    # tendency: after 6 straight passes the run odds go up, after 6 straight runs they go down
    up = co.run_probability(even, down=1, distance=10, recent_families=["pass"] * 6)[0]
    down = co.run_probability(even, down=1, distance=10, recent_families=["run"] * 6)[0]
    assert down < p_even < up
    assert co.run_probability(passes, down=1, distance=10)[0] is None    # one family only: no gate
    # inside the 10 the gate is off: red-zone selection unchanged
    rz = [_fake("civil_rz_pistol_bunch_te_hb_zone", 0.5), _fake("civil_rz_pistol_bunch_te_spot_y_option_1", 0.4)]
    top, mix = co.pick_mixed(rz, "s", down=1, distance=5, yardline=95)
    assert mix["family_gate"].startswith("off") and top.variant.variant_id == "civil_rz_pistol_bunch_te_hb_zone"


def test_24c_same_run_from_another_formation_is_a_separate_call(tmp_path):
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))
    bel = bl.combine({}, 0.1, 0, "none", [])
    a, b = "civil_run_gun_bunch_te_inside_zone", "civil_run_gun_doubles_clamp_stack_inside_zone"
    vs = [CAT.get(a), CAT.get(b)]
    kw = dict(down=1, distance=10, yardline=40, opponent_id="cpu", opponent_type="cpu", belief=bel,
              offset_override={a: 0.5, b: 0.5})
    fresh = {r.variant.variant_id: r for r in s.score(vs, **kw)}
    hist = [a] * 4
    after = {r.variant.variant_id: r for r in s.score(vs, recent_calls=hist, game_calls=hist, **kw)}
    assert after[b].utility == fresh[b].utility and after[b].parts["repeat_penalty"] == 0
    assert after[a].utility < fresh[a].utility


def test_24d_ten_to_twenty_mixes_designed_runs_and_stretch_with_passes(tmp_path, civil_env):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path)
    picks = []
    for i in range(40):
        y = 80 + i % 10
        vid = playcaller.make_call(_sit(1 + i % 2, 10, y, session="b1020"), "cpu", db).scheme_variant["variant_id"]
        assert vid in REG | RZ20, vid
        picks.append(vid)
        _close_snap(db, "b1020", vid, y)
    run_ids = RUNS | {"civil_rz_i_form_close_flex_stretch"}
    assert sum(p in run_ids for p in picks) >= 8 and sum(p in CORE for p in picks) >= 10, picks


def test_25_run_call_shows_its_variation_in_html_live(tmp_path, civil_env):
    from cfb_coach.live_server import LivePlayController, _call_main, render_live_html
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.situation import parse_madden_situation
    from cfb_coach.madden.model.offense_adjustment_designer import validate_live_call
    import html as _html
    db = _db(tmp_path)
    ctrl = LivePlayController(db=db, opponent_id="cpu", make_call=playcaller.make_call,
                              parse_situation=parse_madden_situation, learn_summary=lambda: "ok",
                              brand="Madden 27 Franchise", cpu_only=True, enable_execution_verify=True)
    call = None
    for i in range(40):
        c = playcaller.make_call(_sit(1, 10, 40, session=f"r{i}"), "cpu", db)
        if c.scheme_variant["variant_id"] in RUNS:
            call = c
            break
    assert call is not None
    validate_live_call(db, call, opponent_id="cpu", yardline=40)
    sv = call.scheme_variant
    v = CAT.get(sv["variant_id"])
    assert (call.formation, call.play) == (v.formation, v.base_play) and sv["play_family"] == "run"
    label = f"{v.base_play} [{v.variant_id}]"                # variation = the stock play name
    ctrl.last_call, ctrl.call_text = call, call.format()
    assert _call_main(ctrl) == f"{call.formation} — {call.play} · {label}"
    page = render_live_html(ctrl)
    assert f"VARIATION: {_html.escape(label)}" in page
    assert "none (catalog lists no adjustments: stock play)" in call.format().splitlines()[0]



VERIFIED_NAMES_T358U = {
    "civil_bucs_mtn_shuffle_verts_smash_return_motion_unverified": "MTN Shuffle Verts Smash — Return Motion",
    "civil_bucs_mesh_2_provisional": "Mesh 2",
    "civil_bucs_mesh_4_provisional": "Mesh 4",
    "civil_bucs_mtn_cross_hb_sneak_corner_flat_unverified": "Mtn Cross HB Sneak — Corner/Flat",
}


def test_26_owner_verified_variation_names_t358u():
    """t358u: the four names are owner-verified; catalog bytes and variant_ids stay pinned."""
    assert CAT.version() == "lions_civil_bucs_5wr_v4@15c120decc4b"
    assert CAT.names_owner.get("source") == "t358u"
    for vid, name in VERIFIED_NAMES_T358U.items():
        v = CAT.get(vid)
        assert v is not None, vid
        assert v.variant_id == vid                         # IDs never renamed
        assert v.variant_name == name
        assert v.name_verified is True
        assert "(name unverified)" not in v.variant_name
    # the 26-play catalog file itself was not edited
    raw = json.loads((sc.repo_root() / sc.DATASET_RELPATH).read_text())
    by = {p["variant_id"]: p for p in raw["plays"]}
    assert "(name unverified)" in by["civil_bucs_mesh_2_provisional"]["variant_name"]
    # live cards must show the verified name, not the old unverified label
    v = CAT.get("civil_bucs_mesh_2_provisional")
    from cfb_coach.madden.scheme.coordinator import variation_text
    text = variation_text(v.variant_name, v.variant_id, [a.line() for a in v.adjustments])
    assert "Mesh 2" in text and "(name unverified)" not in text


def test_26b_name_verification_loader_fails_closed_on_a_bad_owner_file():
    owner = json.loads((sc.repo_root() / sc.NAMES_RELPATH).read_text())
    bad = dict(owner, catalog_version="lions_civil_bucs_5wr_v4@deadbeef")
    with pytest.raises(sc.CatalogError, match="name-verification owner file is for"):
        sc.with_owner_verified_names(CAT, bad, digest="x")
    missing = dict(owner, variants=[{"variant_id": "not_a_real_variant", "variant_name": "Nope"}])
    missing["catalog_version"] = CAT.version()
    with pytest.raises(sc.CatalogError, match="unknown variant_id"):
        sc.with_owner_verified_names(CAT, missing, digest="x")



# owner instruction t370u (Michael 59-42 audit): mix sibling variants, get aggressive when trailing late ----------
MESH_OUT, MESH_2, MESH_5 = ("civil_bucs_bunch_te_mesh_post_outside_bunch_wr", "civil_bucs_mesh_2_provisional",
                            "civil_bucs_mesh_5")
ZSS = "civil_5wr_z_spot_shake_1"
BOOMERANG, SAME_SIDE = "civil_run_gun_bunch_te_boomerang_hb_off_tackle", "civil_run_gun_doubles_clamp_stack_same_side_zone"
RZ_PASSES = {"civil_rz_pistol_bunch_te_spot_y_option_1", "civil_rz_i_form_close_flex_mesh"}


def _sit_score(down, distance, yardline, us, them, quarter, session="g1"):
    s = _sit(down, distance, yardline, session=session)
    s.score_us, s.score_them = us, them
    s.extras["quarter"] = quarter
    return s


def _mesh_scorer(tmp_path):
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))
    vs = [CAT.get(v) for v in (MESH_OUT, MESH_2, MESH_5, ZSS)]
    kw = dict(down=1, distance=10, yardline=40, opponent_id="michael", opponent_type="human",
              belief=bl.combine({}, 0.1, 0, "none", []), offset_override={v.variant_id: 0.5 for v in vs})
    return s, vs, kw


def test_27a_history_cannot_freeze_out_unused_siblings(tmp_path):
    """Michael game: Mesh Outside (heavy verified history) ran 6x, Mesh 2-5 never; history lift was up to +0.60."""
    s, vs, kw = _mesh_scorer(tmp_path)
    stats = {MESH_OUT: (20.0, 20.0), ZSS: (20.0, 20.0)}            # strong verified history; Mesh 2/5 unused
    r = {x.variant.variant_id: x for x in s.score(vs, variant_stats=stats, **kw)}
    assert r[MESH_OUT].parts["variant_residual"] <= co.RESIDUAL_POS_CAP < 0.6
    assert r[MESH_OUT].parts["variant_residual_uncapped"] > co.RESIDUAL_POS_CAP     # the cap is what bit
    assert r[MESH_5].parts["novelty_bonus"] > r[MESH_OUT].parts["novelty_bonus"] > 0
    best = max(x.utility for x in r.values())
    for vid in (MESH_2, MESH_5):                                     # unused variants are in contention
        assert best - r[vid].utility <= co.MIX_MARGIN, (vid, best, r[vid].utility)
    # bad history still pulls a variant down by the full cap (only the LIFT is capped)
    bad = {x.variant.variant_id: x for x in s.score(vs, variant_stats={MESH_OUT: (0.0, 40.0)}, **kw)}
    assert bad[MESH_OUT].parts["variant_residual"] < -co.RESIDUAL_POS_CAP
    # runs and one-variant plays get no novelty lift (it only rotates sibling variations)
    run = s.score([CAT.get(BOOMERANG)], **{**kw, "offset_override": {BOOMERANG: 0.5}})[0]
    assert run.parts["novelty_bonus"] == 0.0


def test_27b_mesh_outside_does_not_loop_and_siblings_stay_independent(tmp_path):
    s, vs, kw = _mesh_scorer(tmp_path)
    stats = {MESH_OUT: (20.0, 20.0), ZSS: (20.0, 20.0)}
    hist = [MESH_OUT, ZSS, MESH_OUT]                                 # Mesh Outside called several times already
    ranked = s.score(vs, recent_calls=hist, game_calls=hist, variant_stats=stats, **kw)
    picks = [co.pick(ranked, f"m{i}")[0].variant.variant_id for i in range(200)]
    assert {MESH_2, MESH_5} & set(picks), picks
    assert picks.count(MESH_OUT) < len(picks)
    # t338u: calling variant 1 (and its verified outcomes) never changes variant 2's score
    fresh = {x.variant.variant_id: x for x in s.score(vs, **kw)}
    after = {x.variant.variant_id: x for x in s.score(vs, recent_calls=[MESH_OUT] * 5, game_calls=[MESH_OUT] * 5,
                                                       variant_stats={MESH_OUT: (20.0, 20.0)}, **kw)}
    for vid in (MESH_2, MESH_5):
        assert after[vid].utility == fresh[vid].utility and after[vid].parts["repeat_penalty"] == 0
        assert after[vid].parts["novelty_bonus"] == fresh[vid].parts["novelty_bonus"]
    # ...and does not reduce variant 2's chance of being the call
    p2 = lambda rk: sum(co.pick(rk, f"i{i}")[0].variant.variant_id == MESH_2 for i in range(300))
    assert p2(s.score(vs, recent_calls=[MESH_OUT] * 5, game_calls=[MESH_OUT] * 5, **kw)) >= p2(s.score(vs, **kw))


def test_27c_near_best_siblings_are_mixed_and_logged():
    ranked = [_fake(MESH_OUT, 0.60), _fake(ZSS, 0.55), _fake(MESH_5, 0.50), _fake(MESH_2, 0.49),
              _fake("civil_bucs_mesh_3", 0.30)]                       # Mesh 3 is not near-best: never mixed in
    picks = {}
    for i in range(300):
        top, mix = co.pick(ranked, f"s{i}")
        picks[top.variant.variant_id] = picks.get(top.variant.variant_id, 0) + 1
    assert picks.get(MESH_5, 0) > 0 and picks.get(MESH_2, 0) > 0 and "civil_bucs_mesh_3" not in picks, picks
    assert picks[MESH_OUT] > picks[MESH_5]                           # still ML-ordered, not uniform
    mixes = [co.pick(ranked, f"s{i}")[1] for i in range(50)]
    mesh = [m for m in mixes if m["sibling_mix"]["play_drawn"] != ZSS]
    assert mesh and all({v for v, _ in m["sibling_mix"]["pool"]} == {MESH_OUT, MESH_5, MESH_2} for m in mesh)
    assert "sibling mix" in co.mix_explanation(mesh[0])               # logged for audits


def test_27d_trailing_late_long_yardage_passes_not_a_designed_run():
    """Down 17 in Q4: Boomerang on 2&20 and Same Side Zone on 2&10 at the opp 40 were picked over higher-ranked passes."""
    trail = co.trailing_late(_sit_score(2, 20, 32, 21, 38, 4))
    assert trail and trail["margin"] == -17 and trail["strength"] == 1.0
    for sit_args, flip in (((2, 20, 32), BOOMERANG), ((2, 10, 60), SAME_SIDE)):
        ranked = [_fake("civil_bucs_mesh_3", 0.50), _fake("civil_bucs_texas_y_stutter_wheel_1", 0.49),
                  _fake(flip, 0.45), _fake("civil_run_gun_bunch_te_inside_zone", 0.44)]
        dn, dist, y = sit_args
        p, info = co.run_probability(ranked, down=dn, distance=dist, trail=trail)
        assert p == 0.0 and info["trailing_late"]["rule"]
        for i in range(100):
            top, mix = co.pick_mixed(ranked, f"t{i}", down=dn, distance=dist, yardline=y, trail=trail)
            assert not top.variant.is_run, (sit_args, top.variant.variant_id)
        assert "trailing late 21-38" in co.mix_explanation(mix)
        # same snap tied in the 2nd quarter, or ahead late: the normal run mix still applies
        for other in (co.trailing_late(_sit_score(dn, dist, y, 21, 21, 2)), co.trailing_late(_sit_score(dn, dist, y, 38, 21, 4)),
                      co.trailing_late(_sit_score(dn, dist, y, 21, 38, 2))):
            assert other is None
    # a run that out-ranks every pass on short yardage can still be called late, but rarely (capped)
    short = [_fake("civil_bucs_mesh_3", 0.45), _fake("civil_run_gun_bunch_te_inside_zone", 0.60)]
    assert 0 < co.run_probability(short, down=3, distance=1, trail=trail)[0] <= co.TRAIL_RUN_P_MAX
    assert co.run_probability(short, down=3, distance=1)[0] > co.TRAIL_RUN_P_MAX


def test_27e_trailing_late_live_calls_pass_and_early_keeps_runs(tmp_path, civil_env):
    from cfb_coach.madden import playcaller
    db = _db(tmp_path)
    late = []
    for i in range(30):
        call = playcaller.make_call(_sit_score(2, 20, 32 + i, 21, 38, 4, session=f"late{i}"), "michael", db)
        sv = call.scheme_variant
        assert sv["status"] == "CALL" and sv["variant_id"] in REG
        late.append(sv["variant_id"])
        assert sv["mix"]["trailing_late"]["margin"] == -17 and "trailing late 21-38" in sv["explanation"]
    assert not set(late) & RUNS, late
    row = db.conn.execute("SELECT prediction_json, explanation FROM civil_scheme_decisions ORDER BY id DESC LIMIT 1").fetchone()
    assert json.loads(row[0])["mix"]["trailing_late"]["margin"] == -17 and "trailing late" in row[1]
    early = [playcaller.make_call(_sit_score(1, 10, 40, 14, 14, 2, session=f"e{i}"), "michael", db)
             .scheme_variant["variant_id"] for i in range(40)]
    assert set(early) & RUNS, early                                  # tied / early: designed runs still called


def test_27f_trailing_late_inside_the_10_mixes_in_red_zone_passes(tmp_path, civil_env):
    rz = [_fake("civil_rz_gun_empty_quads_qb_blast", 0.55), _fake("civil_rz_pistol_bunch_te_hb_zone", 0.54),
          _fake("civil_rz_pistol_bunch_te_spot_y_option_1", 0.45), _fake("civil_rz_i_form_close_flex_mesh", 0.44)]
    trail = co.trailing_late(_sit_score(1, 5, 95, 21, 38, 4))
    tl = [co.pick_mixed(rz, f"z{i}", down=1, distance=5, yardline=95, trail=trail)[0].variant.variant_id
          for i in range(200)]
    assert sum(v in RZ_PASSES for v in tl) >= 80 and sum(v not in RZ_PASSES for v in tl) >= 10
    plain = {co.pick_mixed(rz, f"z{i}", down=1, distance=5, yardline=95)[0].variant.variant_id for i in range(200)}
    assert not plain & RZ_PASSES                                     # not trailing late: unchanged
    from cfb_coach.madden import playcaller
    db = _db(tmp_path)
    got = []
    for i in range(20):
        sv = playcaller.make_call(_sit_score(1 + i % 3, 5, 95, 21, 38, 4, session=f"rz{i}"), "michael", db).scheme_variant
        assert sv["variant_id"] in RZ10 | RZ20
        got.append(sv["variant_id"])
    assert set(got) & RZ_PASSES, got
