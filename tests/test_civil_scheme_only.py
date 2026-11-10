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
RZ_FORMS = ["Pistol Bunch TE", "Gun Empty Quads"]
FORMS_ALL = FORMS + RZ_FORMS
Z_SPOT = {"civil_5wr_z_spot_1", "civil_5wr_z_spot_shake_1"}
CORE = {v.variant_id for v in CAT.core_variants}
RZ = {v.variant_id for v in CAT.redzone_variants}


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
    assert len(CAT.core_variants) == 26 and len(CAT.redzone_variants) == 5 and len(set(CAT.ids)) == 31
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
    assert len(ok) == 29 and not (ok & Z_SPOT) and RZ <= ok


# 3 -------------------------------------------------------------- live legality
def test_03_owner_confirmed_adjustments_make_in_book_variants_legal_book_still_required(tmp_path):
    db = _db(tmp_path, formations=FORMS[:3])
    legal, verdicts = lg.legal_set(CAT, db=db)
    ids = {v.variant_id for v in legal}
    assert ids == {v.variant_id for v in CAT.variants if v.formation in FORMS[:3]}      # 17, no per-variant verify
    assert all(verdicts[i].checks["owner_confirmed_adjustments"] for i in ids)
    for v in CAT.variants:
        if v.formation == "Gun 5WR Tight":
            assert not verdicts[v.variant_id].legal                                     # formation not installed
    # the blanket owner confirmation is pinned to this catalog version
    import dataclasses
    other = dataclasses.replace(CAT, digest="0" * 64)
    assert not lg.owner_confirmation_applies(other)
    assert lg.legal_set(other, db=db)[0] == []


# 4 -------------------------------------------------------------- NO_LEGAL, no fallback
def test_04_no_legal_scheme_play_without_fallback(tmp_path, civil_env, monkeypatch):
    from cfb_coach.madden import catalog as mc, playcaller
    other = next(f for f in mc.book_formations("offense", "Buccaneers") if f not in FORMS_ALL)
    db = _db(tmp_path, formations=[other])
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
    for dd in ((1, 10, 25), (3, 2, 60), (2, 7, 85), (4, 1, 99)):
        call = playcaller.make_call(_sit(*dd), "cpu", db)
        sv = call.scheme_variant
        v = CAT.get(sv["variant_id"])
        assert v is not None and sv["variant_id"] not in Z_SPOT
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


def test_14_z_spot_only_with_an_owner_confirmed_book_name(tmp_path):
    db = _db(tmp_path)
    v = CAT.get("civil_5wr_z_spot_1")
    assert not lg.check(v, CAT, db=db, require_verified=False).legal          # no alias is inferred
    with pytest.raises(ValueError):
        lg.record_verified(db, CAT, v.variant_id, book_play="Not A Play")
    lg.record_verified(db, CAT, v.variant_id, book_play="L Spot")              # owner confirms the book name
    assert lg.check(v, CAT, db=db).legal


# owner instruction t333u (1): only the 26 listed variant IDs can ever be output -----------------
def test_15_only_the_listed_catalog_variants_can_be_output(tmp_path, civil_env, monkeypatch):
    import dataclasses
    import random
    from cfb_coach.madden import playcaller
    db = _db(tmp_path)
    allowed = set(CAT.ids)
    assert len(allowed) == 31 and allowed == CORE | RZ
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
        if call_yard(call) < 90:
            assert sv["variant_id"] in CORE
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
            out[0].variant = fake
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


# owner instruction t335u/t336u: red-zone package plays ONLY inside the opponent 10 -------------------------
def test_18_red_zone_plays_only_inside_the_10_before_scoring_and_before_display(tmp_path, civil_env, monkeypatch):
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.model.offense_adjustment_designer import validate_live_call
    db = _db(tmp_path)
    # the one legal filter, before scoring
    for y in range(0, 101):
        ids = {v.variant_id for v in lg.legal_set(CAT, db=db, yardline=y)[0]}
        assert (ids & RZ) == (RZ if y >= 90 else set()), y
    assert not ({v.variant_id for v in lg.legal_set(CAT, db=db, yardline=None)[0]} & RZ)   # unknown spot: out
    # live calls across the field
    rz_seen_in_ranking = False
    for y in list(range(1, 90, 4)) + list(range(90, 100)):
        for d, dist in ((1, 10), (3, 2), (4, 1)):
            call = playcaller.make_call(_sit(d, min(dist, 100 - y), y, session=f"s{y}"), "cpu", db)
            vid = call.scheme_variant["variant_id"]
            if y < 90:
                assert vid in CORE, (y, vid)
                assert not (set(call.scheme_variant["candidates"]) & RZ)
            else:
                assert RZ <= set(call.scheme_variant["candidates"])
                rz_seen_in_ranking = True
    assert rz_seen_in_ranking                       # inside the 10 they are real candidates
    # model-scored, not forced: inside the 10 a core play still wins when it scores higher
    s = co.Scorer(sm.load(_artifact(tmp_path)), CAT, dz.GeometrySource(enabled=False))
    legal = lg.legal_set(CAT, db=db, yardline=95)[0]
    bel = bl.combine({}, 0.1, 0, "none", [])
    over = {v.variant_id: (0.3 if v.variant_id in RZ else 0.6) for v in legal}
    assert s.score(legal, down=1, distance=5, yardline=95, opponent_id="cpu", opponent_type="cpu", belief=bel,
                   offset_override=over)[0].variant.variant_id in CORE
    over = {v.variant_id: (0.8 if v.variant_id == "civil_rz_gun_empty_quads_qb_blast" else 0.4) for v in legal}
    assert s.score(legal, down=1, distance=5, yardline=95, opponent_id="cpu", opponent_type="cpu", belief=bel,
                   offset_override=over)[0].variant.variant_id == "civil_rz_gun_empty_quads_qb_blast"
    # before display: a red-zone play injected outside the 10 is rejected (fail closed, no fallback)
    rzv = CAT.get("civil_rz_pistol_bunch_te_hb_zone")
    orig = co.Scorer.score

    def inject(self, variants, **kw):
        out = orig(self, variants, **kw)
        out[0].variant = rzv
        return out
    monkeypatch.setattr(co.Scorer, "score", inject)
    assert playcaller.make_call(_sit(1, 10, 50), "cpu", db).play == lg.NO_LEGAL
    monkeypatch.setattr(co.Scorer, "score", orig)
    # display gate: a red-zone call scored at the 5 cannot be shown for a snap at midfield
    call = None
    for i in range(40):
        c = playcaller.make_call(_sit(1, 5, 95, session=f"z{i}", look=["cover 2", "cover 3", "man"][i % 3]), "cpu", db)
        if c.scheme_variant["variant_id"] in RZ:
            call = c
            break
    if call is None:   # force one through the scorer for the display-gate check
        over = {v.variant_id: (0.9 if v.variant_id == rzv.variant_id else 0.1) for v in legal}
        top = s.score([rzv], down=1, distance=5, yardline=95, opponent_id="cpu", opponent_type="cpu", belief=bel,
                      offset_override=over)
        call = co.build_call(top[0], top, bel, book=None, shadow=False)
        call.scheme_variant["yardline"] = 95
    validate_live_call(db, call, opponent_id="cpu", yardline=95)
    with pytest.raises(ValueError):
        validate_live_call(db, call, opponent_id="cpu", yardline=50)
    tampered = call
    tampered.scheme_variant["yardline"] = 50
    with pytest.raises(ValueError):
        lg.revalidate_call(db, tampered, CAT)
    tampered.scheme_variant.pop("yardline")
    with pytest.raises(ValueError):
        lg.revalidate_call(db, tampered, CAT)
    # the unnamed 5WR setup stays blocked everywhere, including inside the 10
    for b in CAT.blocked_ids:
        assert not lg.check(None, CAT, db=db, variant_id=b, yardline=95).legal


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
