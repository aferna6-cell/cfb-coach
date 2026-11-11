"""ravens_defense_ml: the Lions Ravens two-formation (Nickel 2-4 Wide / Dime 2-3 Odd) defensive ML coordinator.

All databases here are synthetic tmp files. No real gameplay DB is opened or written.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
import stat
import time
from pathlib import Path

import pytest

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense import confirmations as cf
from cfb_coach.madden.ravens_defense import coordinator as co
from cfb_coach.madden.ravens_defense import data
from cfb_coach.madden.ravens_defense import legality as lg
from cfb_coach.madden.ravens_defense import mode as md
from cfb_coach.madden.ravens_defense import model as rm
from cfb_coach.madden.ravens_defense import opponent as op
from cfb_coach.madden.ravens_defense import outcome as oc
from cfb_coach.madden.ravens_defense import users as us
from cfb_coach.madden.ravens_defense.concepts import CLASSES, classify
from cfb_coach.situation import Situation

CAT = rc.load()
BOOK = rc.ravens_stock_book()                     # exact in-repo Ravens inventory for the two formations
FP = rc.installed_fingerprint(BOOK)
ALL_SETUP_CONF = cf.Confirmations(
    path="test", installed_book_fingerprint=FP,
    macros={m: {"exists": True, "executes": True, "verified_on": "2026-11-10"} for m in ("24_BASE", "DB_FIRE", "ZONE_IT")},
    coaching_profile={"profile_id": "p", "verified_on": "2026-11-10", "advanced_nickel_adjustments_active": False,
                      "compatible_call_ids": list(CAT.calls)})


def _snap(session="g1", order=0, concept="run", *, down=1, distance=10, ytg=60, success=False, yards=3.0,
          explosive=False, execution="recommended_unexecuted", cand=None, started="2026-11-01T00:00:00", opp="jaxon"):
    return data.DefSnap(order, session, opp, "human", "primary", started, f"{started}{order:04d}", order, order,
                        down, distance, ytg, concept, concept, None, True, success, yards, explosive,
                        False, False, None, execution, cand, "ok" if cand else "no_verified_execution")


def _history(n_games=6, per=30):
    out = []
    for g in range(n_games):
        for i in range(per):
            out.append(_snap(f"h{g}", i, CLASSES[(g + i) % len(CLASSES)], down=1 + i % 4, distance=3 + i % 9,
                             ytg=20 + (i * 7) % 70, success=(i % 3 == 0), started=f"2026-10-{10 + g:02d}T00:00:00"))
    return out


def _model(snaps=None):
    snaps = _history() if snaps is None else snaps
    return rm.train(snaps, CAT, belief_eval=None, source={"test": True})


def _decide(model, *, conf=ALL_SETUP_CONF, same=(), down=1, distance=10, ytg=60, shape=None, installed=BOOK):
    legal, _ = lg.legal_set(CAT, installed=installed, conf=conf)
    snap = co.SnapCtx("live", "jaxon", len(same), down, distance, ytg, shape)
    return co.decide(model, legal=legal, snap=snap, same_game=list(same), history=model.history, conf=conf,
                     cat=CAT, installed=installed)


# 1 -------------------------------------------------------------- the approved candidate set only
def test_01_only_twelve_approved_calls_in_two_formations():
    assert rc.contract_errors(CAT.raw) == []
    assert set(CAT.calls) == set(rc.APPROVED_CALL_IDS) and len(rc.APPROVED_CALL_IDS) == 12
    assert {c.formation for c in CAT.candidates} == {rc.NICKEL, rc.DIME}
    assert len(CAT.candidates) == 14                      # 12 calls + Zone It on Quarters and Palms
    assert all(rc.stock_catalog_check(CAT).values())      # every base play exists in the Ravens inventory
    fake = dataclasses.replace(CAT.candidates[0], candidate_id="nickel_over_cover1", base_play="Cover 1 Robber")
    assert lg.check(CAT, fake, installed=BOOK, conf=ALL_SETUP_CONF).reasons == ("NOT_AN_APPROVED_CANDIDATE",)


# 2 -------------------------------------------------------------- macros are part of the candidate
def test_02_dime_is_db_fire_2_blitz_only_and_nickel_stock_requires_24_base():
    by = CAT.by_id()
    dime = [c for c in CAT.candidates if c.formation == rc.DIME]
    assert [c.candidate_id for c in dime] == ["dime_dbfire2"] and dime[0].is_blitz and dime[0].required_macro == "DB_FIRE"
    for cid in ("nickel_cloud", "nickel_cover2", "nickel_tampa2", "nickel_cover9"):
        assert by[cid].required_macro == "24_BASE"
    bad = dataclasses.replace(by["dime_dbfire2"], required_macro=None)
    assert not lg.check(CAT, bad, installed=BOOK, conf=ALL_SETUP_CONF).legal          # a macro-less DB Fire 2 is not a candidate
    no_macro = dataclasses.replace(ALL_SETUP_CONF, macros={})
    v = lg.check(CAT, by["dime_dbfire2"], installed=BOOK, conf=no_macro)
    assert not v.legal and "MACRO_UNCONFIRMED:DB_FIRE" in v.reasons
    assert by["dime_dbfire2"].macro_display(CAT) and by["nickel_cloud"].macro_display(CAT)


# 3 -------------------------------------------------------------- Match Cover 2 vs Nickel Cover 2
def test_03_match_cover2_and_nickel_cover2_stay_distinct():
    by = CAT.by_id()
    m, n = by["match_cover2"], by["nickel_cover2"]
    assert m.base_play == n.base_play and m.formation == n.formation
    assert m.macros == () and n.macros == ("24_BASE",) and m.candidate_id != n.candidate_id
    assert "pkg_nickel_stock" in oc.axes(n) and "pkg_nickel_stock" not in oc.axes(m)
    assert rc.candidate_for_execution(CAT, rc.NICKEL, m.base_play, [])[0] == m
    assert rc.candidate_for_execution(CAT, rc.NICKEL, n.base_play, ["24 Base"])[0] == n
    assert rc.candidate_for_execution(CAT, rc.NICKEL, m.base_play, None) == (None, "macro_unconfirmed")
    assert oc.phi(m, "flood", ()) != oc.phi(n, "flood", ())                             # separate learned terms
    ranked = {x["candidate_id"] for x in _decide(_model())["ranked"]}
    assert {"match_cover2", "nickel_cover2"} <= ranked                                  # both compete, never merged


# 4 -------------------------------------------------------------- installed book, fail closed
def test_04_validates_installed_book_and_fails_closed():
    assert len(lg.legal_set(CAT, installed=BOOK, conf=ALL_SETUP_CONF)[0]) == 13          # Cover 9 alias blocked
    assert lg.legal_set(CAT, installed={}, conf=ALL_SETUP_CONF)[0] == []                 # not installed
    partial = {rc.NICKEL: BOOK[rc.NICKEL]}
    assert lg.legal_set(CAT, installed=partial, conf=ALL_SETUP_CONF)[0] == []           # fingerprint changed
    wrong_fp = dataclasses.replace(ALL_SETUP_CONF, installed_book_fingerprint="RAV-DEF-000000000000")
    assert lg.legal_set(CAT, installed=BOOK, conf=wrong_fp)[0] == []
    assert lg.legal_set(CAT, installed=BOOK, conf=cf.Confirmations())[0] == []          # nothing confirmed
    d = _decide(_model(), installed={})
    assert d["status"] == "NO_LEGAL_RAVENS_CALL" and co.card_lines(d)[0].startswith("PLAY: NO LEGAL RAVENS CALL")


def test_04b_malformed_confirmation_file_confirms_nothing(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"schema": cf.SCHEMA, "installed_book_fingerprint": FP,
                             "user_verifications": [{"call_id": "match_cover3", "source": "guide"}]}))
    c = cf.load(p)
    assert c.errors and c.installed_book_fingerprint is None and not c.present


# 5 -------------------------------------------------------------- Zone It
def test_05_zone_it_only_on_compatible_match_calls_and_never_auto_fires_after_one_explosive():
    zi = [c for c in CAT.candidates if c.optional_macro == "ZONE_IT"]
    assert {c.call_id for c in zi} == {"match_quarters", "match_palms"}
    model = _model()
    explosive = [_snap("live", 0, "verticals", success=True, yards=45.0, explosive=True)]
    d = _decide(model, same=explosive)
    assert d["status"] == "CALL" and not d["zone_it_selected"]
    assert any(s["candidate_id"].endswith("+zone_it") for s in d["ranked"])            # it competed


def test_05b_zone_it_can_win_only_with_verified_evidence():
    base = _history()
    games = [f"v{g}" for g in range(10)]
    verified = []
    for gi, g in enumerate(games):
        for i in range(24):
            cid = ["match_quarters+zone_it", "match_quarters", "match_cover3", "stock_cover4_zone"][i % 4]
            win = cid != "match_quarters+zone_it"                                     # Zone It stops the offense
            verified.append(_snap(g, i, "verticals", success=win, yards=20.0 if win else 1.0,
                                  execution="verified_ravens_execution", cand=cid, started=f"2026-11-{1 + gi:02d}T00:00:00"))
    one = _model(base + verified[:1])
    assert one.outcome.status == "PARTIAL_UNDER_EVIDENCE_BAR" and not _decide(one)["zone_it_selected"]
    full = _model(base + verified)
    assert full.outcome.status == "TRAINED"
    vert = [_snap("live", i, "verticals") for i in range(6)]
    d = _decide(full, same=vert)
    assert d["zone_it_selected"] and d["candidate"]["candidate_id"] == "match_quarters+zone_it"
    assert d["user"]["status"] == us.UNVERIFIED                                         # recalculated, not inherited


# 6 -------------------------------------------------------------- USER
def test_06_eighteen_users_unverified_unless_owner_verified_in_nickel_2_4_wide():
    rows = us.audit(CAT, cf.Confirmations())
    assert len(rows) == 18 and all(r["status"] == us.UNVERIFIED for r in rows)
    cand = CAT.by_id()["match_quarters"]
    ver = {"call_id": "match_quarters", "offense_shape": "trips", "formation": rc.NICKEL, "user": "FS",
           "verified_on": "2026-11-10", "source": "owner_in_game", "check": "Quarters on trips side"}
    conf = dataclasses.replace(ALL_SETUP_CONF, user_verifications=(ver,))
    assert us.resolve(CAT, cand, "trips", conf)["status"] == "LIVE_VALID"
    dime_ver = dataclasses.replace(conf, user_verifications=({**ver, "formation": rc.DIME},))
    assert us.resolve(CAT, cand, "trips", dime_ver)["status"] == us.UNVERIFIED       # Dime guide role does not transfer
    zi = CAT.by_id()["match_quarters+zone_it"]
    assert us.resolve(CAT, zi, "trips", conf)["status"] == us.UNVERIFIED             # Zone It recalculates USER
    assert us.resolve(CAT, cand, None, conf)["status"] == us.UNVERIFIED


# 7 -------------------------------------------------------------- live card
def test_07_live_card_is_play_user_macro_only():
    d = _decide(_model(), same=[_snap("live", i, "crossers") for i in range(8)], down=3, distance=8)
    lines = co.card_lines(d)
    assert lines[0].startswith("PLAY: ") and lines[1] in ("USER: UNVERIFIED",) and len(lines) <= 3
    if len(lines) == 3:
        assert lines[2].startswith("MACRO: ")
    for ln in lines:
        assert not any(w in ln.lower() for w in ("prob", "loss", "belief", "%", "safety"))


# 8 -------------------------------------------------------------- per-snap beliefs, no leakage
def test_08_opponent_belief_uses_only_earlier_snaps():
    model = _model()
    early = [_snap("live", i, "run") for i in range(4)]
    later = [_snap("live", 10 + i, "verticals") for i in range(20)]
    a = op.belief(model.opponent, model.belief_method, model.history, early, co.SnapCtx("live", "j", 5, 1, 10, 60))
    b = op.belief(model.opponent, model.belief_method, model.history, early + later,
                  co.SnapCtx("live", "j", 5, 1, 10, 60))
    assert a == b                                                                   # later snaps ignored
    cur = _snap("live", 5, "verticals")
    c = op.belief(model.opponent, model.belief_method, model.history, early + [cur], co.SnapCtx("live", "j", 5, 1, 10, 60))
    assert c == a                                                                   # the current snap's concept unused
    d = op.belief(model.opponent, model.belief_method, model.history, early + [_snap("live", 4, "verticals")],
                  co.SnapCtx("live", "j", 5, 1, 10, 60))
    assert d != a and abs(sum(d) - 1) < 1e-9                                         # updates per snap


# 9 -------------------------------------------------------------- red zone
def test_09_red_zone_is_model_scored_and_tampa2_never_forced():
    model = _model()
    for ytg in (5, 12, 19):
        d = _decide(model, ytg=ytg, down=2, distance=6)
        assert d["status"] == "CALL" and len(d["ranked"]) == 13
        assert d["ranked"][0]["candidate_id"] == d["candidate"]["candidate_id"]
        tampa = next(s for s in d["ranked"] if s["candidate_id"] == "nickel_tampa2")
        if d["candidate"]["candidate_id"] == "nickel_tampa2":
            assert tampa["defensive_loss"] == min(s["defensive_loss"] for s in d["ranked"])
    assert "rz" in " ".join(op.context_tokens(2, 6, 12)) or op.context_tokens(2, 6, 12) != op.context_tokens(2, 6, 60)


# 10 ------------------------------------------------------------- no invented labels
def test_10_outcome_model_untrained_without_verified_executions():
    model = _model()
    assert model.outcome.status == "UNTRAINED_PRIOR_ONLY"
    assert model.meta["verified_defensive_execution_examples"] == 0
    unknown = [_snap("u", i, "verticals", success=True, execution="unknown", cand=None) for i in range(300)]
    assert _model(_history() + unknown).outcome.status == "UNTRAINED_PRIOR_ONLY"
    other = [_snap("o", i, "run", execution="verified_other_execution") for i in range(300)]
    assert _model(_history() + other).meta["verified_defensive_execution_examples"] == 0
    assert classify("Stick") == "quick" and classify("Mesh Spot") in ("mesh", "quick")


# 11 ------------------------------------------------------------- artifacts + read-only training
def test_11_versioned_artifact_has_required_fields_and_db_is_read_only(tmp_path):
    out = rm.save(_model(), tmp_path / "art", {"artifact_version": "t", "held_out_evaluation": {}})
    man = rm.manifest(out)
    for k in ("architecture", "training_config", "verified_defensive_execution_examples", "training_cutoff_started_ts",
              "dataset", "feature_set", "outcome_model_status", "model_sha256", "artifact_version"):
        assert k in man, k
    assert man["model_sha256"] == hashlib.sha256((out / "model.json").read_bytes()).hexdigest()
    assert rm.load(out).outcome.status == "UNTRAINED_PRIOR_ONLY"
    body = json.loads((out / "model.json").read_text())
    body["meta"]["dataset"]["sha256"] = "0" * 64
    (out / "model.json").write_text(json.dumps(body))
    rm._CACHE.clear()
    with pytest.raises(ValueError):
        rm.load(out)
    p = tmp_path / "copy.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE snaps (id INTEGER PRIMARY KEY, session_id TEXT, side TEXT, result TEXT, down INT, distance INT, ts TEXT)")
    con.execute("CREATE TABLE game_sessions (session_id TEXT, opponent_id TEXT, started_ts TEXT)")
    con.execute("INSERT INTO snaps VALUES (1,'s','defense','+4',1,10,'t')")
    con.commit()
    con.close()
    before = data.sha256_file(p)
    ro = data.open_ro(p)
    assert len(data.read_snaps(ro)) == 1
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("CREATE TABLE x (a)")
    ro.close()
    assert data.sha256_file(p) == before
    from cfb_coach.madden.ravens_defense import cli
    os.chmod(p, 0o444)
    with pytest.raises(SystemExit):
        cli._require_backup_copy(p)                                                    # official mode-444 copies refused


# 12 ------------------------------------------------------------- latency
def test_12_decision_latency_under_150ms():
    model = _model()
    same = [_snap("live", i, CLASSES[i % 8]) for i in range(60)]
    _decide(model, same=same)
    t0 = time.perf_counter()
    d = _decide(model, same=same, down=3, distance=7, ytg=30)
    assert (time.perf_counter() - t0) * 1000 < co.LATENCY_TARGET_MS
    assert d["latency_ms"] < co.LATENCY_TARGET_MS


# 13 ------------------------------------------------------------- offense unchanged, modes independent
# Pinned to the #153 offense (designed runs + online logging, 447b5e3); the defense port must not touch them.
OFFENSE_PINS = {
    "cfb_coach/madden/scheme/coordinator.py": "91facbe3c932e852e1ebe609169be3b0b0f0069320d184aed7bf51cc58e20224",
    "cfb_coach/madden/scheme/legality.py": "e6c959dd4912a7f036c9c831ac31e1c6193c9c587e1fc10ac1167731d2393d36",
    "cfb_coach/madden/scheme/mode.py": "ef99c16e697b1fe25ab5f1709cf3d7ee307762856f370ffab2b48deb844fb031",
    "cfb_coach/madden/scheme/model.py": "af793fd4dfbe03c234ccf11cd0d6afeec8a555507ea041604222e637aab778a7",
    "cfb_coach/madden/scheme/catalog.py": "53e0244f72222785549c47df932e4d06d77c1d8c882db635ff3a482824b5bcaf",
    "datasets/madden27/lions_offensive_scheme_catalog.json": "15c120decc4bba0ae3dc23f1c5a4c33f183f42a1dd8bd44a4a823526692a77ee",
}


def test_13_offense_files_and_restrictions_unchanged():
    for rel, sha in OFFENSE_PINS.items():
        assert hashlib.sha256((rc.REPO / rel).read_bytes()).hexdigest() == sha, rel
    from cfb_coach.madden.scheme import catalog as sc
    off = sc.load()
    assert not ({v.formation for v in off.variants} & set(rc.FORMATIONS))         # separate restrictions
    assert md.ENV_MODE != __import__("cfb_coach.madden.scheme.mode", fromlist=["ENV_MODE"]).ENV_MODE


@pytest.fixture
def live_db(tmp_path, monkeypatch):
    from cfb_coach.madden import playbook
    from cfb_coach.madden.model.cli import open_explicit_madden_db
    db = open_explicit_madden_db(tmp_path / "madden27.db", read_only=False)
    # the live prep catalog has no Ravens defensive stock book, so the applied record is written by hand here
    rec = {"side": "defense", "mode": "custom", "name": "Ravens two-formation (test)",
           "formations": {f: list(BOOK[f]) for f in rc.FORMATIONS}, "audibles": {}, "core": list(rc.FORMATIONS),
           "rev": 1, "locked_ts": "2026-11-10T00:00:00+00:00", "reason": "test"}
    playbook._save_state(db, {"applied": {"defense": rec}, "pending": {}})
    art = rm.save(_model(), tmp_path / "art", {})
    conf = tmp_path / "conf.json"
    inst = rc.installed_two_formation_book(rec)
    conf.write_text(json.dumps({"schema": cf.SCHEMA, "installed_book_fingerprint": rc.installed_fingerprint(inst),
                                "macros": dict(ALL_SETUP_CONF.macros), "coaching_profile": dict(ALL_SETUP_CONF.coaching_profile)}))
    monkeypatch.setenv(cf.ENV, str(conf))
    monkeypatch.setenv(md.ENV_ARTIFACT, str(art))
    return db


def _dsit(down=3, distance=7, yardline=40):
    s = Situation(raw="t", down=down, distance=distance, yardline=yardline)
    s.side = "defense"
    s.extras = {"session_id": "g1"}
    return s


def test_13b_make_call_defense_uses_ravens_only_when_mode_on(live_db, monkeypatch):
    from cfb_coach.madden import playcaller
    monkeypatch.setenv(md.ENV_MODE, md.PREVIOUS)
    hit = []
    monkeypatch.setattr(playcaller, "_pick_defense", lambda *a, **k: hit.append(1) or (_ for _ in ()).throw(LookupError()))
    with pytest.raises(LookupError):
        playcaller.make_call(_dsit(), "jaxon", live_db)
    assert hit == [1]                                                                   # previous caller used by default
    monkeypatch.setenv(md.ENV_MODE, md.RAVENS)
    monkeypatch.setattr(playcaller, "_pick_defense", lambda *a, **k: (_ for _ in ()).throw(AssertionError("old D")))
    call = playcaller.make_call(_dsit(), "jaxon", live_db)
    assert call.ravens_defense["status"] == "CALL" and call.call_label == co.LABEL
    assert call.formation in rc.FORMATIONS
    text = call.format()
    assert text.splitlines()[0] == call.headline() and "USER: UNVERIFIED" in text
    assert call.ravens_defense["end_to_end_ms"] < co.LATENCY_TARGET_MS * 3                # generous CI bound


def test_13c_both_modes_enabled_together_offense_unchanged(live_db, monkeypatch):
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.scheme import mode as smd
    from cfb_coach.madden import playbook
    st = playbook._load_state(live_db)
    st["applied"]["offense"] = playbook.make_record("offense", "stock", "Buccaneers")
    playbook._save_state(live_db, st)
    monkeypatch.setenv(smd.ENV_MODE, smd.CIVIL)
    monkeypatch.setenv(md.ENV_MODE, md.RAVENS)
    import cfb_coach.madden.scheme.coordinator as scoord
    seen = []
    monkeypatch.setattr(scoord, "call_civil", lambda *a, **k: seen.append("civil") or (_ for _ in ()).throw(RuntimeError("stop")))
    d = playcaller.make_call(_dsit(), "jaxon", live_db)
    assert d.ravens_defense["status"] == "CALL" and seen == []                        # defense never enters civil
    s = Situation(raw="t", down=1, distance=10, yardline=35)
    s.extras = {"session_id": "g1"}
    import cfb_coach.madden.ravens_defense.coordinator as rcoord
    monkeypatch.setattr(rcoord, "call_ravens", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ravens on O")))
    with pytest.raises(Exception):
        playcaller.make_call(s, "jaxon", live_db)                                      # offense went to civil
    assert seen == ["civil"]


# 14 ------------------------------------------------------------- enable gates, postgame, revert
def test_14_enable_gates_fail_closed_postgame_never_promotes_and_revert(live_db, tmp_path, monkeypatch):
    from cfb_coach.madden.ravens_defense import cli
    monkeypatch.delenv(md.ENV_MODE, raising=False)
    art = Path(os.environ[md.ENV_ARTIFACT])
    gates = cli.release_gates(art, tmp_path / "madden27.db", cf.load())
    assert not gates["all_pass"] and gates["outcome_model_trained"].startswith("FAIL")
    assert md.defense_mode(live_db) == md.PREVIOUS                                     # default
    md.set_state(live_db, enabled=True, artifact=str(art), model_sha256="x")
    assert md.defense_mode(live_db) == md.RAVENS
    monkeypatch.setenv(md.ENV_MODE, md.PREVIOUS)
    assert md.defense_mode(live_db) == md.PREVIOUS                                     # --previous-defense
    monkeypatch.delenv(md.ENV_MODE)
    st = md.set_state(live_db, enabled=False, artifact=None, model_sha256=None)
    assert md.defense_mode(live_db) == md.PREVIOUS and st["previous"]["enabled"] is True
    a = cf.coaching_audit(CAT, cf.Confirmations())
    assert set(a["per_call"].values()) == {"PROFILE_UNCONFIRMED"}
    inc = dataclasses.replace(ALL_SETUP_CONF, coaching_profile={**ALL_SETUP_CONF.coaching_profile,
                                                                  "incompatible_call_ids": ["nickel_tampa2"]})
    assert cf.coaching_audit(CAT, inc)["per_call"]["nickel_tampa2"] == "INCOMPATIBLE"
    assert "COACHING_PROFILE:INCOMPATIBLE" in lg.check(CAT, CAT.by_id()["nickel_tampa2"], installed=BOOK, conf=inc).reasons
