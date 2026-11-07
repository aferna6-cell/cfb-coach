"""A knowledge-trained artifact inside the EXISTING live loop (route_aware / route_aware_preview), beside a default artifact.

Training data is SYNTHETIC. Nothing here proves model quality on real games; it proves the wiring, isolation and fallbacks.
"""
from __future__ import annotations

import json
import random

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden import playbook
from cfb_coach.madden.model import defense_knowledge_eval as ke
from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import experimental_live as xl
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import inference
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_route_ranker as rr
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_time_correct as tc
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import parse_madden_situation

SNAP = json.loads(oa.SNAPSHOT_PATH.read_text(encoding="utf-8"))["sides"]["offense"]
BOOK = {f: list(p) for f, p in SNAP["formations"].items()}


@pytest.fixture(scope="module")
def rows():
    return tc.annotate(ke.make_pressure_world_rows(n_games=18, snaps=40, seed=7, strength=2.0), knowledge=True)


@pytest.fixture(scope="module")
def artifacts(rows):
    train = [r for r in rows if int(r["game_id"][1:]) < 12]
    return {"default": sm.train(train, penalty_scale=1.0), "knowledge": sm.train(train, penalty_scale=1.0, defense_knowledge=True),
            "exp": em.train_experimental([dict(r) for r in train], side="offense")}


def _world(tmp_path, monkeypatch, artifacts, which):
    tmp_path = tmp_path / which                      # one directory (and database file) per world: the two fixtures must not share state
    tmp_path.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CFB_COACH_DB", str(tmp_path / "coach.db"))
    monkeypatch.delenv(rr.ENV_MODE, raising=False)
    monkeypatch.delenv(rr.ENV_ARTIFACT, raising=False)
    db = CoachDB(tmp_path / "madden27.db", seed={"opponents": {
        "cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu", "confidence": "low", "profile_json": "{}"}}})
    playbook._save_state(db, {"applied": {"offense": {
        "side": "offense", "mode": "custom", "name": "ML Designed Offense (custom)", "rev": 1, "source_book": "Buccaneers",
        "trimmed": True, "formations": BOOK, "audibles": {}, "core": list(BOOK), "locked_ts": "test"}}, "pending": {}})
    exp_path = tmp_path / "offense.json"
    em.save_artifact(artifacts["exp"], exp_path)
    xl.set_artifact_path(db, exp_path)
    db.set_meta(rr.META_ARTIFACT, str(sm.save_artifact(artifacts[which], tmp_path / f"{which}.json")))
    return db


@pytest.fixture()
def kdb(tmp_path, monkeypatch, artifacts):
    db = _world(tmp_path, monkeypatch, artifacts, "knowledge")
    yield db
    db.close()


@pytest.fixture()
def ddb(tmp_path, monkeypatch, artifacts):
    db = _world(tmp_path, monkeypatch, artifacts, "default")
    yield db
    db.close()


def _ctrl(db):
    c = LivePlayController(db=db, opponent_id="cpu", make_call=lambda sit, **kw: make_call(sit, "cpu", db, rng=random.Random(5), **kw),
                           parse_situation=parse_madden_situation, learn_summary=lambda: "ok", brand="Madden 27 Franchise",
                           cpu_only=True, enable_execution_verify=True)
    c.start()
    return c


def _prime(db, mode="route_aware"):
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    rr.set_mode(db, mode)


def test_knowledge_artifact_drives_the_live_ranker_and_scores_all_93_plays(kdb):
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    info = rr.info_of(c.last_call)
    assert info["applied"] is True and info["n_scored"] == 93 and not info.get("fallback_reason")
    assert info["latency_ms"] < 150


def test_typed_calls_pass_through_the_existing_form_and_untyped_pressure_now_reaches_the_model(kdb, ddb):
    """"Fire Zone 3" is stored by today's form as "pressure"; "Zero Blitz" as "Cover 0". Both artifacts see the same stored entries."""
    for db, knowledge in ((kdb, True), (ddb, False)):
        _prime(db)
        c = _ctrl(db)
        c.call_only("1&10 my 25")
        c.result_and_call(outcome="+6", sit_raw="2&4 my 31", their="Fire Zone 3")
        c.result_and_call(outcome="+2", sit_raw="3&2 my 33", their="Zero Blitz")
        rows = [r[0] for r in db.conn.execute("SELECT coverage_seen FROM snaps WHERE side='offense' AND session_id=? AND result IS NOT NULL ORDER BY id", (c.session_id,))]
        assert rows == ["pressure", "Cover 0"]                                 # the lossy channel, observed in the real database
        cov = rr.info_of(c.last_call)["coverage"]
        assert cov["pressure_n"] == 1 and cov["named_n"] == 1
        keys = {k for k, _ in cov["top"]}
        if knowledge:
            assert cov["basis"].endswith("+defense_knowledge") and ds.GENERIC_PRESSURE_KEY in keys
        else:
            assert cov["basis"] == "closed_manual_postplay_coverage_with_historical_prior" and not any(k.startswith("dk:") for k in keys)


def test_unrecognized_and_generic_labels_never_become_a_named_defense(kdb):
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+3", sit_raw="2&7 my 28", their="heavy pressure")
    c.result_and_call(outcome="+1", sit_raw="3&6 my 31", their="some made up call")
    cov = rr.info_of(c.last_call)["coverage"]
    assert cov["pressure_n"] == 1 and cov["named_n"] == 0
    assert "dk:cover_0" not in dict(cov["top"]) and "cover_0" not in dict(cov["top"]) and "dk:parsed:cover_0" not in dict(cov["top"])


def test_beliefs_stay_inside_the_game_and_move_with_what_was_entered(kdb):
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    shares = []
    for sit in ["2&4 my 31", "1&10 my 41", "2&8 my 43"]:
        c.result_and_call(outcome="+6", sit_raw=sit, their="Fire Zone 3")           # stored as "pressure"
        shares.append(dict(rr.info_of(c.last_call)["coverage"]["top"]).get(ds.GENERIC_PRESSURE_KEY, 0.0))
    assert shares == sorted(shares) and shares[-1] > shares[0] > 0
    c2 = _ctrl(kdb)
    c2.call_only("1&10 my 25")
    assert rr.info_of(c2.last_call)["coverage"]["pressure_n"] == 0 and rr.info_of(c2.last_call)["coverage"]["named_n"] == 0


def test_preview_mode_never_changes_the_call_for_a_knowledge_artifact(kdb):
    _prime(kdb, "route_aware_preview")
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    info = rr.info_of(c.last_call)
    assert info["applied"] is False and info["n_scored"] == 93


def test_a_stale_knowledge_artifact_is_refused_and_the_existing_ranker_takes_over(kdb, artifacts, tmp_path):
    stale = json.loads(json.dumps(artifacts["knowledge"].to_dict()))
    stale["defense_knowledge_digest"] = "0" * 20
    p = tmp_path / "stale.json"
    p.write_text(json.dumps(stale), encoding="utf-8")
    kdb.set_meta(rr.META_ARTIFACT, str(p))
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    info = rr.info_of(c.last_call)
    assert info["applied"] is False and "RouteAwareUnavailable" in info["fallback_reason"] and "catalog changed" in info["fallback_reason"]


def test_default_artifact_behaves_exactly_as_before_on_a_known_label(ddb):
    _prime(ddb)
    c = _ctrl(ddb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+6", sit_raw="2&4 my 31", their="cover 2 man")
    cov = rr.info_of(c.last_call)["coverage"]
    assert cov["named_n"] == 1 and cov["basis"] == "closed_manual_postplay_coverage_with_historical_prior"
    assert dict(cov["top"]).get("cover_2_man", 0.0) > 0 and not any(k.startswith("dk:") for k, _ in cov["top"])


def test_no_input_for_opponent_pre_snap_adjustments_was_added(kdb):
    """On offense the opponent's press/shade/drop/override changes are unobservable. This change adds no column, field or form entry for them."""
    cols = {r[1] for r in kdb.conn.execute("PRAGMA table_info(snaps)")}
    assert not {c for c in cols if any(k in c.lower() for k in ("shade", "press_adj", "opp_adjust", "opponent_adjust", "defensive_adjust"))}
    assert not any(k in ds.DK_NAMES for k in ("dk_opp_adjustment", "dk_shade", "dk_press_adjusted"))
    assert not any("adjust" in n for n in ds.DK_NAMES)
