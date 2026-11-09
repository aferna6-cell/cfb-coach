"""Football scenarios, knowledge distinctions, learning validation (synthetic, labelled) and CLI/preflight gates."""
from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

import pytest

from cfb_coach.madden.unified import knowledge as K, postgame as PG, pregame as PRE, preflight as PF, sim
from cfb_coach.madden.unified.cli import main as cli_main
from cfb_coach.madden.unified.coordinator import inventory_fp
from cfb_coach.madden.unified.opponent import OpponentModel
from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, OFF, FootballAction, GameState, OpponentAction, Outcome,
                                            SnapObservation)
from cfb_coach.madden.unified.world_model import DefSig, NotCalibrated, Profile, WorldModel, fit_state_value
from tests.unified.helpers import coordinator, inventory, play_snap, state


# ------------------------------------------------------------------------------ football scenarios
def test_short_yardage_prefers_the_run_and_long_yardage_prefers_the_pass():
    co = coordinator(OFF)
    assert co.recommend(state(OFF, seq=1, down=4, dist=1, yl=60)).reads is not None
    short = co.snaps[state(OFF, seq=1).snap_id].recommended.play
    long_ = co.recommend(state(OFF, seq=2, down=3, dist=12, yl=35))
    long_play = co.snaps[state(OFF, seq=2).snap_id].recommended.play
    assert short.play_type == "run"
    assert long_play.play_type in ("pass", "play_action", "screen", "rpo")


def test_heavy_pressure_belief_favours_quick_game_over_deep_shots():
    pr = K.defense_play("Nickel Over", "DBL Safety Blitz")
    om = OpponentModel(DEF, prior={pr.signature: 1.0})
    inv = {"Gun Trips X Nasty": ["Mtn Stick Wheel", "Quick Unders", "Slot Fade H Wheel", "Mtn Slot Fades", "Switch HB Wheel", "WR Smash"]}
    co = coordinator(OFF, inv=inv, opponent_model=om)
    play = co.recommend(state(OFF, down=3, dist=7)).play
    assert K.offense_play("Gun Trips X Nasty", play).features["deep"] is False


def test_defense_chooses_differently_when_the_offense_is_expected_to_run():
    run_sig, pass_sig = ("run", "inside_zone"), ("pass", "four_verts")
    a = coordinator(DEF, opponent_model=OpponentModel(OFF, prior={run_sig: 1.0})).recommend(state(DEF, down=2, dist=3))
    b = coordinator(DEF, opponent_model=OpponentModel(OFF, prior={pass_sig: 1.0})).recommend(state(DEF, down=2, dist=3))
    assert a.user.primary != b.user.primary  # the USER job follows the predicted threat


# ------------------------------------------------------------------------------ knowledge layer
def _sig(bucket):
    for id_, e in K.defense_catalog().items():
        if id_.startswith("inst:") and e.get("legacy_bucket") == bucket:
            f, p = id_[5:].split("|", 1)
            return K.defense_play(f, p).signature
    pytest.skip(f"{bucket} not in catalog")


def test_defensive_variants_stay_distinct():
    pairs = [("cover_2_man", "tampa_2"), ("cover_3_match", "cover3"), ("cover_4_quarters", "cover_4_palms"),
             ("cover_1_hole", "cover_1_robber"), ("cover_6", "cover_9")]
    for a, b in pairs:
        assert _sig(a) != _sig(b), (a, b)
    sky = K.defense_play("4-3 Over", "Cover 3 Sky").signature
    assert sky != _sig("cover_3_match") and sky[2].startswith("cover3")


def test_unknown_assignments_stay_unknown():
    rep = K.knowledge_coverage_report()
    assert rep["defense_fully_verified_11"] == 0 and rep["offense_with_verified_route"] == 0  # nothing invented
    p = K.defense_play("4-3 Over", "Cover 3 Sky")
    assert "defender_assignments" in p.unknown_fields and p.verified_assignments == 0
    o = K.offense_play("Gun Trips X Nasty", "Mesh Spot")
    assert "routes" in o.unknown_fields and not o.assignments


def test_types_enforce_the_label_rules():
    st = state(OFF)
    obs = SnapObservation(st)
    with pytest.raises(ValueError):
        obs.with_execution(None, EXEC_CONFIRMED)
    with pytest.raises(ValueError):
        OpponentAction(DEF, verification="guess")
    with pytest.raises(ValueError):
        GameState("g", 0, OFF, "x")
    assert obs.executed is None and not obs.labelled_for_own_learning


# ------------------------------------------------------------------------------ learning (SYNTHETIC ground truth)
def test_hierarchical_model_recovers_planted_effects_better_than_the_prior():
    import math
    games = sim.gen_games(30, 60, seed=2)
    rows, _ = PG.validate_labels(games)

    def err(make):
        se = n = 0
        order = sorted({r["game_order"] for r in rows})
        for g in order[2:]:
            m = make([r for r in rows if r["game_order"] < g])
            for r in [r for r in rows if r["game_order"] == g]:
                st, p, d, o = PG._row_objs(r)
                t = sim.true_mean(p.concept, d.family, 4.0 if p.play_type == "run" else 6.0)
                se += (m.predict(st, p, d).e_yards - t) ** 2
                n += 1
        return math.sqrt(se / n)

    assert err(PG.fit_on) < 0.75 * err(lambda tr: WorldModel())


def test_rare_play_borrows_from_its_concept_without_being_treated_as_observed():
    m = WorldModel()
    st = state(OFF)
    d = K.defense_play("4-3 Over", "Cover 1 Hole")
    ds = DefSig.from_signature(d.signature, d.play_id, d.features)
    seen = K.offense_play("Gun Trips X Nasty", "Hi Lo Cross")
    for _ in range(80):
        m.update(st, Profile.off(FootballAction(seen)), ds, Outcome(12.0, True, False, False, False, False))
    rare = K.offense_play("Gun Trips X Nasty", "Mesh Spot")  # never observed; same concept family? use shared concept
    shared = Profile("pass", seen.concept, "off:Other|Rare", dict(seen.features))
    cold = WorldModel().predict(st, shared, ds)
    borrowed = m.predict(st, shared, ds)
    assert borrowed.e_yards > cold.e_yards + 1.0  # borrowed strength from the concept...
    assert borrowed.level < 4 and borrowed.value_sd >= m.predict(st, Profile.off(FootballAction(seen)), ds).value_sd  # ...but more uncertain than the observed play


def test_opponent_belief_beats_a_frequency_baseline_when_a_habit_exists():
    games = sim.gen_games(12, 50, seed=5, habit=0.7)
    res = PG.eval_opponent(games, DEF)
    assert res["n"] > 100 and res["belief_better"], res


def test_state_value_refuses_to_fabricate_epa():
    with pytest.raises(NotCalibrated):
        fit_state_value([{"start": {"yardline": 30, "down": 1, "distance": 10}, "points": 3}] * 10)
    assert WorldModel().objective.calibrated is False


# ------------------------------------------------------------------------------ latency / chronology
@pytest.mark.parametrize("side", [OFF, DEF])
def test_hundred_live_calls_stay_inside_the_latency_budget(side):
    res = PF.run_cycle(coordinator(side), game_id="lat", n=100)
    assert res["latency_ms"]["p95"] <= 150.0, res
    assert res["card_action_mismatches"] == 0 and res["fallbacks"] == 0 and res["belief_evidence_monotone"]


def test_full_cycle_updates_the_next_recommendation_inputs():
    co = coordinator(OFF)
    first = co.opp.beliefs(state(OFF, seq=1))
    play_snap(co, state(OFF, seq=1), opp_call=("4-3 Over", "Cover 1 Hole"), yards=9.0)
    nxt = co.recommend(state(OFF, seq=2))
    assert "game(1)" in nxt.opponent_prediction_source and "game" not in first.prediction_source
    assert co.model.n_records == 1  # verified execution + revealed call + outcome -> one training row


# ------------------------------------------------------------------------------ preflight verdict logic
def _books(tmp_path, tamper=False):
    out = {}
    for s in (OFF, DEF):
        inv = inventory(s)
        out[s] = {"side": s, "fingerprint": ("X" if tamper else inventory_fp(s, inv)), "inventory": inv, "attested_by": "owner"}
    f = tmp_path / "books.json"
    f.write_text(json.dumps(out))
    return str(f)


def _db(tmp_path):
    p = tmp_path / "test.db"
    c = sqlite3.connect(p)
    c.executescript("create table snaps(id integer); create table game_sessions(session_id text, dynasty text); insert into game_sessions values('a','TEST');")
    c.commit(); c.close()
    return str(p)


def test_preflight_verdicts_are_honest(tmp_path):
    shadow = PF.preflight(calls=20)
    assert shadow["verdict"] == "SHADOW ONLY" and "offense_installed_book_confirmed_by_owner" in shadow["blocking"]
    db = _db(tmp_path)
    go = PF.preflight(calls=20, confirmed_books=_books(tmp_path), db=db)
    assert go["verdict"] == "GO", go["blocking"]
    nogo = PF.preflight(calls=20, confirmed_books=_books(tmp_path, tamper=True), db=db)
    assert nogo["verdict"] == "NO-GO"


def test_cli_preflight_and_game_prep(tmp_path, monkeypatch, capsys):
    assert cli_main(["preflight", "--calls", "10", "--out", str(tmp_path / "pf" / "r.md")]) == 0
    assert (tmp_path / "pf" / "r.md").exists() and "SHADOW ONLY" in capsys.readouterr().out
    full = PRE.candidate_records
    monkeypatch.setattr(PRE, "candidate_records", lambda side: full(side)[:40])
    assert cli_main(["game-prep", "--out", str(tmp_path / "prep"), "--opponent", "cpu"]) == 0
    prop = json.loads((tmp_path / "prep" / "proposal_offense.json").read_text())
    assert prop["status"] == "PROPOSAL_NOT_INSTALLED" and prop["confirmed_install"] is False and prop["n_formations"] <= 15
    prop_path = tmp_path / "prep" / "proposal_offense.json"
    assert cli_main(["confirm-install", "--proposal", str(prop_path), "--attested-by", "owner", "--confirmed-books", str(tmp_path / "b.json")]) == 0
    assert json.loads((tmp_path / "b.json").read_text())["offense"]["fingerprint"] == prop["proposed_fingerprint"]
