"""Route-aware offense in the EXISTING live loop: selectable, time-correct, repeatable, safe.

All training data here is SYNTHETIC (planted signal / pipeline checks). Nothing proves model quality on real games.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden import playbook
from cfb_coach.madden.model import experimental_live as xl
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.model import inference
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_route_ranker as rr
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_synthetic as syn
from cfb_coach.madden.model import offense_time_correct as tc
from cfb_coach.madden.model.schema import CoachingMode
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import parse_madden_situation

SNAP = json.loads(oa.SNAPSHOT_PATH.read_text(encoding="utf-8"))["sides"]["offense"]
BOOK = {f: list(p) for f, p in SNAP["formations"].items()}
PAIRS = [(f, p) for f, ps in BOOK.items() for p in ps]


# --------------------------------------------------------------------------- shared fixtures

@pytest.fixture(scope="module")
def planted_rows():
    return tc.annotate(syn.make_rows(truth="contradict", n_games=36, snaps=60, seed=11))


@pytest.fixture(scope="module")
def art(planted_rows):
    return sm.train([r for r in planted_rows if int(r["game_id"][1:]) < 24])


@pytest.fixture()
def world(tmp_path, art, planted_rows, monkeypatch):
    """A real Madden DB with the real 93-play book applied, an experimental artifact, and a route-aware artifact."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CFB_COACH_DB", str(tmp_path / "coach.db"))
    monkeypatch.delenv(rr.ENV_MODE, raising=False)
    monkeypatch.delenv(rr.ENV_ARTIFACT, raising=False)
    db = CoachDB(tmp_path / "madden27.db", seed={"opponents": {
        "cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu", "confidence": "low", "profile_json": "{}"},
        "gavin": {"display_name": "Gavin", "team_now": "DET", "skill": "human", "confidence": "low", "profile_json": "{}"}}})
    playbook._save_state(db, {"applied": {"offense": {
        "side": "offense", "mode": "custom", "name": "ML Designed Offense (custom)", "rev": 1, "source_book": "Buccaneers",
        "trimmed": True, "formations": BOOK, "audibles": {}, "core": list(BOOK), "locked_ts": "test"}}, "pending": {}})
    exp = em.train_experimental([dict(r) for r in planted_rows if int(r["game_id"][1:]) < 24], side="offense")
    exp_path = tmp_path / "offense.json"
    em.save_artifact(exp, exp_path)
    xl.set_artifact_path(db, exp_path)
    art_path = sm.save_artifact(art, tmp_path / "route_aware.json")
    db.set_meta(rr.META_ARTIFACT, str(art_path))
    yield db, tmp_path
    db.close()


def _ctrl(db, opponent="cpu"):
    c = LivePlayController(
        db=db, opponent_id=opponent,
        make_call=lambda sit, **kw: make_call(sit, opponent, db, rng=random.Random(5), **kw),
        parse_situation=parse_madden_situation, learn_summary=lambda: "ok",
        brand="Madden 27 Franchise", cpu_only=True, enable_execution_verify=True)
    c.start()
    return c


def _direct_call(db, raw="3&5 my 50", opponent="cpu"):
    """One call for a FIXED session id, so repeated calls see identical state (the live controller mints a new id)."""
    sit = parse_madden_situation(raw)
    sit.extras = {"session_id": "fixed-session"}
    return make_call(sit, opponent, db, rng=random.Random(5))


def _info(ctrl):
    return rr.info_of(ctrl.last_call)


# --------------------------------------------------------------------------- learning, not lookup

def test_pipeline_learns_a_planted_coverage_effect_that_contradicts_football_convention(planted_rows, art):
    test = [r for r in planted_rows if int(r["game_id"][1:]) >= 24]
    ll = {"base": 0.0, "struct": 0.0}
    for r in test:
        y = he.offense_label(r, "success")
        s = sm.structure_any(r["action_formation"], r["action_play"])
        sc = sm.score(art, s, down=r["down"], distance=r["distance"], yardline=r["yardline"], opponent_id="cpu",
                      opponent_type="cpu", cov=r[tc.KEY])
        for k, p in (("base", sc.base_probability), ("struct", sc.probability)):
            p = min(0.999, max(0.001, p))
            ll[k] += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    assert ll["struct"] / len(test) < ll["base"] / len(test) - 0.003          # held-out GAMES, game-level split
    w = dict(zip(art.names, art.weights[1:]))
    assert w["am_man_shallow"] < -0.1          # learned: shallow shapes LOSE to man (planted truth contradicts convention)


def test_rankings_flip_with_the_coverage_belief_the_model_receives(art):
    shallow = sorted(((af_s, f, p) for f, p in PAIRS for af_s in [pf.af.art_features(f, p)["art_shallow_share"]]
                      if pf.af.art_features(f, p)["art_present"]))
    lo, hi = shallow[0], shallow[-1]
    def p(play, mix):
        return sm.score(art, sm.structure_any(play[1], play[2]), down=2, distance=6, yardline=40, opponent_id="cpu",
                        opponent_type="cpu", mix=mix).probability
    man, zone = {"cover_2_man": 1.0}, {"cover3": 1.0}
    gap_man = p(hi, man) - p(lo, man)
    gap_zone = p(hi, zone) - p(lo, zone)
    assert gap_man < gap_zone                   # same two plays, different belief -> different relative score


def test_tendency_is_learned_from_data_not_a_lookup_table(art):
    src = (Path(rr.__file__).read_text() + Path(sm.__file__).read_text()).lower()
    assert "cover_2_man\": \"" not in src and "beaten_by" not in src and "counter_table" not in src


# --------------------------------------------------------------------------- time-correctness / no leakage

def _row(game, opp, i, cov, result="+5", side="offense"):
    return {"game_id": game, "opponent_id": opp, "side": side, "coverage_seen": cov, "result": result, "i": i}


def test_same_game_coverage_only_flows_forward(planted_rows):
    rows = [_row("g1", "cpu", 0, "Cover 2 Man"), _row("g1", "cpu", 1, "Cover 2 Man"), _row("g1", "cpu", 2, "Cover 3")]
    out = tc.annotate(rows)
    assert [r[tc.KEY]["n_named"] for r in out] == [0, 1, 2]                   # row i never sees itself or later rows
    assert out[1][tc.KEY]["counts"] == {"cover_2_man": 1} and out[0][tc.KEY]["counts"] == {}
    # perturbing the CURRENT and FUTURE rows' coverage cannot change an earlier row's features
    rows2 = [dict(rows[0]), dict(rows[1], coverage_seen="Cover 4 Palms"), dict(rows[2], coverage_seen="Cover 0")]
    out2 = tc.annotate(rows2)
    assert out2[0][tc.KEY] == out[0][tc.KEY] and out2[1][tc.KEY] == out[1][tc.KEY]
    assert out2[2][tc.KEY] != out[2][tc.KEY]


def test_coverage_never_crosses_games_or_opponents_and_unclosed_or_defense_rows_are_ignored():
    rows = [_row("g1", "cpu", 0, "Cover 2 Man"), _row("g2", "cpu", 0, "Cover 3"),
            _row("g1", "gavin", 1, "Cover 4 Palms"), _row("g1", "cpu", 2, "Cover 2 Man", result=""),
            _row("g1", "cpu", 3, "Cover 2 Man", side="defense"), _row("g1", "cpu", 4, "Cover 3")]
    out = tc.annotate(rows)
    assert out[1][tc.KEY]["n_named"] == 0                                      # other game
    assert out[2][tc.KEY]["n_named"] == 0                                      # other opponent
    assert out[5][tc.KEY]["counts"] == {"cover_2_man": 1}                      # unclosed + defense rows excluded


def test_historical_prior_for_a_row_comes_from_earlier_games_only():
    rows = ([_row("g1", "cpu", i, "Cover 2 Man") for i in range(10)] +
            [_row("g2", "cpu", i, "Cover 3") for i in range(10)] + [_row("g3", "cpu", 0, "Cover 3")])
    pri = tc.prior_as_of(rows)
    assert pri[0] == {}                                                         # first game: no history
    assert pri[10] == {"cover_2_man": 1.0}                                      # g2 sees only g1
    assert pri[20] == pytest.approx({"cover_2_man": 0.5, "cover3": 0.5})        # g3 sees g1+g2, never itself


def test_live_blend_is_the_training_blend():
    from cfb_coach.madden.model import offense_live_coverage as lc
    prior = {"cover3": 1.0}
    a = lc.blend_counts({"cover_2_man": 3}, prior)
    b = tc.mix_from(prior, {"counts": {"cover_2_man": 3}})
    assert a == b and 0 < a["cover_2_man"] < 1 and a["cover_2_man"] == pytest.approx(3 / 11)


# --------------------------------------------------------------------------- selectable, default unchanged

def test_default_mode_is_unchanged_and_preview_is_off_by_default(world):
    db, _ = world
    assert rr.resolve_mode(db) == "experimental"
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    assert rr.info_of(c.last_call) is None and "ROUTE-AWARE" not in c.call_text


def test_env_override_wins_for_one_session_and_invalid_values_fall_back(world, monkeypatch):
    db, _ = world
    rr.set_mode(db, "route_aware_preview")
    monkeypatch.setenv(rr.ENV_MODE, "experimental")
    assert rr.resolve_mode(db) == "experimental"
    monkeypatch.setenv(rr.ENV_MODE, "nonsense")
    assert rr.resolve_mode(db) == "experimental"
    with pytest.raises(ValueError):
        rr.set_mode(db, "nonsense")


def test_preview_in_heuristic_mode_changes_nothing_and_writes_no_database_rows(world):
    db, tmp = world
    assert inference.resolve_mode(db) is CoachingMode.HEURISTIC
    plain = _direct_call(db)
    again = _direct_call(db)
    assert (plain.formation, plain.play) == (again.formation, again.play)           # same state -> same call
    n_snaps = db.conn.execute("SELECT COUNT(*) FROM snaps").fetchone()[0]
    rr.set_mode(db, "route_aware_preview")
    c = _direct_call(db)
    assert (c.formation, c.play) == (plain.formation, plain.play)                   # the call is untouched
    info = rr.info_of(c)
    assert info and info["applied"] is False and info["n_scored"] == 93
    assert "ROUTE-AWARE PREVIEW (not the call)" in "\n".join(rr.format_lines(info))
    assert db.conn.execute("SELECT COUNT(*) FROM snaps").fetchone()[0] == n_snaps   # no historical rows touched
    shadow = [json.loads(l) for l in rr.shadow_log_path().read_text().splitlines()]
    assert shadow and shadow[-1]["kind"] == "shadow_decision_not_an_outcome"
    ctrl = _ctrl(db)                                                                 # and it shows in the live window text
    ctrl.call_only("2&7 my 40")
    assert "ROUTE-AWARE PREVIEW (not the call)" in ctrl.call_text


def test_preview_never_runs_for_human_opponents(world):
    db, _ = world
    rr.set_mode(db, "route_aware_preview")
    c = _ctrl(db, opponent="gavin")
    c.call_only("2&7 my 40")
    assert rr.info_of(c.last_call) is None


def test_preview_alongside_experimental_keeps_the_experimental_call(world):
    db, _ = world
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    ref = _direct_call(db)
    assert (_direct_call(db).formation, _direct_call(db).play) == (ref.formation, ref.play)
    rr.set_mode(db, "route_aware_preview")
    c = _direct_call(db)
    info = rr.info_of(c)
    assert info and info["applied"] is False and info["n_scored"] == 93 and not info.get("fallback_reason")
    assert (c.formation, c.play) == (ref.formation, ref.play)


def test_primary_mode_scores_every_installed_play_and_recommends_an_installed_play(world):
    db, _ = world
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    rr.set_mode(db, "route_aware")
    c = _ctrl(db)
    c.call_only("3&7 my 38")
    info = rr.info_of(c.last_call)
    assert info["applied"] is True and info["n_scored"] == 93                       # no situational shortlist
    assert (c.last_call.formation, c.last_call.play) in set(PAIRS)
    ml = c.last_call.ml_experimental
    assert ml["n_eligible_candidates"] == 93 and not ml.get("fell_back")
    assert "ROUTE-AWARE:" in c.call_text and "expected coverage:" in c.call_text
    assert info["card"]["ADJUSTMENT"]                                              # always definitive: NO ADJUSTMENT or a plan


def test_primary_mode_falls_back_to_the_current_ranker_and_says_why(world, tmp_path):
    db, _ = world
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    rr.set_mode(db, "route_aware")
    db.set_meta(rr.META_ARTIFACT, str(tmp_path / "missing.json"))
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    info = rr.info_of(c.last_call)
    assert info["applied"] is False and "RouteAwareUnavailable" in info["fallback_reason"]
    assert (c.last_call.formation, c.last_call.play) in set(PAIRS) and not c.last_call.ml_experimental.get("fell_back")
    assert "unavailable, current ranker used" in c.call_text


def test_stale_artifact_is_refused_not_used(world, art, tmp_path):
    db, _ = world
    stale = sm.StructuralArtifact.from_dict(art.to_dict())
    stale.art_digest = "stale"
    p = sm.save_artifact(stale, tmp_path / "stale.json")
    with pytest.raises(rr.RouteAwareUnavailable):
        rr.load_artifact(p)


# --------------------------------------------------------------------------- the post-play workflow feeds the model

def test_existing_postplay_form_updates_coverage_beliefs_every_snap_and_stays_per_game(world):
    db, _ = world
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    rr.set_mode(db, "route_aware")
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    first = _info(c)
    assert first["coverage"]["named_n"] == 0                                        # nothing known before snap 1
    shares, ranks = [], []
    down = ["2&4 my 31", "1&10 my 41", "2&8 my 43", "3&3 my 48"]
    for i, sit in enumerate(down):
        out = c.result_and_call(outcome="+6" if i % 2 == 0 else "+1", sit_raw=sit, their="cover 2 man")
        assert out.get("ok", True) is not False
        info = _info(c)
        assert info["coverage"]["named_n"] == i + 1                                 # the form's coverage reached the model
        shares.append(dict(info["coverage"]["top"]).get("cover_2_man", 0.0))
        ranks.append([(r["formation"], r["play"], r["p"]) for r in info["route_aware_top"]])
    assert shares == sorted(shares) and shares[-1] > shares[0]                      # belief moves toward what was observed
    assert len({tuple(x) for x in ranks}) > 1                                       # rankings recomputed from the new belief
    # history rows are exactly what the existing form wrote: one closed offense snap per entry, nothing extra
    rows = db.conn.execute("SELECT coverage_seen FROM snaps WHERE side='offense' AND session_id=?", (c.session_id,)).fetchall()
    assert [str(r[0]).lower() for r in rows].count("cover 2 man") == 4
    # a different game (new session) of the same opponent starts with no same-game entries
    c2 = _ctrl(db)
    c2.call_only("1&10 my 25")
    assert _info(c2)["coverage"]["named_n"] == 0 and c2.session_id != c.session_id


def test_coverage_changes_move_the_belief_away_from_the_previous_look(world):
    db, _ = world
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    rr.set_mode(db, "route_aware")
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    seq = ["2&5 my 30"] * 4 + ["1&10 my 45", "2&6 my 49", "1&10 my 60", "2&4 my 66", "1&10 my 70"]
    man_share = []
    for i, sit in enumerate(seq):
        cov = "cover 2 man" if i < 4 else "cover 3 match"
        c.result_and_call(outcome="+4", sit_raw=sit, their=cov)
        man_share.append(dict(_info(c)["coverage"]["top"]).get("cover_2_man", 0.0))
    peak = max(man_share[:4])
    assert man_share[-1] < peak - 0.05          # not assuming the opponent repeats its previous coverage forever
    top = dict(_info(c)["coverage"]["top"])
    assert top.get("cover_3_match", 0.0) > 0.15


def test_pressure_only_and_unknown_labels_do_not_become_cover_zero_or_named_coverage(world):
    db, _ = world
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    rr.set_mode(db, "route_aware")
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+3", sit_raw="2&7 my 28", their="heavy pressure")
    cov = _info(c)["coverage"]
    assert cov["pressure_n"] == 1 and cov["named_n"] == 0
    assert "cover_0" not in dict(cov["top"])


# --------------------------------------------------------------------------- performance + CLI

def test_end_to_end_live_decision_stays_inside_the_150ms_budget(world):
    db, _ = world
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    rr.set_mode(db, "route_aware")
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    times = []
    sits = ["2&7 my 31", "3&2 my 40", "1&10 my 52", "2&9 my 61", "3&11 my 33", "1&10 my 85", "2&1 my 98"]
    for i in range(14):
        t = time.perf_counter()
        c.result_and_call(outcome="+3" if i % 2 else "incomplete", sit_raw=sits[i % len(sits)], their="cover 3 match")
        times.append((time.perf_counter() - t) * 1000)
        assert _info(c)["latency_ms"] < 150 and not _info(c)["fallback_reason"]
    assert sorted(times)[int(0.95 * (len(times) - 1))] < 400     # whole form submit+call, a generous CI bound


def test_cli_selects_previews_and_rolls_back_the_ranker(world, monkeypatch, capsys):
    from cfb_coach.madden.model import cli as ml_cli
    db, tmp = world
    monkeypatch.setattr(ml_cli, "_resolved_madden_db_path", lambda: tmp / "madden27.db")
    ns = lambda **k: argparse.Namespace(mode=k.get("mode"), artifact=k.get("artifact"))
    assert ml_cli.cmd_ml_offense_ranker(ns(mode="route_aware_preview")) == 0
    assert "stored): route_aware_preview" in capsys.readouterr().out
    assert ml_cli.cmd_ml_offense_ranker(ns(mode="experimental")) == 0
    assert "stored): experimental" in capsys.readouterr().out
    db.set_meta(rr.META_ARTIFACT, str(tmp / "nope.json"))
    monkeypatch.setattr(rr, "default_artifact_path", lambda: tmp / "none.json")
    assert ml_cli.cmd_ml_offense_ranker(ns(mode="route_aware")) == 2           # refuses without a valid artifact


# --------------------------------------------------------------------------- adjustments + legality in the live card

def test_live_card_adjustment_is_legal_and_reads_only_name_real_routes(world):
    db, _ = world
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    rr.set_mode(db, "route_aware")
    c = _ctrl(db)
    for i in range(8):
        c.call_only("2&8 my 35") if i == 0 else c.result_and_call(outcome="+4", sit_raw="2&8 my 35", their="cover 3")
        info = _info(c)
        # diagnostics: an intermittent failure of this test was seen only in long full-suite runs; keep everything needed to explain a recurrence
        ctx = f"snap {i}: call={c.last_call.formation}/{c.last_call.play} fallback={info.get('fallback_reason')!r} latency_ms={info.get('latency_ms')} card={info.get('card')}"
        card = info["card"]
        assert card["ADJUSTMENT"] == od.NO_ADJUSTMENT or card["ADJUSTMENT"].startswith(("HOT ROUTE", "MACRO")), ctx
        assert card["EXECUTABLE NOW"] == "yes", ctx
        rec = oa.index(oa.load_database())[(c.last_call.formation, c.last_call.play)]
        known = {s for s, v in rec["slots"].items() if v["category"] == "route"}
        for field in ("PRIMARY READ", "SECONDARY READ"):
            txt = card[field]
            assert txt.startswith(("UNKNOWN", "RUN", "RPO", "n/a")) or txt.split()[0] in known, f"{field}={txt!r} known={sorted(known)} {ctx}"   # never a made-up route


# --------------------------------------------------------------------------- nested model selection ("do no harm")

def test_selection_keeps_the_residual_off_on_pure_noise_and_on_for_a_real_signal(planted_rows):
    noise = tc.annotate(syn.make_rows(truth="expected", strength=0.0, n_games=14, snaps=50, seed=3))
    a = sm.train(noise)
    assert a.residual_active is False and "residual OFF" in a.selection["decision"]
    assert sm.score(a, sm.structure_any(*PAIRS[0]), down=1, distance=10, yardline=30).probability == pytest.approx(
        sm.score(a, sm.structure_any(*PAIRS[0]), down=1, distance=10, yardline=30).base_probability)   # = existing coordinator
    b = sm.train([r for r in planted_rows if int(r["game_id"][1:]) < 24])
    assert b.residual_active is True and "residual ON" in b.selection["decision"]
    assert b.selection["inner_games"] >= 2 and any(v is not None for v in b.selection["scales"].values())


def test_too_few_games_leaves_the_existing_coordinator_in_charge(planted_rows):
    few = [r for r in planted_rows if int(r["game_id"][1:]) < 4]
    a = sm.train(few)
    assert a.residual_active is False and "fewer than" in a.selection["decision"]


def test_identical_training_data_is_fitted_once(planted_rows):
    rows = [r for r in planted_rows if int(r["game_id"][1:]) < 10]
    a, b = sm.train(rows, penalty_scale=2.0), sm.train(rows, penalty_scale=2.0)
    assert sm.artifact_digest(a) == sm.artifact_digest(b) and a is not b
    changed = [dict(r) for r in rows]
    changed[0]["result"] = "+99"
    assert sm._rows_key(changed, 2.0, True, 12) != sm._rows_key(rows, 2.0, True, 12)


# --------------------------------------------------------------------------- display surfaces

def test_live_window_state_is_json_serializable_and_carries_the_route_aware_block(world):
    db, _ = world
    inference.set_mode(db, CoachingMode.EXPERIMENTAL)
    rr.set_mode(db, "route_aware")
    c = _ctrl(db)
    c.call_only("2&6 my 44")
    c.result_and_call(outcome="+3", sit_raw="3&3 my 47", their="cover 4 palms")
    state = json.loads(json.dumps(c.state()))
    block = state["ml_experimental"]["route_aware"]
    assert block["applied"] is True and block["coverage"]["named_n"] == 1
    assert {"route_aware_top", "current_ranker_top", "card"} <= set(block)
    assert "ROUTE-AWARE:" in state["call_text"]


def test_terminal_loop_helper_prints_the_same_route_aware_lines(world):
    from cfb_coach.madden.cli import _route_aware_text
    db, _ = world
    rr.set_mode(db, "route_aware_preview")
    call = _direct_call(db)
    text = _route_aware_text(call)
    assert "ROUTE-AWARE PREVIEW (not the call)" in text and "expected coverage:" in text and "adjustment:" in text
    rr.set_mode(db, "experimental")
    assert _route_aware_text(_direct_call(db)) == ""
