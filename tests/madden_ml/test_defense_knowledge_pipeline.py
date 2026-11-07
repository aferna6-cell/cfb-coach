"""The defense-knowledge features inside the REAL offense model pipeline: default behavior untouched, structures change inputs,
learned scores respond, controls stay flat, nothing leaks, the live defensive caller cannot load the offline artifact.

The planted-signal worlds are SYNTHETIC. They prove plumbing and the absence of invented signal; they say nothing about real games.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time

import pytest

from cfb_coach.madden.model import defense_catalog as dc
from cfb_coach.madden.model import defense_knowledge_eval as ke
from cfb_coach.madden.model import defense_knowledge_train as kt
from cfb_coach.madden.model import defense_logistic_train as dlt
from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import defense_v1
from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.model import offense_art_features as af
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_live_coverage as lc
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_time_correct as tc

GAMES = 18
TRAIN_GAMES = 12


def _games(rows):
    gs = []
    for r in rows:
        if r["game_id"] not in gs:
            gs.append(r["game_id"])
    return gs


def _split(rows):
    gs = _games(rows)
    return [r for r in rows if r["game_id"] in gs[:TRAIN_GAMES]], [r for r in rows if r["game_id"] not in gs[:TRAIN_GAMES]]


@pytest.fixture(scope="module")
def world():
    return tc.annotate(ke.make_pressure_world_rows(n_games=GAMES, snaps=40, seed=7, strength=2.0), knowledge=True)


@pytest.fixture(scope="module")
def arts(world):
    train, _ = _split(world)
    return sm.train(train, penalty_scale=1.0), sm.train(train, penalty_scale=1.0, defense_knowledge=True)


def _ll(art, rows):
    total = 0.0
    for r in rows:
        s = sm.structure_any(r["action_formation"], r["action_play"])
        p = sm.score(art, s, down=r["down"], distance=r["distance"], yardline=r["yardline"], opponent_id=r["opponent_id"],
                     opponent_type=r["opponent_type"], cov=r[tc.KEY]).probability
        y = he.offense_label(r, "success")
        p = min(1 - 1e-6, max(1e-6, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(rows)


def _play(shallow_rank):
    plays = sorted(((af.art_features(p["formation"], p["play"])["art_shallow_share"], p["formation"], p["play"])
                    for p in oa.load_database()["plays"] if af.art_features(p["formation"], p["play"])["art_present"]))
    return plays[shallow_rank]


# ------------------------------------------------------------------------------------------------ the default path is untouched

def test_default_feature_set_and_version_are_unchanged():
    names = pf.feature_names()
    assert len(names) == 135 and not any(n.startswith("dk_") for n in names)
    assert pf.FEATURE_VERSION == "offense_play_features.v4_situation_interactions"
    assert pf.feature_names(defense_knowledge=True)[:135] == names and len(pf.feature_names(True)) == 135 + len(ds.DK_NAMES)


def test_legacy_mix_gives_the_same_distribution_as_before_and_no_family_mix():
    d, src = pf.scheme_distribution({"cover3": 0.5, "cover_4_palms": 0.25, "cover_2_man": 0.25})
    assert dict(d) == {"cover_3_spot": 0.25, "cover_3_match": 0.25, "cover_4_palms": 0.25, "cover_2_man": 0.25}
    assert d.family_mix is None and src == "opponent_coverage_mix_uniform_within_bucket"
    e, esrc = pf.scheme_distribution({})
    assert dict(e) == {"unknown": 1.0} and esrc == "no_pre_snap_information"


def test_default_vector_ignores_defense_knowledge_keys(arts):
    p = _play(40)
    s = sm.structure_any(p[1], p[2])
    dk_dist, _ = pf.scheme_distribution({"dk:cover_3_sky": 0.5, "dk:fire_zone_3": 0.5})
    legacy_dist, _ = pf.scheme_distribution(ds.project_legacy({"dk:cover_3_sky": 0.5, "dk:fire_zone_3": 0.5}))
    assert pf.full_vector(s, dk_dist) == pf.full_vector(s, legacy_dist)
    assert pf.full_vector(s, dk_dist) == pf.full_vector(s, dk_dist, names=pf.feature_names())


def test_a_default_artifact_serializes_and_validates_exactly_as_before(arts):
    base, _ = arts
    d = base.to_dict()
    assert "defense_knowledge_digest" not in d and d["feature_version"] == pf.FEATURE_VERSION
    assert base.names == list(pf.feature_names()) and sm.validate_artifact(base) == [] and base.uses_defense_knowledge is False
    assert "dk" not in base.penalties


# ------------------------------------------------------------------------------------------------ structures change the inputs

CALLS = ["Cover 3 Sky", "Cover 3 Cloud", "Cover 3 Buzz Match", "Cover 2 Man", "Cover 4 Quarters", "Cover 4 Palms", "Fire Zone 3",
         "Zero Blitz", "Nickel Sim 2", "Blitz Loop 3"]


def _block(label, play):
    s = sm.structure_any(play[1], play[2])
    d, _ = pf.scheme_distribution({ds.resolve(label).key: 1.0})
    names = pf.feature_names(True)
    v = pf.full_vector(s, d, None, names=names)
    return dict(zip(names, v)), v


def test_every_pair_of_different_defensive_structures_gives_a_different_model_input():
    play = _play(60)
    vecs = {c: tuple(_block(c, play)[1]) for c in CALLS}
    for a in CALLS:
        for b in CALLS:
            if a < b:
                assert vecs[a] != vecs[b], f"{a} == {b}"


def test_the_legacy_block_cannot_tell_sky_from_cloud_but_the_knowledge_block_can():
    play = _play(60)
    (fa, va), (fb, vb) = _block("Cover 3 Sky", play), _block("Cover 3 Cloud", play)
    legacy = [n for n in pf.feature_names()]
    assert [fa[n] for n in legacy] == [fb[n] for n in legacy]            # same parent class -> same legacy inputs
    assert [fa[n] for n in ds.DK_NAMES] != [fb[n] for n in ds.DK_NAMES]


def test_pressure_type_moves_the_interaction_inputs_and_scales_with_play_geometry():
    deep, shallow = _play(0), _play(-1)
    zd, _ = _block("Fire Zone 3", deep)
    zs, _ = _block("Fire Zone 3", shallow)
    cs, _ = _block("Cover 3 Cloud", shallow)
    assert zs["dk_xa_zblitz_shallow"] > zd["dk_xa_zblitz_shallow"] > 0
    assert zs["dk_xa_zblitz_shallow"] / zd["dk_xa_zblitz_shallow"] == pytest.approx(shallow[0] / deep[0], rel=1e-6)
    assert cs["dk_xa_zblitz_shallow"] == cs["dk_xa_press_shallow"] == 0.0     # no pressure axis on a coverage-only call
    mb, _ = _block("Zero Blitz", shallow)
    assert mb["dk_xa_mblitz_shallow"] > 0 and zs["dk_xa_mblitz_shallow"] == 0.0


def test_untyped_pressure_now_reaches_the_model_instead_of_vanishing():
    play = _play(60)
    f, _ = _block("pressure", play)
    assert f["dk_press_unspecified"] == 1.0 and f["dk_press_any"] == 1.0
    g, _ = _block("totally unknown label", play)
    assert g["dk_ev_info"] == 0.0 and all(g[n] == 0.0 for n in ds.AXIS_NAMES)    # unknown adds NO information


def test_unknown_mix_is_zero_information_not_a_default_coverage():
    s = sm.structure_any(*_play(10)[1:])
    d, _ = pf.scheme_distribution({})
    v = dict(zip(pf.feature_names(True), pf.full_vector(s, d, None, names=pf.feature_names(True))))
    assert all(v[n] == 0.0 for n in ds.DK_NAMES)


# ------------------------------------------------------------------------------------------------ the knowledge artifact

def test_knowledge_artifact_is_versioned_pinned_and_validated(arts, tmp_path):
    _, dk = arts
    assert dk.feature_version == pf.FEATURE_VERSION_DK and dk.uses_defense_knowledge and len(dk.names) == 135 + len(ds.DK_NAMES)
    assert dk.defense_knowledge_digest == ds.data_digest() == dc.load_catalog()["digest"]
    assert sm.validate_artifact(dk) == [] and dk.penalties["dk"] == pytest.approx(30.0 * 1.0)
    back = sm.load_artifact(sm.save_artifact(dk, tmp_path / "dk.json"))
    assert sm.validate_artifact(back) == [] and back.uses_defense_knowledge


def test_a_stale_catalog_or_edited_schema_invalidates_the_knowledge_artifact(arts):
    import copy
    _, dk = arts
    stale = copy.deepcopy(dk)
    stale.defense_knowledge_digest = "0" * 20
    assert any("catalog changed" in e for e in sm.validate_artifact(stale))
    wrong = copy.deepcopy(dk)
    wrong.names = wrong.names[:-1]
    assert any("feature names differ" in e for e in sm.validate_artifact(wrong))
    mixed = copy.deepcopy(arts[0])
    mixed.defense_knowledge_digest = ds.data_digest()
    assert any("non-knowledge artifact" in e for e in sm.validate_artifact(mixed))


def test_a_default_artifact_cannot_be_scored_with_the_knowledge_vector_by_accident(arts):
    base, dk = arts
    s = sm.structure_any(*_play(30)[1:])
    p_base = sm.score(base, s, down=3, distance=7, yardline=45, opponent_id="cpu", mix={"dk:fire_zone_3": 1.0}).probability
    p_leg = sm.score(base, s, down=3, distance=7, yardline=45, opponent_id="cpu", mix={"unknown": 1.0}).probability
    assert p_base == pytest.approx(p_leg)          # the default artifact only ever sees the legacy projection (unknown here)


# ------------------------------------------------------------------------------------------------ learned scores respond (SYNTHETIC)

def test_trained_scores_respond_to_pressure_versus_coverage_and_the_baseline_cannot(arts):
    base, dk = arts
    lo, hi = _play(0), _play(-1)

    def contrast(art, mix):
        f = lambda play: sm.score(art, sm.structure_any(play[1], play[2]), down=3, distance=7, yardline=45,
                                  opponent_id="cpu", opponent_type="cpu", mix=mix).probability
        return f(hi) - f(lo)

    press, cover = {"dk:fire_zone_3": 1.0}, {"dk:cover_3_cloud": 1.0}
    assert contrast(dk, press) > 0.3 and contrast(dk, cover) < -0.2        # the planted effect, recovered with the right signs
    assert abs(contrast(base, {"unknown": 1.0}) - contrast(base, {"cover3": 1.0})) < 0.15   # the legacy pipeline is blind to it


def test_knowledge_model_beats_the_identical_model_without_it_when_there_is_signal(world, arts):
    base, dk = arts
    _, test = _split(world)
    assert _ll(dk, test) < _ll(base, test) - 0.01


def test_no_gain_in_a_null_world_or_after_a_global_label_shuffle():
    for rows in (ke.make_pressure_world_rows(n_games=GAMES, snaps=40, seed=7, strength=2.0, null=True),
                 ke.permute_coverage_labels(ke.make_pressure_world_rows(n_games=GAMES, snaps=40, seed=7, strength=2.0), seed=3, mode="global")):
        rows = tc.annotate(rows, knowledge=True)
        train, test = _split(rows)
        b, d = sm.train(train, penalty_scale=1.0), sm.train(train, penalty_scale=1.0, defense_knowledge=True)
        assert _ll(d, test) > _ll(b, test) - 0.01, "the knowledge model must not invent a gain from noise"


def test_the_within_game_shuffle_is_documented_as_not_a_null_control():
    doc = ke.permute_coverage_labels.__doc__
    assert "NOT a null control" in doc and "global" in doc


# ------------------------------------------------------------------------------------------------ no future leakage, isolation

def _row(game, opp, i, label, result="+5", side="offense"):
    return {"game_id": game, "session_id": game, "opponent_id": opp, "side": side, "result": result, "coverage_seen": label, "id": i}


def test_annotation_uses_only_earlier_closed_same_game_same_opponent_labels():
    # lossless names (the form cannot produce them), so each reads exactly
    rows = [_row("g1", "A", 1, "Cover 3 Cloud"), _row("g1", "B", 2, "Cover 4 Palms"), _row("g1", "A", 3, "Fire Zone 3"),
            _row("g1", "A", 4, "Zero Blitz", result=""), _row("g1", "A", 5, "Zero Blitz"), _row("g2", "A", 6, "Cover 6 Willie")]
    out = tc.annotate(rows, knowledge=True)
    dk = [r[tc.KEY]["dk_counts"] for r in out]
    assert dk[0] == {} and dk[1] == {}                                         # first snap per game/opponent sees nothing
    assert dk[2] == {"dk:cover_3_cloud": 1.0}                                  # opponent B and later rows excluded
    assert dk[3] == {"dk:cover_3_cloud": 1.0, "dk:fire_zone_3": 1.0}
    assert dk[4] == {"dk:cover_3_cloud": 1.0, "dk:fire_zone_3": 1.0}           # the OPEN snap (no result) never counts
    assert dk[5] == {}                                                         # a different game starts clean


def test_changing_a_future_label_never_changes_an_earlier_rows_features():
    a = [_row("g1", "A", i, lab) for i, lab in enumerate(["Cover 3 Cloud", "Fire Zone 3", "Cover 4 Palms", "Zero Blitz"])]
    b = [dict(r) for r in a]
    b[3]["coverage_seen"] = "Cover 4 Drop"
    ra, rb = tc.annotate(a, knowledge=True), tc.annotate(b, knowledge=True)
    assert [r[tc.KEY] for r in ra[:4]] == [r[tc.KEY] for r in rb[:4]]


def test_a_games_own_labels_never_enter_its_own_prior():
    g1 = [_row("g1", "A", i, "Cover 3 Cloud") for i in range(3)]
    g2a = [_row("g2", "A", 10 + i, "Fire Zone 3") for i in range(3)]
    g2b = [_row("g2", "A", 10 + i, "Cover 4 Palms") for i in range(3)]
    pa = tc.prior_as_of(g1 + g2a, knowledge=True)
    pb = tc.prior_as_of(g1 + g2b, knowledge=True)
    assert pa[3:] == pb[3:] and pa[0] == {}                                    # game 2's prior ignores game 2's own labels
    assert pa[3]["dk:cover_3_cloud"] == 1.0


def test_collapsed_stored_labels_enter_the_history_as_mixtures_not_as_an_exact_call():
    out = tc.annotate([_row("g1", "A", 1, "Cover 3 Sky"), _row("g1", "A", 2, "Cover 3 Sky")], knowledge=True)
    c = out[1][tc.KEY]["dk_counts"]
    assert sum(c.values()) == pytest.approx(1.0) and len(c) >= 5 and c.get("dk:cover_3_sky", 0) < 0.25


def test_legacy_annotated_rows_are_recomputed_not_reused_for_knowledge():
    rows = [_row("g1", "A", i, lab) for i, lab in enumerate(["Cover 3 Cloud", "Fire Zone 3", "Cover 4 Palms"])]
    legacy = tc.annotate(rows)
    assert "dk_counts" not in legacy[2][tc.KEY]
    upgraded = tc.annotate(legacy, knowledge=True)
    assert upgraded[2][tc.KEY]["dk_counts"] == {"dk:cover_3_cloud": 1.0, "dk:fire_zone_3": 1.0}
    with pytest.raises(ValueError):
        tc.mix_from({}, legacy[2][tc.KEY], knowledge=True)


@pytest.fixture
def snaps():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE snaps(id INTEGER PRIMARY KEY,session_id TEXT,opponent_id TEXT,side TEXT,snap_seq INT,result TEXT,coverage_seen TEXT)")
    return c


def _add(c, seq, cov, result="+5", session="game_a", opp="cpu", side="offense"):
    c.execute("INSERT INTO snaps(session_id,opponent_id,side,snap_seq,result,coverage_seen) VALUES(?,?,?,?,?,?)", (session, opp, side, seq, result, cov))


def test_live_context_with_knowledge_keeps_the_same_isolation_rules(snaps):
    _add(snaps, 1, "Fire Zone 3")
    _add(snaps, 2, "Cover 3 Cloud")
    _add(snaps, 3, "pressure")
    _add(snaps, 4, "gibberish")
    _add(snaps, 5, "Cover 6", session="other")
    _add(snaps, 6, "Cover 6", opp="John")
    _add(snaps, 7, "Cover 3 Match", side="defense")
    _add(snaps, 8, "Cover 4 Palms", result="")
    _add(snaps, 9, "Cover 2 Man")                                              # at/after the next snap: must not count
    x = lc.coverage_for_next_snap(snaps, session_id="game_a", opponent_id="cpu", next_snap_seq=9, knowledge=True)
    assert x.named_n == 2 and x.pressure_n == 1 and x.unknown_n == 1 and x.history_n == 4
    assert set(k for k in x.mix if k.startswith("dk:")) == {"dk:fire_zone_3", "dk:cover_3_cloud", ds.GENERIC_PRESSURE_KEY}
    assert "dk:cover_2_man" not in x.mix and "dk:cover_6" not in x.mix and "dk:cover_4_palms" not in x.mix
    assert x.basis.endswith("+defense_knowledge")
    legacy = lc.coverage_for_next_snap(snaps, session_id="game_a", opponent_id="cpu", next_snap_seq=9)
    assert legacy.named_n == 0 and legacy.pressure_n == 1 and legacy.unknown_n == 3        # the old recognizer names none of these
    assert not any(k.startswith("dk:") for k in legacy.mix) and legacy.basis == "closed_manual_postplay_coverage_with_historical_prior"


# ------------------------------------------------------------------------------------------------ safety and budget

def test_the_live_defensive_caller_rejects_the_offline_knowledge_artifact():
    snap = json.loads(dc.SNAPSHOT_PATH.read_text())["sides"]["defense"]["formations"]
    book = [(f, p) for f, ps in snap.items() for p in ps]
    tmpl = {"features": ["down"], "call_families": ["a"], "packages": ["b"]}
    labels = {k: {"family": "a", "package": "b"} for k in book}
    from cfb_coach.madden.situation import parse_madden_situation
    rows = [dlt.SnapRow(i, f"g{i % 3}", "John", f"2026-10-0{i % 3 + 1}", parse_madden_situation("2&7 my 40", default_side="defense"),
                        *book[i], bool(i % 2), dlt.PROVENANCE_VERIFIED, True) for i in range(12)]
    art = kt.fit_knowledge_artifact(rows, labels, tmpl, 100.0)
    assert art["schema"] == kt.SCHEMA and art["schema"] != defense_v1.SCHEMA_V2 and art["catalog_digest"] == ds.data_digest()
    with pytest.raises(defense_v1.DefenseV1Error):
        defense_v1.validate_artifact(art)
    assert 0.0 < kt.predict_knowledge(art, rows[0], labels) < 1.0


def test_offline_defensive_comparison_does_not_promote_on_noise():
    import random
    from cfb_coach.madden.situation import parse_madden_situation
    snap = json.loads(dc.SNAPSHOT_PATH.read_text())["sides"]["defense"]["formations"]
    book = [(f, p) for f, ps in snap.items() for p in ps]
    fams = ["cover3", "pressure"]
    tmpl = {"schema": defense_v1.SCHEMA_V2, "policy": defense_v1.POLICY, "target": defense_v1.TARGET, "intercept": 0.0,
            "features": ["down", "distance"], "coefficients": {"down": 0.0, "distance": 0.0}, "call_families": fams,
            "packages": ["nickel", "other"], "family_coefficients": {f: 0.0 for f in fams}, "package_coefficients": {"nickel": 0.0, "other": 0.0},
            "family_situation_coefficients": {f: {i: 0.0 for i in defense_v1.INTERACTION_FEATURES} for f in fams},
            "package_situation_coefficients": {p: {i: 0.0 for i in defense_v1.INTERACTION_FEATURES} for p in ("nickel", "other")}, "candidates": []}
    rnd = random.Random(2)
    labels, rows = {}, []
    for g in range(6):
        for _ in range(20):
            f, p = rnd.choice(book)
            labels[(f, p)] = {"family": "pressure" if any(k in p for k in ("Blitz", "Fire", "Sim")) else "cover3", "package": "nickel" if "Nickel" in f else "other"}
            rows.append(dlt.SnapRow(len(rows), f"g{g}", "John", f"2026-10-0{g + 1}", parse_madden_situation("2&7 my 40", default_side="defense"),
                                    f, p, rnd.random() < 0.5, dlt.PROVENANCE_VERIFIED, True))
    rep = kt.compare_leave_one_game_out(rows, labels, tmpl, lambdas=(30.0, 300.0))
    assert rep["gate"]["result"] in ("INSUFFICIENT", "FAILED") and "none" in rep["gate"]["promotion"]
    assert rep["games"] == 6 and rep["pooled"]["knowledge"]["n"] == rep["pooled"]["baseline"]["n"] == 120


def test_offline_gate_needs_enough_games_and_an_interval_that_excludes_zero():
    def report(games, hi):
        m = lambda ll: {"n": 200, "log_loss": ll, "brier": ll / 2.77, "auc": 0.6}
        return {"games": games, "pooled": {"baseline": m(0.70), "knowledge": m(0.66)},
                "paired_bootstrap": {"lo": -0.07, "hi": hi, "mean_diff": -0.04}}
    assert kt.gate(report(8, -0.01))["result"] == "PASSED"
    assert kt.gate(report(8, +0.01))["result"] == "FAILED"                  # a 0.04 gain whose interval spans zero is NOT a pass
    assert kt.gate(report(7, -0.01))["result"] == "FAILED" or kt.gate(report(7, -0.01))["checks"]["min_games"] is False
    thin = kt.gate(report(3, -0.01))
    assert thin["result"] == "INSUFFICIENT" and "none" in thin["promotion"]


def test_shadow_replay_runs_a_knowledge_artifact_on_recorded_games_and_never_acts(tmp_path):
    from cfb_coach.madden.model import offense_readiness as rd
    rows = ke.make_pressure_world_rows(n_games=9, snaps=20, seed=5, strength=2.0)
    data = he.EvalData(rows=rows, game_order={g: i for i, g in enumerate(dict.fromkeys(r["game_id"] for r in rows))}, source="synthetic")
    out = rd.shadow_replay(data, out_path=tmp_path / "shadow.jsonl", min_train_games=6, max_snaps_per_game=8, defense_knowledge=True)
    recs = [json.loads(line) for line in (tmp_path / "shadow.jsonl").read_text().splitlines()]
    assert out["records"] == len(recs) > 0 and out["errors_or_fallbacks"] == 0 and out["illegal"] == 0
    assert all(r["kind"] == "shadow_decision_not_an_outcome" and r["source"] == "recorded_game_replay" for r in recs)
    assert all(k.startswith("dk:") or k == "unknown" for r in recs for k, _ in r["estimated_coverage"])      # the replay used the knowledge mix
    assert max(out["latency_ms_max"], 0) < 150


def test_a_knowledge_artifact_refuses_rows_annotated_without_knowledge_instead_of_dropping_evidence(arts, world):
    _, dk = arts
    row = next(r for r in world if r["side"] == "offense")
    legacy_cov = {k: v for k, v in row[tc.KEY].items() if k != "dk_counts"}
    s = sm.structure_any(row["action_formation"], row["action_play"])
    with pytest.raises(ValueError, match="annotated without defense knowledge"):
        sm.score(dk, s, down=1, distance=10, yardline=30, opponent_id="cpu", cov=legacy_cov)
    assert 0.0 < sm.score(dk, s, down=1, distance=10, yardline=30, opponent_id="cpu", cov=row[tc.KEY]).probability < 1.0


def test_scoring_every_installed_play_stays_inside_the_150ms_budget(arts):
    book = [(p["formation"], p["play"]) for p in oa.load_effective()["plays"]]
    for art, mix in ((arts[0], {"cover3": 0.5, "cover_2_man": 0.5}), (arts[1], {"dk:cover_3_sky": 0.4, "dk:fire_zone_3": 0.3, "dk:cover_2_man": 0.3})):
        best = min(ke.latency_report(art, book, mix, repeats=3)["ms_max"] for _ in range(3))
        assert best < 150, f"{'knowledge' if art.uses_defense_knowledge else 'default'} artifact took {best:.0f} ms"


def test_running_the_pipeline_does_not_modify_the_catalog_or_the_recorded_playbook(arts):
    paths = [dc.CATALOG_PATH, dc.SNAPSHOT_PATH, dc.ANNOTATIONS_PATH]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    ke.recognition_audit()
    ds.explain("Cover 3 Sky", "Nickel Over")
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def test_recognition_audit_reports_the_improvement_and_the_unchanged_verification_state():
    a = ke.recognition_audit()
    assert a["installed"]["entries"] == 123 and a["installed"]["old_recognizer_named_coverage"] == 38
    assert a["installed"]["catalog_resolved"] == 123 and a["external"]["catalog_resolved"] == 27
    assert a["legacy_disagreements"] == [] and a["unresolved"] == []
    assert a["verification"]["ledger_known_slots"] == 0 and "NOT responsibility verification" in a["verification"]["note"]
    assert sum(b["old"] for b in a["by_formation"].values()) == 38 and sum(b["new"] for b in a["by_formation"].values()) == 123


def test_verdict_demands_enough_games_and_an_interval_that_excludes_zero():
    good = {"mean_diff": -0.02, "lo": -0.04, "hi": -0.005}
    assert ke.verdict(good, games=3).startswith("NO-GO (insufficient data)")
    assert ke.verdict(good, games=12).startswith("GO for a CPU shadow")
    assert ke.verdict({"mean_diff": -0.02, "lo": -0.05, "hi": 0.01}, games=12).startswith("NO-GO: no demonstrated improvement")
    assert ke.verdict({"mean_diff": 0.01, "lo": -0.01, "hi": 0.03}, games=12).startswith("NO-GO")
    assert ke.verdict({}, games=12).startswith("NOT EVALUATED")
