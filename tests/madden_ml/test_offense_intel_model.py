"""Structural model, joint decision, pre-snap advice, evaluation plumbing and the CPU pilot.

All training rows are SYNTHETIC and exist only to exercise code paths (as in the other tests here).
They are never used to claim model quality.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_pilot as pilot
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_presnap_advice as adv
from cfb_coach.madden.model import offense_structural_eval as ev
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model.strategic_context import PostSnapLeak

ROOT = Path(__file__).resolve().parents[2]
DB = oa.load_database()
BOOK = [(p["formation"], p["play"]) for p in DB["plays"]]
ANNOTATED = [(p["formation"], p["play"]) for p in DB["plays"] if oa.play_status(p) != "unknown"]
HEDGES = re.compile(r"\b(maybe|consider|might|perhaps|could try|possibly)\b", re.I)


def _rows(n_games=12, per=25, seed=1, plays=BOOK):
    rnd = random.Random(seed)
    rows = []
    for g in range(n_games):
        for i in range(per):
            f, pl = rnd.choice(plays)
            raw = {"session_id": f"g{g}", "ml_snap_id": f"{g}-{i}", "opponent_id": "cpu", "side": "offense",
                   "down": rnd.choice([1, 2, 3]), "distance": rnd.choice([3, 7, 10]), "yardline": rnd.randint(10, 90),
                   "result": rnd.choice(["+8", "+2", "incomplete", "+12", "-1"]), "formation": f, "play": pl,
                   "executed_status": "identified", "executed_verification": "verified", "executed_formation": f,
                   "executed_play": pl, "coverage_seen": rnd.choice(["Cover 3", "Cover 1", "Cover 4 Quarters"]),
                   "opponent_type": "cpu"}
            rows.append(he.dataset_mod._normalize_row(raw, provenance="test"))
    return rows


@pytest.fixture(scope="module")
def rows():
    return _rows()


@pytest.fixture(scope="module")
def art(rows):
    return sm.train(rows)


SIT = od.Situation(down=3, distance=7, yardline=45, opponent_id="cpu", opponent_type="cpu")


# 11 ---------------------------------------------------------------- no future information

def test_post_snap_fields_are_rejected_as_pre_snap_input():
    for bad in ("coverage_seen", "result", "yards", "success", "concept_seen", "executed_play"):
        with pytest.raises(PostSnapLeak):
            od.Situation.from_mapping({"down": 1, "distance": 10, bad: "x"})
    with pytest.raises(ValueError):
        od.Situation.from_mapping({"down": 1, "mystery": 3})


def test_prediction_ignores_the_current_snaps_coverage_and_result(art, rows):
    r = dict(rows[0])
    s = sm.structure_any(r["action_formation"], r["action_play"])
    kw = dict(down=r["down"], distance=r["distance"], yardline=r["yardline"], opponent_id="cpu", opponent_type="cpu")
    p0 = sm.score(art, s, **kw).probability
    r["coverage_seen"], r["result"], r["success"] = "Cover 0 Blitz", "+40", "true"
    assert sm.score(art, s, **kw).probability == p0


def test_training_uses_only_the_rows_it_is_given_so_folds_cannot_leak(rows):
    early = [r for r in rows if r["game_id"] in {"g0", "g1", "g2", "g3", "g4"}]
    a = sm.train(early)
    assert a.trained_games == 5 and a.n_train <= len(early)
    folds = he.make_folds(rows, {f"g{i}": i for i in range(12)}, is_test_row=he.is_verified_offense, min_train_games=3)
    for f in folds:
        assert all(int(g[1:]) < int(f.test_game[1:]) for g in f.train_games)


# model ---------------------------------------------------------------------------------------------

def test_artifact_roundtrip_digest_and_validation(art, tmp_path):
    assert sm.validate_artifact(art) == []
    p = sm.save_artifact(art, tmp_path / "a.json")
    back = sm.load_artifact(p)
    assert sm.artifact_digest(back) == sm.artifact_digest(art)
    bad = sm.load_artifact(p)
    bad.names = bad.names[:-1]
    assert any("feature names" in e for e in sm.validate_artifact(bad))
    bad = sm.load_artifact(p)
    bad.assignment_digest = "stale"
    assert any("assignment database changed" in e for e in sm.validate_artifact(bad))
    with pytest.raises(ValueError):
        sm.StructuralArtifact.from_dict({"schema": "other"})


def test_too_little_data_disables_the_residual_and_returns_the_base_coordinator(rows):
    tiny = sm.train(rows[:10])
    assert not tiny.residual_active
    s = sm.structure_any(*BOOK[0])
    sc = sm.score(tiny, s)
    assert sc.probability == pytest.approx(sc.base_probability) and sc.logit_se == 0.0


def test_scores_are_probabilities_with_uncertainty_and_regularization_shrinks(art, rows):
    s = sm.structure_any(*ANNOTATED[0])
    sc = sm.score(art, s, down=3, distance=7, yardline=45, opponent_id="cpu", opponent_type="cpu")
    lo, hi = sc.interval
    assert 0 < lo < sc.probability < hi < 1 and sc.logit_se > 0
    strong = sm.train(rows, penalty_scale=50.0)
    weak = sm.train(rows, penalty_scale=0.05)
    n = lambda a: sum(abs(w) for w in a.weights[1:])
    assert n(strong) < n(art) < n(weak)


def test_unseen_and_non_installed_plays_still_score_from_hint_features(art):
    s = sm.structure_any("Some Old Formation", "Mesh")
    assert s.n_known == 0 and s.hint["concepts"] == ["mesh"]
    p = sm.score(art, s, down=1, distance=10, yardline=30).probability
    assert 0 < p < 1


def test_unseen_hot_routed_structures_are_outside_training_support_or_capped(art):
    s = pf.structure_lookup(*ANNOTATED[0])
    scorer = sm.DeltaScorer(art, s, mix=None)
    for slot in pf.eligible_hot_route_targets(s):
        for opt in ("Slant", "Streak", "Block"):
            try:
                m = pf.apply_hot_route(s, slot, opt)
            except pf.StructureError:
                continue
            gain, se, outside = scorer.score(m)
            assert abs(gain) <= sm.MAX_EXTRAPOLATED_LOGIT + 1e-12 and se >= 0


# 12 ---------------------------------------------------------------- definitive decisions

def test_decision_is_always_definitive_and_unconfirmed_adjustments_never_go_live(art):
    d = od.decide(art, SIT, BOOK, mode="live")
    assert d.adjustment == od.NO_ADJUSTMENT or re.match(r"^(HOT ROUTE|MACRO) ", d.adjustment)
    assert not HEDGES.search(d.adjustment)
    d2 = od.decide(art, SIT, ANNOTATED, mode="live", margin=0.0, z=0.0)
    assert d2.adjustment == od.NO_ADJUSTMENT                       # nothing is confirmed in-game yet
    assert d2.actions == () and d2.executable_now
    assert d2.plans and any("not yet confirmed" in p.reject_reason for p in d2.plans)


def test_research_mode_scores_adjustments_but_marks_them_not_executable(art):
    d = od.decide(art, SIT, ANNOTATED, mode="research", margin=0.0, z=0.0)
    assert d.adjustment == od.NO_ADJUSTMENT or not d.executable_now
    if d.adjustment != od.NO_ADJUSTMENT:
        assert d.basis == od.BASIS and d.final_structure.adjustments


def test_confirmed_hot_route_can_be_selected_and_is_exactly_the_registry_plan(art):
    conf = {"off.hot_route": {"confirmed": True}}
    conf.update({f"off.hot_route|{o}": {"confirmed": True} for o in ex.registry()["off.hot_route"].params["route"]["options"]})
    d = od.decide(art, SIT, ANNOTATED, mode="live", confirmations=conf, margin=0.0, z=0.0)
    for a in d.actions:
        assert a.action_id == "off.hot_route" and a.params["target"] in d.final_structure.skill_slots
        assert a.params["route"] in ex.registry()["off.hot_route"].params["route"]["options"]
    if d.actions:
        assert d.executable_now and ex.validate_plan(
            d.actions, ex.PlayContext("offense", d.formation, d.play, "pass", eligible_receivers=d.final_structure.skill_slots),
            confirmations=conf, require_live=True) == []


def test_adjustment_requires_margin_and_positive_lower_bound(art):
    never = od.decide(art, SIT, ANNOTATED, mode="research", margin=0.9)
    assert never.adjustment == od.NO_ADJUSTMENT
    zhigh = od.decide(art, SIT, ANNOTATED, mode="research", margin=0.0, z=50.0)
    assert zhigh.adjustment == od.NO_ADJUSTMENT


def test_learned_model_ranks_all_candidates_and_only_installed_plays_are_chosen(art):
    d = od.decide(art, SIT, BOOK, mode="research")
    assert (d.formation, d.play) in set(BOOK)
    assert len(d.ranking) == 93 and d.ranking[0]["probability"] >= d.ranking[-1]["probability"]
    with pytest.raises(ValueError):
        od.decide(art, SIT, [])


def test_installed_macro_compatibility(art):
    s = pf.structure_lookup(*ANNOTATED[0])
    slot = pf.eligible_hot_route_targets(s)[0]
    eff = od.MacroEffect("SLANTS", {slot: "Slant"}, verified=True, source="user_ingame:test")
    lo = ex.Loadout.from_dict({"offense": [{"name": "SLANTS", "slot": "A", "installed": True, "active": True,
                                            "confirmed_by_user": True}]})
    assert lo.usable("offense", "SLANTS")
    plan = (ex.PlannedAction("off.saved.SLANTS"),)
    ctx = ex.PlayContext("offense", s.formation, s.play, "pass", eligible_receivers=s.skill_slots)
    assert ex.validate_plan(plan, ctx, loadout=lo, require_live=True) == []
    # not installed / not confirmed -> not live
    lo_off = ex.Loadout.from_dict({"offense": [{"name": "SLANTS", "slot": "A", "installed": True, "active": False,
                                                "confirmed_by_user": True}]})
    assert ex.validate_plan(plan, ctx, loadout=lo_off, require_live=True)
    # a macro and a hot route both change routes:all / route:<target> -> cannot be combined
    both = plan + (ex.PlannedAction("off.hot_route", {"target": slot, "route": "Post"}),)
    assert any("cannot both be applied" in e for e in ex.validate_plan(both, ctx, loadout=lo))
    # unverified effect is unscoreable and never selected
    d = od.decide(art, SIT, [(s.formation, s.play)], loadout=lo, mode="research", margin=-1.0, z=-9.0,
                  macro_effects={"SLANTS": od.MacroEffect("SLANTS", {slot: "Slant"}, verified=False, source="x")})
    assert not any(a.action_id == "off.saved.SLANTS" for a in d.actions)
    assert od._apply_actions(s, plan, {"SLANTS": eff}).route_for(slot).family == "slant"
    with pytest.raises(pf.StructureError):
        od._apply_actions(s, plan, {"SLANTS": od.MacroEffect("SLANTS", {slot: "Slant"}, False, "x")})


def test_recent_calls_only_reorder_inside_the_indifference_band(art):
    base = od.decide(art, SIT, BOOK, mode="research", allow_adjustments=False)
    recent = [(base.formation, base.play, "none")] * 6
    varied = od.decide(art, SIT, BOOK, mode="research", allow_adjustments=False, recent_calls=recent)
    gap = base.ranking[0]["probability"] - next(r["probability"] for r in base.ranking
                                                if (r["formation"], r["play"]) == (varied.formation, varied.play))
    assert gap <= od.INDIFFERENCE_BAND + 0.07


# 13 ---------------------------------------------------------------- advice

def test_card_for_a_documented_play_uses_its_real_routes_and_labels_the_basis(art):
    d = od.decide(art, SIT, [("Gun Trips X Nasty", "Hi Lo Cross")], mode="live", allow_adjustments=False)
    card = adv.build_card(d, {"cover3": 1.0})
    assert card["PRIMARY READ"].startswith("WR2 in (deep, right)") and "[Deep In]" in card["PRIMARY READ"]
    assert card["SECONDARY READ"].startswith("TE1")
    assert card["CHECKDOWN"].startswith("HB1 flat")
    assert card["READ BASIS"].startswith("DOCUMENTED")
    text = adv.format_card(card)
    assert not HEDGES.search(text) and len(text.splitlines()) <= 10
    for k in ("FORMATION", "PLAY", "ADJUSTMENT", "PRIMARY READ", "SECONDARY READ", "CHECKDOWN", "DEFENSIVE INDICATOR", "WHY"):
        assert f"{k}:" in text


def test_card_never_fabricates_a_progression_for_unannotated_plays(art):
    d = od.decide(art, SIT, [("Gun 5WR Tight", "Mesh")], mode="live", allow_adjustments=False)
    card = adv.build_card(d, None)
    assert card["PRIMARY READ"].startswith("UNKNOWN") and card["SECONDARY READ"].startswith("UNKNOWN")
    assert "not observable" in card["DEFENSIVE INDICATOR"]
    run = od.decide(art, SIT, [("Gun Trips X Nasty", "Inside Zone")], mode="live", allow_adjustments=False)
    c = adv.build_card(run, None)
    assert "UNKNOWN" in c["PRIMARY READ"] and "name-level hint" in c["PRIMARY READ"]


def test_card_after_a_hot_route_says_the_documented_order_no_longer_applies(art):
    s = pf.structure_lookup("Gun Trips X Nasty", "Hi Lo Cross")
    d = od.decide(art, SIT, [(s.formation, s.play)], allow_adjustments=False)
    d.final_structure = pf.apply_hot_route(s, "WR1", "Slant")
    d.adjustment = "HOT ROUTE WR1 -> Slant"
    card = adv.build_card(d, None)
    assert card["READ BASIS"].startswith("INFERRED")
    assert "slant" in card["WHY"].lower() and "documented order no longer applies" in card["READ BASIS"]


# 14 ---------------------------------------------------------------- latency

def test_decision_latency_stays_inside_the_150ms_budget(art):
    full = []
    for _ in range(15):
        t = time.perf_counter()
        d = od.decide(art, SIT, BOOK, mode="live")
        full.append((time.perf_counter() - t) * 1000)
        assert d.latency_ms < 150
    worst = []
    for _ in range(8):
        d = od.decide(art, SIT, ANNOTATED, mode="research", margin=0.0, z=0.0)
        worst.append(d.latency_ms)
    assert sorted(full)[int(0.95 * 14)] < 150 and max(worst) < 150


def test_search_is_truncated_not_overrun_when_the_budget_is_tiny(art):
    d = od.decide(art, SIT, ANNOTATED, mode="research", budget_ms=0.5, margin=0.0, z=0.0)
    assert d.truncated and any("truncated" in n for n in d.notes)
    assert d.latency_ms < 50


# evaluation plumbing ------------------------------------------------------------------------------

def test_eval_reports_insufficient_data_instead_of_claiming(rows):
    few = [r for r in rows if r["game_id"] in {"g0", "g1", "g2", "g3", "g4"}]
    data = he.EvalData(rows=few, game_order={f"g{i}": i for i in range(5)}, source="synthetic")
    rep = ev.evaluate(data, n_boot=20, run_behavior=False)
    assert rep["claims_allowed"] is False and "INSUFFICIENT DATA" in rep["verdict"]
    none = ev.evaluate(he.EvalData(rows=[], game_order={}, source="x"), run_behavior=False)
    assert none["folds"] == 0 and "No folds" in none["verdict"]


def test_eval_runs_end_to_end_with_game_level_chronological_splits(rows):
    data = he.EvalData(rows=rows[:200], game_order={f"g{i}": i for i in range(8)}, source="synthetic")
    rep = ev.evaluate(data, n_boot=20, run_behavior=True)
    m = rep["metrics"]["success"]
    assert {"base_rate", "experimental_shrinkage", "structural_features", "structural_no_matchup"} <= set(m["models"])
    assert rep["behavior"]["illegal_or_off_book"] == 0 and rep["behavior"]["latency_ms_max"] < 150
    assert "not evaluable" in rep["behavior"]["variant_D_macros"]
    assert "Decision behavior" in ev.render_markdown(rep)


# 15-16 ------------------------------------------------------------------- pilot, compatibility, no side effects

@pytest.fixture()
def pilot_env(tmp_path, art, monkeypatch):
    monkeypatch.setattr(pilot, "LIVE_DIR", tmp_path / "live" / ".cfb-coach")
    ap = sm.save_artifact(art, tmp_path / "art.json")
    cfg = pilot.PilotConfig(artifact_path=str(ap), log_path=str(tmp_path / "log.jsonl"),
                            artifact_digest=sm.artifact_digest(art))
    return cfg, art, tmp_path


def test_pilot_is_inert_until_approved_and_cpu_only(pilot_env):
    cfg, art, _ = pilot_env
    cfg.mode = "pilot"
    assert any("requires explicit approval" in e for e in pilot.validate_config(cfg, artifact=art))
    with pytest.raises(pilot.PilotRefused):
        pilot.PilotSession(cfg, artifact=art)
    pilot.approve(cfg, "Aidan")
    s = pilot.PilotSession(cfg, artifact=art)
    with pytest.raises(pilot.PilotRefused):
        s.recommend({"down": 1, "distance": 10, "yardline": 25, "opponent_id": "gavin", "opponent_type": "human"})
    with pytest.raises(pilot.PilotRefused):
        pilot.approve(pilot.PilotConfig(), "  ")
    bad = pilot.PilotConfig(**{**cfg.to_dict(), "opponent_scope": "human"})
    assert any("CPU only" in e for e in pilot.validate_config(bad, artifact=art))


def test_recommendation_is_logged_and_is_not_an_execution(pilot_env):
    cfg, art, tmp = pilot_env
    s = pilot.PilotSession(cfg, artifact=art)
    rec = s.recommend({"down": 2, "distance": 6, "yardline": 40, "opponent_id": "cpu", "opponent_type": "cpu"}, game_id="g1")
    assert rec["executed"] == "not_confirmed" and rec["baseline_top5"] and rec["fallback_reason"] == ""
    assert (rec["primary"]["formation"], rec["primary"]["play"]) in set(BOOK)
    summ = pilot.summarize_log(cfg.log_path)
    assert summ["recommendations"] == 1 and summ["executions_confirmed"] == 0 and summ["unconfirmed"] == 1
    s.confirm_execution(rec["rec_id"], formation=rec["primary"]["formation"], play=rec["primary"]["play"])
    summ = pilot.summarize_log(cfg.log_path)
    assert summ["executions_confirmed"] == 1 and summ["followed_recommended_play"] == 1
    other = s.confirm_execution(rec["rec_id"], formation="X", play="Y", status="not_executed")
    assert pilot.summarize_log(cfg.log_path)["executions_confirmed"] == 0
    with pytest.raises(pilot.PilotRefused):
        s.confirm_execution("nope", formation="a", play="b")
    for line in Path(cfg.log_path).read_text().splitlines():
        assert json.loads(line)["schema"] == pilot.LOG_SCHEMA


def test_fallback_to_the_existing_coordinator_only_on_technical_failure(pilot_env, monkeypatch):
    cfg, art, _ = pilot_env
    s = pilot.PilotSession(cfg, artifact=art)
    sit = {"down": 2, "distance": 6, "yardline": 40, "opponent_id": "cpu", "opponent_type": "cpu"}
    ok = s.recommend(sit)
    assert ok["fallback_reason"] == ""                              # routine call: no heuristic/baseline override
    monkeypatch.setattr(od, "decide", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    bad = s.recommend(sit)
    assert "boom" in bad["fallback_reason"] and bad["primary"]["adjustment"] == od.NO_ADJUSTMENT
    base_top = em.rank_candidates(art.base, BOOK, down=2, distance=6, yardline=40, opponent_id="cpu", opponent_type="cpu")[0]
    assert (bad["primary"]["formation"], bad["primary"]["play"]) == (base_top["formation"], base_top["play"])
    assert pilot.summarize_log(cfg.log_path)["fallbacks"] == 1


def test_pilot_refuses_unsafe_paths_stale_artifacts_and_non_installed_plays(pilot_env, tmp_path):
    cfg, art, tmp = pilot_env
    live = pilot.PilotConfig(**{**cfg.to_dict(), "log_path": str(pilot.LIVE_DIR / "log.jsonl")})
    assert any("inside the live data directory" in e for e in pilot.validate_config(live, artifact=art))
    stale = pilot.PilotConfig(**{**cfg.to_dict(), "artifact_digest": "deadbeef"})
    assert any("digest differs" in e for e in pilot.validate_config(stale, artifact=art))
    wrong = pilot.PilotConfig(**{**cfg.to_dict(), "installed_fingerprint": "OFF-0000-0000"})
    assert any("fingerprint" in e for e in pilot.validate_config(wrong, artifact=art))
    with pytest.raises(pilot.PilotRefused):
        pilot.PilotSession(cfg, artifact=art, candidates=[("Gun Bunch", "Not Installed")])


def test_pilot_and_decision_code_never_import_the_live_caller_or_write_a_database():
    for name in ("offense_pilot", "offense_decision", "offense_structural_model", "offense_play_features",
                 "offense_presnap_advice", "offense_assignments", "offense_taxonomy", "offense_structural_eval"):
        src = (ROOT / "cfb_coach" / "madden" / "model" / f"{name}.py").read_text()
        assert "experimental_live" not in src and "CoachDB" not in src and "INSERT" not in src.upper().replace("INSERT_", "")
        assert "sqlite3.connect" not in src


def test_existing_coach_modules_are_unchanged_by_this_work():
    out = subprocess.run(["git", "diff", "--name-status", "origin/madden-ml/coverage-semantics-bulk-catalog-20261106", "--",
                          "cfb_coach", "tests", "scripts"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("branch comparison unavailable")
    touched = [l.split("\t", 1)[1] for l in out.stdout.splitlines() if not l.startswith("A")]
    allowed = {"pyproject.toml"}
    assert [t for t in touched if t not in allowed] == [], "existing files were modified"


def test_running_everything_does_not_create_or_change_live_data(tmp_path, art, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    snap = hashlib.sha256(oa.SNAPSHOT_PATH.read_bytes()).hexdigest()
    ann = hashlib.sha256(oa.ANNOTATIONS_PATH.read_bytes()).hexdigest()
    od.decide(art, SIT, BOOK, mode="research")
    assert not (tmp_path / ".cfb-coach").exists()
    assert hashlib.sha256(oa.SNAPSHOT_PATH.read_bytes()).hexdigest() == snap
    assert hashlib.sha256(oa.ANNOTATIONS_PATH.read_bytes()).hexdigest() == ann


def test_backward_compatibility_base_coordinator_scores_are_untouched(art, rows):
    base = em.train_experimental([dict(r) for r in rows], side="offense")
    for f, p in BOOK[:5]:
        a = em.predict_success(base, formation=f, play=p, down=2, distance=5, yardline=40)["probability"]
        b = em.predict_success(art.base, formation=f, play=p, down=2, distance=5, yardline=40)["probability"]
        assert a == pytest.approx(b)
    ranked = em.rank_candidates(art.base, BOOK, down=2, distance=5, yardline=40)
    assert len(ranked) == 93


def test_cli_end_to_end_in_an_isolated_home(tmp_path, art):
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(ROOT)}
    cli = [sys.executable, str(ROOT / "scripts/madden_ml/offense_pilot_cli.py"), "--workdir", str(tmp_path / "w")]

    def run(*a, ok=None):
        r = subprocess.run(cli + list(a), env=env, capture_output=True, text=True)
        if ok is not None:
            assert r.returncode == ok, r.stdout + r.stderr
        return r

    run("config-init", ok=0)
    sm.save_artifact(art, tmp_path / "w" / "offense_structural_v1.json")
    cfgp = tmp_path / "w" / "pilot_config.json"
    cfg = json.loads(cfgp.read_text())
    cfg["artifact_digest"] = sm.artifact_digest(art)
    cfgp.write_text(json.dumps(cfg))
    run("validate", ok=0)
    assert not json.loads(cfgp.read_text())["approved"]
    r = run("pilot", "--down", "1", "--distance", "10", "--yardline", "25")
    assert r.returncode == 2 and "not approved" in r.stderr
    r = run("shadow", "--down", "1", "--distance", "10", "--yardline", "25", ok=0)
    assert "FORMATION:" in r.stdout and "ADJUSTMENT:" in r.stdout and "rec_id=" in r.stdout
    rec_id = re.search(r"rec_id=(\w+)", r.stdout).group(1)
    run("approve", "--by", "Aidan", ok=0)
    r = run("pilot", "--down", "1", "--distance", "10", "--yardline", "25", ok=0)
    assert "PRIMARY READ:" in r.stdout
    run("confirm", "--rec-id", rec_id, "--formation", "Pistol Trips", "--play", "Slants", ok=0)
    rep = json.loads(run("report", ok=0).stdout)
    assert rep["recommendations"] == 2 and rep["executions_confirmed"] == 1
    assert not (tmp_path / ".cfb-coach").exists()
