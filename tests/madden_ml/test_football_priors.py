"""Football priors -> actual model scores, rankings, adjustments and QB reads; adaptive three-level learning.

Everything numeric here is a PLUMBING/BEHAVIOUR test on synthetic or fixed inputs: it proves the knowledge changes decisions and that learning
moves it sensibly, not that the football hypotheses are true in Madden. Real evidence needs held-out games from your database."""
from __future__ import annotations

import pytest

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import adaptive_priors as ap
from cfb_coach.madden.model import football_priors as fp
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_presnap_advice as adv
from cfb_coach.madden.model import offense_structural_model as sm
from tests.madden_ml.test_offense_intel_model import ANNOTATED, BOOK, art, rows  # noqa: F401

MESH = ("Gun Trips X Nasty", "Hi Lo Cross")           # route-derived: mesh + levels + drive (+ name hints)
SMASH = ("Gun Doubles Clamp Stack", "Mtn Shuffle Verts Smash")


def sit(mix, **kw):
    return od.Situation(down=kw.get("down", 2), distance=kw.get("distance", 7), yardline=kw.get("yardline", 45), opponent_id="cpu", opponent_type="cpu", mix=mix)


def edge(play, cov):
    return fp.matchup_edge(sm.structure_any(*play), {cov: 1.0})["edge"]


def test_priors_are_valid_sourced_and_bounded():
    assert fp.validate() == []
    r = fp.report()
    assert r["priors"] >= 60 and r["concepts_covered"] >= 15
    assert all(p.confidence <= 1 and p.mechanism and p.evidence for p in fp.PRIORS)


@pytest.mark.parametrize("a,b,play", [("cover_1_unspecified", "cover_1_robber", MESH), ("cover_2_spot", "cover_2_man", SMASH),
                                      ("cover_3_unspecified", "cover_3_match", ("Gun Trips X Nasty", "Mtn Stick Wheel")),
                                      ("cover_4_quarters", "cover_4_palms", SMASH)])
def test_variants_of_one_family_get_different_edges_for_the_same_play(a, b, play):
    assert abs(edge(play, a) - edge(play, b)) > 0.01, (play, a, b)


def test_cover_2_man_is_not_cover_2_zone_for_smash():
    assert edge(SMASH, "cover_2_spot") > 0 > edge(SMASH, "cover_2_man")


def test_name_only_hints_are_trusted_less_than_route_derived_concepts():
    s = sm.structure_any(*MESH)
    c = fp.play_concepts(s)
    assert c["mesh"] == 1.0 and fp.NAME_TRUST < 1.0 and fp.play_concepts(sm.structure_any(*SMASH))["smash"] == fp.NAME_TRUST


def test_unknown_coverage_and_unknown_concepts_abstain_rather_than_guess():
    assert fp.descriptor("cover_9") is None and fp.descriptor("unknown") is None
    assert fp.matchup_edge(sm.structure_any(*MESH), {"cover_9": 1.0})["edge"] == 0.0
    assert fp.matchup_edge(sm.structure_any(*MESH), None)["edge"] == 0.0


def test_the_edge_is_a_mixture_and_capped():
    m = fp.matchup_edge(sm.structure_any(*MESH), {"cover_1_unspecified": 0.5, "cover_3_match": 0.5})["edge"]
    assert min(edge(MESH, "cover_1_unspecified"), edge(MESH, "cover_3_match")) <= m <= max(edge(MESH, "cover_1_unspecified"), edge(MESH, "cover_3_match"))
    assert abs(m) <= fp.EDGE_CAP


def test_coverage_changes_the_actual_scores_and_ranking_of_the_model(art):
    man, zone = {"cover_1_unspecified": 1.0}, {"cover_3_match": 1.0}
    on_m = od.decide(art, sit(man), BOOK, mode="research", priors=True, allow_adjustments=False)
    on_z = od.decide(art, sit(zone), BOOK, mode="research", priors=True, allow_adjustments=False)
    off_m = od.decide(art, sit(man), BOOK, mode="research", priors=False, allow_adjustments=False)
    off_z = od.decide(art, sit(zone), BOOK, mode="research", priors=False, allow_adjustments=False)
    pm = {(r["formation"], r["play"]): r["probability"] for r in on_m.ranking}
    pz = {(r["formation"], r["play"]): r["probability"] for r in on_z.ranking}
    assert sum(abs(pm[k] - pz[k]) > 1e-3 for k in pm) > 30                                   # many plays' scores move with the coverage
    top = lambda d, n=10: [(r["formation"], r["play"]) for r in d.ranking[:n]]              # noqa: E731
    assert top(on_m) != top(on_z)                                                             # and so does the ranking
    assert max(abs(r["probability"] - q["probability"]) for r, q in zip(sorted(on_m.ranking, key=lambda x: (x["formation"], x["play"])),
                                                                        sorted(off_m.ranking, key=lambda x: (x["formation"], x["play"])))) > 0.005
    assert on_m.football_priors and on_m.matchup is not None and not off_m.football_priors


def test_adjustment_choice_follows_the_coverage_and_pressure_buys_protection(art, monkeypatch):
    s = sm.structure_any(*ANNOTATED[0])
    prot = (ex.PlannedAction("off.pass_protection", {"protection": "Half Slide"}),)
    kw = dict(mode="research", margin=0.0, z=0.0, priors=True, extra_actions=prot, budget_ms=3000)
    press = od.decide(art, sit({"dk:_generic_pressure": 1.0}), [(s.formation, s.play)], **kw)
    calm = od.decide(art, sit({"cover_3_unspecified": 1.0}), [(s.formation, s.play)], **kw)
    assert any(a.action_id == "off.pass_protection" for a in press.actions) or press.adjustment != calm.adjustment
    assert not any(a.action_id == "off.pass_protection" for a in calm.actions)
    plans = {}
    for k in ("cover_1_unspecified", "cover_2_spot", "cover_4_quarters", "dk:_generic_pressure"):
        d = od.decide(art, sit({k: 1.0}), [(s.formation, s.play)], mode="research", margin=0.0, z=0.0, priors=True, budget_ms=3000)
        plans[k] = {p.label: p.gain_prob for p in d.plans}
    common = set.intersection(*(set(v) for v in plans.values()))
    assert common
    moved = [lab for lab in common if max(plans[k][lab] for k in plans) - min(plans[k][lab] for k in plans) > 0.002]
    assert moved, "the same transformed plays should be valued differently under different coverages"      # candidate gains follow the look


def test_qb_reads_are_recomputed_from_the_coverage_and_the_transformed_play(art):
    cards = {}
    for k in ("cover_1_unspecified", "cover_4_quarters", "dk:_generic_pressure"):
        d = od.decide(art, sit({k: 1.0}), [MESH], mode="research", priors=True, allow_adjustments=False)
        cards[k] = adv.build_card(d, {k: 1.0})
    assert cards["cover_1_unspecified"]["COVERAGE-CONDITIONED ORDER"] and any(
        cards[a]["COVERAGE-CONDITIONED ORDER"] != cards[b]["COVERAGE-CONDITIONED ORDER"] for a in cards for b in cards if a < b)
    d0 = od.decide(art, sit({"cover_1_unspecified": 1.0}), [MESH], mode="research", priors=False, allow_adjustments=False)
    assert adv.build_card(d0, {"cover_1_unspecified": 1.0})["COVERAGE-CONDITIONED ORDER"] is None             # off => unchanged card


def test_a_back_kept_in_to_protect_is_not_offered_as_the_checkdown(art):
    s = sm.structure_any(*MESH)
    d = od.Decision(formation=s.formation, play=s.play, adjustment="x", probability=.5, stock_probability=.5, interval=(.4, .6), final_structure=s,
                    actions=(ex.PlannedAction("off.pass_protection", {"protection": "Half Slide"}),))
    card = adv.build_card(d, {"cover_1_unspecified": 1.0}, priors=True)
    assert not card["CHECKDOWN"].startswith(("HB", "FB"))


def test_defensive_inverse_reads_the_same_table_from_the_other_side():
    c2man, c2zone = fp.descriptor("cover_2_man"), fp.descriptor("cover_2_spot")
    smash_heavy, mesh_heavy = {"smash": 1.0}, {"mesh": 1.0}
    assert fp.defense_call_edge(c2zone, smash_heavy)["offense_edge_allowed"] > fp.defense_call_edge(c2man, smash_heavy)["offense_edge_allowed"]
    assert fp.defense_call_edge(c2man, mesh_heavy)["offense_edge_allowed"] > fp.defense_call_edge(c2zone, mesh_heavy)["offense_edge_allowed"]


# ------------------------------------------------------------------------------------------------------------ adaptive learning
def _snap(play, cov, ok, opp="o1", verified=True, result=None):
    return {"side": "offense", "opponent_id": opp, "formation": play[0], "play": play[1], "coverage_seen": cov, "coverage_typed": None,
            "result": result or ("+8" if ok else "incomplete"), "executed_verification": "verified" if verified else "unknown"}


def test_one_failed_snap_barely_moves_a_cell_and_twenty_move_it_gradually():
    cell = ("man", "mesh")
    mu0, _ = ap.prior_cell(*cell)
    one, _ = ap.posterior_cells([_snap(MESH, "Cover 1", False)] + [_snap(MESH, "Cover 1", True)] * 3, "o1")
    many = [_snap(MESH, "Cover 1", False)] * 20 + [_snap(MESH, "Cover 1", True)] * 4
    twenty, _ = ap.posterior_cells(many, "o1")
    assert abs(one[cell] - mu0) < 0.04
    assert twenty[cell] < mu0 - 0.05 and twenty[cell] > -0.5                                  # moves, but is not abandoned or zeroed


def test_opponent_level_is_pooled_toward_madden_level():
    snaps = [_snap(MESH, "Cover 1", True, opp="o1")] * 12 + [_snap(MESH, "Cover 1", False, opp="o2")] * 6
    l1, i1 = ap.posterior_cells(snaps, "o1")
    l2, i2 = ap.posterior_cells(snaps, "o2")
    cell = i1["cells_updated"]["man|mesh"]
    assert l1[("man", "mesh")] != l2[("man", "mesh")] and cell["n_opponent"] > 0
    assert min(cell["madden"], cell["opponent"]) <= l1[("man", "mesh")] <= max(cell["madden"], cell["opponent"]) + 0.2


def test_unattested_execution_and_penalties_count_less_or_not_at_all():
    a, _ = ap.posterior_cells([_snap(MESH, "Cover 1", False, verified=True)] * 10, "o1")
    b, _ = ap.posterior_cells([_snap(MESH, "Cover 1", False, verified=False)] * 10, "o1")
    mu0, _ = ap.prior_cell("man", "mesh")
    assert abs(a[("man", "mesh")] - mu0) > abs(b[("man", "mesh")] - mu0)
    flags = [_snap(MESH, "Cover 1", False, result="penalty holding")] * 10
    assert ap.observations(flags) == []


def test_snaps_without_revealed_coverage_teach_nothing():
    assert ap.observations([_snap(MESH, "", True)]) == []


def test_learned_cells_change_the_models_edge_and_therefore_its_ranking_input():
    s = sm.structure_any(*MESH)
    base = fp.matchup_edge(s, {"cover_1_unspecified": 1.0})["edge"]
    learned, _ = ap.posterior_cells([_snap(MESH, "Cover 1", False)] * 40, "o1")
    after = fp.matchup_edge(s, {"cover_1_unspecified": 1.0}, learned=learned)["edge"]
    assert after < base


def test_exploration_is_cautious_logged_and_never_on_costly_downs():
    rows = [{"formation": "A", "play": "a", "probability": 0.52, "route_aware_se": 0.02}, {"formation": "B", "play": "b", "probability": 0.515, "route_aware_se": 0.30}]
    c = ap.exploration_choice(rows, situation={"down": 2, "yardline": 40})
    assert c["explored"] and c["chosen"] == ("B", "b") and c["candidates"] and c["policy"] == "deterministic_ucb"
    assert not ap.exploration_choice(rows, situation={"down": 4, "yardline": 40})["explored"]
    assert not ap.exploration_choice(rows, situation={"down": 1, "yardline": 97})["explored"]
    far = [rows[0], {**rows[1], "probability": 0.40}]
    assert not ap.exploration_choice(far, situation={"down": 1, "yardline": 40})["explored"]


# ------------------------------------------------------------------------------------------------------------ the real live loop
def _live(tmp_path, monkeypatch, artifacts, label, on, look):
    from tests.madden_ml.test_defense_knowledge_live import _ctrl, _prime, _world
    from cfb_coach.madden.model import offense_route_ranker as rr
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = _world(tmp_path, monkeypatch, artifacts, "default")
    (monkeypatch.setenv if on else monkeypatch.delenv)(*( (rr.ENV_PRIORS, "1") if on else (rr.ENV_PRIORS,) ), **({} if on else {"raising": False}))
    _prime(db)
    c = _ctrl(db)
    c.call_only("1&10 my 25")
    for sit_raw in ("2&4 my 31", "1&10 my 41", "2&8 my 43"):
        c.result_and_call(outcome="+6", sit_raw=sit_raw, their=look)
    info = rr.info_of(c.last_call)
    db.close()
    return info


@pytest.fixture(scope="module")
def live_artifacts(art, rows):
    from tests.madden_ml.test_defense_knowledge_live import artifacts  # noqa: F401
    from cfb_coach.madden.model import experimental_model as em
    import tests.madden_ml.test_offense_intel_model as T
    return {"default": art, "exp": em.train_experimental([dict(r) for r in T._rows()], side="offense")}


def test_live_loop_ranking_and_card_follow_the_reported_defense_only_when_priors_are_on(tmp_path, monkeypatch, live_artifacts):
    on_man = _live(tmp_path / "a", monkeypatch, live_artifacts, "a", True, "Cover 1")
    on_zone = _live(tmp_path / "b", monkeypatch, live_artifacts, "b", True, "Cover 4 Quarters")
    off_man = _live(tmp_path / "c", monkeypatch, live_artifacts, "c", False, "Cover 1")
    off_zone = _live(tmp_path / "d", monkeypatch, live_artifacts, "d", False, "Cover 4 Quarters")
    assert on_man["applied"] and on_man["n_scored"] == 93 and on_man["latency_ms"] < 150 and not on_man.get("fallback_reason")
    assert "football_priors" in on_man and "football_priors" not in off_man
    assert on_man["route_aware_top"] != on_zone["route_aware_top"]                              # different reported defense => different calls
    assert [t["p"] for t in on_man["route_aware_top"]] != [t["p"] for t in on_zone["route_aware_top"]]
    assert on_man["coverage"]["top"] != on_zone["coverage"]["top"]
