"""Deliverable C: identical legal candidates, explicit shells, one portfolio picker.

Run with pytest -s to see numerical proof rows. These are controlled prior-only
counterfactuals, not observed gameplay or claimed improvements in win rate.
"""
from dataclasses import replace
import json
from unittest.mock import patch

import pytest

from cfb_coach.madden import catalog
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import football_priors as fp
from cfb_coach.madden.model import offense_portfolio as portfolio
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model.football_situation import explain_play
from cfb_coach.madden.model.offense_pregame_scoring import score_pair

# A subset of ONE real source book; no synthetic formation or play names.
CANDIDATES = {"Buccaneers": {
    "Gun 5WR Tight": ["Mesh", "Smash Dagger", "Mtn Double Post", "QB Draw"],
    "Gun Doubles Clamp Stack": ["Flood Seam", "Mtn FK Rub", "Mtn Shuffle Verts Smash", "Inside Zone"],
    "Pistol Trips": ["Slants", "Smash", "HB Slip Screen", "HB Dive"],
    "Singleback Deuce Close": ["Flood", "Stick", "HB Dive"],
    "Gun Bunch TE": ["Mesh", "Mtn Return Flood Stick", "Deep Smash", "Inside Zone"],
}}
KEYS = ("cover_1_unspecified", "cover_1_robber", "cover_2_man", "cover_2_spot", "trap_2",
        "cover_3_spot", "cover_3_match", "cover_4_quarters", "cover_4_palms", "cover_1_pressure")
CONTRASTS = (("cover_1_unspecified", "cover_1_robber"),
             ("cover_2_man", "cover_2_spot"), ("cover_2_man", "trap_2"),
             ("cover_3_spot", "cover_3_match"),
             ("cover_4_quarters", "cover_4_palms"),
             ("cover_1_unspecified", "cover_1_pressure"))


def scored_book(mix, priors_on=True, *, name_only=False, probe=None):
    art = em.ExperimentalArtifact()  # no training, history, or runtime opt-in
    rows = []
    for source, forms in CANDIDATES.items():
        for form, plays in forms.items():
            for play in plays:
                structure = sm.structure_any(form, play)
                if name_only:
                    structure = replace(structure, routes=(), read_order=(), sourced_concepts=())
                result = score_pair(art, formation=form, play=play,
                                    probe=probe or portfolio.PORTFOLIO_PROBES["normal"],
                                    mix=mix, opponent_id="unknown", priors_on=priors_on,
                                    structure=structure)
                rows.append({"formation": form, "play": play, "source_book": source,
                             "base": result.base_probability, "score": result.total_probability,
                             "edge": result.matchup_edge, "status": result.assignment_status,
                             "confidence": result.assignment_confidence, "cells": result.cells,
                             "reads": result.components["reads"],
                             "concepts": fp.play_concepts(structure)})
    return sorted(rows, key=lambda r: (-r["score"], r["formation"], r["play"]))


def selected_book(mix, priors_on=True, *, cap=3, active=None):
    return portfolio.select_formation_portfolio(
        art=em.ExperimentalArtifact(), all_books=CANDIDATES,
        active=active or {}, opponent_id="unknown", max_formations=cap,
        coverage_mix=mix, priors_on=priors_on)


def identities(rows, n=1):
    return [(r["formation"], r["play"]) for r in rows[:n]]


def summarize(key, rows, selection):
    return {"mix": key, "scores": [{k: r[k] for k in ("formation", "play", "score", "edge", "concepts", "cells", "status")}
                                    for r in rows],
            "preferred_formations": [r["formation"] for r in selection["chosen"]],
            "portfolio_scores": {r["formation"]: r["score"] for r in selection["ranked"]}}


def test_candidates_are_legal_known_source_book_pairs():
    for source, forms in CANDIDATES.items():
        legal = catalog.book_formations("offense", source)
        for form, plays in forms.items():
            assert form in legal
            assert set(plays) <= set(legal[form])
    assert all(fp.descriptor(k) is not None for k in KEYS)


@pytest.mark.parametrize("left,right", CONTRASTS)
def test_coverage_counterfactual_changes_actual_scores_and_reports_decisions(left, right):
    with patch.object(em, "predict_success", return_value={"probability": 0.5}):
        a, b = scored_book({left: 1}), scored_book({right: 1})
        pa, pb = selected_book({left: 1}), selected_book({right: 1})
    av = {(r["formation"], r["play"]): r["score"] for r in a}
    bv = {(r["formation"], r["play"]): r["score"] for r in b}
    assert av.keys() == bv.keys()
    assert any(abs(av[k] - bv[k]) > 1e-6 for k in av)
    assert {r["formation"]: r["score"] for r in pa["ranked"]} != {
        r["formation"]: r["score"] for r in pb["ranked"]}
    if left == "cover_2_man":
        assert pa["chosen"][0]["formation"] != pb["chosen"][0]["formation"]
        assert {r["formation"] for r in pa["chosen"]} != {r["formation"] for r in pb["chosen"]}
    # Five contrasts have genuine play/formation argmax flips, not prose changes.
    if (left, right) not in (("cover_4_quarters", "cover_4_palms"),):
        assert identities(a) != identities(b)
    print(json.dumps({"left": summarize(left, a, pa), "right": summarize(right, b, pb),
                      "argmax_changed": identities(a) != identities(b),
                      "top3_changed": identities(a, 3) != identities(b, 3),
                      "formation_selection_changed": [r["formation"] for r in pa["chosen"]] !=
                      [r["formation"] for r in pb["chosen"]],
                      "max_probability_delta": max(abs(av[k] - bv[k]) for k in av)}, sort_keys=True))


def test_priors_toggle_changes_model_scores_using_one_picker():
    with patch.object(em, "predict_success", return_value={"probability": 0.5}):
        off = selected_book({"cover_1_robber": 1}, False)
        on = selected_book({"cover_1_robber": 1}, True)
        off_rows = scored_book({"cover_1_robber": 1}, False)
        on_rows = scored_book({"cover_1_robber": 1}, True)
    assert all(r["edge"] == 0 and r["score"] == 0.5 for r in off_rows)
    assert identities(off_rows) != identities(on_rows)
    assert off["ranked"] != on["ranked"]
    print(json.dumps({"priors_off": summarize("cover_1_robber", off_rows, off),
                      "priors_on": summarize("cover_1_robber", on_rows, on)}))


def test_trusted_vs_name_only_uses_real_routes_without_fabrication():
    with patch.object(em, "predict_success", return_value={"probability": 0.5}):
        trusted = scored_book({"cover_1_pressure": 1})
        names = scored_book({"cover_1_pressure": 1}, name_only=True)
    assert all(r["status"] != "verified" and "UNKNOWN" in r["reads"] for r in names)
    assert all(r["confidence"] <= fp.NAME_TRUST for r in names)
    slants = next(r for r in trusted if r["play"] == "Slants")
    stripped = next(r for r in names if r["play"] == "Slants")
    assert slants["status"] == "verified"
    assert slants["score"] != stripped["score"]
    assert identities(trusted) != identities(names)
    print(json.dumps({"trusted_winner": trusted[0], "name_only_winner": names[0],
                      "trusted_slants": slants, "name_only_slants": stripped}))


def test_unknown_opponent_mix_is_negative_control_for_scores_and_picker():
    with patch.object(em, "predict_success", return_value={"probability": 0.5}):
        off, unknown = scored_book({}, False, name_only=True), scored_book({}, True, name_only=True)
        poff, punk = selected_book({}, False), selected_book({}, True)
    assert off == unknown
    assert all(r["edge"] == 0 and r["cells"] == [] for r in unknown)
    assert all(r["status"] != "verified" and "UNKNOWN" in r["reads"] for r in unknown)
    assert poff == punk  # identical algorithm, scores, ranking, marginal decisions


def test_portfolio_does_not_seed_installed_formations_or_fill_fifteen():
    active = {"formations": {"Not a legal candidate": ["No play"]}}
    with patch.object(em, "predict_success", return_value={"probability": 0.5}):
        for enabled in (False, True):
            result = selected_book({"cover_1_unspecified": 1}, enabled, cap=15, active=active)
            assert len(result["chosen"]) < 15
            assert all(r["formation"] in CANDIDATES["Buccaneers"] for r in result["chosen"])


@pytest.mark.parametrize("key", KEYS)
def test_third_and_long_keeps_passing_answers_without_punt_or_run_bias(key):
    probe = portfolio.PORTFOLIO_PROBES["third_and_long"]
    assert probe.down == 3 and probe.distance >= 7
    with patch.object(em, "predict_success", return_value={"probability": 0.5}):
        rows = scored_book({key: 1}, probe=probe)
        chosen = selected_book({key: 1})
    for row in rows:
        fit = explain_play(row["play"], probe)
        row["situational_score"] = row["score"] + fit["selection_delta"] + fit["coordinator_delta"]
    winner = max(rows, key=lambda r: r["situational_score"])
    assert not catalog.is_run(winner["play"])
    assert "punt" not in winner["play"].lower()
    assert any(r["has_pass"] for r in chosen["chosen"])
    assert chosen["situation_coverage"]["third_and_long"]["addressed"]
    assert any(not catalog.is_run(p) for r in chosen["chosen"] for p in r["plays"])


def test_real_predict_success_prior_only_artifact_end_to_end():
    # No mocked ML path and no fitting or saved-artifact mutation.
    before = selected_book({"cover_1_pressure": 1}, False)
    after = selected_book({"cover_1_pressure": 1}, True)
    a, b = scored_book({"cover_1_pressure": 1}, False), scored_book({"cover_1_pressure": 1}, True)
    assert any(r["score"] != r["base"] for r in b)
    assert {r["formation"]: r["score"] for r in before["ranked"]} != {
        r["formation"]: r["score"] for r in after["ranked"]}
    print(json.dumps({"real_prior_only_before": summarize("cover_1_pressure", a, before),
                      "real_prior_only_after": summarize("cover_1_pressure", b, after)}))
