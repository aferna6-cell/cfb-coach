"""Tiny synthetic graphs: geometry, not a play name, drives matchup concepts."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_portfolio as portfolio
from cfb_coach.madden.model.offense_geometry_decode import (
    decode_provisional_geometry, resolve_scoring_structure,
)
from cfb_coach.madden.model.offense_pregame_scoring import score_pair

FORMATION, PLAY = "Gun Test Wide", "NonceGeomAlpha"
PROBE = SimpleNamespace(down=3, distance=10, yardline=45,
                        coverage_hint="Cover 1 Robber", coverage_source="live")


def element(family, points, kind="route_proposal"):
    return dict(kind=kind, route_family_candidate=family, polyline=points, confidence="medium")


def graph(routes):
    markers = [dict(start_px=[x, 200], identity_hypothesis="DUMMY_ID", slot="DUMMY_SLOT",
                    elements=[]) for x in (0, 20, 100, 180, 200)]
    for x, family, points in routes:
        next(m for m in markers if m["start_px"][0] == x)["elements"].append(element(family, points))
    return dict(coordinate_frame="diagram_px_unreviewed", markers=markers, unowned_elements=[],
                qb_read=["DUMMY_READ"], yards=9999, concept_interactions=["DUMMY_CONCEPT"])


def smash():
    return graph([(0, "curl_or_comeback", [[0, 200], [0, 160], [0, 170]]),
                  (20, "stem_then_diagonal", [[20, 200], [20, 100], [-40, 40]])])


def verts():
    return graph([(x, "vertical", [[x, 200], [x, 20]]) for x in (0, 20, 180, 200)])


def decode(g, tier="high_confidence_extracted"):
    return decode_provisional_geometry(g, formation=FORMATION, play=PLAY, feature_tier=tier)


def score(structure, cover):
    return score_pair(em.ExperimentalArtifact(), formation=FORMATION, play=PLAY,
                      probe=PROBE, opponent_id="unknown", priors_on=True,
                      mix={f"cover_{cover}_unspecified": 1.0}, structure=structure)


def test_route_combination_changes_cover_1_2_3_matchup():
    a, b = decode(smash()), decode(verts())
    assert "smash" in a.concepts()
    assert "four_verticals" in b.concepts()
    assert [(r.family, r.side, r.band) for r in a.specs] == [
        ("hitch", "left", "short"), ("corner", "left", "deep")]
    assert [(r.family, r.side, r.band) for r in b.specs] == [
        ("streak", "left", "deep"), ("streak", "left", "deep"),
        ("streak", "right", "deep"), ("streak", "right", "deep")]
    for s in (a, b):
        assert [slot for slot, _, _ in s.routes] == [f"geom_{i}" for i in range(s.n_known)]
        assert {tier for _, _, tier in s.routes} == {"high_confidence_extracted"}
        assert s.read_order == s.sourced_concepts == s.skill_slots == ()
        assert s.personnel_tier == "unknown"
    with patch.object(em, "predict_success", return_value={"probability": .5}):
        ae = {c: score(a, c).matchup_edge for c in (1, 2, 3)}
        be = {c: score(b, c).matchup_edge for c in (1, 2, 3)}
        assert ae[2] > ae[1]
        assert be[3] > ae[3]
        assert ae[2] > be[2]
        assert ae[2] != be[2] and ae[3] != be[3]
        # Both graphs deliberately disappear when no decoded structure is supplied.
        controls = [{c: score(None, c).matchup_edge for c in (1, 2, 3)} for _ in (a, b)]
        assert controls[0] == controls[1]
        assert controls[0] == {1: 0, 2: 0, 3: 0}


@pytest.mark.parametrize("family", ["multi_break", "unknown_complex", "unexpected"])
def test_ambiguous_families_omitted(family):
    assert decode(graph([(0, family, [[0, 200], [0, 20]])])).routes == ()


def test_documented_first_party_wins_unchanged():
    first_party = sm.structure_any("Pistol Trips", "Inside Cross")
    assert any(t in {"in_game_verified", "diagram_verified", "source_documented"}
               for _, _, t in first_party.routes)
    with patch.object(sm, "structure_any", return_value=first_party), patch(
            "cfb_coach.madden.model.offense_geometry_decode.decode_provisional_geometry") as decoder:
        assert resolve_scoring_structure("Pistol Trips", "Inside Cross", geometry=smash(),
                                        feature_tier="high_confidence_extracted",
                                        model_use_permitted=True) is first_party
        decoder.assert_not_called()


@pytest.mark.parametrize("geometry,permitted", [(None, True), ({}, True), (smash(), False)])
def test_fallback_without_usable_permitted_geometry(geometry, permitted):
    s = resolve_scoring_structure(FORMATION, PLAY, geometry=geometry,
                                  feature_tier="high_confidence_extracted", model_use_permitted=permitted)
    assert s == sm.structure_any(FORMATION, PLAY)
    assert s.routes == ()


@pytest.mark.parametrize("candidate,depth,family", [
    ("vertical", 180, "seam"), ("break_in", 30, "slant"),
    ("break_in", 90, "in"), ("break_in", 180, "post"),
    ("break_out", 30, "out"), ("break_out", 180, "corner"),
    ("curl_or_comeback", 30, "hitch"), ("curl_or_comeback", 90, "curl"),
    ("curl_or_comeback", 180, "comeback"), ("backward_or_swing", -30, "swing"),
])
def test_family_bands_and_center_alignment(candidate, depth, family):
    s = decode(graph([(100, candidate, [[100, 200], [100, 200-depth]])]), tier="partial")
    assert s.specs[0].family == family
    assert s.specs[0].side == "middle"
    assert s.routes[0][2] == "partial"


def test_unowned_crossing_and_diagonal_direction():
    g = graph([])
    g["unowned_elements"] = [element("flat_or_horizontal", [[0, 200], [180, 190]], "route")]
    assert decode(g).specs[0].family == "drag"
    assert decode(g).specs[0].side == "left"
    assert decode(graph([(0, "diagonal_upfield", [[0, 200], [0, 100], [70, 20]])])).specs[0].family == "post"
    assert decode(graph([(0, "flat_or_horizontal", [[0, 200], [30, 190]])])).specs[0].family == "flat"


def test_catalog_geometry_is_passed_to_scoring():
    entry = SimpleNamespace(listing_id="synthetic", formation=FORMATION, play=PLAY,
                            feature_tier="high_confidence_extracted", model_use_permitted=True,
                            source_books=("Test",), installable=True)
    geometries = []
    index = SimpleNamespace(entries=[entry], geometry_for=lambda lid: geometries.append(lid) or smash())
    from cfb_coach.madden.model import offense_pregame_scoring as scoring
    with patch.object(em, "predict_success", return_value={"probability": .5}), patch.object(
            scoring, "score_pair", wraps=scoring.score_pair) as spy:
        def run(idx):
            portfolio._catalog_universe(em.ExperimentalArtifact(), idx, {}, "unknown",
                                        coverage_mix={"cover_2_unspecified": 1}, priors_on=True, learned=None)
            return spy.call_args.kwargs["structure"]
        decoded = run(index)
        fallback = run(SimpleNamespace(entries=[entry]))
        assert geometries == ["synthetic"]
        assert "smash" in decoded.concepts() and fallback.routes == ()
        assert score(decoded, 2).matchup_edge > score(fallback, 2).matchup_edge


def test_omitted_routes_do_not_leak_or_displace_anonymous_slots():
    g = smash()
    g["markers"][0]["elements"].extend([
        element("multi_break", [[0, 200], [10, 20]]),
        element("unknown_complex", [[0, 200], [20, 10]])])
    assert decode(g) == decode(smash())


def test_unsupported_frame_or_malformed_polyline_is_not_geometry_evidence():
    g = smash()
    g["coordinate_frame"] = "unknown"
    assert decode(g).routes == ()
    g = graph([(0, "vertical", [[0, 200], [float("nan"), 20]])])
    assert decode(g).routes == ()


def test_documented_tier_claim_is_never_copied_to_decoded_routes():
    assert {t for _, _, t in decode(smash(), tier="source_documented").routes} == {"partial"}
