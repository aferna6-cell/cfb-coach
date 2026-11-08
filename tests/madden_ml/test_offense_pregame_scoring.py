"""Pregame scores preserve model value and never claim hypothetical live evidence."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import football_priors as fp
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model.offense_pregame_scoring import (
    assignment_status, coverage_mix_from_seen, score_pair,
)

PROBE = SimpleNamespace(down=3, distance=10, yardline=45,
                        coverage_hint="Cover 1 Robber", coverage_source="live")


def score(*, mix=None, priors_on=True, **kwargs):
    return score_pair(em.ExperimentalArtifact(), formation="Gun 5WR Tight", play="Mesh",
                      probe=PROBE, opponent_id="unknown", mix=mix,
                      priors_on=priors_on, **kwargs)


@pytest.mark.parametrize("priors_on,mix", [(False, {"cover_1_unspecified": 1}), (True, {}), (False, {})])
def test_disabled_or_unknown_preserves_model_probability(priors_on, mix):
    with patch.object(em, "predict_success", return_value={"probability": 0.37}) as predict:
        result = score(mix=mix, priors_on=priors_on)
    assert result.base_probability == result.total_probability == 0.37
    assert result.matchup_edge == 0
    assert result.cells == []
    assert predict.call_count == 1
    assert predict.call_args.kwargs["coverage_source"] == "none"
    assert predict.call_args.kwargs["coverage_hint"] is None
    assert "UNKNOWN" in result.components["reads"]
    if not mix:
        assert result.mix_source == "unknown"


def test_enabled_uses_uncovered_base_and_same_live_logit_composition():
    with patch.object(em, "predict_success", return_value={"probability": 0.37}) as predict:
        result = score(mix={"cover_1_unspecified": 1}, mix_source="opponent_history")
    assert predict.call_args.kwargs["coverage_source"] not in ("live", "last")
    assert predict.call_args.kwargs["coverage_hint"] is None
    assert result.mix_source == "opponent_history"
    # Mesh name hint: 0.35 trust * 0.18 man prior = 0.063 logit lift.
    assert result.matchup_edge == pytest.approx(0.063)
    import math
    assert result.total_probability == pytest.approx(1 / (1 + math.exp(-(math.log(.37 / .63) + .063))))
    assert result.cells
    assert result.assignment_status == "name_only"
    assert result.assignment_confidence == fp.NAME_TRUST


@pytest.mark.parametrize("formation,play", [("Gun 5WR Tight", "Mesh"),
                                               ("Singleback Bunch TE", "Cross Drag"),
                                               ("Singleback Deuce Close", "Flood")])
def test_robber_is_not_collapsed_to_generic_man(formation, play):
    structure = sm.structure_any(formation, play)
    with patch.object(em, "predict_success", return_value={"probability": 0.5}):
        common = dict(formation=formation, play=play, probe=PROBE, opponent_id="unknown",
                      priors_on=True, structure=structure)
        man = score_pair(em.ExperimentalArtifact(), mix={"cover_1_unspecified": 1}, **common)
        robber = score_pair(em.ExperimentalArtifact(), mix={"cover_1_robber": 1}, **common)
    assert man.matchup_edge != robber.matchup_edge
    assert man.total_probability != robber.total_probability
    assert man.assignment_status == robber.assignment_status == "name_only"


def test_assignment_status_requires_documented_routes_not_name_hints():
    trusted = sm.structure_any("Pistol Trips", "Inside Cross")
    assert trusted.routes and trusted.n_known
    assert assignment_status(trusted) == ("verified", 0.6)
    name_only = replace(trusted, routes=(), read_order=(), sourced_concepts=())
    assert assignment_status(name_only) == ("name_only", fp.NAME_TRUST)
    inferred = replace(trusted, routes=tuple((slot, spec, "football_inference")
                                           for slot, spec, _ in trusted.routes))
    assert assignment_status(inferred)[0] != "verified"
    partial = replace(trusted, routes=inferred.routes[:1] + trusted.routes[1:])
    assert assignment_status(partial)[0] != "verified"
    assert assignment_status(partial)[0] == "partial"
    assert assignment_status(partial)[1] < assignment_status(trusted)[1]
    unknown = sm.structure_any("Unknown formation", "Uncatalogued XYZ")
    assert assignment_status(unknown) == ("unknown", 0.0)
    result = score(mix={}, structure=unknown)
    assert result.assignment_status == "unknown"
    assert result.assignment_confidence == 0
    assert result.cells == []
    assert "UNKNOWN" in result.components["reads"]


def test_learned_residual_is_added_not_substituted():
    learned = {("g", "man", "mesh"): 0.1, ("v", "man_one_high", "mesh"): 0.05}
    with patch.object(em, "predict_success", return_value={"probability": 0.5}):
        prior = score(mix={"cover_1_unspecified": 1})
        updated = score(mix={"cover_1_unspecified": 1}, learned=learned)
    expected = fp.matchup_edge(sm.structure_any("Gun 5WR Tight", "Mesh"),
                               {"cover_1_unspecified": 1}, learned=learned)
    assert updated.matchup_edge == pytest.approx(expected["edge"])
    assert updated.total_probability > prior.total_probability


def test_history_preserves_explicit_variants_and_drops_unspecific_spot():
    labels = ["Cover 1", "Cover 1 Robber", "Cover 2 Man", "Cover 2 Spot",
              "Cover 2 Trap", "Cover 3 Spot", "Cover 3 Match", "Quarters",
              "Palms", "Cover 1 Pressure"]
    expected = {"cover_1_unspecified", "cover_1_robber", "cover_2_man", "cover_2_spot",
                "trap_2", "cover_3_spot", "cover_3_match", "cover_4_quarters",
                "cover_4_palms", "cover_1_pressure"}
    mix = coverage_mix_from_seen(labels)
    assert set(mix) == expected
    assert all(v == pytest.approx(0.1) for v in mix.values())
    assert coverage_mix_from_seen(["spot", "two high", "single_high", "???", None]) == {}
    assert coverage_mix_from_seen(["Cover 2"]) == {"cover_2_unspecified": 1}
    assert coverage_mix_from_seen(["Cover 3"]) == {"cover_3_unspecified": 1}
    assert coverage_mix_from_seen(["Cover 4"]) == {"cover_4_unspecified": 1}
    assert coverage_mix_from_seen(["Cover 0"]) == {"cover_0": 1}


def test_inferred_routes_are_not_verified_or_provisional():
    trusted = sm.structure_any("Pistol Trips", "Inside Cross")
    inferred = replace(trusted, routes=tuple((slot, spec, "football_inference") for slot, spec, _ in trusted.routes))
    assert assignment_status(inferred)[0] not in ("verified", "high_confidence_extracted")


def test_feed_claiming_verified_without_documented_routes_is_not_verified():
    trusted = sm.structure_any("Pistol Trips", "Inside Cross")
    name_only = replace(trusted, routes=(), read_order=(), sourced_concepts=())
    assert assignment_status(name_only, feature_tier="verified_assignments")[0] != "verified"
    # documented routes still verify regardless of a weaker claimed tier
    assert assignment_status(trusted, feature_tier="name_only")[0] == "verified"


def test_high_confidence_extracted_shrinks_edge_and_hides_reads():
    mix = {"cover_1_unspecified": 1}
    with patch.object(em, "predict_success", return_value={"probability": 0.37}):
        plain = score(mix=mix)
        prov = score(mix=mix, feature_tier="high_confidence_extracted", model_use_permitted=True)
        denied = score(mix=mix, feature_tier="high_confidence_extracted", model_use_permitted=False)
    assert prov.assignment_status == "high_confidence_extracted"
    conf = prov.assignment_confidence
    assert conf == fp.TIER_TRUST["football_inference"]
    raw_edge = prov.components["raw_matchup_edge"]
    assert raw_edge == pytest.approx(plain.matchup_edge)
    assert prov.matchup_edge == pytest.approx(raw_edge - (1 - conf) * raw_edge)
    assert prov.components["reads"] == "UNKNOWN (provisional structure; not verified)"
    assert denied.assignment_status != "high_confidence_extracted"


def test_cold_start_shrinks_toward_artifact_global_rate():
    art = em.ExperimentalArtifact(global_rate=0.4)  # synthetic test-only rate
    unknown = sm.structure_any("Unknown formation", "Uncatalogued XYZ")
    common = dict(formation="Gun 5WR Tight", play="Mesh", probe=PROBE, opponent_id="unknown", mix={}, priors_on=True)
    with patch.object(em, "predict_success", return_value={"probability": 0.5, "evidence_quality": "prior_driven"}):
        u = score_pair(art, structure=unknown, **common)
        n = score_pair(art, **common)
    assert u.assignment_status == "unknown"
    assert u.shrunk_probability == pytest.approx(0.4)
    assert u.score_calibrated is False and u.score_available is True
    assert u.components["raw"] == 0.5 and u.components["shrunk"] == pytest.approx(0.4)
    assert u.total_probability == 0.5  # raw composition preserved
    assert n.assignment_status == "name_only"
    assert n.shrunk_probability == pytest.approx(0.4 + fp.NAME_TRUST * (0.5 - 0.4))
    assert n.score_calibrated is False


def test_nested_base_global_rate_preferred_and_calibrated_base_not_shrunk():
    from cfb_coach.madden.model.offense_pregame_scoring import resolve_global_rate
    wrapper = SimpleNamespace(base=em.ExperimentalArtifact(global_rate=0.45), global_rate=0.5)
    assert resolve_global_rate(wrapper) == 0.45
    assert resolve_global_rate(SimpleNamespace(base={"global_rate": 0.42})) == 0.42
    with patch.object(em, "predict_success", return_value={"probability": 0.61, "evidence_quality": "empirical"}):
        r = score(mix={})
    assert r.score_calibrated is True
    assert r.shrunk_probability == r.total_probability == 0.61
