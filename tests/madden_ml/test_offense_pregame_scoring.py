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
