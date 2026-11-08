"""Regression: decoded partial Ace geometry must not get full football matchup credit.

These are synthetic structures. They don't certify third-party route geometry.
"""
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import football_priors as fp
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model.offense_pregame_scoring import score_pair


def _score(structure, *, feature_tier):
    with (
        patch.object(em, "predict_success", return_value={
            "probability": 0.37, "evidence_quality": "empirical",
        }),
        patch.object(fp, "matchup_edge", return_value={
            "edge": 0.09, "se": 0.0, "cells": [], "mass": 1.0,
        }),
    ):
        return score_pair(
            em.ExperimentalArtifact(),
            formation="Pistol Trips",
            play="Inside Cross",
            probe=SimpleNamespace(down=2, distance=7, yardline=40),
            opponent_id="cpu",
            mix={"cover_2_unspecified": 1.0},
            priors_on=True,
            structure=structure,
            feature_tier=feature_tier,
            model_use_permitted=True,
        )


def test_successfully_decoded_partial_routes_receive_confidence_shrink():
    source = sm.structure_any("Pistol Trips", "Inside Cross")
    assert source.n_known > 0  # exercise the originally missed branch
    geometry = replace(
        source,
        routes=tuple((slot, spec, "partial") for slot, spec, _ in source.routes),
        read_order=(),
        sourced_concepts=(),
    )
    result = _score(geometry, feature_tier="partial")
    assert result.assignment_status == "partial"
    assert result.assignment_confidence == pytest.approx(0.35)
    assert result.components["raw_matchup_edge"] == pytest.approx(0.09)
    assert result.matchup_edge == pytest.approx(0.0315)
    assert result.components["reads"] == "UNKNOWN (provisional structure; not verified)"


def test_verified_routes_keep_full_matchup_edge():
    trusted = sm.structure_any("Pistol Trips", "Inside Cross")
    assert trusted.n_known > 0
    result = _score(trusted, feature_tier="partial")
    assert result.assignment_status == "verified"
    assert result.matchup_edge == pytest.approx(0.09)


def test_partly_documented_routes_are_not_mistaken_for_ace_geometry():
    trusted = sm.structure_any("Pistol Trips", "Inside Cross")
    first_slot, first_spec, _ = trusted.routes[0]
    mixed = replace(trusted, routes=((first_slot, first_spec, "football_inference"),) + trusted.routes[1:])
    result = _score(mixed, feature_tier="name_only")
    assert result.assignment_status == "partial"
    assert result.matchup_edge == pytest.approx(0.09)
