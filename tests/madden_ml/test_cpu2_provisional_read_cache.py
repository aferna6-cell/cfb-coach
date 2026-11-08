"""A provisional diagram cache must feed useful but explicitly tentative QB reads."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from cfb_coach.madden.model import offense_route_ranker as rr
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model.offense_decision import Decision
from cfb_coach.madden.model.offense_live_read_cache import (
    SCHEMA, pairs_digest, structure_for_live_read,
)
from cfb_coach.madden.model.offense_presnap_advice import build_card


def _cache(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.write_text(json.dumps({
        "schema": SCHEMA,
        "pairs_digest": pairs_digest(pairs),
        "plays": [{
            "formation": "Gun Bunch X Nasty",
            "play": "Fake Deep Pass",
            "routes": [
                {"family": "corner", "side": "left", "band": "deep", "tier": "partial"},
                {"family": "hitch", "side": "right", "band": "short", "tier": "partial"},
                {"family": "flat", "side": "right", "band": "shallow", "tier": "partial"},
            ],
        }],
    }), encoding="utf-8")


def test_live_geometry_enables_real_play_specific_read_card(tmp_path):
    pair = ("Gun Bunch X Nasty", "Fake Deep Pass")
    path = tmp_path / "reads.json"
    _cache(path, [pair])
    stock = sm.structure_any(*pair)
    assert stock.n_known == 0
    modified = structure_for_live_read(
        *pair, stock, installed_pairs_digest=pairs_digest([pair]), path=path
    )
    assert modified.n_known == 3
    assert modified.skill_slots == ()  # anonymous diagram != eligible Xbox receiver targets
    assert modified.read_order == ()
    dec = Decision(
        formation=pair[0], play=pair[1],
        adjustment="NO ADJUSTMENT", probability=0.55,
        stock_probability=0.55, interval=(0.43, 0.67),
        final_structure=modified,
    )
    card = build_card(dec, {})
    assert "INFERRED from unreviewed play-diagram geometry" in card["READ BASIS"]
    assert "receiver identities UNKNOWN" in card["READ BASIS"]
    assert "receiver unverified" in card["PRIMARY READ"]
    assert "UNKNOWN: this play's receiver routes" not in card["PRIMARY READ"]
    assert "geom_" not in card["PRIMARY READ"]


def test_inventory_mismatch_refuses_stale_route_geometry(tmp_path):
    pair = ("Gun Bunch X Nasty", "Fake Deep Pass")
    path = tmp_path / "reads.json"
    _cache(path, [pair])
    stock = sm.structure_any(*pair)
    result = structure_for_live_read(
        *pair, stock, installed_pairs_digest=pairs_digest([
            pair, ("Gun Doubles", "Different Play")
        ]), path=path,
    )
    assert result is stock


def test_missing_cache_falls_back_without_misleading_read(tmp_path):
    stock = sm.structure_any("Gun Bunch X Nasty", "Fake Deep Pass")
    result = structure_for_live_read(
        "Gun Bunch X Nasty", "Fake Deep Pass", stock,
        installed_pairs_digest="not-an-inventory", path=tmp_path / "absent.json",
    )
    assert result is stock


def test_documented_read_wins_over_provisional_cache(tmp_path):
    from cfb_coach.madden.model.offense_play_features import PlayStructure
    from cfb_coach.madden.model.offense_taxonomy import RouteSpec
    pair = ("Gun Bunch X Nasty", "Fake Deep Pass")
    path = tmp_path / "reads.json"
    _cache(path, [pair])
    stock = PlayStructure(
        formation=pair[0], play=pair[1],
        routes=(("WR1", RouteSpec("slant", "left", "short"), "diagram_verified"),),
        skill_slots=("WR1",), personnel_tier="verified", hint={},
        read_order=("WR1",),
    )
    result = structure_for_live_read(
        *pair, stock, installed_pairs_digest=pairs_digest([pair]), path=path,
    )
    assert result is stock


def test_prior_scoring_without_active_route_residual_does_not_crash():
    from types import SimpleNamespace
    import time
    from cfb_coach.madden.model import adaptive_priors

    pair = ("Gun Bunch X Nasty", "Fake Deep Pass")
    base = [{"formation": pair[0], "play": pair[1],
             "probability": 0.45, "uncertainty": 0.2, "components": {}}]
    art = SimpleNamespace(residual_active=False)
    with (
        mock.patch.object(rr, "resolve_artifact_path", return_value=Path("mock")),
        mock.patch.object(rr, "load_artifact", return_value=art),
        mock.patch.object(rr, "_coverage", return_value=({}, {})),
        mock.patch.object(rr, "football_priors_on", return_value=True),
        mock.patch.object(adaptive_priors, "learned_cells", return_value=({}, {})),
        mock.patch.object(rr.sm, "structure_any", return_value=SimpleNamespace(n_known=0)),
        mock.patch.object(rr.fpri, "matchup_edge", return_value={"edge": 0.1, "se": 0, "cells": []}),
    ):
        rows, _, ctx = rr.apply(
            "route_aware", base, [pair], [pair],
            sit=SimpleNamespace(down=1, distance=10, yardline=35),
            opponent_id="cpu", db=None, session_id="s", snap_seq=1,
            rank_kwargs={}, started=time.perf_counter(), budget_ms=1000,
        )
    assert ctx.applied, ctx.error
    assert rows[0]["football_prior_edge"] == 0.1
    assert rows[0]["probability"] > base[0]["probability"]
