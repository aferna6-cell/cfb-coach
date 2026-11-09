"""Defensive structural learner: named assignment features are never free priors."""
from __future__ import annotations

from cfb_coach.madden.model import defense_v2_pilot as v2


def snap(game: int, seq: int, good: bool) -> v2.Snap:
    return v2.Snap(
        f"g{game}", "cpu", f"2026-11-{game+1:02}", seq,
        "Dime" if good else "Nickel Over",
        "Cover 2 Man" if good else "Cover 3 Sky",
        "man" if good else "single_high",
        "long", "vert", not good
    )


def test_structural_model_inactive_without_many_verified_games():
    model = v2.fit([snap(1, i, i % 2 == 0) for i in range(1, 30)])
    assert not model["structural"]["active"]
    assert v2.structural_delta(model, "Dime", "Cover 2 Man") == 0.0


def test_structural_feature_weights_are_fitted_only_after_threshold(monkeypatch):
    from cfb_coach.madden.model import defense_semantics as ds

    # A schematic mock: test training math, not inferred real Madden duties.
    monkeypatch.setattr(ds, "CANDIDATE_NAMES", ("dk_cand_in_catalog", "dk_mock_man"))
    monkeypatch.setattr(ds, "candidate_features", lambda form, play: {
        "dk_cand_in_catalog": 1.0,
        "dk_mock_man": 1.0 if play == "Cover 2 Man" else 0.0
    })
    data = [snap(game, seq, seq % 2 == 0) for game in range(8) for seq in range(1, 26)]
    model = v2.fit(data)
    assert model["structural"]["active"]
    assert model["structural"]["n_train"] == 200
    assert model["structural"]["n_games"] == 8
    # Positive offensive success corresponds to "Cover 3 Sky" in this synthetic
    # world; structural contribution should distinguish the two actions.
    assert v2.structural_delta(model, "Dime", "Cover 2 Man") < 0
    assert v2.structural_delta(model, "Nickel Over", "Cover 3 Sky") > 0
    assert abs(v2.structural_delta(model, "Dime", "Cover 2 Man")) <= 0.07
