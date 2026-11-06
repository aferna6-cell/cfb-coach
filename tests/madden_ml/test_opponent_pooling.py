"""League -> style -> individual run/pass prior and its sequential evaluation."""

from __future__ import annotations

import random

import pytest

from cfb_coach.madden.model import heldout_eval as he
from cfb_coach.madden.model import opponent_tendencies as ot


def _obs(opp: str, game: str, passes: int, runs: int, down: int = 1, distance: int = 10) -> list[ot.RunPassObservation]:
    plays = [1] * passes + [0] * runs
    random.Random(f"{opp}{game}").shuffle(plays)
    return [ot.RunPassObservation(opp, game, down, distance, y) for y in plays]


def _league() -> list[ot.RunPassObservation]:
    out = []
    for i, (p, r) in enumerate([(24, 6), (25, 5), (23, 7), (6, 24), (5, 25), (7, 23)]):
        out += _obs(f"o{i}", f"g{i}", p, r)
    return out


def test_new_opponent_gets_the_league_prior_with_zero_confidence() -> None:
    prior = ot.RunPassPrior().fit(_league())
    est = ot.estimate_run_pass(prior, [], 1, 10, use_styles=False)
    assert est["p_pass"] == pytest.approx(prior.league_rate(1, 10))
    assert est["confidence"] == 0.0
    assert est["source"] == "league"


def test_individual_evidence_moves_the_estimate_gradually() -> None:
    prior = ot.RunPassPrior().fit(_league())
    base = prior.league_rate(1, 10)
    few = ot.estimate_run_pass(prior, _obs("new", "x", 3, 0), 1, 10, use_styles=False)
    many = ot.estimate_run_pass(prior, _obs("new", "x", 30, 0), 1, 10, use_styles=False)
    assert base < few["p_pass"] < many["p_pass"] < 1.0
    assert few["confidence"] < many["confidence"]
    stiff = ot.estimate_run_pass(prior, _obs("new", "x", 3, 0), 1, 10, use_styles=False, handoff=1e6)
    assert stiff["p_pass"] == pytest.approx(base, abs=1e-4)


def test_styles_are_learned_from_evidence_and_identify_a_passer() -> None:
    prior = ot.RunPassPrior().fit(_league())
    assert len(prior.styles) == 2
    passer = max(prior.styles, key=lambda s: s["offset"])
    assert passer["opponents"] == 3
    est = ot.estimate_run_pass(prior, _obs("new", "x", 8, 1), 1, 10, use_individual=False)
    weights = dict(zip([s["offset"] for s in prior.styles], est["style_weights"]))
    assert weights[passer["offset"]] > 0.9


def test_styles_need_enough_qualifying_opponents() -> None:
    thin = _obs("a", "g0", 20, 5) + _obs("b", "g1", 5, 20) + _obs("c", "g2", 2, 2)
    assert ot.RunPassPrior().fit(thin).styles == []


def _rows(game: str, opp: str, plays: list[str]) -> list[dict]:
    return [
        {"side": "defense", "game_id": game, "opponent_id": opp, "down": 1, "distance": 10,
         "concept_seen": c, "executed_status": "unknown", "executed_verification": "unknown"}
        for c in plays
    ]


def test_sequential_eval_uses_only_earlier_snaps_and_unverified_defense_rows() -> None:
    rows = _rows("g0", "ann", ["Mesh"] * 25 + ["Inside Zone"] * 20) + _rows("g1", "bob", ["Inside Zone"] * 6 + ["Mesh"] * 6)
    order = {"g0": 0, "g1": 1}
    preds, folds = he.opponent_sequential_eval(rows, order, min_train_obs=10)
    assert [f["game"] for f in folds] == ["g1"]
    assert [p["n_history"] for p in preds] == list(range(12))
    changed = rows[:45] + _rows("g1", "bob", ["Inside Zone"] * 6 + ["Inside Zone"] * 6)
    preds2, _ = he.opponent_sequential_eval(changed, order, min_train_obs=10)
    for a, b in zip(preds[:7], preds2[:7]):
        assert a["p"] == b["p"]


def test_leave_one_opponent_out_never_trains_on_the_test_opponent() -> None:
    rows = (_rows("g0", "ann", ["Mesh"] * 30) + _rows("g1", "bob", ["Inside Zone"] * 30)
            + _rows("g2", "ann", ["Mesh"] * 30))
    order = {"g0": 0, "g1": 1, "g2": 2}
    _, folds = he.opponent_sequential_eval(rows, order, scheme="leave_one_opponent_out", min_train_obs=10)
    by_game = {f["game"]: f["train_plays"] for f in folds}
    assert by_game == {"g0": 30, "g1": 60, "g2": 30}
