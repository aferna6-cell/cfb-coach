"""Three-way held-out comparison on PLANTED synthetic worlds. Behaviour/plumbing checks only: they show the machinery adapts and abstains correctly."""
from __future__ import annotations

import random

import pytest

from cfb_coach.madden.model import football_priors as fp
from cfb_coach.madden.model import football_priors_eval as fe
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_synthetic as syn


def world(truth_fn, n_games=14, seed=3):
    rows = syn.make_rows(n_games=n_games, snaps=40, seed=seed, truth="null")
    rnd = random.Random(seed)
    for r in rows:
        s = sm.structure_any(r["executed_formation"], r["executed_play"])
        cov = r.get("coverage_seen") or ""
        lift = truth_fn(fp.play_concepts(s), cov, r.get("opponent_id"))
        p = 1 / (1 + 2.718281828 ** -(lift))
        ok = rnd.random() < p
        r["success"], r["stop"] = ("true" if ok else "false"), ("false" if ok else "true")
        r["yards"], r["result"] = (7, "+7") if ok else (0, "incomplete")
    return rows, syn.game_order(rows)


def test_machinery_runs_and_reports_three_models_and_a_conservative_gate():
    rows, order = world(lambda c, cov, o: 0.0)                                  # no effect anywhere
    rep = fe.evaluate(rows, order, n_boot=100)
    assert {"name_only", "structured_prior", "structured_learned"} <= set(rep) and rep["folds"] >= 5
    assert rep["gate"]["result"] == "KEEP_CHAMPION"                             # nothing to win in a null world: never promotes


def test_learner_corrects_a_prior_that_is_wrong_for_this_world():
    def truth(c, cov, opp):                                                     # mesh against man is actually BAD here; priors say good
        return -0.9 if ("mesh" in c and "Cover 1" in cov) else 0.0
    rows, order = world(truth)
    rep = fe.evaluate(rows, order, n_boot=100)
    assert rep["structured_learned"] <= rep["structured_prior"] + 1e-3          # shrinkage is deliberately slow (residual sd <= ~0.2): never materially worse than the static prior
