"""Chronological comparison of Sprint 12 and Sprint 13 policies.

Snap N receives only earlier verified snaps. These scenarios are synthetic.
They are not a win-rate result and not a held-out study of a real season.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.offense_joint_decision import choose_joint_action
from cfb_coach.madden.model.opponent_learning import LEARNING_CAP, fit_opponent_model

EVAL_VERSION = "opponent_learning_eval.v1"


def _sit(**kwargs: Any) -> SimpleNamespace:
    base = dict(
        down=3, distance=8, yardline=40, red_zone=False, goal_line=False,
        two_minute=False, score_us=14, score_them=14, coverage_hint=None,
        coverage_source="none", extras={}, quarter=2,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _record(seq: int, *, pressure: bool, concept: str = "mesh", yards: int = 4,
            down: int = 3, distance: int = 8, probability: float = 0.55) -> dict[str, Any]:
    success = yards >= {1: 0.4, 2: 0.6, 3: 1.0, 4: 1.0}[down] * distance
    return {
        "snap_seq": seq,
        "down": down,
        "distance": distance,
        "yards": yards,
        "formation": "Gun Bunch",
        "play": "Mesh" if concept == "mesh" else concept,
        "concept_id": concept,
        "observed_defense": "blitz" if pressure else "no pressure",
        "observation": {
            "coverage_shell": None,
            "pressure": pressure,
            "blitz": pressure,
            "legacy_ambiguous": False,
        },
        "verified_execution": True,
        "recommendation_only": False,
        "prelabeled": True,
        "success": success,
        "conversion": yards >= distance,
        "turnover": False,
        "sack": False,
        "model_probability": probability,
        "source": "synthetic",
    }


def _changing_game() -> list[dict[str, Any]]:
    rows = []
    seq = 1
    for _ in range(8):
        rows.append(_record(seq, pressure=False, yards=11))
        seq += 1
    for _ in range(8):
        rows.append(_record(seq, pressure=True, yards=2))
        seq += 1
    for _ in range(8):
        rows.append(_record(seq, pressure=False, yards=9))
        seq += 1
    return rows


def _policies() -> dict[str, dict[str, bool]]:
    return {
        "sprint12_baseline": {
            "use_knowledge": True, "use_strategy": True, "use_opponent_learning": False,
        },
        "sprint13_opponent_learning": {
            "use_knowledge": True, "use_strategy": False, "use_opponent_learning": True,
        },
        "sprint13_learning_and_strategy": {
            "use_knowledge": True, "use_strategy": True, "use_opponent_learning": True,
        },
    }


def _choose(flags: Mapping[str, bool], memory: Mapping[str, Any], sit: Any) -> dict[str, Any]:
    book = {"Gun Bunch": ["Mesh", "Power"]}
    ranked = [
        {"formation": "Gun Bunch", "play": play, "probability": 0.55, "selection_score": 0.55}
        for play in book["Gun Bunch"]
    ]
    return choose_joint_action(
        ranked=ranked, anchor=ranked[0], sit=sit, book=book, memory=memory,
        active=[], db=None, session_id="synthetic", snap_seq=900,
        **dict(flags),
    )


def evaluate_opponent_learning() -> dict[str, Any]:
    """Replay a synthetic tendency shift. Report coverage, not wins."""
    game = _changing_game()
    leakage_failures = 0
    samples = []
    for index, _row in enumerate(game):
        prior = game[:index]
        model = fit_opponent_model(records=prior)
        if model["usable_verified_snaps"] != index:
            leakage_failures += 1
        if any(record.get("snap_seq") == game[index]["snap_seq"] for record in prior):
            leakage_failures += 1
        samples.append(model["usable_verified_snaps"])
    early = fit_opponent_model(records=game[:3])
    mid = fit_opponent_model(records=game[:16])
    late = fit_opponent_model(records=game)
    sit = _sit(coverage_hint=None, coverage_source="none")
    calls = {}
    for name, flags in _policies().items():
        decision = _choose(flags, {"verified_snap_records": game[:16]}, sit)
        winner = (decision.get("football_intelligence") or {}).get("winner") or {}
        calls[name] = {
            "play": (decision.get("joint") or {}).get("play"),
            "learning_delta": winner.get("learning_delta"),
            "strategy_hypothesis": (
                (decision.get("football_intelligence") or {}).get("strategy") or {}
            ).get("hypothesis_id"),
        }
    labeled = [row for row in game if row.get("model_probability") is not None and row.get("success") is not None]
    brier = None
    if len(labeled) >= 8:
        brier = round(sum(
            (float(row["model_probability"]) - (1.0 if row["success"] else 0.0)) ** 2
            for row in labeled
        ) / len(labeled), 4)
    return {
        "version": EVAL_VERSION,
        "synthetic": True,
        "held_out_evaluation": "insufficient",
        "win_rate_claim": False,
        "confirmed_complete_historical_games_in_workspace": 0,
        "reason": (
            "One user-reported CPU game is not in this workspace and would not "
            "be a held-out season. These comparisons are synthetic chronological replays."
        ),
        "temporal_leakage_failures": leakage_failures,
        "sample_sizes_before_each_snap": samples,
        "sparse_three_snaps_published_change": bool(early.get("changes")),
        "after_16_snaps": {
            "changes": mid.get("changes"),
            "hypothesis": (mid.get("strategy_hypothesis") or {}).get("id"),
        },
        "after_reversal": {
            "changes": [row.get("metric") for row in late.get("changes") or []],
            "hypothesis": (late.get("strategy_hypothesis") or {}).get("id"),
        },
        "policy_calls_at_snap_16": calls,
        "learning_cap": LEARNING_CAP,
        "calibration": {
            "brier": brier,
            "n": len(labeled),
            "note": "Synthetic labels with a constant 0.55 prediction. Not a season calibration.",
        },
        "evidence_coverage": {
            "snaps": len(game),
            "pressure_labeled": len(game),
            "ambiguous_labels": 0,
        },
        "learned_from_video": False,
    }


def replay_prior_only(snaps: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    """Public helper so tests can prove snap N cannot see itself."""
    return fit_opponent_model(records=list(snaps[:index]))
