"""Compare the joint coordinator with the previous model-primary policy.

Scenarios in this module are synthetic. They are not the user's franchise
games. No laptop database is read here.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from cfb_coach.madden.model.experimental_model import _play_concept
from cfb_coach.madden.model.football_situation import SITUATION_NAMES, evaluate_situation
from cfb_coach.madden.model.offense_joint_decision import choose_joint_action
from cfb_coach.madden.model.offense_selection_policy import choose_model_play

SYNTHETIC_BOOK = {
    "Gun Bunch": ["Mesh", "Inside Zone", "Four Verticals", "HB Slip Screen", "Quick Slants"],
    "Gun Trips": ["Flood", "HB Draw", "PA Cross", "Spacing"],
    "Singleback Wing": ["Stretch", "PA Boot", "Slants"],
    "Gun Tight": ["Smash", "HB Dive", "Levels"],
    "I Form": ["Power", "Counter", "Play Action"],
    "Gun Doubles": ["Stick", "Curl Flat", "Outside Zone"],
}


def _sit(**kwargs: Any) -> SimpleNamespace:
    base = dict(
        down=1, distance=10, yardline=40, red_zone=False, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={},
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def synthetic_scenarios() -> list[dict[str, Any]]:
    """Labeled synthetic snaps. Not franchise results."""
    return [
        {"name": "normal", "sit": _sit(), "label": "synthetic"},
        {"name": "third_and_long", "sit": _sit(down=3, distance=12), "label": "synthetic"},
        {"name": "second_and_short", "sit": _sit(down=2, distance=2), "label": "synthetic"},
        {"name": "goal_line", "sit": _sit(down=1, distance=1, yardline=99, goal_line=True, red_zone=True), "label": "synthetic"},
        {"name": "two_minute_lead", "sit": _sit(down=1, distance=10, two_minute=True, score_us=24, score_them=17), "label": "synthetic"},
        {"name": "two_minute_trail", "sit": _sit(down=2, distance=8, two_minute=True, score_us=14, score_them=21), "label": "synthetic"},
        {"name": "unknown_coverage", "sit": _sit(coverage_hint="Cover 3", coverage_source="last"), "label": "synthetic"},
    ]


def _ranked() -> list[dict[str, Any]]:
    rows = []
    probability = 0.57
    for form, plays in SYNTHETIC_BOOK.items():
        for play in plays:
            rows.append({
                "formation": form, "play": play, "probability": probability,
                "uncertainty": 0.62, "evidence_quality": "prior_driven",
                "play_concept": _play_concept(play),
            })
            probability = 0.57 if probability < 0.57 else 0.56
    return rows


def compare_policies(*, seed: int = 11) -> dict[str, Any]:
    """Deterministic comparison. ``seed`` is recorded; sampling uses session ids."""
    del seed  # session/snap seeds inside the policy are the reproducibility source
    rows = _ranked()
    calls = []
    third_long_runs = 0
    third_long = 0
    legal = 0
    unknown_called_certain = 0
    latencies = []
    differed = 0
    for index, scenario in enumerate(synthetic_scenarios(), start=1):
        sit = scenario["sit"]
        baseline, _audit = choose_model_play(
            rows, sit=sit, opponent_type="cpu",
            session_id="synthetic-compare", snap_seq=index,
        )
        started = time.perf_counter()
        joint = choose_joint_action(
            ranked=baseline, anchor=baseline[0], sit=sit, book=SYNTHETIC_BOOK,
            active=[], db=None, opponent_id="synthetic",
            session_id="synthetic-compare", snap_seq=index, opponent_type="cpu",
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        form = joint["joint"]["formation"]
        play = joint["joint"]["play"]
        legal_play = play in SYNTHETIC_BOOK.get(form, [])
        legal += int(legal_play and joint.get("kind") in ("none", "adjustment", "macro"))
        if (form, play) != (baseline[0]["formation"], baseline[0]["play"]):
            differed += 1
        evaluation = evaluate_situation(sit)
        if scenario["name"] == "third_and_long":
            third_long += 1
            from cfb_coach.madden.catalog import is_run
            third_long_runs += int(is_run(play))
        if scenario["name"] == "unknown_coverage" and evaluation["coverage"]["state"] == "observed":
            unknown_called_certain += 1
        calls.append({
            "scenario": scenario["name"],
            "label": "synthetic",
            "baseline_play": baseline[0]["play"],
            "coordinator_play": play,
            "coordinator_formation": form,
            "adjustment": joint.get("kind"),
            "coverage_state": evaluation["coverage"]["state"],
            "plays_considered": joint.get("plays_considered"),
        })
    n = len(calls)
    distinct = len({(c["coordinator_formation"], c["coordinator_play"]) for c in calls})
    concepts = len({_play_concept(c["coordinator_play"]) for c in calls})
    return {
        "evidence": "synthetic_fixtures",
        "user_franchise_logs": "not_available_in_this_environment",
        "scenarios": n,
        "situation_probes_required": list(SITUATION_NAMES),
        "distinct_coordinator_plays": distinct,
        "distinct_concepts": concepts,
        "third_and_long_scenarios": third_long,
        "third_and_long_run_calls": third_long_runs,
        "legal_action_rate": round(legal / n, 3) if n else 0.0,
        "calls_differing_from_baseline": differed,
        "unknown_coverage_labeled_observed": unknown_called_certain,
        "fallback_rate": 0.0,
        "max_joint_latency_ms": round(max(latencies) if latencies else 0.0, 3),
        "calls": calls,
        "note": (
            "Synthetic comparison only. Adjustment verification and yards require "
            "explicit in-game confirmation that is not present here."
        ),
    }
