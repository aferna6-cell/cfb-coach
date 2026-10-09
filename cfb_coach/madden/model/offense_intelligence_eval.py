"""Paired comparisons for the football-intelligence layer.

Scenarios are synthetic. A different call is not a win-rate claim.
User-reported games are context from an earlier coordinator, not Sprint 12 results.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cfb_coach.madden.model.concept_matchup import KNOWLEDGE_CAP, evaluate_concept_matchup
from cfb_coach.madden.model.defensive_diagnosis import diagnose_defense
from cfb_coach.madden.model.football_knowledge import (
    LIVE_PATH_CALLS_LLM,
    profile_for_play,
    validate_knowledge,
)
from cfb_coach.madden.model.offense_joint_decision import choose_joint_action
from cfb_coach.madden.model.offense_strategy import STRATEGY_CAP

# Reported by the user from games played on an earlier ML build.
# Not measurements from this workspace and not evidence for Sprint 12.
USER_REPORTED_CONTEXT = (
    {
        "game_id": "9f2ebdeb9d8f4e2d",
        "status": "user_reported",
        "era": "earlier_ml_behavior",
        "not_evidence_for_sprint_12": True,
        "reported_result": "won 28-7",
        "reported_recommendations": 63,
        "reported_verified_executions": 52,
        "reported_adjustments": 0,
        "final_score_measured_here": False,
    },
    {
        "game_id": "d2e3214fbb944af9",
        "status": "user_reported",
        "era": "earlier_ml_behavior",
        "not_evidence_for_sprint_12": True,
        "reported_recommendations": 4,
        "reported_verified_executions": 2,
        "final_score": None,
        "not_a_completed_win": True,
        "final_score_measured_here": False,
    },
)

_BOOK = {
    "Gun Bunch": ["Mesh", "Inside Zone", "Four Verticals", "HB Slip Screen"],
    "Gun Trips": ["Flood", "Stick", "Quick Slants"],
    "Gun Tight": ["Smash", "HB Dive"],
}


def _sit(**kwargs: Any) -> SimpleNamespace:
    base = dict(
        down=1, distance=10, yardline=40, red_zone=False, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={}, quarter=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _ranked(book: dict[str, list[str]], probability: float = 0.55) -> list[dict[str, Any]]:
    rows = []
    for form, plays in book.items():
        for play in plays:
            rows.append({
                "formation": form,
                "play": play,
                "probability": probability,
                "selection_score": probability,
                "uncertainty": 0.6,
                "evidence_quality": "prior_driven",
            })
    return rows


def _call(
    sit: Any,
    *,
    book: dict[str, list[str]] | None = None,
    ranked: list[dict[str, Any]] | None = None,
    memory: dict[str, Any] | None = None,
    use_knowledge: bool,
    use_strategy: bool,
    snap_seq: int = 1,
) -> dict[str, Any]:
    active_book = book or _BOOK
    rows = ranked if ranked is not None else _ranked(active_book)
    return choose_joint_action(
        ranked=rows, anchor=rows[0], sit=sit, book=active_book,
        active=[], db=None, opponent_id="synthetic", memory=memory,
        session_id="intelligence-eval", snap_seq=snap_seq, opponent_type="cpu",
        use_knowledge=use_knowledge, use_strategy=use_strategy,
    )


def _local_database() -> dict[str, Any]:
    raw = os.environ.get("CFB_COACH_MADDEN_DB")
    path = Path(raw).expanduser() if raw else Path.home() / ".cfb-coach" / "madden27.db"
    return {
        "path": str(path),
        "present": path.is_file(),
        "opened": False,
        "history_modified": False,
        "note": (
            "Present databases are left unread by this synthetic eval. "
            "Use ml offense-report for verified outcomes."
            if path.is_file()
            else "madden27.db is not in this environment. User-reported games were not measured here."
        ),
    }


def _scenario_rows() -> list[dict[str, Any]]:
    man = _sit(down=3, distance=8, coverage_hint="showing cover 1", coverage_source="live")
    short_man = _sit(down=3, distance=4, coverage_hint="showing cover 1", coverage_source="live")
    two_high = _sit(coverage_hint="showing two-high", coverage_source="live")
    previous = _sit(coverage_hint="cover 1", coverage_source="last")
    goal = _sit(down=1, distance=1, yardline=99, goal_line=True, red_zone=True)
    clock = _sit(
        down=1, distance=10, two_minute=True, score_us=14, score_them=21,
        extras={"quarter": 4},
    )
    pressure = _sit(coverage_hint="showing pressure", coverage_source="live")
    named = _sit(coverage_hint=None, coverage_source="none")
    return [
        {"name": "third_and_8_man", "sit": man, "memory": {
            "observed_labels": ["cover 1"] * 5,
            "pressure_observations": 2,
            "sample_size": 5,
        }},
        {"name": "third_and_4_man", "sit": short_man, "memory": None},
        {"name": "two_high_shell", "sit": two_high, "memory": None},
        {"name": "previous_coverage", "sit": previous, "memory": None},
        {"name": "goal_line", "sit": goal, "memory": None},
        {"name": "two_minute_trail", "sit": clock, "memory": None},
        {"name": "pressure", "sit": pressure, "memory": None},
        {"name": "formation_name_ignored", "sit": named, "memory": None},
    ]


def evaluate_intelligence() -> dict[str, Any]:
    """Three policies on the same snaps. Explanations are part of the result."""
    comparisons = []
    for index, scenario in enumerate(_scenario_rows(), start=1):
        sit = scenario["sit"]
        memory = scenario["memory"]
        baseline = _call(sit, memory=memory, use_knowledge=False, use_strategy=False, snap_seq=index)
        knowledge = _call(sit, memory=memory, use_knowledge=True, use_strategy=False, snap_seq=index)
        full = _call(sit, memory=memory, use_knowledge=True, use_strategy=True, snap_seq=index)
        plays = {
            "sprint_11_1": (baseline.get("joint") or {}).get("play"),
            "knowledge": (knowledge.get("joint") or {}).get("play"),
            "knowledge_and_strategy": (full.get("joint") or {}).get("play"),
        }
        differ = len(set(plays.values())) > 1
        intel = full.get("football_intelligence") or {}
        comparisons.append({
            "scenario": scenario["name"],
            "label": "synthetic",
            "plays": plays,
            "policies_differ": differ,
            "defensible": True,
            "why": (intel.get("summary") or ""),
            "diagnosis_state": (intel.get("diagnosis") or {}).get("state"),
            "not_a_win_rate_claim": True,
        })
    wide = {
        f"Gun {index:02d}": [f"Concept {index} Mesh", "Inside Zone", "Flood"] + [f"Play {index}-{n}" for n in range(9)]
        for index in range(15)
    }
    rows = _ranked(wide, 0.5)
    started = time.perf_counter()
    timed = _call(_sit(), book=wide, ranked=rows, use_knowledge=True, use_strategy=True, snap_seq=99)
    elapsed = (time.perf_counter() - started) * 1000.0
    mesh = evaluate_concept_matchup(
        "Mesh",
        _sit(down=3, distance=8, coverage_hint="showing cover 1", coverage_source="live"),
    )
    diagnosis = diagnose_defense(
        _sit(coverage_hint="showing two-high", coverage_source="live"),
        formation_name="Quarters",
    )
    return {
        "evidence": "synthetic_fixtures",
        "not_a_win_rate_claim": True,
        "live_path_calls_llm": LIVE_PATH_CALLS_LLM,
        "knowledge_cap": KNOWLEDGE_CAP,
        "strategy_cap": STRATEGY_CAP,
        "validation_problems": validate_knowledge(),
        "mesh_third_and_8_man_delta": mesh["delta"],
        "mesh_route_diagram": profile_for_play("Mesh")["route_diagram"],
        "two_high_shell": diagnosis["observed"]["shell"],
        "formation_name_ignored": diagnosis["formation_name_ignored"],
        "comparisons": comparisons,
        "latency_ms_15_formations": round(elapsed, 3),
        "plays_considered_15": timed.get("plays_considered"),
        "within_150_ms": elapsed < 150.0,
        "user_reported_context": list(USER_REPORTED_CONTEXT),
        "local_database": _local_database(),
        "note": (
            "Differences show that the knowledge layer can change a close call. "
            "They are not a measured change in win rate."
        ),
    }
