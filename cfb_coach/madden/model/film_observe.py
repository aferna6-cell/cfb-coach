"""Offline visual notes. This is not a Madden coverage classifier.

Synthetic fixture frames use a documented pixel protocol so tests can check
the reader. A real recording without that marker stays unknown. Safety depth
never becomes Cover 2, Cover 3, or Quarters.
"""
from __future__ import annotations

from typing import Any, Mapping

from cfb_coach.madden.model.defensive_observation import structure_observation
from cfb_coach.madden.model.video_observation import (
    VIDEO_OBSERVATION_VERSION,
    validate_video_observation,
)

OBSERVER_VERSION = "film_observer.v1"
_MAGIC = (1, 2, 3)


def interpret_frame(
    rgb: bytes,
    *,
    width: int,
    height: int,
    game_id: str,
    recording_id: str,
    video_timestamp: float,
    snap_id: str | None = None,
    unresolved_snap_association: str | None = None,
) -> dict[str, Any]:
    cells = _cells(rgb, width, height)
    fixture = cells.get((0, 0)) == _MAGIC
    if not fixture:
        state = {"quarter": None, "clock_seconds": None, "score_us": None, "score_them": None,
                 "down": None, "distance": None, "source": "not_a_fixture_frame"}
        alignment = structure_observation(
            None, timing="unknown", source="video", confidence=0.0,
        )
        timing = "unknown"
        confidence = 0.0
        field_confidence = {}
    else:
        state, alignment, timing, confidence, field_confidence = _read_fixture(cells)
    raw = {
        "schema": VIDEO_OBSERVATION_VERSION,
        "game_id": game_id,
        "recording_id": recording_id,
        "video_timestamp": video_timestamp,
        "observation_time_relative_to_snap": timing,
        "observed_game_state": state,
        "defensive_alignment": alignment,
        "source_model": OBSERVER_VERSION,
        "source_version": OBSERVER_VERSION,
        "source_layer": "model",
        "confidence": confidence,
        "field_confidence": field_confidence,
        "human_verification": "unverified",
        "raw_model_observation": {"state": state, "alignment": alignment},
        "human_label": None,
    }
    if snap_id:
        raw["snap_id"] = snap_id
    else:
        raw["unresolved_snap_association"] = unresolved_snap_association or f"t:{video_timestamp}"
    label = alignment.get("structure_label")
    validated = validate_video_observation(raw)
    if label and isinstance(validated.get("defensive_alignment"), dict):
        validated["defensive_alignment"]["structure_label"] = label
        validated["defensive_alignment"]["coverage_shell"] = None
    validated["recognition_accuracy_claim"] = False
    validated["fixture_protocol"] = fixture
    validated["coverage_call_inferred_from_safety_depth"] = False
    return validated


def _cells(rgb: bytes, width: int, height: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    cells = {}
    if width <= 0 or height <= 0 or len(rgb) < width * height * 3:
        return cells
    for y in range(height):
        for x in range(width):
            index = (y * width + x) * 3
            cells[(x, y)] = (rgb[index], rgb[index + 1], rgb[index + 2])
    return cells


def _read_fixture(cells: Mapping[tuple[int, int], tuple[int, int, int]]) -> tuple[dict, dict, str, float, dict]:
    quarter_down = cells.get((1, 0), (0, 0, 0))
    clock = cells.get((2, 0), (0, 0, 0))
    score = cells.get((3, 0), (0, 0, 0))
    safety = cells.get((4, 0), (0, 0, 0))[0]
    pressure = cells.get((5, 0), (255, 0, 0))[0]
    outcome = cells.get((6, 0), (0, 0, 0))
    post = cells.get((7, 0), (0, 0, 0))[0] == 1
    timing = "post_snap" if post else "pre_snap"
    depth = {2: "two_high", 1: "single_high"}.get(safety)
    pressure_value = None if pressure == 255 else bool(pressure)
    alignment = structure_observation(
        None,
        timing=timing,
        source="video",
        confidence=0.7,
        fields={
            "safety_depth": depth,
            "box_count": score[2] if 5 <= score[2] <= 9 else None,
            "pressure": pressure_value,
        },
    )
    if depth == "two_high":
        alignment["coverage_shell"] = None
        alignment["structure_label"] = "two_high_safety_structure"
    state: dict[str, Any] = {
        "quarter": quarter_down[0] or None,
        "down": quarter_down[1] or None,
        "distance": quarter_down[2] or None,
        "clock_seconds": clock[1] * 256 + clock[0],
        "score_us": score[0],
        "score_them": score[1],
        "fixture": True,
    }
    if post and outcome[0] == 1:
        state["outcome"] = f"gain {outcome[1]}"
        state["yards"] = outcome[1]
    elif post:
        state["outcome"] = None
    field_confidence = {
        "quarter": 0.7, "down": 0.7, "distance": 0.7, "clock_seconds": 0.7,
        "safety_depth": 0.6 if depth else 0.0,
        "coverage_shell": 0.0,
        "box_count": 0.6 if alignment.get("box_count") else 0.0,
    }
    return state, alignment, timing, 0.7, field_confidence


def pre_snap_features(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Features a coordinator could see before the snap. Post-snap facts stay out."""
    if observation.get("observation_time_relative_to_snap") == "post_snap":
        return {
            "available_before_snap": False,
            "withheld": ["post_snap_observation"],
        }
    if observation.get("available_before_snap") is False:
        return {"available_before_snap": False, "withheld": ["marked_unavailable_before_snap"]}
    state = dict(observation.get("observed_game_state") or {})
    for key in ("outcome", "yards", "result", "success", "turnover"):
        state.pop(key, None)
    alignment = dict(observation.get("defensive_alignment") or {})
    return {
        "available_before_snap": True,
        "game_state": state,
        "coverage_shell": alignment.get("coverage_shell"),
        "safety_depth": alignment.get("safety_depth"),
        "structure_label": alignment.get("structure_label"),
        "box_count": alignment.get("box_count"),
        "pressure": alignment.get("pressure"),
        "source_layer": observation.get("source_layer"),
        "human_label_used": False,
    }
