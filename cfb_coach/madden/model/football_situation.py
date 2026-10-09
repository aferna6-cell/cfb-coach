"""Explicit, testable football situation knowledge for Madden offense.

This module contains no model calls and no post-snap access for the current
decision. It turns known game state into labeled proxy components. The proxy
is intentionally not presented as expected points or win probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from cfb_coach.madden.catalog import is_deep, is_run
from cfb_coach.madden.model.experimental_model import _play_concept, _play_family


KNOWLEDGE_VERSION = "madden.football_situation.v1"


@dataclass(frozen=True)
class FootballSituation:
    down: int | None
    distance: int | None
    yardline: int | None
    yards_to_goal: int | None
    field_zone: str
    goal_to_go: bool | None
    score_differential: int | None
    quarter: int | None
    clock_seconds: int | None
    timeouts_us: int | None
    timeouts_them: int | None
    possession_objective: str
    clock_objective: str
    conversion_objective: str
    coverage_observation: str | None
    coverage_observation_source: str
    coverage_is_current: bool
    tendency_coverage: str | None
    tendency_confidence: float | None
    tendency_sample_size: int
    uncertainty: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extra(sit: Any, *keys: str) -> Any:
    extras = getattr(sit, "extras", None) or {}
    for key in keys:
        if key in extras and extras[key] is not None:
            return extras[key]
    return None


def evaluate_situation(
    sit: Any,
    *,
    opponent_evidence: Mapping[str, Any] | None = None,
) -> FootballSituation:
    """Describe only facts known before this snap and labeled historical priors."""
    down = _integer(getattr(sit, "down", None))
    distance = _integer(getattr(sit, "distance", None))
    yardline = _integer(getattr(sit, "yardline", None))
    quarter = _integer(getattr(sit, "quarter", None) or _extra(sit, "quarter"))
    clock = _integer(
        getattr(sit, "clock_seconds", None)
        or _extra(sit, "clock_seconds", "quarter_clock_seconds")
    )
    timeouts_us = _integer(
        getattr(sit, "timeouts_us", None) or _extra(sit, "timeouts_us")
    )
    timeouts_them = _integer(
        getattr(sit, "timeouts_them", None) or _extra(sit, "timeouts_them")
    )
    score_us = _integer(getattr(sit, "score_us", None))
    score_them = _integer(getattr(sit, "score_them", None))
    score_diff = (
        score_us - score_them
        if score_us is not None and score_them is not None
        else None
    )
    yards_to_goal = 100 - yardline if yardline is not None else None
    if bool(getattr(sit, "goal_line", False)) or (
        yards_to_goal is not None and yards_to_goal <= 5
    ):
        field_zone = "goal_line"
    elif bool(getattr(sit, "red_zone", False)) or (
        yards_to_goal is not None and yards_to_goal <= 20
    ):
        field_zone = "red_zone"
    elif yardline is not None and yardline <= 10:
        field_zone = "backed_up"
    elif yardline is None:
        field_zone = "unknown"
    else:
        field_zone = "open_field"

    late = bool(getattr(sit, "two_minute", False)) or (
        quarter is not None and quarter >= 4 and clock is not None and clock <= 150
    )
    if late and score_diff is not None and score_diff > 0:
        possession_objective = "protect_lead"
        clock_objective = "consume_clock"
    elif late and score_diff is not None and score_diff < 0:
        possession_objective = "chase_score"
        clock_objective = "preserve_clock"
    else:
        possession_objective = "neutral"
        clock_objective = "normal"

    if down in (3, 4):
        conversion_objective = (
            "short_conversion" if distance is not None and distance <= 2
            else "long_conversion" if distance is not None and distance >= 7
            else "conversion"
        )
    elif down == 2 and distance is not None and distance <= 2:
        conversion_objective = "ahead_of_sticks"
    else:
        conversion_objective = "normal_down"

    coverage = getattr(sit, "coverage_hint", None)
    coverage_source = str(getattr(sit, "coverage_source", None) or "none")
    coverage_current = bool(coverage) and coverage_source == "live"
    evidence = dict(opponent_evidence or {})
    tendency_n = max(0, _integer(evidence.get("sample_size")) or 0)
    tendency_conf = evidence.get("confidence")
    try:
        tendency_conf = (
            max(0.0, min(1.0, float(tendency_conf)))
            if tendency_conf is not None else None
        )
    except (TypeError, ValueError):
        tendency_conf = None

    unknown: list[str] = []
    if down is None or distance is None:
        unknown.append("down_distance")
    if yardline is None:
        unknown.append("field_position")
    if score_diff is None:
        unknown.append("score")
    if quarter is None or clock is None:
        unknown.append("clock")
    if not coverage_current:
        unknown.append("current_coverage")
    return FootballSituation(
        down=down,
        distance=distance,
        yardline=yardline,
        yards_to_goal=yards_to_goal,
        field_zone=field_zone,
        goal_to_go=(
            True if bool(getattr(sit, "goal_line", False))
            else None if distance is None or yards_to_goal is None
            else distance >= yards_to_goal
        ),
        score_differential=score_diff,
        quarter=quarter,
        clock_seconds=clock,
        timeouts_us=timeouts_us,
        timeouts_them=timeouts_them,
        possession_objective=possession_objective,
        clock_objective=clock_objective,
        conversion_objective=conversion_objective,
        coverage_observation=str(coverage) if coverage else None,
        coverage_observation_source=coverage_source,
        coverage_is_current=coverage_current,
        tendency_coverage=(
            str(evidence.get("dominant_coverage"))
            if evidence.get("dominant_coverage") else None
        ),
        tendency_confidence=tendency_conf,
        tendency_sample_size=tendency_n,
        uncertainty=tuple(unknown),
    )


def score_play_suitability(play: str, state: FootballSituation) -> dict[str, Any]:
    """Return an interpretable situation proxy; never claim causal lift."""
    name = play.lower()
    run = is_run(play)
    family = _play_family(play)
    concept = _play_concept(play)
    screen = family == "screen"
    quick = any(token in name for token in ("slant", "mesh", "stick", "spacing", "flat"))
    sideline = any(token in name for token in ("out", "corner", "flood", "sail"))
    deep = is_deep(play) or any(token in name for token in ("vert", "post", "fade", "wheel"))
    components: list[dict[str, Any]] = []

    def add(name_: str, value: float, reason: str) -> None:
        if value:
            components.append(
                {"name": name_, "value": round(value, 5), "reason": reason}
            )

    d, yards = state.down, state.distance
    if d in (3, 4) and yards is not None and yards >= 7:
        add("conversion_distance", -0.48 if run else 0.0,
            "late-down long yardage requires a realistic first-down path")
        add("screen_sticks_risk", -0.10 if screen else 0.0,
            "screen starts behind a long conversion marker")
        add("route_depth", 0.04 if deep and not screen else 0.0,
            "route family can threaten the conversion line")
        add("quick_pressure_answer", 0.02 if quick else 0.0,
            "quick-game spacing can preserve a pressure answer")
    elif d in (3, 4) and yards is not None and yards <= 2:
        add("short_conversion", 0.045 if run or quick else 0.0,
            "efficient short-yardage answer")
        add("development_cost", -0.045 if deep else 0.0,
            "long development is unnecessary for short conversion")
    elif d == 2 and yards is not None and yards <= 2:
        add("ahead_of_sticks_efficiency", 0.025 if run or quick else 0.0,
            "keeps a high-probability conversion available")
        add("ahead_of_sticks_upside", 0.025 if deep else 0.0,
            "second-and-short permits a selective higher-upside attempt")

    if state.field_zone == "goal_line":
        add("compressed_space", -0.055 if deep else 0.0,
            "goal-line spacing reduces room for long-developing routes")
        add("goal_line_fit", 0.04 if run or quick else 0.0,
            "run or quick-spacing family fits compressed field")
    elif state.field_zone == "red_zone":
        add("red_zone_spacing", -0.025 if deep else 0.0,
            "reduced vertical space lowers generic deep-route value")
        add("red_zone_fit", 0.025 if run or quick else 0.0,
            "run or quick-spacing family addresses reduced space")
    elif state.field_zone == "backed_up":
        add("backed_up_security", 0.025 if run or quick else 0.0,
            "limits development and possession risk near our goal")
        add("backed_up_development", -0.035 if deep else 0.0,
            "deep development carries extra backed-up risk")

    if state.clock_objective == "preserve_clock":
        add("clock_preservation", -0.09 if run and (yards or 0) >= 4 else 0.0,
            "trailing late values clock and timeout preservation")
        add("sideline_access", 0.045 if sideline else 0.0,
            "sideline route family can preserve clock when completed")
    elif state.clock_objective == "consume_clock":
        add("clock_consumption", 0.055 if run else 0.0,
            "late lead values a running clock and possession")
        add("lead_possession_risk", -0.03 if deep else 0.0,
            "late lead discounts lower-control deep attempts")

    if state.coverage_is_current:
        cov = (state.coverage_observation or "").lower()
        pressure = any(x in cov for x in ("pressure", "blitz", "zero"))
        man = "man" in cov or "cover 1" in cov or "cover 0" in cov
        zone = any(x in cov for x in ("cover 2", "cover 3", "cover 4", "zone"))
        add("observed_pressure_quick_game", 0.045 if pressure and quick else 0.0,
            "credible current pressure observation favors quick separation")
        add("observed_pressure_development", -0.05 if pressure and deep else 0.0,
            "credible current pressure raises long-development cost")
        add("observed_man_spacing", 0.025 if man and concept in ("mesh", "slant", "cross") else 0.0,
            "current man observation fits crossing/separation family")
        add("observed_zone_spacing", 0.02 if zone and concept in ("flood", "smash", "spacing") else 0.0,
            "current zone observation fits horizontal/vertical stretch family")
    elif (
        state.tendency_coverage
        and state.tendency_confidence is not None
        and state.tendency_sample_size >= 4
    ):
        # Historical tendencies inform a small prior, never a claim about now.
        tendency = state.tendency_coverage.lower()
        strength = min(0.015, 0.015 * state.tendency_confidence)
        pressure_tendency = any(
            token in tendency for token in ("pressure", "blitz", "zero")
        )
        add(
            "historical_pressure_quick_game",
            strength if pressure_tendency and quick else 0.0,
            "repeated pressure tendency modestly favors quick answers; current look unknown",
        )
        add(
            "historical_pressure_development",
            -strength if pressure_tendency and deep else 0.0,
            "repeated pressure tendency discounts long development; current look unknown",
        )
        if not pressure_tendency:
            add(
                "historical_tendency_prior",
                strength,
                f"historical {state.tendency_coverage} tendency; current coverage remains unknown",
            )

    total = sum(float(c["value"]) for c in components)
    return {
        "knowledge_version": KNOWLEDGE_VERSION,
        "proxy_score": round(total, 5),
        "proxy_label": "situation_aware_football_proxy_not_ep_or_win_probability",
        "components": components,
        "concept": concept,
        "family": family,
        "reasons": [str(c["reason"]) for c in components] or ["normal-down neutral fit"],
    }


def game_context_from_database(
    db: Any,
    *,
    session_id: str | None,
    opponent_id: str,
    limit: int = 40,
) -> dict[str, Any]:
    """Summarize completed prior snaps only; never read a future/current outcome."""
    if db is None or not session_id:
        return {
            "sample_size": 0, "confidence": 0.0, "dominant_coverage": None,
            "coverage_counts": {}, "pressure_rate": None, "source": "no_game_history",
        }
    try:
        rows = db.conn.execute(
            "SELECT coverage_seen FROM snaps "
            "WHERE session_id=? AND opponent_id=? AND coverage_seen IS NOT NULL "
            "AND TRIM(coverage_seen)<>'' ORDER BY id DESC LIMIT ?",
            (session_id, opponent_id, limit),
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    counts: dict[str, int] = {}
    for row in rows:
        coverage = str(row["coverage_seen"] or "").strip()
        if coverage:
            counts[coverage] = counts.get(coverage, 0) + 1
    n = sum(counts.values())
    dominant = max(counts, key=lambda k: (counts[k], k)) if counts else None
    pressure_n = sum(
        count for coverage, count in counts.items()
        if any(x in coverage.lower() for x in ("pressure", "blitz", "zero"))
    )
    return {
        "sample_size": n,
        "confidence": round(n / (n + 8.0), 4) if n else 0.0,
        "dominant_coverage": dominant,
        "coverage_counts": counts,
        "pressure_rate": round(pressure_n / n, 4) if n else None,
        "source": "completed_prior_snaps_in_current_session",
        "current_coverage_known": False,
    }


__all__ = [
    "FootballSituation",
    "KNOWLEDGE_VERSION",
    "evaluate_situation",
    "game_context_from_database",
    "score_play_suitability",
]
