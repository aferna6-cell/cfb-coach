"""Pre-snap football situation evaluator for the offensive coordinator.

Observations, inferred tendencies and unknowns stay separate. A previous
coverage look is not the current coverage. Strategic effects are named
situation proxies with explicit magnitudes. They are not expected points,
win probability, or hidden bonuses.

No language model runs on this path. Pregame research may call it offline;
the live call only evaluates these rules.
"""
from __future__ import annotations

from typing import Any, Mapping

from cfb_coach.madden.catalog import is_deep, is_run
from cfb_coach.madden.model.experimental_model import _play_family

# Situation families the pregame portfolio must be able to discuss.
SITUATION_NAMES: tuple[str, ...] = (
    "normal",
    "short_yardage",
    "third_and_long",
    "red_zone",
    "goal_line",
    "backed_up",
    "two_minute",
    "clock_management",
)

# Magnitudes below match the Sprint 7 selection policy exactly. Do not retune
# them silently; tests lock the signs and the two-minute lead/trail split.
SELECTION_PRIORS: dict[str, dict[str, Any]] = {
    "third_long_run": {
        "delta": -0.45,
        "reason": "third/fourth-and-long ground gain risk",
        "claim": "conversion_proxy_not_expected_points",
    },
    "third_long_screen": {
        "delta": -0.09,
        "reason": "screen behind conversion distance",
        "claim": "conversion_proxy_not_expected_points",
    },
    "third_long_reachable": {
        "delta": 0.025,
        "reason": "route potentially reaches sticks",
        "claim": "conversion_proxy_not_expected_points",
    },
    "short_yardage_run": {
        "delta": 0.025,
        "reason": "short-yardage run option",
        "claim": "conversion_proxy_not_expected_points",
    },
    "short_yardage_deep": {
        "delta": -0.035,
        "reason": "long-developing route on short yardage",
        "claim": "conversion_proxy_not_expected_points",
    },
    "two_minute_trail_run": {
        "delta": -0.07,
        "reason": "trailing in two-minute drill",
        "claim": "clock_proxy_not_win_probability",
    },
    "two_minute_lead_run": {
        "delta": 0.04,
        "reason": "protecting lead / keeping clock running",
        "claim": "clock_proxy_not_win_probability",
    },
}

# Additional coordinator priors. They are NOT folded into the legacy
# selection delta, so existing model-primary sampling stays reproducible.
COORDINATOR_PRIORS: dict[str, dict[str, Any]] = {
    "goal_line_run": {
        "delta": 0.03,
        "reason": "goal-line spacing is compressed; a run fits personnel and the short field",
        "claim": "spacing_proxy_not_expected_points",
    },
    "goal_line_deep": {
        "delta": -0.04,
        "reason": "long-developing route in compressed goal-line space",
        "claim": "spacing_proxy_not_expected_points",
    },
    "red_zone_deep": {
        "delta": -0.02,
        "reason": "red-zone field is shorter than an open-field vertical",
        "claim": "spacing_proxy_not_expected_points",
    },
    "backed_up_deep": {
        "delta": -0.03,
        "reason": "backed-up offense; a shot play risks field position",
        "claim": "field_position_proxy",
    },
    "second_short_run": {
        "delta": 0.02,
        "reason": "second-and-short can convert without abandoning a run",
        "claim": "conversion_proxy_not_expected_points",
    },
    "second_short_explosive": {
        "delta": 0.012,
        "reason": "second-and-short also leaves room for an explosive play",
        "claim": "upside_proxy_not_expected_points",
    },
    "late_lead_run": {
        "delta": 0.02,
        "reason": "late lead; a run can consume clock when the score is known",
        "claim": "clock_proxy_not_win_probability",
    },
    "late_trail_pass": {
        "delta": 0.015,
        "reason": "late deficit; value a play with a realistic gain",
        "claim": "conversion_proxy_not_expected_points",
    },
    "observed_pressure_quick": {
        "delta": 0.02,
        "reason": "credible live pressure; quick game is a supported response",
        "claim": "protection_proxy_requires_live_look",
    },
    "observed_pressure_deep": {
        "delta": -0.025,
        "reason": "credible live pressure; long-developing route needs time",
        "claim": "protection_proxy_requires_live_look",
    },
    "inferred_pressure_quick": {
        "delta": 0.008,
        "reason": "inferred pressure tendency only; not proof of this snap's coverage",
        "claim": "tendency_not_current_coverage",
    },
}


def _extra(sit: Any, key: str) -> Any:
    extras = getattr(sit, "extras", None) or {}
    if not isinstance(extras, Mapping):
        return None
    return extras.get(key)


def _num(sit: Any, name: str, extra_key: str | None = None) -> int | None:
    value = getattr(sit, name, None)
    if value is None and extra_key:
        value = _extra(sit, extra_key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def coverage_evidence(sit: Any, memory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Separate a live look from a prior snap and from a tendency.

    ``last`` coverage is an inference about the past, not an observation of
    this snap. A memory tendency is labeled inferred and never upgrades itself
    into the current shell.
    """
    from cfb_coach.madden.playcaller import coverage_class

    hint = getattr(sit, "coverage_hint", None)
    source = getattr(sit, "coverage_source", None) or "none"
    live_class = coverage_class(hint) if hint and source == "live" else None
    if live_class:
        return {
            "state": "observed",
            "value": live_class,
            "label": hint,
            "source": "live",
            "confidence": 1.0,
            "sample_size": 1,
            "note": "pre-snap look on this snap",
        }
    if hint and source == "last":
        return {
            "state": "inferred",
            "value": None,
            "label": hint,
            "source": "last",
            "confidence": 0.15,
            "sample_size": 1,
            "note": "previous snap only; not the current coverage",
        }
    mem = memory or {}
    sample = int(mem.get("sample_size") or 0)
    inferred = mem.get("inferred_look")
    if inferred and sample >= 3 and mem.get("state") == "inferred":
        return {
            "state": "inferred",
            "value": inferred,
            "label": inferred,
            "source": "tendency",
            "confidence": float(mem.get("confidence") or 0.0),
            "sample_size": sample,
            "note": "historical tendency from earlier snaps; not this snap's coverage",
        }
    return {
        "state": "unknown",
        "value": None,
        "label": None,
        "source": "none",
        "confidence": 0.0,
        "sample_size": sample,
        "note": "current coverage was not observed",
    }


def roster_evidence(roster: Mapping[str, Any] | None) -> dict[str, Any]:
    """Verified roster ratings only. Missing data stays unknown."""
    if not roster or roster.get("verified") is not True:
        return {
            "state": "unknown",
            "note": "no verified roster snapshot",
            "pass_strength": None,
            "run_strength": None,
        }
    try:
        passing = float(roster["pass_strength"])
        running = float(roster["run_strength"])
    except (KeyError, TypeError, ValueError):
        return {
            "state": "unknown",
            "note": "roster snapshot missing pass_strength or run_strength",
            "pass_strength": None,
            "run_strength": None,
        }
    return {
        "state": "verified",
        "note": str(roster.get("source") or "verified roster snapshot"),
        "pass_strength": passing,
        "run_strength": running,
    }


def _phase(score_us: int | None, score_them: int | None) -> str:
    if score_us is None or score_them is None:
        return "unknown"
    if score_us > score_them:
        return "protect_lead"
    if score_us < score_them:
        return "trailing"
    return "neutral"


def evaluate_situation(
    sit: Any,
    *,
    memory: Mapping[str, Any] | None = None,
    roster: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured pre-snap picture. Unknown fields stay unknown."""
    down = _num(sit, "down")
    distance = _num(sit, "distance")
    yardline = _num(sit, "yardline")
    quarter = _num(sit, "quarter", "quarter")
    clock = _num(sit, "clock_seconds", "clock_seconds")
    timeouts = _num(sit, "timeouts_us", "timeouts_us")
    score_us = _num(sit, "score_us")
    score_them = _num(sit, "score_them")
    goal_line = bool(getattr(sit, "goal_line", False))
    red_zone = bool(getattr(sit, "red_zone", False) or goal_line)
    if yardline is not None and yardline >= 80:
        red_zone = True
    if yardline is not None and yardline >= 97:
        goal_line = True
    two_minute = bool(getattr(sit, "two_minute", False))
    if clock is not None and quarter is not None and quarter >= 4 and clock <= 120:
        two_minute = True
    short = bool(distance is not None and distance <= 2 and down in (2, 3, 4))
    long = bool(down in (3, 4) and distance is not None and distance >= 7 and not goal_line)
    backed_up = bool(yardline is not None and yardline <= 12)
    phase = _phase(score_us, score_them)
    look = coverage_evidence(sit, memory)
    personnel = roster_evidence(roster)
    names: list[str] = ["normal"]
    if short:
        names.append("short_yardage")
    if long:
        names.append("third_and_long")
    if red_zone and not goal_line:
        names.append("red_zone")
    if goal_line:
        names.append("goal_line")
    if backed_up:
        names.append("backed_up")
    if two_minute:
        names.append("two_minute")
    if phase in ("protect_lead", "trailing") and (two_minute or (quarter or 0) >= 4):
        names.append("clock_management")
    return {
        "down": down,
        "distance": distance,
        "yardline": yardline,
        "quarter": quarter,
        "clock_seconds": clock,
        "timeouts_us": timeouts,
        "score_us": score_us,
        "score_them": score_them,
        "score_phase": phase,
        "goal_to_go": bool(goal_line or (yardline is not None and distance is not None and yardline + distance >= 100)),
        "red_zone": red_zone,
        "goal_line": goal_line,
        "two_minute": two_minute,
        "short_yardage": short,
        "long_yardage": long,
        "backed_up": backed_up,
        "situations": names,
        "coverage": look,
        "roster": personnel,
        "claim": "situation_proxies_not_expected_points",
    }


def _selection_parts(play: str, sit: Any) -> list[dict[str, Any]]:
    """Legacy magnitudes used by model-primary sampling."""
    if sit is None:
        return []
    down, distance = getattr(sit, "down", None), getattr(sit, "distance", None)
    if down is None or distance is None:
        return []
    try:
        d, yards = int(down), int(distance)
    except (TypeError, ValueError):
        return []
    run = is_run(play)
    screen = _play_family(play) == "screen"
    deep = is_deep(play)
    parts: list[dict[str, Any]] = []
    if d in (3, 4) and yards >= 7:
        if run:
            parts.append({"id": "third_long_run", **SELECTION_PRIORS["third_long_run"]})
        if screen:
            parts.append({"id": "third_long_screen", **SELECTION_PRIORS["third_long_screen"]})
        if not run and not screen and deep:
            parts.append({"id": "third_long_reachable", **SELECTION_PRIORS["third_long_reachable"]})
    elif d in (3, 4) and yards <= 2:
        if run:
            parts.append({"id": "short_yardage_run", **SELECTION_PRIORS["short_yardage_run"]})
        if deep:
            parts.append({"id": "short_yardage_deep", **SELECTION_PRIORS["short_yardage_deep"]})
    if bool(getattr(sit, "two_minute", False)) and yards >= 4 and run:
        score_us, score_them = getattr(sit, "score_us", None), getattr(sit, "score_them", None)
        if score_us is not None and score_them is not None:
            if score_us < score_them:
                parts.append({"id": "two_minute_trail_run", **SELECTION_PRIORS["two_minute_trail_run"]})
            elif score_us > score_them:
                parts.append({"id": "two_minute_lead_run", **SELECTION_PRIORS["two_minute_lead_run"]})
    return parts


def selection_adjustment(play: str, sit: Any) -> tuple[float, str]:
    """Delta consumed by the existing model-primary policy."""
    parts = _selection_parts(play, sit)
    if sit is None:
        return 0.0, "situation unavailable"
    if getattr(sit, "down", None) is None or getattr(sit, "distance", None) is None:
        return 0.0, "down/distance unknown"
    try:
        int(getattr(sit, "down"))
        int(getattr(sit, "distance"))
    except (TypeError, ValueError):
        return 0.0, "down/distance unparseable"
    total = round(sum(float(p["delta"]) for p in parts), 5)
    reason = "; ".join(str(p["reason"]) for p in parts) or "normal situation"
    return total, reason


def coordinator_components(
    play: str,
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extra football reasons. Each row names its proxy and its claim limit."""
    run = is_run(play)
    screen = _play_family(play) == "screen"
    deep = is_deep(play)
    parts: list[dict[str, Any]] = []
    if evaluation.get("goal_line"):
        if run:
            parts.append({"id": "goal_line_run", **COORDINATOR_PRIORS["goal_line_run"]})
        if deep:
            parts.append({"id": "goal_line_deep", **COORDINATOR_PRIORS["goal_line_deep"]})
    elif evaluation.get("red_zone") and deep and "rz" not in play.lower():
        parts.append({"id": "red_zone_deep", **COORDINATOR_PRIORS["red_zone_deep"]})
    if evaluation.get("backed_up") and deep:
        parts.append({"id": "backed_up_deep", **COORDINATOR_PRIORS["backed_up_deep"]})
    if evaluation.get("down") == 2 and evaluation.get("short_yardage"):
        if run:
            parts.append({"id": "second_short_run", **COORDINATOR_PRIORS["second_short_run"]})
        elif not screen:
            parts.append({"id": "second_short_explosive", **COORDINATOR_PRIORS["second_short_explosive"]})
    quarter = evaluation.get("quarter")
    phase = evaluation.get("score_phase")
    if quarter is not None and int(quarter) >= 4 and not evaluation.get("two_minute"):
        if phase == "protect_lead" and run:
            parts.append({"id": "late_lead_run", **COORDINATOR_PRIORS["late_lead_run"]})
        if phase == "trailing" and not run and (evaluation.get("distance") or 0) >= 4:
            parts.append({"id": "late_trail_pass", **COORDINATOR_PRIORS["late_trail_pass"]})
    look = evaluation.get("coverage") or {}
    if look.get("state") == "observed" and look.get("value") == "pressure":
        if screen or (not run and not deep):
            parts.append({"id": "observed_pressure_quick", **COORDINATOR_PRIORS["observed_pressure_quick"]})
        if deep:
            parts.append({"id": "observed_pressure_deep", **COORDINATOR_PRIORS["observed_pressure_deep"]})
    elif (
        look.get("state") == "inferred"
        and look.get("source") == "tendency"
        and look.get("value") == "pressure"
        and float(look.get("confidence") or 0) >= 0.5
        and (screen or (not run and not deep))
    ):
        parts.append({"id": "inferred_pressure_quick", **COORDINATOR_PRIORS["inferred_pressure_quick"]})
    elif look.get("state") != "observed":
        parts.append({
            "id": "coverage_unknown",
            "delta": 0.0,
            "reason": look.get("note") or "coverage unknown",
            "claim": "no_fabricated_coverage",
        })
    return parts


def explain_play(play: str, sit: Any, *, memory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Why this play fits the snap, with selection and coordinator parts split."""
    evaluation = evaluate_situation(sit, memory=memory)
    selected = _selection_parts(play, sit)
    extra = coordinator_components(play, evaluation)
    return {
        "play": play,
        "situations": evaluation["situations"],
        "coverage": evaluation["coverage"],
        "score_phase": evaluation["score_phase"],
        "selection_delta": round(sum(float(p["delta"]) for p in selected), 5),
        "coordinator_delta": round(sum(float(p["delta"]) for p in extra), 5),
        "selection_parts": selected,
        "coordinator_parts": extra,
        "claim": evaluation["claim"],
    }
