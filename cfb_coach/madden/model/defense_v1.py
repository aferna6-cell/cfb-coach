"""Offline defense v1 artifact: a shrunk logistic model of offense success.

The file is local. This repo does not ship a trained artifact or game rows.
The policy name is ``logged_unverified_v1`` because offline eval did not beat
the situation baseline. A loaded file is not a verified defensive result.

JSON schema ``madden.defense.logistic.v1``
------------------------------------------
Required keys:

``schema``
    Exactly ``madden.defense.logistic.v1``.

``policy``
    Exactly ``logged_unverified_v1``.

``target``
    Exactly ``offense_success``. The score is P(offense success), not a stop rate.

``intercept``
    Finite number. Booleans are rejected.

``features``
    Unique situation-feature names. Each name is one of
    ``down``, ``distance``, ``yardline``, ``score_diff``, ``quarter``,
    ``red_zone``, ``goal_line``, ``two_minute``, ``short_yardage``,
    ``long_yardage``, or ``<numeric>_missing`` for a numeric name above.
    A ``*_missing`` flag is 1 when that number was not supplied and 0 when
    it was. The base numeric name must also be listed. A numeric feature
    with no missing flag cannot be scored when the situation lacks it.

``coefficients``
    Object whose keys are exactly ``features``. Values are finite numbers.

``call_families``
    Unique non-empty family vocabulary.

``packages``
    Unique non-empty package vocabulary.

``family_coefficients``
    Object whose keys are exactly ``call_families``.

``package_coefficients``
    Object whose keys are exactly ``packages``.

``candidates``
    Non-empty list of ``{formation, play, family, package}``. ``family`` and
    ``package`` must be in the vocabularies. Formation+play pairs are unique.

The logit is::

    intercept
    + sum(coefficients[name] * feature[name])
    + family_coefficients[family]
    + package_coefficients[package]
    + family_situation_coefficients[family] · interaction(situation)
    + package_situation_coefficients[package] · interaction(situation)

Schema ``madden.defense.logistic.v1`` has no interaction maps, so the
situation term is the same for every candidate and the ranking cannot change
with the situation. Schema ``madden.defense.logistic.v2`` adds those maps.
v1 files still load.

Interaction features, present on every family and every package in v2:

``short_yardage``, ``long_yardage``
    1 when the situation flag is set, else 0.
``down3plus``
    1 when the down is 3 or 4, else 0.
``distance``
    The distance value, or 0 when it was not supplied.
``yardline_own``, ``yardline_mid``, ``yardline_opp``
    One-hot yardline bucket. Own is yardline <= 20, mid is 21..79, opp is
    >= 80. All three are 0 when the yardline is missing.

v2 also samples. Optional keys, with defaults:

``top_k``
    How many leading candidates can be shown. Default 4.
``temperature``
    Softmax temperature on predicted offense success. Lower success is more
    likely. 0 is greedy. Default 0.25.
``max_repeat``
    The same formation and play is not shown more than this many snaps in a
    row. Default 2.
``family_spread``
    Added to the sampling score of every call in the previous snap's family.
    The lowest score is preferred, so repeating a family is penalized.
    Default 0.35.

A rolling window of the last 8 snaps also caps the mix. One call may not be
more than 25 percent of that window, and one coverage family or personnel
package may not be more than 50 percent. The same limits apply while the
window is still filling, so a game cannot build an illegal window. Short
yardage and the goal line are restricted to heavy fronts. A clear run down,
including a run-heavy recent opponent tendency, does not call a Dime or other
light box.

P(offense success) is the logistic of the logit. v1 shows the lowest
probability among the calls the situation still allows. v2 draws from the
legal top-k with a seeded softmax. Ties and the greedy case break by
formation, then play. The best candidate must be in the installed defensive
book; an illegal best call falls back and does not silently take the next
legal candidate.

Default path: ``~/.cfb-coach/models/defense_v1.json``.
Override: ``--defense-model-path``, else ``CFB_COACH_DEFENSE_V1``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_V1 = "madden.defense.logistic.v1"
SCHEMA_V2 = "madden.defense.logistic.v2"
SCHEMA = SCHEMA_V1
SCHEMAS = (SCHEMA_V1, SCHEMA_V2)
POLICY = "logged_unverified_v1"
TARGET = "offense_success"
INTERACTION_FEATURES = (
    "short_yardage",
    "long_yardage",
    "down3plus",
    "distance",
    "yardline_own",
    "yardline_mid",
    "yardline_opp",
)
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.25
DEFAULT_MAX_REPEAT = 2
DEFAULT_FAMILY_SPREAD = 0.35
VARIETY_WINDOW = 8
MAX_CALL_SHARE = 0.25
MAX_FAMILY_SHARE = 0.50
MAX_PACKAGE_SHARE = 0.50
RUN_HEAVY_WINDOW = 6
RUN_HEAVY_MIN_SNAPS = 3
RUN_HEAVY_SHARE = 0.50
_LIGHT_BOX = re.compile(r"dime|dollar|quarter|light[\s-]*box", re.I)
_HEAVY_FRONT = re.compile(
    r"goal[\s-]*line|\b4-3\b|\b4-4\b|\b3-4\b|\b46\b|\bbear\b|\bheavy\b|\bbig\b",
    re.I,
)

NUMERIC_FEATURES = ("down", "distance", "yardline", "score_diff", "quarter")
BINARY_FEATURES = (
    "red_zone",
    "goal_line",
    "two_minute",
    "short_yardage",
    "long_yardage",
)
KNOWN_SITUATION_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES


class DefenseV1Error(Exception):
    """A defense v1 load or score failed closed.

    ``reason`` is ``missing``, ``invalid``, ``missing_feature``, ``illegal``,
    ``installed_gap``, ``timeout``, or ``exception``.
    """

    def __init__(self, reason: str, detail: str, pick: Mapping[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.pick = None if pick is None else dict(pick)


def default_artifact_path() -> Path:
    """``~/.cfb-coach/models/defense_v1.json``."""
    return Path.home() / ".cfb-coach" / "models" / "defense_v1.json"


def resolve_artifact_path(explicit: str | None = None) -> Path:
    """Flag, then ``CFB_COACH_DEFENSE_V1``, then the default home path."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("CFB_COACH_DEFENSE_V1")
    if env:
        return Path(env).expanduser()
    return default_artifact_path()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DefenseV1Error("invalid", f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise DefenseV1Error("invalid", f"{label} must be a finite number")
    return number


def _unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise DefenseV1Error("invalid", f"{label} must be a non-empty list of strings")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise DefenseV1Error("invalid", f"{label} entries must be non-empty strings")
        names.append(item)
    if len(set(names)) != len(names):
        raise DefenseV1Error("invalid", f"{label} entries must be unique")
    return names


def _coef_map(value: Any, expected: Sequence[str], label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise DefenseV1Error("invalid", f"{label} must be an object")
    if set(value) != set(expected):
        raise DefenseV1Error("invalid", f"{label} keys must match the vocabulary exactly")
    return {name: _finite_number(value[name], f"{label}.{name}") for name in expected}


def _validate_feature_names(features: Sequence[str]) -> None:
    allowed = set(KNOWN_SITUATION_FEATURES)
    for name in features:
        if name in allowed:
            continue
        if name.endswith("_missing"):
            base = name[: -len("_missing")]
            if base in NUMERIC_FEATURES and base in features:
                continue
        raise DefenseV1Error("invalid", f"unknown or unpaired situation feature: {name}")


def _interaction_map(value: Any, names: Sequence[str], label: str) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        raise DefenseV1Error("invalid", f"{label} must be an object")
    if set(value) != set(names):
        raise DefenseV1Error("invalid", f"{label} keys must match the vocabulary exactly")
    out: dict[str, dict[str, float]] = {}
    for name in names:
        row = value[name]
        if not isinstance(row, dict):
            raise DefenseV1Error("invalid", f"{label}.{name} must be an object")
        if set(row) != set(INTERACTION_FEATURES):
            raise DefenseV1Error(
                "invalid",
                f"{label}.{name} keys must be {', '.join(INTERACTION_FEATURES)}",
            )
        out[name] = {
            feature: _finite_number(row[feature], f"{label}.{name}.{feature}")
            for feature in INTERACTION_FEATURES
        }
    return out


def _optional_int(raw: Mapping[str, Any], key: str, default: int, minimum: int) -> int:
    if key not in raw or raw.get(key) is None:
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DefenseV1Error("invalid", f"{key} must be an integer >= {minimum}")
    return value


def _optional_float(raw: Mapping[str, Any], key: str, default: float, minimum: float) -> float:
    if key not in raw or raw.get(key) is None:
        return default
    number = _finite_number(raw[key], key)
    if number < minimum:
        raise DefenseV1Error("invalid", f"{key} must be >= {minimum}")
    return number


def validate_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized artifact or raise ``DefenseV1Error``."""
    if not isinstance(raw, Mapping):
        raise DefenseV1Error("invalid", "defense artifact must be a JSON object")
    schema = raw.get("schema")
    if schema not in SCHEMAS:
        raise DefenseV1Error("invalid", f"defense schema must be {SCHEMA_V1} or {SCHEMA_V2}")
    if raw.get("policy") != POLICY:
        raise DefenseV1Error(
            "invalid",
            f"defense policy must be {POLICY}",
        )
    if raw.get("target") != TARGET:
        raise DefenseV1Error("invalid", f"defense target must be {TARGET}")
    features = _unique_strings(raw.get("features"), "features")
    _validate_feature_names(features)
    families = _unique_strings(raw.get("call_families"), "call_families")
    packages = _unique_strings(raw.get("packages"), "packages")
    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise DefenseV1Error("invalid", "candidates must be a non-empty list")
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(candidates_raw):
        if not isinstance(row, Mapping):
            raise DefenseV1Error("invalid", f"candidates[{index}] must be an object")
        formation = row.get("formation")
        play = row.get("play")
        family = row.get("family")
        package = row.get("package")
        if not all(isinstance(item, str) and item.strip() for item in (formation, play, family, package)):
            raise DefenseV1Error("invalid", f"candidates[{index}] needs formation, play, family, and package")
        if family not in families or package not in packages:
            raise DefenseV1Error("invalid", f"candidates[{index}] family or package is outside the vocabulary")
        key = (str(formation), str(play))
        if key in seen:
            raise DefenseV1Error("invalid", f"duplicate candidate {formation} / {play}")
        seen.add(key)
        candidates.append({
            "formation": str(formation),
            "play": str(play),
            "family": str(family),
            "package": str(package),
        })
    artifact = {
        "schema": schema,
        "policy": POLICY,
        "target": TARGET,
        "intercept": _finite_number(raw.get("intercept"), "intercept"),
        "features": features,
        "coefficients": _coef_map(raw.get("coefficients"), features, "coefficients"),
        "call_families": families,
        "packages": packages,
        "family_coefficients": _coef_map(raw.get("family_coefficients"), families, "family_coefficients"),
        "package_coefficients": _coef_map(raw.get("package_coefficients"), packages, "package_coefficients"),
        "candidates": candidates,
    }
    if schema == SCHEMA_V2:
        artifact["family_situation_coefficients"] = _interaction_map(
            raw.get("family_situation_coefficients"), families, "family_situation_coefficients",
        )
        artifact["package_situation_coefficients"] = _interaction_map(
            raw.get("package_situation_coefficients"), packages, "package_situation_coefficients",
        )
        artifact["top_k"] = _optional_int(raw, "top_k", DEFAULT_TOP_K, 1)
        artifact["temperature"] = _optional_float(raw, "temperature", DEFAULT_TEMPERATURE, 0.0)
        artifact["max_repeat"] = _optional_int(raw, "max_repeat", DEFAULT_MAX_REPEAT, 1)
        artifact["family_spread"] = _optional_float(raw, "family_spread", DEFAULT_FAMILY_SPREAD, 0.0)
    return artifact


def load_artifact(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate one local JSON file. Does not train and does not search."""
    dest = Path(path) if path else default_artifact_path()
    if not dest.is_file():
        raise DefenseV1Error("missing", f"defense v1 artifact missing: {dest}")
    try:
        raw = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DefenseV1Error("invalid", f"defense v1 artifact is not valid JSON: {dest}") from exc
    try:
        artifact = validate_artifact(raw)
    except DefenseV1Error:
        raise
    except Exception as exc:  # noqa: BLE001 — a bad file is an invalid artifact
        raise DefenseV1Error("invalid", "defense v1 artifact could not be validated") from exc
    artifact["path"] = str(dest)
    return artifact


def _raw_feature(sit: Any, name: str) -> float | None:
    if name == "down":
        value = getattr(sit, "down", None)
        return None if value is None else float(value)
    if name == "distance":
        value = getattr(sit, "distance", None)
        return None if value is None else float(value)
    if name == "yardline":
        value = getattr(sit, "yardline", None)
        return None if value is None else float(value)
    if name == "score_diff":
        us = getattr(sit, "score_us", None)
        them = getattr(sit, "score_them", None)
        if us is None or them is None:
            return None
        return float(us) - float(them)
    if name == "quarter":
        value = getattr(sit, "quarter", None)
        if value is None:
            extras = getattr(sit, "extras", None) or {}
            value = extras.get("quarter") if isinstance(extras, Mapping) else None
        return None if value is None else float(value)
    if name in BINARY_FEATURES:
        return 1.0 if bool(getattr(sit, name, False)) else 0.0
    raise DefenseV1Error("invalid", f"unknown situation feature: {name}")


def situation_feature_vector(sit: Any, features: Sequence[str]) -> dict[str, float]:
    """Values for ``features``. A missing number without a flag raises."""
    names = set(features)
    values: dict[str, float] = {}
    for name in features:
        if name.endswith("_missing"):
            base = name[: -len("_missing")]
            raw = _raw_feature(sit, base)
            values[name] = 1.0 if raw is None else 0.0
            continue
        raw = _raw_feature(sit, name)
        if raw is None:
            if f"{name}_missing" in names:
                values[name] = 0.0
            else:
                raise DefenseV1Error("missing_feature", f"defense v1 missing situation feature: {name}")
        else:
            values[name] = float(raw)
    return values


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp = math.exp(logit)
    return exp / (1.0 + exp)


def interaction_values(sit: Any) -> dict[str, float]:
    """Situation-by-call features. These differ across candidates only via coefficients."""
    down = getattr(sit, "down", None)
    distance = getattr(sit, "distance", None)
    yardline = getattr(sit, "yardline", None)
    values = {
        "short_yardage": 1.0 if getattr(sit, "short_yardage", False) else 0.0,
        "long_yardage": 1.0 if getattr(sit, "long_yardage", False) else 0.0,
        "down3plus": 1.0 if down is not None and int(down) >= 3 else 0.0,
        "distance": 0.0 if distance is None else float(distance),
        "yardline_own": 0.0,
        "yardline_mid": 0.0,
        "yardline_opp": 0.0,
    }
    if yardline is not None:
        spot = int(yardline)
        if spot <= 20:
            values["yardline_own"] = 1.0
        elif spot >= 80:
            values["yardline_opp"] = 1.0
        else:
            values["yardline_mid"] = 1.0
    return values


def offense_success_probability(
    artifact: Mapping[str, Any],
    sit: Any,
    candidate: Mapping[str, str],
) -> float:
    """P(offense success) for one candidate. Lower is the defensive pick."""
    features = situation_feature_vector(sit, artifact["features"])
    logit = float(artifact["intercept"])
    coefficients = artifact["coefficients"]
    for name, value in features.items():
        logit += float(coefficients[name]) * value
    logit += float(artifact["family_coefficients"][candidate["family"]])
    logit += float(artifact["package_coefficients"][candidate["package"]])
    fam_ix = artifact.get("family_situation_coefficients")
    if fam_ix:
        vector = interaction_values(sit)
        for name, coef in fam_ix[candidate["family"]].items():
            logit += float(coef) * vector[name]
        for name, coef in artifact["package_situation_coefficients"][candidate["package"]].items():
            logit += float(coef) * vector[name]
    return _sigmoid(logit)


def _check_budget(started: float, budget_ms: float) -> None:
    if (time.perf_counter() - started) * 1000.0 > budget_ms:
        raise DefenseV1Error("timeout", "defense v1 exceeded 150 ms")


def play_in_book(book: Mapping[str, Sequence[str]], formation: str, play: str) -> bool:
    plays = book.get(formation)
    return plays is not None and play in list(plays)


def _call_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("formation") or ""), str(row.get("play") or ""))


def _family_of(artifact: Mapping[str, Any], formation: str, play: str) -> str | None:
    for candidate in artifact["candidates"]:
        if candidate["formation"] == formation and candidate["play"] == play:
            return str(candidate["family"])
    return None


def _banned_repeat(
    recent: Sequence[Sequence[Any]],
    max_repeat: int,
) -> tuple[str, str] | None:
    if max_repeat < 1 or len(recent) < max_repeat:
        return None
    tail = [_call_key({"formation": row[0], "play": row[1]}) for row in recent[-max_repeat:]]
    if tail and all(item == tail[0] for item in tail):
        return tail[0]
    return None


def _package_of(artifact: Mapping[str, Any], formation: str, play: str) -> str | None:
    for candidate in artifact["candidates"]:
        if candidate["formation"] == formation and candidate["play"] == play:
            return str(candidate["package"])
    return None


def _recent_parts(artifact: Mapping[str, Any], row: Sequence[Any]) -> tuple[str, str, str, str]:
    formation = str(row[0]) if row else ""
    play = str(row[1]) if len(row) > 1 else ""
    family = str(row[2]) if len(row) > 2 and row[2] else (_family_of(artifact, formation, play) or "")
    package = str(row[3]) if len(row) > 3 and row[3] else (_package_of(artifact, formation, play) or "")
    return formation, play, family, package


def _window_cap(share: float) -> int:
    """Largest count that is not strictly above ``share`` of a full window."""
    return int(math.floor(float(share) * VARIETY_WINDOW))


def _variety_ok(
    recent: Sequence[Sequence[Any]],
    row: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> bool:
    """False when this call, family, or package would break the rolling mix."""
    prior = [_recent_parts(artifact, item) for item in list(recent)[-(VARIETY_WINDOW - 1):]]
    cand = (
        str(row.get("formation") or ""),
        str(row.get("play") or ""),
        str(row.get("family") or ""),
        str(row.get("package") or ""),
    )
    window = prior + [cand]
    calls = sum(1 for item in window if (item[0], item[1]) == (cand[0], cand[1]))
    if calls > _window_cap(MAX_CALL_SHARE):
        return False
    if cand[2]:
        families = sum(1 for item in window if item[2] == cand[2])
        if families > _window_cap(MAX_FAMILY_SHARE):
            return False
    if cand[3]:
        packages = sum(1 for item in window if item[3] == cand[3])
        if packages > _window_cap(MAX_PACKAGE_SHARE):
            return False
    return True


def _is_light_box(row: Mapping[str, Any]) -> bool:
    package = str(row.get("package") or "")
    formation = str(row.get("formation") or "")
    return bool(_LIGHT_BOX.search(package) or _LIGHT_BOX.search(formation))


def _is_heavy_front(row: Mapping[str, Any]) -> bool:
    if _is_light_box(row):
        return False
    package = str(row.get("package") or "")
    formation = str(row.get("formation") or "")
    if package.lower() in {"goal", "goal_line", "goalline", "heavy"}:
        return True
    return bool(_HEAVY_FRONT.search(formation) or _HEAVY_FRONT.search(package))


RED_ZONE_YARDS = 20
GOAL_LINE_YARDS = 5
LIGHT_BOX_PASS_DISTANCE = 7
LIGHT_BOX_SHORT_DISTANCE = 4
LIGHT_BOX_MAX_RUN_SHARE = 0.40


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _yards_to_goal(sit: Any) -> int | None:
    """Defensive snaps are entered as "my N": the offense is N yards from scoring."""
    spot = _int_or_none(getattr(sit, "yardline", None))
    return spot if spot is not None and 0 < spot < 100 else None


def _defense_red_zone(sit: Any) -> bool:
    if getattr(sit, "red_zone", False) or getattr(sit, "goal_line", False):
        return True
    spot = _yards_to_goal(sit)
    return spot is not None and spot <= RED_ZONE_YARDS


def _wants_heavy_front(sit: Any) -> bool:
    if getattr(sit, "goal_line", False) or getattr(sit, "short_yardage", False):
        return True
    spot = _yards_to_goal(sit)
    if spot is not None and spot <= GOAL_LINE_YARDS:
        return True
    down = getattr(sit, "down", None)
    distance = getattr(sit, "distance", None)
    return bool(down in (3, 4) and distance is not None and int(distance) <= 2)


def light_box_allowed(sit: Any, run_share: float | None) -> bool:
    """Dime and other light boxes: obvious passing downs, or a pass-first opponent.

    Never inside the 20 and never on short yardage. An unknown run share
    keeps the box honest except on 3rd/4th-and-7+.
    """
    if _defense_red_zone(sit):
        return False
    down = _int_or_none(getattr(sit, "down", None))
    distance = _int_or_none(getattr(sit, "distance", None))
    if distance is not None and distance <= LIGHT_BOX_SHORT_DISTANCE:
        return False
    if down in (3, 4) and distance is not None and distance >= LIGHT_BOX_PASS_DISTANCE:
        return True
    return run_share is not None and run_share < LIGHT_BOX_MAX_RUN_SHARE


def live_run_share(concepts: Sequence[str] | None, profile: Mapping[str, Any] | None = None) -> float | None:
    """This game's run share from named concepts, else the opponent profile."""
    from cfb_coach.madden.situation import concept_family

    tail = [concept_family(str(c)) for c in (concepts or []) if str(c).strip()][-LIVE_WINDOW:]
    tail = [f for f in tail if f]
    if len(tail) >= LIVE_MIN_CONCEPTS:
        return tail.count("run") / len(tail)
    if profile and int(profile.get("n") or 0) > 0 and profile.get("run_share") is not None:
        return float(profile["run_share"])
    return None


def recent_run_heavy(concepts: Sequence[str] | None) -> bool:
    """True when the opponent's recent named concepts are at least half runs.

    Fewer than three named concepts is not a tendency. Inside zone, stretch,
    and dive are runs.
    """
    if not concepts:
        return False
    from cfb_coach.madden.situation import concept_family

    tail = [str(item) for item in concepts if str(item).strip()][-RUN_HEAVY_WINDOW:]
    if len(tail) < RUN_HEAVY_MIN_SNAPS:
        return False
    runs = sum(1 for name in tail if concept_family(name) == "run")
    return (runs / len(tail)) >= RUN_HEAVY_SHARE


def _guarded_ranked(
    ranked: Sequence[Mapping[str, Any]],
    sit: Any,
    run_heavy: bool,
    light_ok: bool = True,
) -> tuple[list[Mapping[str, Any]], str]:
    """Drop light boxes on run downs, and keep heavy fronts on goal-line/short."""
    rows = list(ranked)
    if _wants_heavy_front(sit):
        heavy = [row for row in rows if _is_heavy_front(row)]
        if heavy:
            return heavy, "heavy_front"
        non_light = [row for row in rows if not _is_light_box(row)]
        if non_light:
            return non_light, "no_light_box"
        return rows, ""
    if run_heavy or not light_ok:
        non_light = [row for row in rows if not _is_light_box(row)]
        if non_light:
            return non_light, "no_light_box" if run_heavy else "no_light_box_situation"
    return rows, ""


def _sampling_score(
    row: Mapping[str, Any],
    artifact: Mapping[str, Any],
    recent: Sequence[Sequence[Any]],
) -> float:
    """Lower is a better defensive call.

    ``family_spread`` is added for the previous snap's family. Subtracting it
    made that family win, which is the bug this score corrects.
    """
    score = float(row["offense_success"])
    spread = float(artifact.get("family_spread") or 0.0)
    if not recent or spread == 0.0:
        return score
    last = recent[-1]
    last_key = (str(last[0]), str(last[1]))
    last_family = str(last[2]) if len(last) > 2 and last[2] else _family_of(artifact, last_key[0], last_key[1])
    row_family = row.get("family") or _family_of(
        artifact, str(row.get("formation") or ""), str(row.get("play") or ""),
    )
    if last_family and row_family == last_family:
        return score + spread
    return score


def _rng(seed: Any) -> random.Random:
    if isinstance(seed, int) and not isinstance(seed, bool):
        return random.Random(seed)
    digest = hashlib.sha256(str(seed).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _sample_v2(
    artifact: Mapping[str, Any],
    ranked: Sequence[Mapping[str, Any]],
    book: Mapping[str, Sequence[str]],
    recent: Sequence[Sequence[Any]],
    seed: Any,
    top_k: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    """Seeded softmax over the legal top-k. An illegal best call is not replaced."""
    k = int(artifact.get("top_k") or DEFAULT_TOP_K) if top_k is None else int(top_k)
    if isinstance(top_k, bool) or k < 1:
        raise DefenseV1Error("invalid", "top_k must be an integer >= 1")
    temp = float(artifact.get("temperature") if temperature is None else temperature)
    if temp < 0 or not math.isfinite(temp):
        raise DefenseV1Error("invalid", "temperature must be >= 0")
    banned = _banned_repeat(recent, int(artifact.get("max_repeat") or DEFAULT_MAX_REPEAT))
    legal = [
        row for row in ranked
        if play_in_book(book, str(row["formation"]), str(row["play"])) and _call_key(row) != banned
    ]
    if not legal:
        blocked = banned or _call_key(ranked[0])
        raise DefenseV1Error(
            "invalid",
            f"defense v2 repeat limit blocked {blocked[0]} / {blocked[1]}",
            pick=dict(ranked[0]),
        )
    varied = [row for row in legal if _variety_ok(recent, row, artifact)]
    pool = (varied or legal)[:k]
    order = sorted(pool, key=lambda row: (_sampling_score(row, artifact, recent), row["formation"], row["play"]))
    if temp == 0 or len(order) == 1:
        chosen = order[0]
    else:
        adjusted = [_sampling_score(row, artifact, recent) for row in order]
        floor = min(adjusted)
        weights = [math.exp(-(value - floor) / temp) for value in adjusted]
        chosen = _rng(0 if seed is None else seed).choices(list(order), weights=weights, k=1)[0]
    return dict(chosen)


_MODEL_FAMILY = {"single_high": "cover3", "two_high": "quarters_c4", "cover2": "two_high_zone",
                 "man": "one_high_man", "pressure": "pressure"}
_COVERAGE_OF = {v: k for k, v in _MODEL_FAMILY.items()}


def _installed_package(formation: str, packages: Sequence[str]) -> str | None:
    if _LIGHT_BOX.search(formation):
        pkg = "dime"
    elif re.search(r"nickel", formation, re.I):
        pkg = "nickel"
    else:
        pkg = "four_three"
    return pkg if pkg in packages else None


def installed_candidates(artifact: Mapping[str, Any], book: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """Every installed play the model can score.

    Trained candidates keep their labels. Other installed plays get the
    model family of their coverage (Cover 3 -> cover3, quarters -> quarters_c4,
    Cover 2/Tampa -> two_high_zone, man -> one_high_man, pressure) and the
    package of their formation (base fronts score as four_three). Plays with
    no recognizable coverage are left to the established caller.
    """
    from cfb_coach.madden.defense_select import call_family

    trained = {(c["formation"], c["play"]): dict(c) for c in artifact["candidates"]}
    out = [c for key, c in trained.items() if play_in_book(book, *key)]
    families = set(artifact["call_families"])
    for form, plays in book.items():
        pkg = _installed_package(str(form), artifact["packages"])
        if pkg is None:
            continue
        for play in plays or []:
            if (form, play) in trained:
                continue
            fam = _MODEL_FAMILY.get(call_family(play) or "")
            if fam in families:
                out.append({"formation": form, "play": play, "family": fam, "package": pkg, "mapped": True})
    return out


LIVE_WINDOW = 8
LIVE_MIN_CONCEPTS = 3
LIVE_COUNTER_WEIGHT = 0.06
LIVE_RUN_TILT = 0.15
LIVE_NEUTRAL_RUN_SHARE = 0.4


def _run_tilt(row: Mapping[str, Any]) -> float:
    """How much a heavier opponent run share hurts this call (+) or helps it (-)."""
    if _is_light_box(row):
        return 1.0
    if str(row.get("package") or "") == "nickel":
        return 0.3
    if _is_heavy_front(row):
        return -0.8
    return -0.5


def _live_opponent_adjust(ranked: list[dict[str, Any]], concepts: Sequence[str] | None) -> str:
    """Shift each call by what this opponent ran in the last eight named concepts.

    Uses only the opponent's own plays (concepts logged this game), never
    whether our displayed call was executed.
    """
    from cfb_coach.madden.defense_select import COUNTERS
    from cfb_coach.madden.situation import concept_family

    tail = [concept_family(str(c)) for c in (concepts or []) if str(c).strip()][-LIVE_WINDOW:]
    tail = [f for f in tail if f]
    if len(tail) < LIVE_MIN_CONCEPTS:
        return ""
    shares = {f: tail.count(f) / len(tail) for f in set(tail)}
    run = shares.get("run", 0.0)
    for row in ranked:
        cov = _COVERAGE_OF.get(str(row.get("family") or ""))
        delta = -sum(LIVE_COUNTER_WEIGHT * s * float((COUNTERS.get(their) or {}).get(cov, 0.0))
                     for their, s in shares.items())
        delta += LIVE_RUN_TILT * (run - LIVE_NEUTRAL_RUN_SHARE) * _run_tilt(row)
        if delta:
            row["offense_success"] = min(0.999, max(0.001, row["offense_success"] + delta))
            row["live_adjust"] = round(delta, 4)
    ranked.sort(key=lambda row: (row["offense_success"], row["formation"], row["play"]))
    top = sorted(shares.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return f"this game: {', '.join(f'{k} {round(100 * v)}%' for k, v in top)} of last {len(tail)} named concepts"


SPREAD_WINDOW = 10
SPREAD_MIN_CALLS = 4
SPREAD_FREE_SHARE = 0.4
SPREAD_PENALTY = 0.15


def _spread_installed(ranked: list[dict[str, Any]], recent: Sequence[Sequence[Any]]) -> None:
    """Penalize a formation past 40% of the last ten calls, so a clear leader still wins.

    At 80% of the window the penalty is 0.06 offense success, enough to let a
    close installed alternative through, not enough to override a big gap.
    """
    window = [str(r[0]) for r in list(recent)[-SPREAD_WINDOW:] if r]
    if len(window) < SPREAD_MIN_CALLS:
        return
    counts: dict[str, int] = {}
    for form in window:
        counts[form] = counts.get(form, 0) + 1
    changed = False
    for row in ranked:
        share = counts.get(str(row["formation"]), 0) / len(window)
        penalty = SPREAD_PENALTY * max(0.0, share - SPREAD_FREE_SHARE)
        if penalty:
            row["offense_success"] = min(0.999, row["offense_success"] + penalty)
            row["concentration_penalty"] = round(penalty, 4)
            changed = True
    if changed:
        ranked.sort(key=lambda row: (row["offense_success"], row["formation"], row["play"]))


RESULT_WINDOW = 10
RESULT_MIN_CALLS = 3
RESULT_WEIGHT = 0.15
_RESULT_NAMES = {"one_high_man": "man", "cover3": "single-high", "quarters_c4": "quarters",
                 "two_high_zone": "cover 2", "pressure": "pressure"}


def offense_won(result: Any, down: Any, distance: Any) -> bool | None:
    """Did the offense win this snap? TD yes; stops and turnovers no; gains vs the sticks.

    First down needs 40% of the distance, second down 60%, third/fourth all
    of it. A bare "+N" is not enough to call a 2-yard gain on 1st-and-10 a win.
    """
    from cfb_coach.outcome import parse_outcome

    parsed = parse_outcome(None if result is None else str(result))
    if parsed.kind == "td":
        return True
    if parsed.kind in ("int", "fumble", "sack", "incomplete", "stop", "loss"):
        return False
    if parsed.kind == "convert":
        return True
    if parsed.kind != "gain" or parsed.yards is None:
        return None
    d, togo = _int_or_none(down), _int_or_none(distance)
    if d is None or togo is None or togo <= 0:
        return parsed.yards >= 4
    need = {1: 0.4, 2: 0.6}.get(d, 1.0) * togo
    return parsed.yards >= need


def _live_result_adjust(
    ranked: list[dict[str, Any]], results: Sequence[Sequence[Any]] | None,
) -> str:
    """Shift coverage families and light boxes by how they fared this game.

    ``results`` rows are (formation, play, offense_won) for this game's shown
    calls, newest last. They are an in-game signal only, never training labels.
    """
    from cfb_coach.madden.defense_select import call_family

    tail = [r for r in (results or []) if len(r) >= 3 and r[2] is not None][-RESULT_WINDOW:]
    if len(tail) < RESULT_MIN_CALLS:
        return ""
    families: dict[str, list[int]] = {}
    light = [0, 0]
    for form, play, won in ((str(r[0]), str(r[1]), bool(r[2])) for r in tail):
        fam = _MODEL_FAMILY.get(call_family(play) or "")
        if fam:
            stat = families.setdefault(fam, [0, 0])
            stat[0] += won
            stat[1] += 1
        if _LIGHT_BOX.search(form):
            light[0] += won
            light[1] += 1

    def shift(won: int, n: int) -> float:
        if n < RESULT_MIN_CALLS:
            return 0.0
        return RESULT_WEIGHT * ((won + 1) / (n + 2) - 0.5)

    changed = False
    for row in ranked:
        stat = families.get(str(row.get("family") or ""))
        delta = shift(*stat) if stat else 0.0
        if _is_light_box(row):
            delta += shift(*light)
        if delta:
            row["offense_success"] = min(0.999, max(0.001, row["offense_success"] + delta))
            row["result_adjust"] = round(delta, 4)
            changed = True
    if changed:
        ranked.sort(key=lambda row: (row["offense_success"], row["formation"], row["play"]))
    beaten = [
        f"{_RESULT_NAMES.get(fam, fam)} {won}/{n}"
        for fam, (won, n) in sorted(families.items()) if n >= RESULT_MIN_CALLS and won * 2 > n
    ]
    if light[1] >= RESULT_MIN_CALLS and light[0] * 2 > light[1]:
        beaten.append(f"light box {light[0]}/{light[1]}")
    return f"beaten this game: {', '.join(beaten)}" if beaten else ""


def pick_call(
    artifact: Mapping[str, Any],
    sit: Any,
    book: Mapping[str, Sequence[str]],
    *,
    budget_ms: float = 150.0,
    recent: Sequence[Sequence[Any]] = (),
    seed: Any = None,
    top_k: int | None = None,
    temperature: float | None = None,
    recent_concepts: Sequence[str] | None = None,
    run_heavy: bool | None = None,
    opponent_profile: Mapping[str, Any] | None = None,
    installed_scope: bool = False,
    recent_results: Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Lowest predicted offense success, then v2 variety among that legal top.

    An illegal best candidate falls back. v2 does not walk from an illegal
    winner onto a later legal call. Run downs and the goal line narrow the
    menu before that draw.

    ``installed_scope`` scores every installed play (``installed_candidates``)
    and ranks only those, then shifts them by this game's opponent concepts
    and spreads a formation past 40% of recent calls. It fails closed
    (``installed_gap``) when nothing installed can be scored, or when a
    heavy-front down has no heavy front installed.
    """
    started = time.perf_counter()
    if installed_scope:
        artifact = dict(artifact, candidates=installed_candidates(artifact, book))
        if not artifact["candidates"]:
            raise DefenseV1Error("installed_gap", "no installed play has a coverage the model can score")
    ranked: list[dict[str, Any]] = []
    try:
        for candidate in artifact["candidates"]:
            _check_budget(started, budget_ms)
            probability = offense_success_probability(artifact, sit, candidate)
            ranked.append({
                "formation": candidate["formation"],
                "play": candidate["play"],
                "family": candidate["family"],
                "package": candidate["package"],
                "offense_success": probability,
            })
        _check_budget(started, budget_ms)
        ranked.sort(key=lambda row: (row["offense_success"], row["formation"], row["play"]))
    except DefenseV1Error as exc:
        if exc.pick is None and ranked:
            exc.pick = ranked[0]
        raise
    except Exception as exc:  # noqa: BLE001 — score failures fall back
        raise DefenseV1Error("exception", "defense v1 scoring raised", pick=ranked[0] if ranked else None) from exc
    if not ranked:
        raise DefenseV1Error("invalid", "defense v1 produced no candidate")
    if installed_scope:
        scoped = [row for row in ranked if play_in_book(book, str(row["formation"]), str(row["play"]))]
        if not scoped:
            raise DefenseV1Error("installed_gap", "no model-known call in the confirmed defense book", pick=ranked[0])
        if _wants_heavy_front(sit) and not any(_is_heavy_front(row) for row in scoped):
            raise DefenseV1Error(
                "installed_gap", "heavy-front down with no model-known heavy front installed", pick=scoped[0],
            )
        ranked = scoped
    best = ranked[0]
    if not play_in_book(book, best["formation"], best["play"]):
        raise DefenseV1Error(
            "illegal",
            f"illegal play not in installed book: {best['formation']} / {best['play']}",
            pick=best,
        )
    tendency_summary = ""
    if opponent_profile and int(opponent_profile.get("n") or 0) > 0:
        from cfb_coach.madden.model.opponent_tendencies import (
            defense_success_delta,
            format_defense_summary,
        )

        changed = False
        for row in ranked:
            delta = defense_success_delta(row, opponent_profile)
            row["tendency_delta"] = delta
            if delta:
                row["offense_success"] = min(0.999, max(0.001, row["offense_success"] + delta))
                changed = True
        if changed:
            ranked.sort(key=lambda row: (row["offense_success"], row["formation"], row["play"]))
        tendency_summary = str(opponent_profile.get("summary") or format_defense_summary(opponent_profile))
    live_summary = ""
    if installed_scope:
        live_summary = _live_opponent_adjust(ranked, recent_concepts)
        _spread_installed(ranked, recent)
    result_summary = _live_result_adjust(ranked, recent_results)
    if result_summary:
        live_summary = f"{live_summary}; {result_summary}" if live_summary else result_summary
    heavy = recent_run_heavy(recent_concepts) if run_heavy is None else bool(run_heavy)
    light_ok = True
    if installed_scope or recent_results is not None:
        light_ok = light_box_allowed(sit, live_run_share(recent_concepts, opponent_profile))
    guarded, guard = _guarded_ranked(ranked, sit, heavy, light_ok)
    chosen = guarded[0]
    if artifact.get("schema") == SCHEMA_V2:
        chosen = _sample_v2(artifact, guarded, book, recent, seed, top_k, temperature)
    elif not play_in_book(book, str(chosen["formation"]), str(chosen["play"])):
        legal = [
            row for row in guarded
            if play_in_book(book, str(row["formation"]), str(row["play"]))
        ]
        chosen = legal[0] if legal else best
    if guard or tendency_summary or live_summary:
        chosen = dict(chosen)
        if guard:
            chosen["situational_guard"] = guard
        if tendency_summary:
            chosen["tendency_summary"] = tendency_summary
        if live_summary:
            chosen["live_summary"] = live_summary
    return {
        "pick": chosen,
        "legal": True,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "tendency_summary": tendency_summary,
        "live_summary": live_summary,
    }


def comparison_record(
    *,
    model: Mapping[str, Any] | None,
    heuristic_formation: Any,
    heuristic_play: Any,
    executed_formation: Any,
    executed_play: Any,
    fell_back: bool,
    reason: str | None,
    artifact_path: str | None = None,
    offense_success: float | None = None,
    schema: str | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    """Model pick, heuristic pick, and the call actually shown."""

    def _pair(formation: Any, play: Any) -> dict[str, str] | None:
        if formation in (None, "") and play in (None, ""):
            return None
        return {"formation": "" if formation is None else str(formation),
                "play": "" if play is None else str(play)}

    model_pick = None if not model else _pair(model.get("formation"), model.get("play"))
    return {
        "policy": policy or POLICY,
        "schema": schema or SCHEMA,
        "artifact_path": artifact_path,
        "model_pick": model_pick,
        "heuristic_pick": _pair(heuristic_formation, heuristic_play),
        "executed_pick": _pair(executed_formation, executed_play),
        "fell_back": fell_back,
        "reason": reason,
        "offense_success": offense_success,
    }


def comparison_config(record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """String pairs stored on the coaching decision for the next retrain."""

    def _fields(prefix: str, pick: Mapping[str, Any] | None) -> tuple[tuple[str, str], tuple[str, str]]:
        formation = "" if not pick else str(pick.get("formation") or "")
        play = "" if not pick else str(pick.get("play") or "")
        return ((f"{prefix}_formation", formation), (f"{prefix}_play", play))

    rows: list[tuple[str, str]] = [
        ("policy", str(record.get("policy") or POLICY)),
        ("schema", str(record.get("schema") or SCHEMA)),
    ]
    rows.extend(_fields("model", record.get("model_pick")))
    rows.extend(_fields("heuristic", record.get("heuristic_pick")))
    rows.extend(_fields("executed", record.get("executed_pick")))
    rows.append(("fell_back", "true" if record.get("fell_back") else "false"))
    rows.append(("reason", "" if record.get("reason") is None else str(record.get("reason"))))
    path = record.get("artifact_path")
    rows.append(("artifact_path", "" if path is None else str(path)))
    return tuple(rows)


def comparison_detail(record: Mapping[str, Any]) -> str:
    def _text(pick: Mapping[str, Any] | None) -> str:
        if not pick or not (pick.get("formation") or pick.get("play")):
            return "none"
        return f"{pick.get('formation')} / {pick.get('play')}"

    return (
        f"policy={record.get('policy') or POLICY}; "
        f"model={_text(record.get('model_pick'))}; "
        f"heuristic={_text(record.get('heuristic_pick'))}; "
        f"executed={_text(record.get('executed_pick'))}"
    )
