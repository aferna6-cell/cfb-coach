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

P(offense success) is the logistic of that logit. The model's pick is the
candidate with the lowest probability. Ties break by formation, then play.
The live caller shows that pick only when it is in the installed defensive
book. It does not walk down to the next legal candidate.

Default path: ``~/.cfb-coach/models/defense_v1.json``.
Override: ``--defense-model-path``, else ``CFB_COACH_DEFENSE_V1``.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "madden.defense.logistic.v1"
POLICY = "logged_unverified_v1"
TARGET = "offense_success"

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
    ``timeout``, or ``exception``.
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


def validate_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized artifact or raise ``DefenseV1Error``."""
    if not isinstance(raw, Mapping):
        raise DefenseV1Error("invalid", "defense v1 artifact must be a JSON object")
    if raw.get("schema") != SCHEMA:
        raise DefenseV1Error("invalid", f"defense v1 schema must be {SCHEMA}")
    if raw.get("policy") != POLICY:
        raise DefenseV1Error(
            "invalid",
            f"defense v1 policy must be {POLICY}",
        )
    if raw.get("target") != TARGET:
        raise DefenseV1Error("invalid", f"defense v1 target must be {TARGET}")
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
    return {
        "schema": SCHEMA,
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
    return _sigmoid(logit)


def _check_budget(started: float, budget_ms: float) -> None:
    if (time.perf_counter() - started) * 1000.0 > budget_ms:
        raise DefenseV1Error("timeout", "defense v1 exceeded 150 ms")


def play_in_book(book: Mapping[str, Sequence[str]], formation: str, play: str) -> bool:
    plays = book.get(formation)
    return plays is not None and play in list(plays)


def pick_call(
    artifact: Mapping[str, Any],
    sit: Any,
    book: Mapping[str, Sequence[str]],
    *,
    budget_ms: float = 150.0,
) -> dict[str, Any]:
    """Lowest predicted offense success. Does not substitute a later legal call."""
    started = time.perf_counter()
    best: dict[str, Any] | None = None
    best_key: tuple[float, str, str] | None = None
    try:
        for candidate in artifact["candidates"]:
            _check_budget(started, budget_ms)
            probability = offense_success_probability(artifact, sit, candidate)
            key = (probability, candidate["formation"], candidate["play"])
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "formation": candidate["formation"],
                    "play": candidate["play"],
                    "family": candidate["family"],
                    "package": candidate["package"],
                    "offense_success": probability,
                }
        _check_budget(started, budget_ms)
    except DefenseV1Error as exc:
        if exc.pick is None and best is not None:
            exc.pick = best
        raise
    except Exception as exc:  # noqa: BLE001 — score failures fall back
        raise DefenseV1Error("exception", "defense v1 scoring raised", pick=best) from exc
    if best is None or best_key is None:
        raise DefenseV1Error("invalid", "defense v1 produced no candidate")
    legal = play_in_book(book, best["formation"], best["play"])
    if not legal:
        raise DefenseV1Error(
            "illegal",
            f"illegal play not in installed book: {best['formation']} / {best['play']}",
            pick=best,
        )
    return {
        "pick": best,
        "legal": True,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
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
) -> dict[str, Any]:
    """Model pick, heuristic pick, and the call actually shown."""

    def _pair(formation: Any, play: Any) -> dict[str, str] | None:
        if formation in (None, "") and play in (None, ""):
            return None
        return {"formation": "" if formation is None else str(formation),
                "play": "" if play is None else str(play)}

    model_pick = None if not model else _pair(model.get("formation"), model.get("play"))
    return {
        "policy": POLICY,
        "schema": SCHEMA,
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

    rows: list[tuple[str, str]] = [("policy", POLICY), ("schema", SCHEMA)]
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
        f"policy={POLICY}; "
        f"model={_text(record.get('model_pick'))}; "
        f"heuristic={_text(record.get('heuristic_pick'))}; "
        f"executed={_text(record.get('executed_pick'))}"
    )
