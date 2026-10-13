"""Madden call and outcome logging.

Columns and helpers here record what a live session actually did. They do
not change the coaching mode, the model, or the play that was shown.

Unknown yards stay NULL. A displayed adjustment is a recommendation, and it
is stored only on ``recommended_adjustment_id``. ``executed_adjustment_id``
is written only when the caller confirms the adjustment was run.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from cfb_coach.outcome import parse_outcome

NO_SNAP_RECORDED = "no_snap_recorded"
LINKED = "linked"
SCHEMA_META_KEY = "madden_call_outcome_schema"
SCHEMA_VERSION = "1"

# Post-snap labels. None of these may be decision-time features.
POST_SNAP_LOG_FIELDS = frozenset(
    {
        "yards",
        "penalty",
        "executed_adjustment_id",
        "snap_link",
        "accepted",
        "outcome",
        "outcome_json",
    }
)

DECISION_COLUMNS = {
    "snap_link": "TEXT",
    "drive_id": "TEXT",
    "quarter": "INTEGER",
    "game_clock": "TEXT",
    "score_for": "INTEGER",
    "score_against": "INTEGER",
    "accepted": "TEXT",
    "behavior_propensity": "REAL",
    "propensity_method": "TEXT",
    "propensity_override": "INTEGER",
    "propensity_override_reason": "TEXT",
    "recommended_adjustment_id": "TEXT",
}
OUTCOME_COLUMNS = {
    "yards": "INTEGER",
    "penalty": "TEXT",
    "drive_id": "TEXT",
    "quarter": "INTEGER",
    "game_clock": "TEXT",
    "score_for": "INTEGER",
    "score_against": "INTEGER",
    "accepted": "TEXT",
    "recommended_adjustment_id": "TEXT",
    "executed_adjustment_id": "TEXT",
}
SNAP_COLUMNS = {
    "drive_id": "TEXT",
    "game_clock": "TEXT",
    "score_for": "INTEGER",
    "score_against": "INTEGER",
    "penalty": "TEXT",
    "yards": "INTEGER",
    "accepted": "TEXT",
    "recommended_adjustment_id": "TEXT",
    "executed_adjustment_id": "TEXT",
}

_YARDED_KINDS = frozenset({"td", "sack", "int", "fumble"})
_EXPLICIT_YARDS = re.compile(r"([+-]?\d+)")
_DRIVE_ID = re.compile(
    r"\bdrive[\s_]*(?:id|#|no\.?|number)\s*[:=]?\s*([A-Za-z0-9][\w-]*)",
    re.I,
)
_NO_PENALTY = re.compile(r"\bno\s+penalty\b|\bno\s+flag\b", re.I)
_PENALTY_DECLINED = re.compile(
    r"\b(?:penalty|flag)\b.{0,40}\bdeclin|\bdeclin\w*.{0,20}\b(?:penalty|flag)\b",
    re.I,
)
_PENALTY_ACCEPTED = re.compile(
    r"\b(?:penalty|flag)\b.{0,40}\baccept|\baccept\w*.{0,20}\b(?:penalty|flag)\b",
    re.I,
)
_PENALTY_VALUES = frozenset({"no_penalty", "accepted", "declined"})


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _text(value: Any) -> str | None:
    if _blank(value):
        return None
    return str(value).strip()


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def yards_for_log(text: str | None) -> int | None:
    """Yards for a scoring, sack, or turnover result.

    A result with no yard figure is NULL. ``0`` is stored only when the text
    itself contains a zero. The outcome parser's ``yards=0`` placeholder for
    an interception, fumble, incomplete, or stop is not a yardage.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    parsed = parse_outcome(raw)
    if parsed.kind in _YARDED_KINDS:
        match = _EXPLICIT_YARDS.search(raw)
        if match is None:
            return None
        number = int(match.group(1))
        if parsed.kind == "sack":
            return -abs(number)
        if parsed.kind == "td":
            return abs(number)
        if match.group(1).startswith("-"):
            return -abs(number)
        return number
    if parsed.kind in {"gain", "loss"} and parsed.yards is not None:
        return int(parsed.yards)
    return None


def coerce_logged_yards(result: str | None, yards: int | None) -> int | None:
    """Keep an explicit yardage. Drop a ``0`` that the text does not state."""
    derived = yards_for_log(result)
    if yards is None:
        return derived
    if isinstance(yards, bool):
        return derived
    try:
        number = int(yards)
    except (TypeError, ValueError):
        return derived
    if number == 0 and derived is None:
        return None
    return number


def penalty_for_log(text: str | None, explicit: str | None = None) -> str | None:
    """Penalty handling. Silence is NULL, which is not ``no_penalty``."""
    token = _text(explicit)
    if token is not None:
        lowered = token.lower().replace(" ", "_")
        if lowered in _PENALTY_VALUES or lowered == "unknown":
            return None if lowered == "unknown" else lowered
        return None
    raw = text or ""
    if _NO_PENALTY.search(raw):
        return "no_penalty"
    if _PENALTY_DECLINED.search(raw):
        return "declined"
    if _PENALTY_ACCEPTED.search(raw):
        return "accepted"
    return None


def drive_id_from_text(text: str | None) -> str | None:
    """An explicit drive id. A play named Drive is not a drive id."""
    match = _DRIVE_ID.search(text or "")
    if match is None:
        return None
    return match.group(1)


def format_game_clock(seconds: int | None, raw: str | None = None) -> str | None:
    if not _blank(raw):
        return str(raw).strip()
    if seconds is None or isinstance(seconds, bool):
        return None
    try:
        whole = int(seconds)
    except (TypeError, ValueError):
        return None
    if whole < 0:
        return None
    return f"{whole // 60}:{whole % 60:02d}"


def context_from_situation(sit: Any) -> dict[str, Any]:
    """Pre-snap drive, clock, quarter, and score. Missing values stay None."""
    if sit is None:
        return {
            "drive_id": None,
            "quarter": None,
            "game_clock": None,
            "score_for": None,
            "score_against": None,
        }
    extras = getattr(sit, "extras", None) or {}
    if not isinstance(extras, dict):
        extras = {}
    raw = getattr(sit, "raw", None)
    drive = _text(extras.get("drive_id")) or drive_id_from_text(str(raw or ""))
    clock_seconds = extras.get("clock_seconds")
    quarter = _int_or_none(extras.get("quarter"))
    return {
        "drive_id": drive,
        "quarter": quarter,
        "game_clock": format_game_clock(clock_seconds if isinstance(clock_seconds, int) else _int_or_none(clock_seconds), extras.get("game_clock")),
        "score_for": _int_or_none(getattr(sit, "score_us", None)),
        "score_against": _int_or_none(getattr(sit, "score_them", None)),
    }


def _mapping(decision: Any) -> dict[str, Any]:
    if decision is None:
        return {}
    if hasattr(decision, "__dataclass_fields__"):
        from cfb_coach.madden.model.schema import to_dict

        payload = to_dict(decision)
        return payload if isinstance(payload, dict) else {}
    if isinstance(decision, dict):
        return decision
    return {}


def recommended_adjustment_id(decision: Any) -> str | None:
    """The adjustment the coach showed. This is not an execution."""
    payload = _mapping(decision)
    macro = payload.get("macro") or {}
    if isinstance(macro, dict):
        macro_id = _text(macro.get("macro_id") or macro.get("id"))
        if macro_id:
            return macro_id
    final = payload.get("final_pick") or {}
    if isinstance(final, dict):
        adjustment = _text(final.get("adjustment_id"))
        if adjustment:
            return adjustment
    return None


def recommended_from_call(call: Any) -> str | None:
    """Displayed adjustment on a live call object. Not an execution."""
    if call is None:
        return None
    adjustment = getattr(call, "adjustment", None)
    if isinstance(adjustment, dict):
        token = _text(adjustment.get("id"))
        if token:
            return token
    macro = getattr(call, "macro", None)
    token = _text(macro)
    if token:
        return token
    action = getattr(call, "ml_offense_action", None)
    if isinstance(action, dict):
        kind = str(action.get("kind") or "").strip().lower()
        if kind and kind != "none":
            return _text(action.get("id"))
    return None


def executed_adjustment_for_log(
    executed_id: str | None,
    *,
    confirmed: bool,
) -> str | None:
    """Store an executed adjustment only when the caller confirms it ran."""
    if not confirmed:
        return None
    return _text(executed_id)


def decision_log_values(decision: Any) -> dict[str, Any]:
    """Accepted and propensity fields, plus the recommended adjustment.

    Keys absent from a plain dict are omitted so an update does not wipe a
    value the caller did not send. A schema decision always has the keys.
    """
    payload = _mapping(decision)
    values: dict[str, Any] = {}
    if "accepted" in payload:
        accepted = payload.get("accepted")
        values["accepted"] = None if _blank(accepted) else str(accepted)
    if "behavior_propensity" in payload:
        raw = payload.get("behavior_propensity")
        if raw is None or isinstance(raw, bool):
            values["behavior_propensity"] = None
        else:
            try:
                values["behavior_propensity"] = float(raw)
            except (TypeError, ValueError):
                values["behavior_propensity"] = None
    if "propensity_method" in payload:
        method = payload.get("propensity_method")
        values["propensity_method"] = None if _blank(method) else str(method)
    if "propensity_override" in payload:
        flag = payload.get("propensity_override")
        values["propensity_override"] = None if flag is None else int(bool(flag))
    if "propensity_override_reason" in payload:
        reason = payload.get("propensity_override_reason")
        values["propensity_override_reason"] = None if _blank(reason) else str(reason)
    recommended = recommended_adjustment_id(decision)
    if recommended is not None:
        values["recommended_adjustment_id"] = recommended
    return values


def current_run_flags() -> dict[str, bool]:
    """Effective VOD flags for this process. Reading them does not change them."""
    from cfb_coach.vod_model.flags import book_frozen, prior_enabled

    return {
        "no_vod_prior": not prior_enabled(),
        "freeze_vod_book": bool(book_frozen()),
    }


def git_revision(cwd: Path | None = None) -> tuple[str | None, str | None]:
    """``(commit, branch)``. Missing git metadata stays NULL."""
    root = cwd or Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    if not commit:
        commit = None
    if not branch or branch == "HEAD":
        branch = branch or None
    return commit, branch


def playbook_version_of(db: Any) -> str | None:
    """Applied offense and defense book revisions. No book stays NULL."""
    if db is None or not hasattr(db, "get_meta"):
        return None
    raw = db.get_meta("active_playbook_json")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    applied = payload.get("applied") if isinstance(payload, dict) else None
    if not isinstance(applied, dict):
        return None
    parts: list[str] = []
    for side in ("offense", "defense"):
        record = applied.get(side) or {}
        if not isinstance(record, dict):
            continue
        name = record.get("name") or record.get("mode")
        rev = record.get("rev")
        if name is None and rev is None:
            continue
        parts.append(f"{side}:{name}@rev{rev}")
    return ",".join(parts) or None


def record_session_run_config(
    db: Any,
    session_id: str,
    *,
    flags: dict[str, Any] | None = None,
    mode: str | None = None,
    model_id: str | None = None,
    playbook_version: str | None = None,
) -> None:
    """Write one run-config row for a session. A second call keeps the first."""
    if db is None or not session_id or not hasattr(db, "log_ml_run_config"):
        return
    from cfb_coach.madden.model.inference import resolve_mode, resolve_model_version

    resolved_mode = mode if mode is not None else resolve_mode(db).value
    resolved_model = model_id if model_id is not None else resolve_model_version(db)
    merged = current_run_flags()
    if flags:
        merged.update(flags)
    commit, branch = git_revision()
    book = playbook_version if playbook_version is not None else playbook_version_of(db)
    db.log_ml_run_config(
        session_id,
        mode=resolved_mode,
        model_id=resolved_model or None,
        flags=merged,
        git_commit=commit,
        git_branch=branch,
        playbook_version=book,
    )


def apply_seal_logging(
    db: Any,
    tracker: Any,
    *,
    decision_id: int | None,
    sit: Any,
    call: Any,
    is_new: bool,
) -> None:
    """Stamp pre-snap context, and close a call that was replaced before a snap."""
    if db is None:
        return
    if is_new and tracker is not None:
        abandoned = getattr(tracker, "abandoned_decision_row_id", None)
        if abandoned is not None and abandoned != decision_id and hasattr(db, "mark_call_no_snap"):
            db.mark_call_no_snap(abandoned)
    if decision_id is not None and hasattr(db, "stamp_call_context"):
        db.stamp_call_context(
            decision_id,
            context_from_situation(sit),
            recommended_adjustment_id=recommended_from_call(call),
        )


def orphan_call_ids(rows: list[dict[str, Any]]) -> list[int]:
    """Decision ids that are neither linked to a snap/outcome nor explicitly closed."""
    orphans: list[int] = []
    for row in rows:
        link = row.get("snap_link")
        has_snap = bool(row.get("has_snap"))
        has_outcome = bool(row.get("has_outcome"))
        if link == NO_SNAP_RECORDED and not has_snap and not has_outcome:
            continue
        if link == LINKED and (has_snap or has_outcome):
            continue
        orphans.append(int(row["id"]))
    return orphans
