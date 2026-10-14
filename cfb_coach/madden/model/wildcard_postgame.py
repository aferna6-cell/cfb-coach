"""Wild Card / issue #75 postgame reconciliation helpers.

Does not invent snaps. Interception attribution stays ``unknown`` unless the
caller supplies supporting evidence. Missing drive/clock/score fields are
reported, not filled with guesses.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from cfb_coach.outcome import parse_outcome

AUDIT_SCHEMA = "madden.wildcard_postgame_audit.v1"
_INT_HINT = re.compile(r"\b(int|intercept|pick(?:ed|six)?)\b", re.I)


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _side(row: Mapping[str, Any]) -> str:
    return str(row.get("side") or "").strip().lower()


def _result_text(row: Mapping[str, Any]) -> str:
    for key in ("result", "outcome", "outcome_text", "raw_outcome"):
        if not _blank(row.get(key)):
            return str(row.get(key))
    payload = row.get("outcome_json")
    if isinstance(payload, Mapping) and not _blank(payload.get("raw")):
        return str(payload.get("raw"))
    if isinstance(payload, str) and payload.strip():
        return payload
    return ""


def _is_int_event(row: Mapping[str, Any]) -> bool:
    text = _result_text(row)
    parsed = parse_outcome(text)
    if parsed.kind == "int":
        return True
    kind = str(row.get("outcome_kind") or row.get("kind") or "").strip().lower()
    if kind in {"int", "interception"}:
        return True
    return bool(text and _INT_HINT.search(text))


def _field_completeness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    missing = {
        "drive_id": 0,
        "game_clock": 0,
        "score_for": 0,
        "score_against": 0,
        "accepted": 0,
        "executed_verification": 0,
    }
    for row in rows:
        for key in missing:
            value = row.get(key)
            if key == "accepted":
                if _blank(value) or str(value).strip().lower() == "unknown":
                    missing[key] += 1
            elif _blank(value):
                missing[key] += 1
    return {
        "n": total,
        "missing": missing,
        "complete_drive_id": total - missing["drive_id"],
        "complete_game_clock": total - missing["game_clock"],
        "complete_score": total - max(missing["score_for"], missing["score_against"]),
    }


def reconcile_interceptions(
    *,
    snaps: Sequence[Mapping[str, Any]] = (),
    decisions: Sequence[Mapping[str, Any]] = (),
    outcomes: Sequence[Mapping[str, Any]] = (),
    reported_offense_ints: int | None = 3,
    user_reported_ints: int | None = 4,
) -> dict[str, Any]:
    """Compare logged INT events with the issue aggregate and the user report.

    Never invents a fourth snap. Attribution is ``unknown`` unless a row
    already carries an explicit attribution that is not playcall blame.
    """
    offense_snap_ints = [
        row for row in snaps
        if _side(row) in ("offense", "o") and _is_int_event(row)
    ]
    defense_forced = [
        row for row in snaps
        if _side(row) in ("defense", "d") and _is_int_event(row)
    ]
    # Decisions / outcomes may hold an INT that never received a snap link
    # (superseded / no_result). Those are candidates for the missing event.
    orphan_ints = []
    for row in list(decisions) + list(outcomes):
        if not _is_int_event(row):
            continue
        if not _blank(row.get("snap_id")) or not _blank(row.get("linked_snap_id")):
            continue
        orphan_reason = str(row.get("orphan_reason") or "").strip().lower()
        snap_link = str(row.get("snap_link") or "").strip().lower()
        if orphan_reason in {"superseded", "no_result"} or snap_link in {
            "no_snap_recorded", "superseded", "no_result",
        }:
            orphan_ints.append(dict(row))
        elif _blank(row.get("snap_id")) and _side(row) in ("offense", "o", ""):
            orphan_ints.append(dict(row))

    logged = len(offense_snap_ints)
    candidates = logged + len(orphan_ints)
    events = []
    for row in offense_snap_ints:
        events.append({
            "source": "offense_snap",
            "snap_id": row.get("snap_id") or row.get("id"),
            "result": _result_text(row),
            "attribution": row.get("attribution") or "unknown",
            "play": row.get("play") or row.get("executed_play"),
            "note": (
                "Individual read/execution mistakes are not assumed to be "
                "bad playcalls without supporting evidence."
            ),
        })
    for row in orphan_ints:
        events.append({
            "source": "orphan_decision_or_outcome",
            "decision_id": row.get("decision_id") or row.get("id"),
            "orphan_reason": row.get("orphan_reason") or row.get("snap_link"),
            "result": _result_text(row),
            "attribution": "unknown",
            "note": (
                "Logged without a snap link. Candidate for a missing INT "
                "event; not invented as a new snap."
            ),
        })

    discrepancy = None
    if user_reported_ints is not None and reported_offense_ints is not None:
        if user_reported_ints != reported_offense_ints:
            discrepancy = {
                "issue_recorded": reported_offense_ints,
                "user_reported": user_reported_ints,
                "logged_offense_snap_ints": logged,
                "orphan_int_candidates": len(orphan_ints),
                "reconcilable_without_invention": candidates >= user_reported_ints,
                "action": (
                    "Inspect orphan INT-linked decisions/outcomes; do not "
                    "silently rewrite or invent a snap."
                    if candidates >= user_reported_ints else
                    "Logged evidence is short of the user report; leave the "
                    "count unresolved rather than inventing a snap."
                ),
            }

    return {
        "offense_snap_interceptions": logged,
        "defense_forced_interceptions": len(defense_forced),
        "orphan_int_candidates": len(orphan_ints),
        "events": events,
        "issue_recorded_offense_ints": reported_offense_ints,
        "user_reported_ints": user_reported_ints,
        "discrepancy": discrepancy,
        "attribution_policy": (
            "Preserve unknown attribution unless supporting evidence exists. "
            "Do not assume every interception demonstrates a bad playcall."
        ),
    }


def audit_wildcard_session(
    *,
    snaps: Sequence[Mapping[str, Any]] = (),
    decisions: Sequence[Mapping[str, Any]] = (),
    outcomes: Sequence[Mapping[str, Any]] = (),
    session: Mapping[str, Any] | None = None,
    reported_final: Mapping[str, Any] | None = None,
    reported_offense_ints: int | None = 3,
    user_reported_ints: int | None = 4,
    armed_defense_macros: Sequence[str] = (),
) -> dict[str, Any]:
    """Issue #75 field audit: INTs, drive/clock/score, unarmed macros."""
    session = dict(session or {})
    last_snap_score = None
    for row in reversed(list(snaps)):
        us, them = row.get("score_for"), row.get("score_against")
        if us is None:
            us = row.get("score_us")
        if them is None:
            them = row.get("score_them")
        if us is not None and them is not None:
            last_snap_score = {"for": us, "against": them, "snap_id": row.get("snap_id") or row.get("id")}
            break

    stored_final = None
    for key in ("final_score", "score", "result_score"):
        raw = session.get(key)
        if isinstance(raw, Mapping) and raw:
            stored_final = dict(raw)
            break
        if isinstance(raw, str) and raw.strip():
            stored_final = {"raw": raw.strip()}
            break

    shown_macros = []
    unarmed_shown = []
    armed = {str(item).strip() for item in armed_defense_macros if str(item).strip()}
    for row in list(snaps) + list(decisions):
        if _side(row) not in ("defense", "d", ""):
            continue
        macro = (
            row.get("recommended_adjustment_id")
            or row.get("macro")
            or row.get("adjustment_id")
        )
        if isinstance(macro, Mapping):
            macro = macro.get("macro_id") or macro.get("id")
        token = None if _blank(macro) else str(macro).strip()
        if not token or token.lower() in {"none", "no adj", "no_adj"}:
            continue
        shown_macros.append(token)
        if armed and token not in armed:
            unarmed_shown.append({
                "macro": token,
                "snap_id": row.get("snap_id") or row.get("id"),
                "reason": "shown_but_not_in_armed_defense_list",
            })

    ints = reconcile_interceptions(
        snaps=snaps,
        decisions=decisions,
        outcomes=outcomes,
        reported_offense_ints=reported_offense_ints,
        user_reported_ints=user_reported_ints,
    )
    return {
        "schema": AUDIT_SCHEMA,
        "completeness": _field_completeness(list(snaps)),
        "interceptions": ints,
        "scoring": {
            "last_logged_snap_score": last_snap_score,
            "session_stored_final": stored_final,
            "reported_final": dict(reported_final) if reported_final else None,
            "note": (
                "Final points after the last logged snap must come from "
                "session close / end_game, not invented snap rows."
            ),
        },
        "macros": {
            "armed_defense": sorted(armed),
            "shown_defense": sorted(set(shown_macros)),
            "unarmed_shown": unarmed_shown,
            "policy": (
                "Never recommend a macro that was not physically created, "
                "verified, and armed."
            ),
        },
        "invented_snaps": False,
        "auto_promote": False,
    }
