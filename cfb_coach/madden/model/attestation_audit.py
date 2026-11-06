"""Read-only audit of human-game play execution evidence.

Every human-game snap (offense and defense) is matched to the call that was
displayed, its decision record and its outcome, then placed in one tier:

``independently_verified``
    A per-snap confirmation of the executed formation and play.
``user_attested``
    The displayed call is known, the outcome is labelled, and nothing
    contradicts the owner's statement that the recommended play was run.
    ``attestation_source`` says which statement applies:
    ``standing_prospective`` (the live toggle was on when the snap was logged)
    or ``retrospective_statement`` (:data:`RETROSPECTIVE_ATTESTATION`).
    ``confidence`` is ``high`` when a decision record confirms the displayed
    call and ``medium`` when only the snap row records it.
``excluded``
    Missing call, missing outcome, or conflicting / explicitly unknown evidence.

An attested row never becomes a verified row. Nothing here writes a database.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from cfb_coach.madden.model.defense_v1 import offense_won
from cfb_coach.opponents import is_cpu_opponent

AUDIT_VERSION = "madden-attestation-audit.v1"
RETROSPECTIVE_ATTESTATION = {
    "id": "owner-statement-2026-11-06-human-games",
    "text": "In games against human opponents, I consistently ran the play recommended by the model.",
    "scope": "human-opponent games, offense and defense, displayed formation + play",
    "not_covered": "pre-snap adjustments, hot routes, macros; any snap with contrary evidence",
}

TIER_VERIFIED = "independently_verified"
TIER_ATTESTED = "user_attested"
TIER_EXCLUDED = "excluded"

_NEUTRAL_STATUS = frozenset({"", "none", "unknown", "shown", "used_recommended"})


@dataclass
class AuditRow:
    snap_id: int
    session_id: str
    opponent_id: str
    side: str
    ml_snap_id: str | None
    situation_raw: str | None
    down: Any
    distance: Any
    yardline: Any
    quarter: Any
    displayed_formation: str | None
    displayed_play: str | None
    result: str | None
    offense_success: bool | None
    tier: str
    attestation_source: str | None = None
    confidence: str | None = None
    executed_formation: str | None = None
    executed_play: str | None = None
    recommendation_source: str = ""
    reasons: list[str] = field(default_factory=list)
    adjustment_unconfirmed: bool = False


def connect_readonly(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _latest(conn: sqlite3.Connection, table: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        for r in conn.execute(f"SELECT * FROM {table} ORDER BY id ASC"):
            if r["snap_id"]:
                out[str(r["snap_id"])] = dict(r)
    except sqlite3.DatabaseError:
        pass
    return out


def _text(v: Any) -> str | None:
    s = "" if v is None else str(v).strip()
    return s or None


def _outcome_source(outcome: Mapping[str, Any] | None) -> str | None:
    if not outcome or not outcome.get("outcome_json"):
        return None
    try:
        return json.loads(outcome["outcome_json"]).get("execution_attestation_source")
    except (TypeError, ValueError):
        return None


def classify(snap: Mapping[str, Any], decision: Mapping[str, Any] | None,
             outcome: Mapping[str, Any] | None) -> AuditRow:
    """Tier one human-game snap from its own row, decision and latest outcome."""
    form, play = _text(snap.get("formation")), _text(snap.get("play"))
    status = str((outcome or {}).get("executed_status") or snap.get("executed_status") or "").lower()
    verification = str((outcome or {}).get("executed_verification") or snap.get("executed_verification") or "").lower()
    ex_form = _text((outcome or {}).get("executed_formation") or snap.get("executed_formation"))
    ex_play = _text((outcome or {}).get("executed_play") or snap.get("executed_play"))
    source = _outcome_source(outcome)
    won = offense_won(snap.get("result"), snap.get("down"), snap.get("distance"))
    row = AuditRow(
        snap_id=int(snap["id"]), session_id=str(snap.get("session_id") or ""),
        opponent_id=str(snap.get("opponent_id") or ""), side=str(snap.get("side") or ""),
        ml_snap_id=_text(snap.get("ml_snap_id")), situation_raw=_text(snap.get("situation_raw")),
        down=snap.get("down"), distance=snap.get("distance"), yardline=snap.get("yardline"),
        quarter=snap.get("quarter"), displayed_formation=form, displayed_play=play,
        result=_text(snap.get("result")), offense_success=won, tier=TIER_EXCLUDED,
    )
    if decision is not None:
        row.recommendation_source = "decision_final_pick"
    elif snap.get("ml_snap_id"):
        row.recommendation_source = "snap_call_without_decision_record"
    else:
        row.recommendation_source = "legacy_snap_call"
    rec_adj = _text(snap.get("recommended_adjustment_id"))
    row.adjustment_unconfirmed = bool(rec_adj and rec_adj != _text(snap.get("executed_adjustment_id")))

    reasons: list[str] = []
    if not form or not play:
        reasons.append("missing_displayed_call")
    if won is None:
        reasons.append("missing_outcome_label")
    if decision is not None and (
        _text(decision.get("final_play")) != play or _text(decision.get("final_formation")) != form
    ):
        reasons.append("decision_disagrees_with_displayed_call")
    if str(snap.get("play_mismatch") or "0") == "1":
        reasons.append("play_mismatch_flag")
    if str((outcome or {}).get("accepted") or snap.get("accepted") or "").lower() == "false":
        reasons.append("recommendation_not_accepted")

    verified = status == "identified" and verification == "verified" and ex_form and ex_play
    if verified and source != "standing_user_attestation":
        row.executed_formation, row.executed_play = ex_form, ex_play
        if won is None:
            row.reasons = ["missing_outcome_label"]
            return row
        row.tier = TIER_VERIFIED
        row.reasons = []
        return row
    if verified and (ex_form, ex_play) != (form, play):
        reasons.append("standing_attestation_records_a_different_play")
    if not verified and status not in _NEUTRAL_STATUS:
        reasons.append(f"executed_status_{status}")
    if not verified and source == "per_snap_user_confirmation":
        reasons.append("user_marked_execution_unknown_for_this_snap")
    if reasons:
        row.reasons = reasons
        return row
    row.tier = TIER_ATTESTED
    row.attestation_source = "standing_prospective" if verified else "retrospective_statement"
    row.confidence = "high" if row.recommendation_source == "decision_final_pick" else "medium"
    row.executed_formation, row.executed_play = form, play
    return row


def audit_rows(conn: sqlite3.Connection) -> list[AuditRow]:
    decisions = _latest(conn, "ml_decisions")
    outcomes = _latest(conn, "ml_outcomes")
    out = []
    for snap in conn.execute("SELECT * FROM snaps ORDER BY id ASC"):
        snap = dict(snap)
        if is_cpu_opponent(str(snap.get("opponent_id") or "")):
            continue
        key = str(snap.get("ml_snap_id") or "")
        out.append(classify(snap, decisions.get(key), outcomes.get(key)))
    return out


def _tier_key(r: AuditRow) -> str:
    if r.tier == TIER_ATTESTED:
        return f"{r.tier}:{r.attestation_source}:{r.confidence}"
    return r.tier


def summarize(rows: Iterable[AuditRow], *, alias: Mapping[str, str] | None = None) -> dict[str, Any]:
    rows = list(rows)
    name = (lambda o: alias.get(o, o)) if alias else (lambda o: o)

    def counts(group) -> dict[str, dict[str, int]]:
        out: dict[str, Counter] = {}
        for r in rows:
            out.setdefault(group(r), Counter())[_tier_key(r)] += 1
        return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}

    return {
        "audit_version": AUDIT_VERSION,
        "retrospective_attestation": RETROSPECTIVE_ATTESTATION,
        "snaps": len(rows),
        "by_tier": dict(sorted(Counter(_tier_key(r) for r in rows).items())),
        "by_side": counts(lambda r: r.side),
        "by_opponent": counts(lambda r: f"{name(r.opponent_id)}|{r.side}"),
        "by_game": counts(lambda r: f"{r.session_id}|{name(r.opponent_id)}|{r.side}"),
        "excluded_reasons": dict(Counter(x for r in rows if r.tier == TIER_EXCLUDED for x in r.reasons)),
        "recommendation_source": dict(Counter(r.recommendation_source for r in rows)),
        "attested_with_unconfirmed_adjustment": sum(r.adjustment_unconfirmed for r in rows if r.tier == TIER_ATTESTED),
    }


def opponent_aliases(rows: Iterable[AuditRow]) -> dict[str, str]:
    seen = sorted({r.opponent_id for r in rows})
    return {o: f"human_{i + 1}" for i, o in enumerate(seen)}


def as_dicts(rows: Iterable[AuditRow]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
