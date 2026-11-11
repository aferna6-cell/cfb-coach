"""Per-snap Ravens defensive call: legal filter -> pre-snap opponent belief -> score every complete
candidate -> pick the lowest expected defensive loss -> re-validate -> USER -> 3-line card.

The live card shows ONLY:
    PLAY: <Formation> — <Play>
    USER: <defender> | UNVERIFIED
    MACRO: <required macro, and Zone It only when selected>     (line omitted when none)
Full scores, beliefs, uncertainty and the decision trace stay on ``call.ravens_defense``.
There is no fallback to another formation or caller: an empty legal set returns NO_LEGAL_RAVENS_CALL.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense import legality as lg
from cfb_coach.madden.ravens_defense import opponent as op
from cfb_coach.madden.ravens_defense import users as us
from cfb_coach.madden.ravens_defense.concepts import CLASSES
from cfb_coach.madden.ravens_defense.confirmations import Confirmations

LABEL = "RAVENS_DEFENSE_ML"
LATENCY_TARGET_MS = 150.0


@dataclass
class SnapCtx:
    """The pre-snap facts of the CURRENT snap (no result, no post-play concept)."""
    session: str
    opponent: str
    order: int
    down: int | None
    distance: int | None
    yards_to_goal: int | None
    shape: str | None = None
    closed: bool = False
    concept: str | None = None


def decide(model: Any, *, legal: Sequence[rc.Candidate], snap: SnapCtx, same_game: Sequence[Any],
           history: op.History, conf: Confirmations, cat: rc.RavensCatalog | None = None,
           installed: Mapping[str, Sequence[str]] | None = None, assume_setup: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    cat = cat or model.cat
    if not legal:
        return {"status": "NO_LEGAL_RAVENS_CALL", "ranked": [], "latency_ms": 0.0}
    earlier = [r for r in same_game if r.order < snap.order and r.closed]
    belief = op.belief(model.opponent, model.belief_method, history, earlier, snap)
    ctx = op.context_tokens(snap.down, snap.distance, snap.yards_to_goal)
    order = {c.candidate_id: i for i, c in enumerate(cat.candidates)}
    scored = [model.outcome.score(c, belief, ctx) for c in legal]
    scored.sort(key=lambda s: (s["defensive_loss"], order[s["candidate_id"]]))
    top = cat.by_id()[scored[0]["candidate_id"]]
    # re-validate the chosen candidate with the same filter before anything is displayed
    v = lg.check(cat, top, installed=installed, conf=conf, assume_setup=assume_setup)
    if not v.legal:
        return {"status": "NO_LEGAL_RAVENS_CALL", "ranked": scored, "reasons": list(v.reasons),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}
    user = us.resolve(cat, top, snap.shape, conf)
    best_without_zone_it = next((s for s in scored if not cat.by_id()[s["candidate_id"]].optional_macro), None)
    return {
        "status": "CALL", "candidate": top.as_dict(), "macro_display": top.macro_display(cat),
        "user": user, "belief": {c: round(belief[k], 4) for k, c in enumerate(CLASSES)},
        "belief_method": model.belief_method, "context": ctx,
        "earlier_same_game_snaps": len(earlier),
        "ranked": scored, "runner_up": scored[1]["candidate_id"] if len(scored) > 1 else None,
        "margin_vs_runner_up": round(scored[1]["defensive_loss"] - scored[0]["defensive_loss"], 5) if len(scored) > 1 else None,
        "zone_it_selected": top.optional_macro == "ZONE_IT",
        "zone_it_margin_vs_best_without": (round(best_without_zone_it["defensive_loss"] - scored[0]["defensive_loss"], 5)
                                           if top.optional_macro and best_without_zone_it else None),
        "blitz_selected": top.is_blitz,
        "outcome_model_status": model.outcome.status,
        "setup_assumed_for_shadow": bool(assume_setup),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


def card_lines(decision: Mapping[str, Any]) -> list[str]:
    if decision.get("status") != "CALL":
        return ["PLAY: NO LEGAL RAVENS CALL (setup incomplete; run `ml ravens-defense verify-book`)"]
    c = decision["candidate"]
    lines = [f"PLAY: {c['formation']} — {c['base_play']}",
             "USER: " + (decision["user"]["user"] if decision["user"]["status"] == "LIVE_VALID" else us.UNVERIFIED)]
    if decision.get("macro_display"):
        lines.append(f"MACRO: {decision['macro_display']}")
    return lines


# ------------------------------------------------------------------ live integration
_SESSION_HISTORY: dict[tuple[str, str, str], op.History] = {}


def _live_rows(conn: Any, session_id: str | None) -> list[Any]:
    from cfb_coach.madden.ravens_defense import data
    if not session_id:
        return []
    return data.read_snaps(conn, session_filter=session_id)


def _history_for(model: Any, conn: Any, session_id: str | None) -> op.History:
    """Training history + any later, already-finished games before this session (never this game)."""
    key = (str(model.meta.get("training_cutoff_started_ts")), str(session_id), str(id(model)))
    if key in _SESSION_HISTORY:
        return _SESSION_HISTORY[key]
    hist = model.history
    try:
        from cfb_coach.madden.ravens_defense import data
        trained = set(model.meta.get("trained_sessions") or [])
        cur = conn.execute("SELECT started_ts FROM game_sessions WHERE session_id=?", (session_id,)).fetchone() if session_id else None
        start = cur[0] if cur else None
        later = [r for r in data.read_snaps(conn) if r.session not in trained and r.session != session_id
                 and (start is None or r.started_ts < start)]
        hist = hist.extend(later)
    except Exception:  # noqa: BLE001 — history extension is best effort; training history remains
        pass
    _SESSION_HISTORY.clear()
    _SESSION_HISTORY[key] = hist
    return hist


def log_decision(conn: Any, rec: Mapping[str, Any]) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS ravens_defense_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session_id TEXT, ml_snap_id TEXT, snap_seq INTEGER,
        opponent_id TEXT, status TEXT, candidate_id TEXT, call_id TEXT, formation TEXT, base_play TEXT,
        macro TEXT, zone_it INTEGER, user_status TEXT, user TEXT, model_sha256 TEXT, outcome_model_status TEXT,
        trace_json TEXT)""")
    conn.execute(
        "INSERT INTO ravens_defense_decisions (ts, session_id, ml_snap_id, snap_seq, opponent_id, status, candidate_id,"
        " call_id, formation, base_play, macro, zone_it, user_status, user, model_sha256, outcome_model_status,"
        " trace_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rec.get("ts"), rec.get("session_id"), rec.get("ml_snap_id"), rec.get("snap_seq"), rec.get("opponent_id"),
         rec.get("status"), rec.get("candidate_id"), rec.get("call_id"), rec.get("formation"), rec.get("base_play"),
         rec.get("macro"), rec.get("zone_it"), rec.get("user_status"), rec.get("user"), rec.get("model_sha256"),
         rec.get("outcome_model_status"), json.dumps(rec.get("trace") or {}, default=str)))
    try:
        conn.commit()
    except Exception:  # noqa: BLE001
        pass


def call_ravens(sit: Any, opponent_id: str, db: Any, defense_book: Mapping[str, Any] | None, *,
                artifact: str | None = None, confirmations_path: str | None = None, shadow: bool = False,
                assume_setup: bool = False) -> Any:
    """Build the live MaddenCall for one defensive snap. Never calls another defensive picker."""
    t0 = time.perf_counter()
    from cfb_coach.madden.playcaller import MaddenCall
    from cfb_coach.madden.ravens_defense import confirmations as cf
    from cfb_coach.madden.ravens_defense import mode as md
    from cfb_coach.madden.ravens_defense import model as rm
    extras = getattr(sit, "extras", None) or {}
    decision: dict[str, Any]
    try:
        cat = rc.load()
        conf = cf.load(confirmations_path)
        installed = rc.installed_two_formation_book(defense_book)
        art = artifact or (str(md.artifact_dir(db)) if md.artifact_dir(db) else None)
        if not art:
            raise ValueError("no trained Ravens defensive artifact (run `ml ravens-defense train`)")
        model = rm.load(art)
        legal, verdicts = lg.legal_set(cat, installed=installed, conf=conf, assume_setup=assume_setup)
        conn = getattr(db, "conn", None)
        session = str(extras.get("session_id") or "")
        same_game = _live_rows(conn, session) if conn is not None else []
        snap = SnapCtx(session, opponent_id, 10 ** 9, getattr(sit, "down", None), getattr(sit, "distance", None),
                       getattr(sit, "yardline", None),
                       extras.get("offense_shape") or us.parse_offense_shape(getattr(sit, "raw", "")))
        hist = _history_for(model, conn, session) if conn is not None else model.history
        decision = decide(model, legal=legal, snap=snap, same_game=same_game, history=hist, conf=conf, cat=cat,
                          installed=installed, assume_setup=assume_setup)
        if decision["status"] != "CALL":
            decision["blocked"] = {k: list(v.reasons) for k, v in verdicts.items() if not v.legal}
        decision["model_sha256"] = rm.manifest(art).get("model_sha256")
        decision["artifact"] = art
    except Exception as exc:  # noqa: BLE001 — fail closed: no call rather than an illegal one
        decision = {"status": "NO_LEGAL_RAVENS_CALL", "error": f"{type(exc).__name__}: {exc}"}
    decision["end_to_end_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    lines = card_lines(decision)
    if decision.get("status") == "CALL":
        c = decision["candidate"]
        call = MaddenCall("defense", c["formation"], c["base_play"], decision.get("macro_display") or "No macro",
                          lines[1], f"{LABEL} ({decision['outcome_model_status']})")
        call.macro = decision.get("macro_display")
    else:
        call = MaddenCall("defense", "—", "NO_LEGAL_RAVENS_CALL", "none", "USER: UNVERIFIED", LABEL)
    call.ravens_defense = {**decision, "card": lines}
    call.call_label = LABEL
    if not shadow and getattr(db, "conn", None) is not None:
        try:
            c = decision.get("candidate") or {}
            log_decision(db.conn, {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "session_id": extras.get("session_id"),
                "ml_snap_id": extras.get("snap_id") or extras.get("ml_snap_id"), "snap_seq": extras.get("snap_seq"),
                "opponent_id": opponent_id, "status": decision.get("status"), "candidate_id": c.get("candidate_id"),
                "call_id": c.get("call_id"), "formation": c.get("formation"), "base_play": c.get("base_play"),
                "macro": decision.get("macro_display"), "zone_it": int(bool(decision.get("zone_it_selected"))),
                "user_status": (decision.get("user") or {}).get("status"), "user": (decision.get("user") or {}).get("user"),
                "model_sha256": decision.get("model_sha256"), "outcome_model_status": decision.get("outcome_model_status"),
                "trace": decision})
        except Exception:  # noqa: BLE001 — logging must never change the call
            pass
    return call
