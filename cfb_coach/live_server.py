"""Localhost HTML live play window — submit outcomes + situations without the terminal.

Stdlib only (http.server). Started by `play` (default ON); `--terminal` / `--no-html`
keeps the classic typed loop.
"""

from __future__ import annotations

import contextvars
import json
import re
import socket
import threading
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from cfb_coach.browser_open import open_url
from cfb_coach.session import start_session


MakeCallFn = Callable[..., Any]
ParseSitFn = Callable[..., Any]
LearnFn = Callable[..., str]


@dataclass
class LogRow:
    formation: str
    play: str
    result: str
    look: str = ""
    call_text: str = ""
    side: str = "offense"

    def to_dict(self) -> dict[str, Any]:
        return {
            "formation": self.formation,
            "play": self.play,
            "result": self.result,
            "look": self.look,
            "call_text": self.call_text,
            "side": self.side,
            "label": f"{self.formation} — {self.play}"
            + (f" · {self.result}" if self.result else "")
            + (f" vs {self.look}" if self.look else ""),
        }


_RAW_DOWN = re.compile(
    r"\b([1-4])(?:st|nd|rd|th)?\s*(?:&|and)\s*(?:\d+|goal|g|inches)\b",
    re.I,
)
_RAW_SPOT = re.compile(
    r"\b(?:(?:my|our|own|opp(?:onent)?s?|their)\s*(?:yl|yard\s*line)?\s*\d{1,2}"
    r"|(?:yl|yardline)\s*\d{1,2})\b",
    re.I,
)


def _sync_raw_to_spot(sit: Any) -> None:
    """Rewrite the typed down and yardline so the logged line matches the snap.

    The form often still shows the previous down. The numeric spot is already the
    ball after the result; the raw line was left behind, so ``situation_raw`` was
    one snap late. Coverage and concept words stay.
    """
    raw = getattr(sit, "raw", "") or ""
    down = getattr(sit, "down", None)
    dist = getattr(sit, "distance", None)
    if not raw or down is None or dist is None:
        return
    updated = _RAW_DOWN.sub(f"{int(down)}&{int(dist)}", raw, count=1)
    yl = getattr(sit, "yardline", None)
    if yl is not None and _RAW_SPOT.search(updated):
        phrase = f"opp {100 - int(yl)}" if int(yl) > 50 else f"my {int(yl)}"
        updated = _RAW_SPOT.sub(phrase, updated, count=1)
    if updated != raw:
        sit.raw = updated


def _same_spot(sit: Any, other: Any) -> bool:
    """True when the form still shows the down and yardline of the snap just logged."""
    return (
        getattr(sit, "down", None) == getattr(other, "down", None)
        and getattr(sit, "distance", None) == getattr(other, "distance", None)
        and getattr(sit, "yardline", None) == getattr(other, "yardline", None)
    )


@dataclass
class LivePlayController:
    """In-memory live session shared by the HTTP handlers."""

    db: Any
    opponent_id: str
    make_call: MakeCallFn
    parse_situation: ParseSitFn
    learn_summary: LearnFn
    brand: str = "CFB Coach"
    play_cmd: str = "cfb-coach play"
    dynasty: str = "alabama"
    dynasty_label: str = ""  # e.g. "Ohio State (experimental)"
    dynasty_source: str = ""  # where the dynasty came from (--dynasty / latest prep / default)
    cpu_only: bool = False
    default_side: str = "offense"
    session_id: str = ""
    last_call: Any | None = None
    last_sit: Any | None = None
    last_coverage: str | None = None
    last_concept: str | None = None
    call_text: str = "waiting for first situation…"
    heard: str = ""
    log: list[LogRow] = field(default_factory=list)
    ended: bool = False
    retrain_summary: str = ""
    result_wl: str | None = None
    score: str | None = None  # final game-over score (winner-first); not the live us-them score
    quarter: int | None = None
    live_score: Any | None = None  # GameScore us-them, persists across snaps
    lock: threading.Lock = field(default_factory=threading.Lock)
    # v1.13 (CFB): playbook-of-record hooks — optional so Madden is unaffected
    book_info: Callable[[], dict[str, Any] | None] | None = None
    book_apply: Callable[[int | None], dict[str, Any] | None] | None = None
    # Madden ML: optional shared shadow scorer. Never mutates the displayed call.
    # Signature: (sit, call, *, game_id, snap_id, snap_seq, session_id, is_new) -> row_id|None
    shadow_evaluate: Callable[..., int | None] | None = None
    enable_execution_verify: bool = False
    ml_tracker: Any | None = None
    # Durable HTTP idempotency: request_key -> last response (also persisted in DB).
    _idempotency_mem: dict[str, dict[str, Any]] = field(default_factory=dict)

    def book_state(self) -> dict[str, Any] | None:
        if self.book_info is None:
            return None
        try:
            return self.book_info()
        except Exception as exc:  # noqa: BLE001 — never break live play
            return {"error": f"{type(exc).__name__}: {exc}"}

    def apply_book(self, rev: int | None = None) -> dict[str, Any]:
        if self.book_apply is None:
            return {"ok": False, "error": "no playbook of record for this game"}
        with self.lock:
            rec = self.book_apply(rev)
        if not rec:
            return {"ok": False, "error": "no matching pending playbook edits (run prep, or they were already applied)",
                    "state": self.state()}
        return {"ok": True, "applied_rev": rec.get("rev"), "dynasty": self.dynasty, "state": self.state()}

    def start(self) -> None:
        sess = start_session(
            self.opponent_id, dynasty=self.dynasty, db=self.db, notes="html-live"
        )
        self.session_id = sess.session_id

    def state(self) -> dict[str, Any]:
        return {
            "opponent_id": self.opponent_id,
            "brand": self.brand,
            "play_cmd": self.play_cmd,
            "session_id": self.session_id,
            "dynasty": self.dynasty,
            "dynasty_label": self.dynasty_label or self.dynasty,
            "dynasty_source": self.dynasty_source,
            "call_text": self.call_text,
            "heard": self.heard,
            "cpu_only": self.cpu_only,
            "default_side": self.default_side,
            "has_pending_call": self.last_call is not None and not self.ended,
            "ended": self.ended,
            "result_wl": self.result_wl,
            "score": self.score,
            "live_score": (
                None if self.live_score is None else {
                    "us": self.live_score.us,
                    "them": self.live_score.them,
                    "label": self.live_score.label,
                }
            ),
            "retrain_summary": self.retrain_summary,
            "log": [r.to_dict() for r in self.log],
            "book": self.book_state(),
            "macro": self.macro_state(),
            "quarter": self.quarter,
            "benched": self.benched_state(),
            "ball": self._book().spot.as_dict(),
            "execution_verify": bool(self.enable_execution_verify),
            "pending_recommendation": self._pending_recommendation(),
            "pending_offense_action": self._pending_offense_action(),
            "ml_experimental": self._ml_experimental_state(),
        }

    def _pending_recommendation(self) -> dict[str, Any] | None:
        call = self.last_call
        if call is None or self.ended:
            return None
        return {
            "formation": getattr(call, "formation", None),
            "play": getattr(call, "play", None),
            "macro": getattr(call, "macro", None),
            "side": getattr(call, "side", None),
        }

    def _pending_offense_action(self) -> dict[str, Any] | None:
        """Research-grounded action details are optional and never the main call."""
        call = self.last_call
        if call is None or self.ended or str(getattr(call, "side", "")).startswith("d"):
            return None
        info = getattr(call, "macro_info", None)
        if isinstance(info, dict) and getattr(call, "macro", None):
            return {
                "kind": "macro", "id": getattr(call, "macro"),
                "label": info.get("name") or info.get("id"),
                "buttons": info.get("buttons") or "",
                "why": info.get("why") or "",
            }
        adj = getattr(call, "adjustment", None)
        if isinstance(adj, dict) and adj.get("id"):
            return {
                "kind": "adjustment", "id": adj["id"],
                "label": adj.get("label") or adj["id"],
                "buttons": adj.get("buttons") or "",
                "why": adj.get("why") or "",
            }
        return None

    def _ml_experimental_state(self) -> dict[str, Any] | None:
        """Heuristic vs ML explanation when experimental mode produced the call."""
        call = self.last_call
        if call is None or self.ended:
            return None
        info = getattr(call, "ml_experimental", None)
        if not isinstance(info, dict):
            return None
        return {
            "heuristic": f"{info.get('heuristic_formation')}/{info.get('heuristic_play')}",
            "ml": f"{info.get('ml_formation')}/{info.get('ml_play')}",
            "final": f"{info.get('final_formation')}/{info.get('final_play')}",
            "evidence_quality": info.get("evidence_quality"),
            "explanation": info.get("explanation"),
            "model_version": info.get("model_version"),
            "knowledge_version": info.get("knowledge_version"),
            "data_version": info.get("data_version"),
            "probability": info.get("probability"),
            "fell_back": bool(info.get("fell_back")),
        }

    @staticmethod
    def _format_call_text(call: Any) -> str:
        """Call text plus experimental ML explanation when present."""
        text = call.format() if hasattr(call, "format") else str(call)
        info = getattr(call, "ml_experimental", None)
        if isinstance(info, dict) and info.get("explanation"):
            eq = info.get("evidence_quality") or "unknown"
            text += (
                f"\n  ML experimental [{eq}]: {info['explanation']}"
                f"\n  heuristic kept visible: {info.get('heuristic_formation')}/{info.get('heuristic_play')}"
                f"\n  (opt-in pilot — not a validated competitive model)"
            )
        return text

    def _ensure_tracker(self) -> Any:
        if self.ml_tracker is not None:
            return self.ml_tracker
        needs = bool(self.enable_execution_verify or self.shadow_evaluate is not None)
        # Experimental calls carry a pending decision that must be sealed+committed.
        if getattr(self.last_call, "_pending_ml_decision", None) is not None:
            needs = True
        if not needs:
            # Still allocate identity when experimental mode is active.
            try:
                from cfb_coach.madden.model import inference as ml_inference
                from cfb_coach.madden.model.schema import CoachingMode

                if ml_inference.resolve_mode(self.db) is CoachingMode.EXPERIMENTAL:
                    needs = True
            except Exception:  # noqa: BLE001
                pass
        if not needs:
            return None
        from cfb_coach.madden.model.identity import LiveDecisionTracker, next_seq_from_db

        sid = self.session_id or ""
        if not sid:
            return None
        self.ml_tracker = LiveDecisionTracker.from_session(
            sid, next_seq=next_seq_from_db(self.db, sid)
        )
        return self.ml_tracker

    def _seal_and_shadow(self, sit: Any, call: Any) -> tuple[str | None, int | None, int | None]:
        """Allocate snap identity; commit experimental decision or run shadow once.

        Experimental decisions are produced in ``make_call`` without a snap id.
        This is the single identity-aware logging point.
        """
        # Ensure tracker when this call has a pending experimental decision.
        if getattr(call, "_pending_ml_decision", None) is not None:
            self.enable_execution_verify = True
        tracker = self._ensure_tracker()
        if tracker is None:
            return None, None, None
        snap_id, seq, _key, is_new = tracker.seal_call(
            side=getattr(call, "side", "offense") or "offense",
            formation=getattr(call, "formation", None),
            play=getattr(call, "play", None),
            situation_raw=getattr(sit, "raw", None),
        )
        row_id = tracker.pending_ml_decision_row_id
        # Commit experimental decision with sealed identity (once per new seal).
        if is_new and getattr(call, "_pending_ml_decision", None) is not None:
            try:
                from cfb_coach.madden.model.experimental_live import commit_experimental_decision

                row_id = commit_experimental_decision(
                    self.db,
                    call,
                    game_id=tracker.game_id,
                    snap_id=snap_id,
                    snap_seq=seq,
                    session_id=self.session_id,
                )
            except Exception:  # noqa: BLE001 — never break live play
                row_id = None
            tracker.bind_decision_row(row_id)
        elif self.shadow_evaluate is not None and is_new:
            try:
                row_id = self.shadow_evaluate(
                    sit,
                    call,
                    game_id=tracker.game_id,
                    snap_id=snap_id,
                    snap_seq=seq,
                    session_id=self.session_id,
                    is_new=is_new,
                )
            except Exception:  # noqa: BLE001 — never break live play
                row_id = None
            tracker.bind_decision_row(row_id)
        return snap_id, seq, row_id

    def benched_state(self) -> dict[str, list[str]]:
        """Calls benched for the rest of this half, per side."""
        from cfb_coach.ingame import bench_report

        out: dict[str, list[str]] = {}
        for side in ("offense",) if self.cpu_only else ("offense", "defense"):
            try:
                rep = bench_report(self.db, self.session_id, side, self.quarter)
            except Exception:  # noqa: BLE001 — never break live play
                continue
            if rep.benched:
                out[side] = rep.lines()
        return out

    def _set_quarter(self, quarter: Any) -> None:
        try:
            q = int(quarter) if quarter not in (None, "") else None
        except (TypeError, ValueError):
            q = None
        if q is not None and 1 <= q <= 5:
            self.quarter = q

    def set_live_score(self, us: Any, them: Any) -> None:
        """Set or clear the session score. Both blank clears; one blank is ignored."""
        from cfb_coach.game_score import GameScore

        def _blank(v: Any) -> bool:
            return v is None or str(v).strip() == ""

        if _blank(us) and _blank(them):
            self.live_score = None
            return
        if _blank(us) or _blank(them):
            return
        try:
            self.live_score = GameScore(int(us), int(them))
        except (TypeError, ValueError):
            return

    def _stamp(self, sit: Any) -> Any:
        from cfb_coach.game_score import LiveContext, absorb_and_stamp

        extras = getattr(sit, "extras", None)
        if isinstance(extras, dict):
            extras["session_id"] = self.session_id or None
            # Quarter from the dropdown wins unless the sit line itself named one.
            if extras.get("quarter") in (None, "") and self.quarter is not None:
                extras["quarter"] = self.quarter
        ctx = LiveContext(score=self.live_score, quarter=self.quarter)
        absorb_and_stamp(sit, ctx)
        self.live_score = ctx.score
        if ctx.quarter is not None:
            self.quarter = ctx.quarter
        return sit

    def macro_state(self) -> dict[str, Any] | None:
        """v1.15: the Active-8 macro on the pending call (CFB offense; Madden defense) or None.
        v1.17 (Madden): also a pre-snap adjustment (hot route / audible / contain…) with its buttons."""
        call = self.last_call
        if call is None or self.ended:
            return None
        head = call.headline() if hasattr(call, "headline") else f"PLAY: {call.play} ({call.formation}) + MACRO: {call.macro}"
        adj = getattr(call, "adjustment", None)
        if not getattr(call, "macro", None) and adj:
            return {"id": adj.get("id"), "name": adj.get("label"), "headline": head, "key": "",
                    "press": adj.get("buttons") or "", "why": adj.get("why") or "", "fire_when": "",
                    "settings": None, "kind": "adjustment"}
        if not getattr(call, "macro", None) or getattr(call, "macro_info", None) is None:
            return None  # custom adjustments that carry macro_info
        mi = dict(getattr(call, "macro_info", None) or {})
        return {
            "id": call.macro,
            "name": mi.get("name") or call.macro,
            "headline": head,
            "key": mi.get("key") or "",
            "press": mi.get("buttons") or f"LB → {mi.get('name') or call.macro}",
            "why": mi.get("why") or "",
            "fire_when": mi.get("fire_when") or "",
            "settings": [{"section": r.get("section"),
                          "setting": "" if r.get("setting") == r.get("section") else r.get("setting"),
                          "value": r.get("value")} for r in mi.get("settings") or []],
            "kind": "macro",
        }

    def _macro_of(self, call: Any) -> str | None:
        from cfb_coach.last_snap import shown_macro

        if call is None:
            return None
        return shown_macro(call)

    def _book(self):
        from cfb_coach.last_snap import LastSnapBook

        book = getattr(self, "_snap_book", None)
        if book is None:
            book = LastSnapBook(
                self.db, self.opponent_id, self.parse_situation, session_id=self.session_id or None,
            )
            if self.live_score is not None:
                book.spot.score_us = self.live_score.us
                book.spot.score_them = self.live_score.them
            self._snap_book = book
        elif self.session_id and not book.session_id:
            book.session_id = self.session_id
        return book

    def _push_score(self) -> None:
        from cfb_coach.game_score import GameScore

        spot = self._book().spot
        if spot.score_us is None:
            return
        self.live_score = GameScore(int(spot.score_us), int(spot.score_them or 0))

    def _apply_form_score(self, score_us: Any, score_them: Any, score_set: bool) -> None:
        """Copy the score boxes onto the ball. Both blank clears the session score."""
        if not score_set:
            return

        def _blank(v: Any) -> bool:
            return v is None or str(v).strip() == ""

        self.set_live_score(score_us, score_them)
        book = self._book()
        if _blank(score_us) and _blank(score_them):
            book.spot.score_us = None
            book.spot.score_them = None
        elif self.live_score is not None:
            book.spot.score_us = self.live_score.us
            book.spot.score_them = self.live_score.them

    @staticmethod
    def _parse_snap_seq(snap_id: str | None) -> int | None:
        try:
            from cfb_coach.madden.model.identity import parse_snap_seq

            return parse_snap_seq(snap_id)
        except Exception:  # noqa: BLE001
            return None

    def _idempotency_get(self, request_key: str | None) -> dict[str, Any] | None:
        if not request_key:
            return None
        cached = self._idempotency_mem.get(request_key)
        if cached is not None:
            out = dict(cached)
            out["idempotent_replay"] = True
            return out
        if hasattr(self.db, "get_idempotent_response"):
            return self.db.get_idempotent_response(request_key)
        return None

    def _idempotency_put(
        self, request_key: str | None, endpoint: str, response: dict[str, Any]
    ) -> dict[str, Any]:
        if not request_key:
            return response
        stored = dict(response)
        stored["idempotency_key"] = request_key
        self._idempotency_mem[request_key] = stored
        if hasattr(self.db, "store_idempotent_response"):
            self.db.store_idempotent_response(
                request_key,
                endpoint=endpoint,
                response=stored,
                session_id=self.session_id or None,
            )
        return stored

    def undo_last(self, *, request_key: str | None = None) -> dict[str, Any]:
        with self.lock:
            cached = self._idempotency_get(request_key)
            if cached is not None:
                return cached
            book = self._book()
            before = len(book._undos)
            # Capture ml identity before undo so the tracker can restore pending.
            undone_ml_snap = book.ml_snap_id
            undone_decision = book.ml_decision_id
            msg = book.undo()
            if len(book._undos) < before and self.log:
                self.log.pop()
            # After voiding the outcome, re-bind the pending call identity so a
            # retry can re-log against the same snap rather than skipping ahead.
            tracker = self.ml_tracker
            if tracker is not None and undone_ml_snap and book.call is not None:
                tracker.pending_snap_id = undone_ml_snap
                tracker.pending_ml_decision_row_id = undone_decision
                tracker.pending_decision_key = tracker.decision_key(
                    side=getattr(book.call, "side", "offense") or "offense",
                    formation=getattr(book.call, "formation", None),
                    play=getattr(book.call, "play", None),
                    situation_raw=getattr(book.sit, "raw", None) if book.sit else None,
                    seq=self._parse_snap_seq(undone_ml_snap),
                )
            self._push_score()
            out = {"ok": True, "message": msg.strip(), "state": self.state()}
            return self._idempotency_put(request_key, "/api/undo", out)

    def _log_pending_result(
        self,
        outcome_raw: str,
        their: str = "",
        next_raw: str | None = None,
        *,
        executed_status: str = "unknown",
        executed_formation: str | None = None,
        executed_play: str | None = None,
        executed_macro: str | None = None,
        applied_recommended_action: bool = False,
    ) -> dict[str, Any] | None:
        """Log the snap that just ended. The form's last-play field belongs to THAT snap."""
        if not self.last_call or not self.last_sit:
            return None
        book = self._book()
        if book.call is None:
            book.remember_call(self.last_call, self.last_sit)
        closed = book.close_from_form(
            outcome_raw,
            their=their,
            next_raw=next_raw,
            executed_status=executed_status,
            executed_formation=executed_formation,
            executed_play=executed_play,
            executed_macro=executed_macro,
            applied_recommended_macro=(
                applied_recommended_action
                if str(self.brand).lower().startswith("madden") else None
            ),
        )
        if closed is None:
            return None
        side = getattr(self.last_call, "side", "") or ""
        if book.concept and side.startswith("d"):
            self.last_concept = book.concept
        if book.coverage and not side.startswith("d"):
            self.last_coverage = book.coverage
        self._push_score()
        # Link ML outcome to the sealed decision identity.
        tracker = self._ensure_tracker()
        ml_snap_id = book.ml_snap_id
        decision_id = book.ml_decision_id
        if tracker is not None:
            pending_snap, pending_dec = tracker.consume_pending()
            ml_snap_id = ml_snap_id or pending_snap
            decision_id = decision_id or pending_dec
        if ml_snap_id and hasattr(self.db, "log_ml_outcome"):
            try:
                from cfb_coach.outcome import parse_outcome

                parsed = parse_outcome(closed.get("result") or outcome_raw)
                self.db.log_ml_outcome(
                    snap_id=ml_snap_id,
                    game_id=self.session_id or None,
                    decision_id=decision_id,
                    executed_status=closed.get("executed_status") or "unknown",
                    executed_formation=closed.get("executed_formation"),
                    executed_play=closed.get("executed_play"),
                    executed_verification=(
                        closed.get("executed_verification")
                        or (
                            "verified"
                            if (closed.get("executed_status") or "") == "identified"
                            else "unknown"
                        )
                    ),
                    outcome={
                        "result": closed.get("result"),
                        "yards": parsed.yards,
                        "kind": parsed.kind,
                        "coverage_seen": book.coverage,
                        "concept_seen": book.concept,
                        "executed_macro": closed.get("executed_macro"),
                        "executed_adjustment_id": (
                            (getattr(self.last_call, "adjustment", None) or {}).get("id")
                            if applied_recommended_action
                            and (closed.get("executed_status") or "") == "identified"
                            and (closed.get("executed_play") or "") == getattr(self.last_call, "play", None)
                            else None
                        ),
                        "offense_action_explicitly_confirmed": bool(
                            applied_recommended_action
                            and (closed.get("executed_status") or "") == "identified"
                        ),
                    },
                    replace=True,
                )
            except Exception:  # noqa: BLE001
                pass
        row = LogRow(
            formation=closed["formation"],
            play=closed["play"],
            result=closed["result"],
            look=closed["look"],
            call_text=closed["call_text"],
            side=closed["side"],
        )
        self.log.append(row)
        out = row.to_dict()
        out["ml_snap_id"] = ml_snap_id
        out["executed_status"] = closed.get("executed_status")
        out["executed_formation"] = closed.get("executed_formation")
        out["executed_play"] = closed.get("executed_play")
        return out

    def _make(self, sit: Any) -> Any:
        kwargs: dict[str, Any] = {}
        # CFB + Madden accept last_coverage / last_concept
        if getattr(sit, "side", "offense") == "offense":
            kwargs["last_coverage"] = self.last_coverage
        else:
            kwargs["last_concept"] = self.last_concept
        try:
            return self.make_call(sit, **kwargs)
        except TypeError:
            return self.make_call(sit)

    def call_only(
        self,
        sit_raw: str,
        *,
        side: str | None = None,
        quarter: Any = None,
        score_us: Any = None,
        score_them: Any = None,
        score_set: bool = False,
        request_key: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            cached = self._idempotency_get(request_key)
            if cached is not None:
                return cached
            if self.ended:
                return {"ok": False, "error": "game already ended"}
            self._set_quarter(quarter)
            self._apply_form_score(score_us, score_them, score_set)
            side_use = side or self.default_side
            if self.cpu_only:
                side_use = "offense"
            sit = self._stamp(self.parse_situation(sit_raw, default_side=side_use))
            self._book().stamp_situation(sit)
            from cfb_coach.situation import format_heard

            try:
                heard = format_heard(sit)
            except Exception:
                heard = getattr(sit, "label", sit_raw)
            call = self._make(sit)
            self.last_call, self.last_sit = call, sit
            ml_snap_id, snap_seq, decision_row = self._seal_and_shadow(sit, call)
            self._book().remember_call(
                call, sit, ml_snap_id=ml_snap_id, snap_seq=snap_seq, ml_decision_id=decision_row
            )
            self.call_text = self._format_call_text(call)
            self.heard = heard
            if getattr(sit, "coverage_hint", None) and sit.side == "offense":
                self.last_coverage = sit.coverage_hint
            if getattr(sit, "concept_hint", None) and sit.side == "defense":
                self.last_concept = sit.concept_hint
            self.default_side = sit.side
            out = {"ok": True, "call_text": self.call_text, "heard": heard, "state": self.state()}
            return self._idempotency_put(request_key, "/api/call", out)

    def result_and_call(
        self,
        *,
        outcome: str,
        sit_raw: str,
        side: str | None = None,
        quarter: Any = None,
        their: str = "",
        score_us: Any = None,
        score_them: Any = None,
        score_set: bool = False,
        executed_status: str = "unknown",
        executed_formation: str | None = None,
        executed_play: str | None = None,
        executed_macro: str | None = None,
        applied_recommended_action: bool = False,
        request_key: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            cached = self._idempotency_get(request_key)
            if cached is not None:
                return cached
            if self.ended:
                return {"ok": False, "error": "game already ended"}
            self._apply_form_score(score_us, score_them, score_set)
            logged = None
            closed_sit = self.last_sit
            if self.last_call is not None and (outcome or "").strip():
                logged = self._log_pending_result(
                    outcome,
                    their=their,
                    next_raw=sit_raw,
                    executed_status=executed_status,
                    executed_formation=executed_formation,
                    executed_play=executed_play,
                    executed_macro=executed_macro,
                    applied_recommended_action=applied_recommended_action,
                )
            elif self.last_call is not None and not (outcome or "").strip():
                # Allow first snap without prior outcome
                pass
            elif (outcome or "").strip() and self.last_call is None:
                return {"ok": False, "error": "no prior call to attach outcome to"}

            self._set_quarter(quarter)
            side_use = side or self.default_side
            if self.cpu_only:
                side_use = "offense"
            sit = self._stamp(self.parse_situation(sit_raw, default_side=side_use))
            book = self._book()
            book.stamp_situation(sit)
            # The down boxes still describe the snap we just logged — the result moved the ball.
            if logged and closed_sit is not None and book.spot.down and _same_spot(sit, closed_sit):
                from cfb_coach.last_snap import refresh_marks

                sit.down = book.spot.down
                sit.distance = book.spot.distance
                if book.spot.yardline is not None:
                    sit.yardline = book.spot.yardline
                refresh_marks(sit)
                _sync_raw_to_spot(sit)
            from cfb_coach.situation import format_heard

            try:
                heard = format_heard(sit)
            except Exception:
                heard = getattr(sit, "label", sit_raw)
            call = self._make(sit)
            self.last_call, self.last_sit = call, sit
            ml_snap_id, snap_seq, decision_row = self._seal_and_shadow(sit, call)
            book.remember_call(
                call, sit, ml_snap_id=ml_snap_id, snap_seq=snap_seq, ml_decision_id=decision_row
            )
            self._push_score()
            self.call_text = self._format_call_text(call)
            self.heard = heard
            if getattr(sit, "coverage_hint", None) and sit.side == "offense":
                self.last_coverage = sit.coverage_hint
            if getattr(sit, "concept_hint", None) and sit.side == "defense":
                self.last_concept = sit.concept_hint
            self.default_side = sit.side
            out = {
                "ok": True,
                "logged": logged,
                "call_text": self.call_text,
                "heard": heard,
                "state": self.state(),
            }
            return self._idempotency_put(request_key, "/api/result_call", out)

    def end_game(
        self, *, result_wl: str, score: str = "", request_key: str | None = None
    ) -> dict[str, Any]:
        with self.lock:
            cached = self._idempotency_get(request_key)
            if cached is not None:
                return cached
            if self.ended:
                return {"ok": True, "retrain_summary": self.retrain_summary, "state": self.state()}
            wl = (result_wl or "").lower().strip()
            if wl not in ("win", "loss", "tie", "w", "l"):
                return {"ok": False, "error": "result_wl must be win or loss"}
            if wl in ("w",):
                wl = "win"
            if wl in ("l",):
                wl = "loss"
            self.result_wl = wl
            self.score = (score or "").strip()
            # Flush dangling call without outcome? leave unlogged.
            play_count = len(self.log)
            try:
                self.db.end_game_session(
                    self.session_id,
                    play_count=play_count,
                    result_wl=wl,
                    score=self.score,
                )
            except Exception:
                pass
            # Persist W/L note in meta too
            try:
                self.db.set_meta(
                    f"last_game:{self.opponent_id}",
                    json.dumps(
                        {
                            "session_id": self.session_id,
                            "result_wl": wl,
                            "score": self.score,
                            "snaps": play_count,
                        }
                    ),
                )
            except Exception:
                pass

            # Session grades (display) + full learn via learn_summary
            try:
                from cfb_coach.retrain import format_grades_summary, grade_play_vs_look

                snaps = list(self.db.get_session_snaps(self.session_id))
                grades = grade_play_vs_look(snaps)
                grade_block = "\n".join(format_grades_summary(grades)) or "  (no graded snaps)"
            except Exception:
                grades = []
                grade_block = "  (grade unavailable)"

            try:
                learn_txt = self.learn_summary()
            except Exception as exc:
                learn_txt = f"(learn error: {exc})"

            header = (
                f"# GAME OVER — {wl.upper()}"
                + (f"  {self.score}" if self.score else "")
                + f"\nvs {self.opponent_id} · session {self.session_id} · {play_count} snaps\n"
                f"## This game — play vs coverage/look\n{grade_block}\n"
            )
            self.retrain_summary = header + "\n" + (learn_txt or "")
            self.ended = True
            self.call_text = f"GAME OVER — {wl.upper()}" + (
                f"  {self.score}" if self.score else ""
            )
            out = {
                "ok": True,
                "retrain_summary": self.retrain_summary,
                "grades": grades,
                "state": self.state(),
            }
            return self._idempotency_put(request_key, "/api/end_game", out)


def _esc(s: Any) -> str:
    return (
        ("" if s is None else str(s))
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_live_html(ctrl: LivePlayController) -> str:
    compact_madden = str(ctrl.brand).lower().startswith("madden")
    brand = _esc(ctrl.brand)
    sc = ctrl.live_score
    us_v = "" if sc is None else str(sc.us)
    them_v = "" if sc is None else str(sc.them)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{brand} — Live Play</title>
<style>
  :root {{ --bg:#0e1117; --fg:#e6edf3; --muted:#8b949e; --accent:#3fb950;
    --call:#58a6ff; --danger:#f85149; --panel:#161b22; --border:#30363d; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin:0; min-height:100%; background:var(--bg); color:var(--fg);
    font-family: ui-sans-serif, system-ui, Segoe UI, sans-serif; }}
  main {{ padding: .75rem 1rem 2rem; max-width: 920px; margin: 0 auto; }}
  h1 {{ font-size: .7rem; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; margin: 0 0 .35rem; }}
  .badge {{ display:inline-block; padding:.1rem .4rem; border-radius:4px;
    background:#21262d; color:var(--accent); font-size:.7rem; }}
  .heard {{ font-family: ui-monospace, Consolas, monospace; font-size: .95rem;
    color: var(--muted); margin-bottom: .35rem; }}
  .call-label {{ font-size: .7rem; letter-spacing: .1em; color: var(--accent);
    text-transform: uppercase; font-weight: 700; margin-top: .25rem; }}
  .call {{ font-size: clamp(2rem, 4.8vw, 3rem); font-weight: 800; line-height: 1.25;
    color: var(--call); margin: .2rem 0 .85rem; white-space: pre-wrap; word-break: break-word; }}
  section {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: .75rem .9rem; margin: .75rem 0; }}
  section h2 {{ margin: 0 0 .55rem; font-size: .75rem; letter-spacing: .06em;
    text-transform: uppercase; color: var(--muted); }}
  .row {{ display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; margin: .35rem 0; }}
  label {{ font-size: .8rem; color: var(--muted); }}
  input, select {{ background:#0d1117; color:var(--fg); border:1px solid var(--border);
    border-radius: 6px; padding: .4rem .55rem; font-size: .95rem; }}
  input[type=number] {{ width: 4.2rem; }}
  input.wide {{ width: min(100%, 22rem); }}
  button {{ background:#21262d; color:var(--fg); border:1px solid var(--border);
    border-radius: 6px; padding: .4rem .7rem; font-weight: 600; cursor: pointer; }}
  button:hover {{ border-color: var(--call); }}
  button.primary {{ background:#1f6feb; border-color:#1f6feb; color:#fff; }}
  button.outcome {{ font-size: .85rem; }}
  button.danger {{ background:#3d1214; border-color: var(--danger); color:#ffa198; }}
  #log {{ max-height: 240px; overflow-y: auto; font-family: ui-monospace, Consolas, monospace;
    font-size: .82rem; }}
  #log div {{ padding: .28rem 0; border-bottom: 1px solid #21262d; color: var(--muted); }}
  #log div b {{ color: var(--fg); font-weight: 600; }}
  #summary {{ white-space: pre-wrap; font-family: ui-monospace, Consolas, monospace;
    font-size: .8rem; color: var(--fg); background:#0d1117; border-radius: 6px;
    padding: .65rem; border: 1px solid var(--border); max-height: 320px; overflow-y: auto; }}
  .err {{ color: var(--danger); font-size: .85rem; min-height: 1.1rem; }}
  .dyn {{ font-size: .85rem; color: var(--fg); margin: 0 0 .3rem; }}
  .dyn b {{ color: var(--accent); }}
  .dyn .src {{ color: var(--muted); font-size: .75rem; }}
  #macro-box {{ border: 2px solid #d29922; background: #2a1f0a; border-radius: 8px; padding: .6rem .8rem; margin: -.35rem 0 .8rem; }}
  #macro-box .mh {{ font-size: clamp(1.05rem, 2.4vw, 1.5rem); font-weight: 800; color: #f2cc60; }}
  #macro-box .mk {{ font-family: ui-monospace, Consolas, monospace; font-size: .95rem; color: var(--fg); margin-top: .3rem; }}
  #macro-box .mw {{ font-size: .8rem; color: var(--muted); margin-top: .25rem; }}
  #macro-box table {{ font-size: .82rem; border-collapse: collapse; margin-top: .35rem; }}
  #macro-box td {{ padding: .12rem .6rem .12rem 0; color: var(--fg); }}
  #macro-box td.s {{ color: var(--muted); }}
  footer {{ margin-top: 1rem; font-size: .72rem; color: var(--muted); line-height: 1.4; }}
  .sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0;
    margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap;
    border: 0; }}
  #live-admin {{ margin-top: 1rem; color: var(--muted); }}
  #live-admin > summary {{ cursor: pointer; font-size: .8rem; padding: .5rem 0; }}
  #call {{ margin: .65rem 0 1.15rem; }}
</style>
</head>
<body>
<main>
  {'<h1 class="sr-only">Live Play — ' + brand + '</h1>' if compact_madden else '<h1>Live Play <span class="badge" id="badge">' + brand + '</span> · keep sticks · type less</h1>'}
  <div class="dyn" id="dyn"{' hidden' if compact_madden else ''}>{_dyn_line(ctrl)}</div>
  <div class="heard" id="heard"{' hidden' if compact_madden else ''}>vs {_esc(ctrl.opponent_id)}</div>
  <div class="call" id="call"{' aria-live="polite"' if compact_madden else ''}>{_esc(_call_main(ctrl))}</div>
  {_macro_box_html(None if compact_madden else ctrl.macro_state())}
  <div class="heard" id="ml-experimental" hidden></div>
  <div class="err" id="err"></div>

  <section id="snap-panel">
    <h2>Last snap outcome</h2>
    <div class="row" id="outcome-btns">
      <button type="button" class="outcome" data-out="gain">gain</button>
      <input type="number" id="gain-n" value="5" title="yards gained"/>
      <button type="button" class="outcome" data-out="loss">loss</button>
      <input type="number" id="loss-n" value="2" title="yards lost"/>
      <button type="button" class="outcome" data-out="incomplete">incomplete</button>
      <button type="button" class="outcome" data-out="sack">sack</button>
      <button type="button" class="outcome" data-out="td">TD</button>
      <button type="button" class="outcome" data-out="int">INT</button>
      <button type="button" class="outcome" data-out="fumble lost">fumble lost</button>
      <button type="button" class="outcome" data-out="stop">stop</button>
      <button type="button" class="outcome" data-out="convert">convert</button>
    </div>
    <div class="row">
      <label>or free text</label>
      <input class="wide" id="outcome-text" placeholder="+13 / gain 13 / incomplete / sack …"/>
    </div>
    <div class="row">
      <label title="Their call on the snap you just logged: their coverage/blitz when you had the ball, their concept when they had it">they ran</label>
      <input class="wide" id="their" placeholder="cover 6 · A-gap blitz · cross wheels · mesh"/>
    </div>
    <div id="exec-verify" hidden>
      <h2 style="margin-top:.85rem">What I ran <span style="font-weight:500;text-transform:none;letter-spacing:0">(change only when you ran something else)</span></h2>
      <div class="row" id="exec-radios" style="gap:.75rem;flex-wrap:wrap">
        <label style="font-weight:600"><input type="radio" name="exec" value="used_recommended" checked/> used recommended</label>
        <label><input type="radio" name="exec" value="used_different"/> used different</label>
        <label><input type="radio" name="exec" value="unknown"/> unknown</label>
      </div>
      <div class="row" id="exec-diff" hidden>
        <label>formation</label>
        <input class="wide" id="exec-formation" placeholder="from applied book or other"/>
        <label>play</label>
        <input class="wide" id="exec-play" placeholder="actual play name"/>
      </div>
    </div>

    <h2 style="margin-top:1rem">Next situation</h2>
    <div class="row">
      <label>quarter</label>
      <select id="quarter">
        <option value="1">1st</option>
        <option value="2">2nd</option>
        <option value="3">3rd</option>
        <option value="4">4th</option>
        <option value="5">OT</option>
      </select>
      <label title="Our score. Sticks until you change it. Us-them, not the final winner-first score.">us</label>
      <input type="number" id="score-us" min="0" max="99" placeholder="—" value="{_esc(us_v)}"/>
      <label title="Their score">them</label>
      <input type="number" id="score-them" min="0" max="99" placeholder="—" value="{_esc(them_v)}"/>
      <label>down</label>
      <input type="number" id="down" min="1" max="4" value="1"/>
      <label>distance</label>
      <input type="number" id="distance" min="1" max="40" value="10"/>
      <label>yard line</label>
      <select id="yl-side">
        <option value="my">my</option>
        <option value="opp">opp</option>
      </select>
      <input type="number" id="yl" min="1" max="50" value="25"/>
      <label id="side-label" style="display:none">side</label>
      <select id="side" style="display:none">
        <option value="offense">O</option>
        <option value="defense">D</option>
      </select>
    </div>
    <div class="row">
      <label>last play / look</label>
      <input class="wide" id="look" placeholder="mesh spot · cover 2 · showing cover 3"/>
      <label title="Bare name = previous snap; check for live pre-snap look">live</label>
      <input type="checkbox" id="live-mark"/>
    </div>
    <div class="row">
      <button type="button" class="primary" id="btn-submit">Submit → log + new PLAY</button>
      <button type="button" id="btn-undo">Undo last snap</button>
      <button type="button" id="btn-call-only">Call only (no log)</button>
    </div>
  </section>

  {'<details id="live-admin"><summary>Game settings, log and finish game</summary>' if compact_madden else ''}
  <section id="bench-panel" hidden>
    <h2>Benched this half (kept failing)</h2>
    <div id="bench-list" style="font-size:.85rem"></div>
  </section>

  <section id="book-panel" hidden>
    <h2>Playbook of record</h2>
    <div id="book-status"></div>
    <div class="row" id="book-apply-row" hidden>
      <button type="button" class="primary" id="btn-book-apply">I applied these edits in CFB 27</button>
    </div>
    <pre id="book-edits" style="white-space:pre-wrap;font-size:.78rem;color:var(--muted)" hidden></pre>
    <details><summary style="font-size:.8rem;color:var(--muted)">Callable plays (live calls never leave this list)</summary>
      <div id="book-plays" style="font-size:.8rem;color:var(--muted)"></div></details>
  </section>

  <section>
    <h2>Game log</h2>
    <div id="log"><div class="muted">No snaps yet.</div></div>
  </section>

  <section id="end-panel">
    <h2>End game → retrain</h2>
    <div class="row">
      <label>result</label>
      <select id="wl">
        <option value="win">Win</option>
        <option value="loss">Loss</option>
      </select>
      <label>score</label>
      <input class="wide" id="score" placeholder="24-17"/>
      <button type="button" class="danger" id="btn-end">Game over → retrain</button>
    </div>
    <div id="summary" hidden></div>
  </section>

  {'</details>' if compact_madden else ''}

  <footer{' hidden' if compact_madden else ''}>
    Aidan keeps sticks · this window is the live input pad<br/>
    Server: <code>{_esc(ctrl.play_cmd)}</code> · outcomes save to SQLite · Game over runs smarter retrain
  </footer>
</main>
<script>
const $ = (id) => document.getElementById(id);
const compactMadden = {str(compact_madden).lower()};
let outcomeChoice = "";

function setErr(msg) {{ $("err").textContent = msg || ""; }}

function renderState(st) {{
  if (!st) return;
  if (compactMadden) {{
    const rec = st.pending_recommendation;
    $("call").textContent = rec && rec.formation && rec.play
      ? rec.formation + " — " + rec.play
      : (st.ended ? "GAME OVER" : "Waiting for next play");
    renderMacro(null);
    const mlBox = $("ml-experimental");
    if (mlBox) {{ mlBox.hidden = true; mlBox.textContent = ""; }}
  }} else {{
    let callTxt = st.call_text || "";
    if (st.macro) callTxt = callTxt.split("\\n").filter(l => !l.trim().startsWith("MACRO:")).join("\\n");  // shown in #macro-box
    $("call").textContent = callTxt;
    renderMacro(st.macro);
    const mlBox = $("ml-experimental");
    if (mlBox) {{
      if (st.ml_experimental && st.ml_experimental.explanation) {{
        mlBox.hidden = false;
        mlBox.textContent = "ML experimental [" + (st.ml_experimental.evidence_quality || "?") + "] · "
          + "heuristic " + (st.ml_experimental.heuristic || "") + " · "
          + "ML " + (st.ml_experimental.ml || "") + " · "
          + "shown " + (st.ml_experimental.final || "") + " · "
          + (st.ml_experimental.explanation || "");
      }} else {{
        mlBox.hidden = true;
        mlBox.textContent = "";
      }}
    }}
  }}
  $("heard").textContent = st.heard || ("vs " + (st.opponent_id || ""));
  const log = $("log");
  if (!st.log || !st.log.length) {{
    log.innerHTML = "<div>No snaps yet.</div>";
  }} else {{
    log.innerHTML = st.log.map(r => {{
      const look = r.look ? " vs " + r.look : "";
      return "<div><b>" + (r.formation||"?") + " — " + (r.play||"?") + "</b> · "
        + (r.result||"") + look + "</div>";
    }}).join("");
    log.scrollTop = log.scrollHeight;
  }}
  if (st.ended) {{
    $("snap-panel").style.opacity = "0.45";
    $("snap-panel").style.pointerEvents = "none";
    $("btn-end").disabled = true;
    if (st.retrain_summary) {{
      $("summary").hidden = false;
      $("summary").textContent = st.retrain_summary;
    }}
  }}
  renderBook(st.book);
  renderDyn(st);
  renderBench(st.benched);
  if (st.quarter) $("quarter").value = String(st.quarter);
  if (st.live_score) {{
    $("score-us").value = st.live_score.us;
    $("score-them").value = st.live_score.them;
  }} else if (st.ball && st.ball.score_us != null) {{
    $("score-us").value = st.ball.score_us;
    $("score-them").value = st.ball.score_them || 0;
  }} else if (document.activeElement !== $("score-us") && document.activeElement !== $("score-them")) {{
    $("score-us").value = "";
    $("score-them").value = "";
  }}
  const ball = st.ball;
  if (ball && ball.down) {{
    $("down").value = ball.down;
    if (ball.distance) $("distance").value = ball.distance;
  }}
  if (ball && ball.yardline) {{
    const yl = Number(ball.yardline);
    if (yl <= 50) {{ $("yl-side").value = "my"; $("yl").value = yl; }}
    else {{ $("yl-side").value = "opp"; $("yl").value = 100 - yl; }}
  }}
  if (st.cpu_only) {{
    $("side").style.display = "none";
    $("side-label").style.display = "none";
  }} else {{
    $("side").style.display = "";
    $("side-label").style.display = "";
  }}
  const ev = $("exec-verify");
  if (ev) ev.hidden = !st.execution_verify;
  // Never rewrite an explicit used_different selection on state refresh.
  syncExecDiffVisibility();
}}

function currentExecSelection() {{
  const el = document.querySelector('input[name="exec"]:checked');
  return el ? el.value : "used_recommended";
}}

function setExecSelection(value) {{
  const target = value || "used_recommended";
  const radio = document.querySelector('input[name="exec"][value="' + target + '"]');
  if (radio) radio.checked = true;
  syncExecDiffVisibility();
}}

function syncExecDiffVisibility() {{
  const diff = $("exec-diff");
  if (!diff) return;
  diff.hidden = currentExecSelection() !== "used_different";
}}

function restoreExecDefaultAfterSubmit() {{
  // Default stays used_recommended after each submit so unknown does not reappear.
  // Clearing formation/play only — do not leave a stale used_different form visible.
  setExecSelection("used_recommended");
  if ($("exec-formation")) $("exec-formation").value = "";
  if ($("exec-play")) $("exec-play").value = "";
}}

function renderMacro(m) {{
  const box = $("macro-box");
  if (!m) {{ box.hidden = true; return; }}
  box.hidden = false;
  $("macro-head").textContent = m.headline || ("+ MACRO: " + m.name);
  $("macro-key").textContent = (m.press || ("LB → " + m.name)) + (m.key ? "  |  " + m.key : "");
  $("macro-why").textContent = m.why || "";
  $("macro-rows").innerHTML = (m.settings || []).map(r =>
    "<tr><td class='s'>" + esc(r.section) + "</td><td class='s'>" + esc(r.setting) + "</td><td>" + esc(r.value) + "</td></tr>").join("")
    + (m.settings ? "<tr><td class='s'></td><td class='s'>Everything else</td><td>Default</td></tr>" : "");
}}

function renderBench(b) {{
  const sides = Object.entries(b || {{}}).filter(([, rows]) => rows && rows.length);
  $("bench-panel").hidden = !sides.length;
  $("bench-list").innerHTML = sides.map(([side, rows]) =>
    "<div><b>" + (side === "offense" ? "O" : "D") + "</b>: " + rows.map(esc).join(" · ") + "</div>").join("");
}}

function esc(s) {{ return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }}

function renderDyn(st) {{
  if (!st.book) {{ $("dyn").innerHTML = ""; return; }}
  const b = st.book;
  let bk = b.name ? (b.name + (b.callable_rev ? " rev " + b.callable_rev + (b.confirmed ? " (applied)" : " (unconfirmed)") : "")) : "no custom book yet";
  $("dyn").innerHTML = esc(b.context_label || "Dynasty") + ": <b>" + esc(st.dynasty_label || st.dynasty || "?") + "</b> · Book: <b>" + esc(bk) + "</b>"
    + (st.dynasty_source ? " <span class='src'>(" + esc(st.dynasty_source) + ")</span>" : "");
}}

function renderBook(b) {{
  if (!b) {{ $("book-panel").hidden = true; return; }}
  $("book-panel").hidden = false;
  if (b.error) {{ $("book-status").textContent = "Playbook unavailable: " + b.error; return; }}
  let msg = "";
  if (!b.callable_rev) msg = "No custom playbook yet — run prep.";
  const gl = b.game_label || "CFB 27";
  $("btn-book-apply").textContent = "I applied these edits in " + gl;
  if (!b.callable_rev) msg = b.empty_text || msg;
  else if (!b.confirmed) msg = (b.name || "Book") + " rev " + b.callable_rev + " (" + (b.dynasty || "") + ") is UNCONFIRMED (first build): build it in " + gl + ", then confirm below.";
  else msg = "Locked to your applied " + (b.name || "book") + " rev " + b.callable_rev + " (" + (b.dynasty || "") + ").";
  if (b.pending_rev && b.confirmed) msg += " " + b.pending_edits + " pending edit(s) (rev " + b.pending_rev + ") are NOT callable until you confirm.";
  $("book-status").textContent = msg;
  $("book-apply-row").hidden = !b.pending_rev;
  $("btn-book-apply").dataset.rev = b.pending_rev || "";
  $("book-edits").hidden = !b.pending_text;
  $("book-edits").textContent = b.pending_text || "";
  $("book-plays").innerHTML = Object.entries(b.formations || {{}}).map(([f, ps]) =>
    "<div><b>" + f + "</b>: " + ps.join(", ") + "</div>").join("");
}}

/* ---- Idempotency for browser actions (durable across retries / refresh) ---- */
const IDEM_STORE = "cfb_coach_live_idempotency_v1";
const ACTION_BTN_IDS = ["btn-submit", "btn-undo", "btn-call-only", "btn-end", "btn-book-apply"];
let uiLocked = false;

function newIdempotencyKey() {{
  if (window.crypto && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return "k-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 12);
}}

function actionFingerprint(path, body) {{
  const copy = Object.assign({{}}, body || {{}});
  delete copy.idempotency_key;
  delete copy.request_key;
  return path + "|" + JSON.stringify(copy);
}}

function loadPending() {{
  try {{ return JSON.parse(sessionStorage.getItem(IDEM_STORE) || "null"); }}
  catch (e) {{ return null; }}
}}

function savePending(p) {{
  try {{
    if (p) sessionStorage.setItem(IDEM_STORE, JSON.stringify(p));
    else sessionStorage.removeItem(IDEM_STORE);
  }} catch (e) {{ /* private mode / quota — in-memory only this page */ }}
}}

function setBusy(busy) {{
  ACTION_BTN_IDS.forEach(id => {{
    const el = $(id);
    if (!el) return;
    if (busy) {{
      el.dataset.wasDisabled = el.disabled ? "1" : "0";
      el.disabled = true;
    }} else {{
      el.disabled = el.dataset.wasDisabled === "1";
      delete el.dataset.wasDisabled;
    }}
  }});
}}

function claimIdempotencyKey(path, body) {{
  const fp = actionFingerprint(path, body);
  const pending = loadPending();
  // Retry / double-submit of the same intentional action reuses the key.
  if (pending && pending.path === path && pending.fingerprint === fp && pending.key) {{
    pending.status = "in_flight";
    pending.ts = Date.now();
    savePending(pending);
    return pending.key;
  }}
  // New intentional action → new key.
  const key = newIdempotencyKey();
  savePending({{ path: path, fingerprint: fp, key: key, status: "in_flight", ts: Date.now() }});
  return key;
}}

async function apiAction(path, body) {{
  const fp = actionFingerprint(path, body);
  const key = claimIdempotencyKey(path, body);
  const payload = Object.assign({{}}, body || {{}}, {{ idempotency_key: key }});
  let res;
  try {{
    res = await fetch(path, {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(payload),
    }});
  }} catch (netErr) {{
    // Ambiguous network failure: server may have applied — keep key for retry.
    savePending({{ path: path, fingerprint: fp, key: key, status: "ambiguous", ts: Date.now() }});
    throw new Error("Network error — retry the same action (idempotency key kept)");
  }}
  let data = null;
  try {{
    data = await res.json();
  }} catch (parseErr) {{
    savePending({{ path: path, fingerprint: fp, key: key, status: "ambiguous", ts: Date.now() }});
    throw new Error("Ambiguous response — retry the same action (idempotency key kept)");
  }}
  if (!res.ok || !data.ok) {{
    if (res.status >= 500) {{
      savePending({{ path: path, fingerprint: fp, key: key, status: "ambiguous", ts: Date.now() }});
    }} else {{
      // Clear client/validation errors so a corrected form gets a fresh key.
      savePending(null);
    }}
    throw new Error((data && data.error) || ("HTTP " + res.status));
  }}
  savePending(null);
  return data;
}}

async function withUiLock(fn) {{
  if (uiLocked) return null; // ignore double-click while in flight
  uiLocked = true;
  setBusy(true);
  try {{
    return await fn();
  }} finally {{
    uiLocked = false;
    setBusy(false);
  }}
}}

function buildSit() {{
  const d = $("down").value || "1";
  const dist = $("distance").value || "10";
  const ylSide = $("yl-side").value;
  const yl = $("yl").value || "";
  let look = ($("look").value || "").trim();
  if (look && $("live-mark").checked && !/^\\s*(showing|live|pre-snap|aligned)\\b/i.test(look)) {{
    look = "showing " + look;
  }}
  const parts = [d + "&" + dist];
  if (yl) parts.push(ylSide + " " + yl);
  if (look) parts.push(look);
  return parts.join(" ");
}}

function currentOutcome() {{
  const free = ($("outcome-text").value || "").trim();
  if (free) return free;
  if (outcomeChoice === "gain") return "gain " + ($("gain-n").value || "0");
  if (outcomeChoice === "loss") return "loss " + ($("loss-n").value || "0");
  return outcomeChoice || "";
}}

document.querySelectorAll("#outcome-btns button.outcome").forEach(btn => {{
  btn.addEventListener("click", () => {{
    outcomeChoice = btn.getAttribute("data-out");
    $("outcome-text").value = "";
    document.querySelectorAll("#outcome-btns button.outcome").forEach(b => b.style.outline = "");
    btn.style.outline = "2px solid #58a6ff";
  }});
}});

$("btn-submit").addEventListener("click", async () => {{
  setErr("");
  try {{
    const data = await withUiLock(async () => {{
      const out = currentOutcome();
      const sit = buildSit();
      const exec = (document.querySelector('input[name="exec"]:checked') || {{}}).value || "unknown";
      const body = {{
        outcome: out, sit: sit, side: $("side").value, quarter: $("quarter").value,
        their: ($("their").value || "").trim(), score_us: $("score-us").value, score_them: $("score-them").value,
        executed_status: exec,
        executed_formation: ($("exec-formation").value || "").trim() || null,
        executed_play: ($("exec-play").value || "").trim() || null,
      }};
      return await apiAction("/api/result_call", body);
    }});
    if (!data) return; // double-click ignored
    outcomeChoice = "";
    $("outcome-text").value = "";
    $("their").value = "";
    $("look").value = "";
    $("live-mark").checked = false;
    document.querySelectorAll("#outcome-btns button.outcome").forEach(b => b.style.outline = "");
    restoreExecDefaultAfterSubmit();
    renderState(data.state);
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

document.querySelectorAll('input[name="exec"]').forEach(r => {{
  r.addEventListener("change", () => {{
    syncExecDiffVisibility();
  }});
}});
syncExecDiffVisibility();

$("btn-undo").addEventListener("click", async () => {{
  setErr("");
  try {{
    const data = await withUiLock(async () => apiAction("/api/undo", {{}}));
    if (!data) return;
    renderState(data.state);
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

$("btn-call-only").addEventListener("click", async () => {{
  setErr("");
  try {{
    const data = await withUiLock(async () => apiAction("/api/call", {{
      sit: buildSit(), side: $("side").value, quarter: $("quarter").value,
      score_us: $("score-us").value, score_them: $("score-them").value,
    }}));
    if (!data) return;
    renderState(data.state);
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

$("btn-end").addEventListener("click", async () => {{
  setErr("");
  if (!confirm("End game and run retrain on this log?")) return;
  try {{
    const data = await withUiLock(async () => apiAction("/api/end_game", {{
      result_wl: $("wl").value,
      score: $("score").value || "",
    }}));
    if (!data) return;
    renderState(data.state);
    $("summary").hidden = false;
    $("summary").textContent = data.retrain_summary || data.state.retrain_summary || "";
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

$("btn-book-apply").addEventListener("click", async () => {{
  setErr("");
  if (!confirm("Confirm you made every edit in the " + ($("btn-book-apply").textContent.replace("I applied these edits in ", "") || "CFB 27") + " custom playbook editor?")) return;
  try {{
    const data = await withUiLock(async () => {{
      const rev = parseInt($("btn-book-apply").dataset.rev || "0", 10) || null;
      return await apiAction("/api/book_apply", {{ rev: rev }});
    }});
    if (!data) return;
    renderState(data.state);
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

fetch("/api/state").then(r => r.json()).then(st => renderState(st)).catch(() => {{}});
</script>
</body>
</html>
"""


def _dyn_line(ctrl: LivePlayController) -> str:
    """Top-of-window line: which dynasty and custom book this live session plays."""
    if ctrl.book_info is None:  # Madden / no playbook of record: nothing to show
        return ""
    b = ctrl.book_state() or {}
    if b.get("name"):
        bk = f"{b['name']}" + (f" rev {b['callable_rev']} ({'applied' if b.get('confirmed') else 'unconfirmed'})" if b.get("callable_rev") else "")
    else:
        bk = "no custom book yet"
    src = f" <span class='src'>({_esc(ctrl.dynasty_source)})</span>" if ctrl.dynasty_source else ""
    return f"{_esc(b.get('context_label') or 'Dynasty')}: <b>{_esc(ctrl.dynasty_label or ctrl.dynasty)}</b> · Book: <b>{_esc(bk)}</b>{src}"


def _call_main(ctrl: LivePlayController) -> str:
    """Madden gets formation/play only; CFB keeps its existing macro-rich pad."""
    if not str(ctrl.brand).lower().startswith("madden"):
        txt = ctrl.call_text or ""
        if ctrl.macro_state():
            txt = "\n".join(
                ln for ln in txt.split("\n")
                if not ln.strip().startswith(("MACRO:", "ADJ:"))
            )
        return txt
    recommendation = ctrl._pending_recommendation()
    if recommendation:
        formation, play = recommendation.get("formation"), recommendation.get("play")
        if formation and play:
            return f"{formation} — {play}"
    return "GAME OVER" if ctrl.ended else "Waiting for next play"


def _macro_box_html(m: dict[str, Any] | None) -> str:
    """Prominent PLAY + MACRO banner with the macro's key settings (expandable rows)."""
    m = m or {}
    rows = "".join(
        f"<tr><td class='s'>{_esc(r.get('section'))}</td><td class='s'>{_esc(r.get('setting'))}</td><td>{_esc(r.get('value'))}</td></tr>"
        for r in m.get("settings") or []
    ) + ("<tr><td class='s'></td><td class='s'>Everything else</td><td>Default</td></tr>" if m.get("settings") else "")
    return f"""<div id="macro-box"{"" if m else " hidden"}>
    <div class="mh" id="macro-head">{_esc(m.get("headline") or "")}</div>
    <div class="mk" id="macro-key">{(_esc(m.get("press") or ("LB → " + str(m.get("name")))) + (("  |  " + _esc(m.get("key"))) if m.get("key") else "")) if m else ""}</div>
    <div class="mw" id="macro-why">{_esc(m.get("why") or "")}</div>
    <details id="macro-details"><summary class="mw">Macro settings (everything else Default)</summary>
      <table id="macro-rows">{rows}</table></details>
  </div>"""


def make_handler(ctrl: LivePlayController) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Quiet default logging; parent can print on demand
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            data = json.dumps(obj).encode("utf-8")
            self._send(code, data, "application/json; charset=utf-8")

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                return json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    html = render_live_html(ctrl).encode("utf-8")
                    self._send(200, html, "text/html; charset=utf-8")
                    return
                if path == "/api/state":
                    self._json(200, ctrl.state())
                    return
                self._json(404, {"ok": False, "error": "not found"})
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc), "trace": traceback.format_exc()})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = self._read_json()
                if path == "/api/call":
                    sit = (body.get("sit") or body.get("situation") or "").strip()
                    if not sit:
                        self._json(400, {"ok": False, "error": "sit required"})
                        return
                    self._json(200, ctrl.call_only(
                        sit, side=body.get("side"), quarter=body.get("quarter"),
                        score_us=body.get("score_us"), score_them=body.get("score_them"),
                        score_set=("score_us" in body or "score_them" in body),
                        request_key=body.get("idempotency_key") or body.get("request_key"),
                    ))
                    return
                if path == "/api/undo":
                    self._json(
                        200,
                        ctrl.undo_last(
                            request_key=body.get("idempotency_key") or body.get("request_key"),
                        ),
                    )
                    return
                if path in ("/api/result_call", "/api/result+call"):
                    sit = (body.get("sit") or body.get("situation") or "").strip()
                    if not sit:
                        self._json(400, {"ok": False, "error": "sit required"})
                        return
                    self._json(
                        200,
                        ctrl.result_and_call(
                            outcome=body.get("outcome") or body.get("result") or "",
                            sit_raw=sit,
                            side=body.get("side"),
                            quarter=body.get("quarter"),
                            their=body.get("their") or "",
                            score_us=body.get("score_us"),
                            score_them=body.get("score_them"),
                            score_set=("score_us" in body or "score_them" in body),
                            executed_status=body.get("executed_status") or "unknown",
                            executed_formation=body.get("executed_formation"),
                            executed_play=body.get("executed_play"),
                            executed_macro=body.get("executed_macro"),
                            applied_recommended_action=body.get("applied_recommended_action") is True,
                            request_key=body.get("idempotency_key") or body.get("request_key"),
                        ),
                    )
                    return
                if path == "/api/book_apply":
                    rev = body.get("rev")
                    res = ctrl.apply_book(int(rev) if rev else None)
                    self._json(200 if res.get("ok") else 409, res)
                    return
                if path == "/api/end_game":
                    self._json(
                        200,
                        ctrl.end_game(
                            result_wl=body.get("result_wl") or body.get("result") or "",
                            score=body.get("score") or "",
                            request_key=body.get("idempotency_key") or body.get("request_key"),
                        ),
                    )
                    return
                self._json(404, {"ok": False, "error": "not found"})
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc), "trace": traceback.format_exc()})

    return Handler


class ContextThreadingHTTPServer(ThreadingHTTPServer):
    """Request threads see the ContextVars from when this server was created.

    ``threading.Thread`` does not copy contextvars on this runtime. The HTML
    server handles each request on a worker, so ``--no-vod-prior`` and
    ``--freeze-vod-book`` set on the command thread never reached ``make_call``.
    The snapshot is taken at construction, which is the flagged thread in
    ``run_live_server``, and each worker enters that snapshot before the handler.
    """

    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)
        self._request_context = contextvars.copy_context()

    def process_request_thread(self, request, client_address):
        self._request_context.run(super().process_request_thread, request, client_address)


def pick_port(host: str = "127.0.0.1", preferred: int = 8765) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 20)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    # Last resort: ephemeral
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def run_live_server(
    ctrl: LivePlayController,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    open_browser: bool = True,
) -> int:
    """Serve the live HTML page until Ctrl+C. Returns 0."""
    ctrl.start()
    port = port or pick_port(host)
    handler = make_handler(ctrl)
    server = ContextThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"HTML live play ON → {url}")
    print("  Submit outcome + next situation in the browser. Ctrl+C to stop the server.")
    if open_browser:
        open_url(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live server…")
    finally:
        server.shutdown()
        server.server_close()
    return 0


__all__ = [
    "ContextThreadingHTTPServer",
    "LivePlayController",
    "LogRow",
    "make_handler",
    "pick_port",
    "render_live_html",
    "run_live_server",
]
