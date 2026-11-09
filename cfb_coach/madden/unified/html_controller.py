"""Adapter that lets the ESTABLISHED live HTML page and HTTP handler drive ``UnifiedSession``.

``live_server.render_live_html`` / ``make_handler`` only talk to the controller through a small duck-typed surface
(``state``, ``call_only``, ``result_and_call``, ``undo_last``, ``end_game`` ...). This class implements that surface on
top of the unified two-sided session, so the page, its layout and its interactions are the same ones used in earlier games.

Guarantees:
* every recommendation comes from ``UnifiedSession.start_snap`` (never ``playcaller.make_call``);
* this module never imports or opens ``CoachDB`` / the primary database;
* a recommendation is only recorded as executed when the owner explicitly says so ("used recommended" / "used different");
* everything is validated before anything is written, and every mutating request is idempotent (key persisted in the store).
"""
from __future__ import annotations

import html as _html
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cfb_coach.madden.unified.inputs import InputError
from cfb_coach.madden.unified.session import SessionError, UnifiedSession
from cfb_coach.madden.unified.types import DEF, OFF, EXEC_CONFIRMED

LATENCY_NOTE = "decision latency is shown per snap; request latency is measured by `html-check`"


def _e(v: Any) -> str:
    """The established page writes log rows and book lines with innerHTML, so every string we hand it is HTML-escaped here."""
    return _html.escape(str(v if v is not None else ""), quote=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_OUTCOME_ALIASES = {"fumble lost": "fumble", "convert": "convert", "stop": "stop"}


class UnifiedLiveController:
    """Duck-typed stand-in for ``live_server.LivePlayController`` (unified_v1 only)."""

    unified_ui = True  # tells render_live_html to include the unified status panel
    # Owner-authorized standing attestation: submitting a completed snap means
    # the recommendation was executed unless the owner explicitly overrides it.
    standing_execution_attestation = True
    book_info = None  # the page's legacy 'playbook of record' line stays hidden; books are shown from the confirmed file

    def __init__(self, session: UnifiedSession, *, model_path: str | None = None, play_cmd: str = "python -m cfb_coach.madden.unified html") -> None:
        self.session = session
        self.model_path = model_path
        self.opponent_id = session.opponent_id
        self.brand = "Madden 27 · Unified V1"  # "madden…" selects the compact Madden layout in the page
        self.play_cmd = play_cmd
        self.dynasty = "unified"
        self.dynasty_label = "unified_v1"
        self.dynasty_source = "owner-confirmed books"
        self.cpu_only = False
        self.default_side = "offense"
        self.session_id = session.game_id
        self.ended = False
        self.quarter: int | None = None
        self.lock = threading.RLock()
        self._score = (None, None)
        self.errors: list[str] = []

    # ---- surface used by render_live_html ------------------------------------------------------------------------
    @property
    def live_score(self) -> Any:
        if self._score == (None, None):
            return None
        from types import SimpleNamespace
        return SimpleNamespace(us=self._score[0], them=self._score[1], label=f"{self._score[0]}-{self._score[1]}")

    def _pending_recommendation(self) -> dict[str, Any] | None:
        if self.session.pending is None or self.ended:
            return None
        st, _ = self.session.pending
        c = self._pending_card()
        return {"formation": c["formation"], "play": c["play"], "side": "offense" if st.our_side == OFF else "defense", "snap_id": st.snap_id}

    def macro_state(self) -> None:
        return None

    @property
    def call_text(self) -> str:
        p = self.session.pending
        if p is None:
            return "GAME OVER" if self.ended else "Waiting for next play"
        c = self._pending_card()
        return f"{c.get('formation')} — {c.get('play')}"

    def start(self) -> None:  # the established server calls this; there is nothing legacy to start
        return None

    def book_state(self) -> dict[str, Any]:
        forms = {}
        for side, tag in ((OFF, "O"), (DEF, "D")):
            for f, ps in self.session.coord[side].inventory.items():
                forms[_e(f"{tag}: {f}")] = [_e(x) for x in ps]
        return {"name": "owner-confirmed books (unified)", "callable_rev": 1, "confirmed": True, "formations": forms, "game_label": "Madden 27",
                "context_label": "Franchise", "dynasty": "unified"}

    def apply_book(self, rev: int | None = None) -> dict[str, Any]:
        return {"ok": False, "error": "books are installed and attested with `confirm-install`, not in this window", "state": self.state()}

    # ---- helpers -----------------------------------------------------------------------------------------------------
    def _pending_card(self) -> dict[str, Any]:
        st, card = self.session.pending  # type: ignore[misc]
        return card if isinstance(card, dict) else card.to_dict()

    def _side_name(self, side: str) -> str:
        return "OFFENSE" if side == OFF else "DEFENSE"

    def _rows(self) -> list[dict[str, Any]]:
        s = self.session
        voided = s.store.voided()
        out = []
        for sid in s.history.get(s.game_id, []):
            o = s.snaps[sid]
            if o.recommended is None:
                continue
            side = "O" if o.state.our_side == OFF else "D"
            ran = {"confirmed": "ran as called", "modified": "ran different", "not_executed": "not run", "unknown": "execution unknown"}.get(o.executed_status, o.executed_status)
            res = o.outcome.result_text if o.outcome and o.outcome.result_text else ("(no result yet)" if o.outcome is None else "(voided)")
            if sid in voided:
                res = "UNDONE"
            opp = o.opponent_action
            look = (f"{opp.formation} | {opp.play}" if opp and opp.formation else (opp.play if opp and opp.play else "")) if opp and opp.verification != "unverified" else ""
            rec = o.recommended.play
            out.append({"formation": _e(f"{side}{o.state.seq}: {rec.formation}"), "play": _e(rec.play), "result": _e(f"{res} · {ran}"), "look": _e(look)})
        return out

    def _unified_panel(self) -> dict[str, Any]:
        s = self.session
        mode = ("LEARNED weights from the selected artifact (" + str(s.model_rows_at_start) + " rows at start)") if s.model_rows_at_start else \
            "UNCALIBRATED COLD-START PRIOR (no trained weights in this artifact)"
        panel: dict[str, Any] = {"decision_model": "unified_v1", "solver": None, "artifact": s.artifact, "model_mode": mode,
                                 "game_id": s.game_id, "next_snap": f"{s.game_id}-{s.seq:04d}", "world_model_rows": s.model.n_records,
                                 "provenance": dict(s.model.provenance), "history": s.history_report, "pending": None,
                                 "result_note": "Enter the OFFENSE's result on the snap you just ran: on offense that is OUR yards; on defense it is the OPPONENT OFFENSE's yards.",
                                 "result_perspective": None, "fallback": {"occurred": False, "reason": None}, "heuristic_fallback_possible": False}
        if s.pending is not None:
            st, _ = s.pending
            c = self._pending_card()
            panel["pending"] = {k: c.get(k) for k in ("snap_id", "side", "formation", "play", "adjustments", "changed_routes", "user", "reads", "explanation",
                                                       "opponent_prediction_source", "football_evidence_confidence", "evidence_note", "latency_ms", "expected",
                                                       "alternatives", "policy_prob")}
            panel["solver"] = c.get("model_version")
            panel["fallback"] = c.get("fallback") or panel["fallback"]
            panel["result_perspective"] = ("OPPONENT OFFENSE — enter the yards THEY gained on this snap" if st.our_side == DEF
                                           else "OUR OFFENSE — enter the yards WE gained on this snap")
            panel["decision_model"] = c.get("decision_model", "unified_v1")
        return panel

    # ---- state ---------------------------------------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        with self.lock:
            s = self.session
            pend = s.pending
            panel = self._unified_panel()
            rec = self._pending_recommendation()
            act = reads = None
            ball = {}
            if pend is not None:
                st, _ = pend
                c = self._pending_card()
                adjs = c.get("adjustments") or []
                if adjs:
                    act = {"id": "unified.adjust", "label": "; ".join(adjs), "buttons": "ADJUST: " + "; ".join(adjs)
                           + ("  [route change " + ", ".join(f"{k}→{v}" for k, v in (c.get("changed_routes") or {}).items()) + "]" if c.get("changed_routes") else ""),
                           "why": f"{c.get('opponent_prediction_source')}: {c.get('explanation')}"}
                r = c.get("reads")
                if st.our_side == OFF and r:
                    reads = {"status": "known" if r.get("identity_known") else "concept", "label": "UNIFIED " + ("(documented routes)" if r.get("identity_known") else "(concept level — receiver identity UNKNOWN)"),
                             "reads": [{"order": "1", "text": r["primary"]}, {"order": "2", "text": r["secondary"]}], "basis": r.get("basis") or "", "note": r.get("basis") or ""}
                yl = st.yardline if st.our_side == OFF else 100 - st.yardline
                ball = {"down": st.down, "distance": st.distance, "yardline": yl, "score_us": st.score_us, "score_them": st.score_opp}
            return {
                "opponent_id": self.opponent_id, "brand": self.brand, "play_cmd": self.play_cmd, "session_id": s.game_id, "dynasty": self.dynasty,
                "dynasty_label": self.dynasty_label, "dynasty_source": self.dynasty_source, "call_text": self.call_text, "heard": f"vs {self.opponent_id} · game {s.game_id}",
                "cpu_only": False, "default_side": s.last_side or "offense", "has_pending_call": pend is not None and not self.ended, "ended": self.ended,
                "result_wl": None, "score": None, "live_score": None if self._score == (None, None) else {"us": self._score[0], "them": self._score[1], "label": f"{self._score[0]}-{self._score[1]}"},
                "retrain_summary": getattr(self, "retrain_summary", ""), "log": self._rows(), "book": self.book_state(), "macro": None, "quarter": self.quarter,
                "benched": {}, "ball": ball, "execution_verify": True, "standing_execution_attestation": True, "pending_recommendation": rec,
                "pending_offense_action": act, "conditional_action": None,
                "ml_experimental": {"football_reason": (pend and self._pending_card().get("explanation")) or None},
                "qb_reads": reads, "unified": panel,
            }

    # ---- request handling ------------------------------------------------------------------------------------------------
    def _idem(self, key: str | None) -> dict[str, Any] | None:
        return self.session.store.idem_get(key) if key else None

    def _done(self, key: str | None, path: str, out: dict[str, Any]) -> dict[str, Any]:
        if key:
            self.session.store.idem_put(key, path, out, _now())
        return out

    def _fail(self, msg: str) -> dict[str, Any]:
        return {"ok": False, "error": msg}

    @staticmethod
    def _situation_line(sit: str, side: str | None) -> str:
        tok = "d" if str(side or "").lower().startswith("d") else "o"
        return f"{tok} {sit}"

    def _executed_text(self, executed_status: str, executed_formation: str | None, executed_play: str | None,
                       applied_action: bool, card: dict[str, Any]) -> str:
        e = (executed_status or "unknown").lower()
        if e == "used_recommended":
            return "y" if (not card.get("adjustments") or applied_action) else "y noadj"
        if e == "not_executed":
            return "n"
        if e == "used_recommended_no_adjustment":
            return "y noadj"
        if e == "used_different":
            if not (executed_formation and executed_play):
                raise InputError("'used different' needs both the formation and the play you actually ran")
            return f"{executed_formation.strip()} | {executed_play.strip()}"
        return "u"

    def _finish_pending(self, **kw: Any) -> dict[str, Any] | None:
        s = self.session
        if s.pending is None:
            return None
        card = self._pending_card()
        ex = self._executed_text(kw.get("executed_status") or "unknown", kw.get("executed_formation"), kw.get("executed_play"),
                                 bool(kw.get("applied_recommended_action")), card)
        outcome = (kw.get("outcome") or "").strip()
        outcome = _OUTCOME_ALIASES.get(outcome.lower(), outcome)
        return s.finish_snap(executed=ex, result=outcome or None, opponent_call=(kw.get("their") or "").strip() or None)

    def _remember_form(self, quarter: Any, score_us: Any, score_them: Any, score_set: bool) -> None:
        def _i(v: Any) -> int | None:
            try:
                return int(v) if str(v).strip() != "" else None
            except (TypeError, ValueError):
                return None
        q = _i(quarter)
        if q:
            self.quarter = q
        if score_set:
            su, st_ = _i(score_us), _i(score_them)
            if su is not None and st_ is not None:
                self._score = (su, st_)

    def result_and_call(self, *, outcome: str, sit_raw: str, side: str | None = None, quarter: Any = None, their: str = "", score_us: Any = None,
                        score_them: Any = None, score_set: bool = False, executed_status: str = "unknown", executed_formation: str | None = None,
                        executed_play: str | None = None, executed_macro: str | None = None, applied_recommended_action: bool = False,
                        confirmed_no_adjustment: bool = False, verify_touched: bool = False,
                        request_key: str | None = None, **_ignored: Any) -> dict[str, Any]:
        """Log the previous snap (if any) AND produce the next recommendation — the established 'Submit' button."""
        with self.lock:
            cached = self._idem(request_key)
            if cached is not None:
                return cached
            if self.ended:
                return self._fail("this game is closed")
            from cfb_coach.madden.unified.inputs import parse_snap_line
            try:
                line = self._situation_line(sit_raw, side)
                parse_snap_line(line, self.session.last_side)  # validate the NEXT situation before changing anything
                # Owner explicitly authorized following every model call. Only the
                # completed-snap Submit applies this standing attestation:
                # call-only, undo, resume, and page rendering do not.
                status = executed_status
                if self.session.pending is not None and not verify_touched and status in ("", "unknown"):
                    status = "used_recommended"
                # 'used recommended' includes its adjustments unless the owner
                # specifically marked that they skipped the adjustment.
                applied = (applied_recommended_action or
                           (status == "used_recommended" and not confirmed_no_adjustment))
                if self.session.pending is not None:  # validate the previous snap's entries without writing
                    self._dry_validate(outcome, their, status, executed_formation, executed_play, applied)
                self._remember_form(quarter, score_us, score_them, score_set)
                logged = self._finish_pending(outcome=outcome, their=their, executed_status=status, executed_formation=executed_formation,
                                              executed_play=executed_play, applied_recommended_action=applied)
                self.session.start_snap(line, quarter=self.quarter, score_us=self._score[0], score_opp=self._score[1])
            except (InputError, SessionError) as e:
                return self._fail(str(e))
            except Exception as e:  # noqa: BLE001 — surfaced, never replaced by the legacy caller
                self.errors.append(repr(e))
                return self._fail(f"UNIFIED COORDINATOR ERROR (no fallback caller was used): {e!r}")
            out = {"ok": True, "logged": logged, "call_text": self.call_text, "heard": f"vs {self.opponent_id}", "state": self.state()}
            return self._done(request_key, "/api/result_call", out)

    def _dry_validate(self, outcome: str, their: str, executed_status: str, ef: str | None, ep: str | None, applied: bool) -> None:
        from cfb_coach.madden.unified.inputs import parse_opponent_call, parse_result
        st, _ = self.session.pending  # type: ignore[misc]
        card = self._pending_card()
        ex = self._executed_text(executed_status, ef, ep, applied, card)
        if "|" in ex:
            self.session.coord[st.our_side].stock_action(*[x.strip() for x in ex.split("|", 1)])
        o = _OUTCOME_ALIASES.get((outcome or "").strip().lower(), (outcome or "").strip())
        if o:
            parse_result(o, st)
        if (their or "").strip():
            parse_opponent_call(their, DEF if st.our_side == OFF else OFF)

    def call_only(self, sit_raw: str, *, side: str | None = None, quarter: Any = None, score_us: Any = None, score_them: Any = None,
                  score_set: bool = False, request_key: str | None = None, **_ignored: Any) -> dict[str, Any]:
        """'Call only (no log)': a NEW recommendation for a corrected situation. The waiting snap is left decided-but-unlearned."""
        with self.lock:
            cached = self._idem(request_key)
            if cached is not None:
                return cached
            from cfb_coach.madden.unified.inputs import parse_snap_line
            try:
                line = self._situation_line(sit_raw, side)
                parse_snap_line(line, self.session.last_side)
                self._remember_form(quarter, score_us, score_them, score_set)
                self.session.start_snap(line, quarter=self.quarter, score_us=self._score[0], score_opp=self._score[1])
            except (InputError, SessionError) as e:
                return self._fail(str(e))
            except Exception as e:  # noqa: BLE001
                self.errors.append(repr(e))
                return self._fail(f"UNIFIED COORDINATOR ERROR (no fallback caller was used): {e!r}")
            out = {"ok": True, "call_text": self.call_text, "heard": f"vs {self.opponent_id}", "state": self.state()}
            return self._done(request_key, "/api/call", out)

    def undo_last(self, *, request_key: str | None = None) -> dict[str, Any]:
        """Void the most recent snap that carries facts (append-only revisions), drop any waiting call, rebuild from disk."""
        with self.lock:
            cached = self._idem(request_key)
            if cached is not None:
                return cached
            s = self.session
            target = None
            for sid in reversed(s.history.get(s.game_id, [])):
                if sid in s.store.voided():
                    continue
                if any(s.store.has_row(t, sid) for t in ("executions", "outcomes", "opponent_obs")):
                    target = sid
                    break
            if target is None:
                return self._fail("nothing to undo yet")
            if s.pending is not None:  # the call waiting on screen belongs to the situation after the undone snap: drop it too
                s.store.void(s.pending[0].snap_id, _now(), "dropped with undo")
            s.pending = None
            s.void_snap(target)
            self.session = s.reload()
            out = {"ok": True, "message": f"undid {target}: it no longer trains anything and its id stays used. Enter the situation again.", "state": self.state()}
            return self._done(request_key, "/api/undo", out)

    def end_game(self, *, result_wl: str = "", score: str = "", request_key: str | None = None, last_outcome: str = "") -> dict[str, Any]:
        with self.lock:
            cached = self._idem(request_key)
            if cached is not None:
                return cached
            s = self.session
            if s.pending is not None:
                s.abandon_pending("game ended")
            s.end_game()
            self.ended = True
            self.retrain_summary = (f"Unified game {s.game_id} closed ({result_wl or 'result not given'} {score}). "
                                    f"No legacy retrain was run. Evaluate it with:\n"
                                    f"python -m cfb_coach.madden.unified postgame --store {s.store.path} --out <new folder> --champion <model you played>")
            out = {"ok": True, "retrain_summary": self.retrain_summary, "grades": [], "state": self.state()}
            return self._done(request_key, "/api/end_game", out)
