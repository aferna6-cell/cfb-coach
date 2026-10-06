"""Localhost HTML live play window — submit outcomes + situations without the terminal.

Stdlib only (http.server). Started by `play` (default ON); `--terminal` / `--no-html`
keeps the classic typed loop.
"""

from __future__ import annotations

import json
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
        }

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
        if call is None:
            return None
        if getattr(call, "macro", None):
            return call.macro
        if getattr(call, "side", "") == "defense":
            return getattr(call, "adj_or_macro", None)
        return None

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

    def undo_last(self) -> dict[str, Any]:
        with self.lock:
            book = self._book()
            before = len(book._undos)
            msg = book.undo()
            if len(book._undos) < before and self.log:
                self.log.pop()
            self._push_score()
            return {"ok": True, "message": msg.strip(), "state": self.state()}

    def _log_pending_result(self, outcome_raw: str, their: str = "", next_raw: str | None = None) -> dict[str, Any] | None:
        """Log the snap that just ended. The form's last-play field belongs to THAT snap."""
        if not self.last_call or not self.last_sit:
            return None
        book = self._book()
        if book.call is None:
            book.remember_call(self.last_call, self.last_sit)
        closed = book.close_from_form(outcome_raw, their=their, next_raw=next_raw)
        if closed is None:
            return None
        side = getattr(self.last_call, "side", "") or ""
        if book.concept and side.startswith("d"):
            self.last_concept = book.concept
        if book.coverage and not side.startswith("d"):
            self.last_coverage = book.coverage
        self._push_score()
        row = LogRow(
            formation=closed["formation"],
            play=closed["play"],
            result=closed["result"],
            look=closed["look"],
            call_text=closed["call_text"],
            side=closed["side"],
        )
        self.log.append(row)
        return row.to_dict()

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
    ) -> dict[str, Any]:
        with self.lock:
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
            self._book().remember_call(call, sit)
            self.call_text = call.format()
            self.heard = heard
            if getattr(sit, "coverage_hint", None) and sit.side == "offense":
                self.last_coverage = sit.coverage_hint
            if getattr(sit, "concept_hint", None) and sit.side == "defense":
                self.last_concept = sit.concept_hint
            self.default_side = sit.side
            return {"ok": True, "call_text": self.call_text, "heard": heard, "state": self.state()}

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
    ) -> dict[str, Any]:
        with self.lock:
            if self.ended:
                return {"ok": False, "error": "game already ended"}
            self._apply_form_score(score_us, score_them, score_set)
            logged = None
            closed_sit = self.last_sit
            if self.last_call is not None and (outcome or "").strip():
                logged = self._log_pending_result(outcome, their=their, next_raw=sit_raw)
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
            from cfb_coach.situation import format_heard

            try:
                heard = format_heard(sit)
            except Exception:
                heard = getattr(sit, "label", sit_raw)
            call = self._make(sit)
            self.last_call, self.last_sit = call, sit
            book.remember_call(call, sit)
            self._push_score()
            self.call_text = call.format()
            self.heard = heard
            if getattr(sit, "coverage_hint", None) and sit.side == "offense":
                self.last_coverage = sit.coverage_hint
            if getattr(sit, "concept_hint", None) and sit.side == "defense":
                self.last_concept = sit.concept_hint
            self.default_side = sit.side
            return {
                "ok": True,
                "logged": logged,
                "call_text": self.call_text,
                "heard": heard,
                "state": self.state(),
            }

    def end_game(self, *, result_wl: str, score: str = "") -> dict[str, Any]:
        with self.lock:
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
            return {
                "ok": True,
                "retrain_summary": self.retrain_summary,
                "grades": grades,
                "state": self.state(),
            }


def _esc(s: Any) -> str:
    return (
        ("" if s is None else str(s))
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_live_html(ctrl: LivePlayController) -> str:
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
  .call {{ font-size: clamp(1.35rem, 3.2vw, 2.15rem); font-weight: 800; line-height: 1.25;
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
</style>
</head>
<body>
<main>
  <h1>Live Play <span class="badge" id="badge">{brand}</span> · keep sticks · type less</h1>
  <div class="dyn" id="dyn">{_dyn_line(ctrl)}</div>
  <div class="heard" id="heard">vs {_esc(ctrl.opponent_id)}</div>
  <div class="call-label">PLAY</div>
  <div class="call" id="call">{_esc(_call_main(ctrl))}</div>
  {_macro_box_html(ctrl.macro_state())}
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

  <footer>
    Aidan keeps sticks · this window is the live input pad<br/>
    Server: <code>{_esc(ctrl.play_cmd)}</code> · outcomes save to SQLite · Game over runs smarter retrain
  </footer>
</main>
<script>
const $ = (id) => document.getElementById(id);
let outcomeChoice = "";

function setErr(msg) {{ $("err").textContent = msg || ""; }}

function renderState(st) {{
  if (!st) return;
  let callTxt = st.call_text || "";
  if (st.macro) callTxt = callTxt.split("\\n").filter(l => !l.trim().startsWith("MACRO:")).join("\\n");  // shown in #macro-box
  $("call").textContent = callTxt;
  renderMacro(st.macro);
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

async function api(path, body) {{
  const res = await fetch(path, {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify(body || {{}}),
  }});
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || ("HTTP " + res.status));
  return data;
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
    const out = currentOutcome();
    const sit = buildSit();
    const body = {{ outcome: out, sit: sit, side: $("side").value, quarter: $("quarter").value, their: ($("their").value || "").trim(), score_us: $("score-us").value, score_them: $("score-them").value }};
    const data = await api("/api/result_call", body);
    outcomeChoice = "";
    $("outcome-text").value = "";
    $("their").value = "";
    $("look").value = "";
    $("live-mark").checked = false;
    document.querySelectorAll("#outcome-btns button.outcome").forEach(b => b.style.outline = "");
    renderState(data.state);
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

$("btn-undo").addEventListener("click", async () => {{
  setErr("");
  try {{
    const data = await api("/api/undo", {{}});
    renderState(data.state);
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

$("btn-call-only").addEventListener("click", async () => {{
  setErr("");
  try {{
    const data = await api("/api/call", {{ sit: buildSit(), side: $("side").value, quarter: $("quarter").value, score_us: $("score-us").value, score_them: $("score-them").value }});
    renderState(data.state);
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

$("btn-end").addEventListener("click", async () => {{
  setErr("");
  if (!confirm("End game and run retrain on this log?")) return;
  try {{
    const data = await api("/api/end_game", {{
      result_wl: $("wl").value,
      score: $("score").value || "",
    }});
    renderState(data.state);
    $("summary").hidden = false;
    $("summary").textContent = data.retrain_summary || data.state.retrain_summary || "";
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

$("btn-book-apply").addEventListener("click", async () => {{
  setErr("");
  if (!confirm("Confirm you made every edit in the " + ($("btn-book-apply").textContent.replace("I applied these edits in ", "") || "CFB 27") + " custom playbook editor?")) return;
  try {{
    const rev = parseInt($("btn-book-apply").dataset.rev || "0", 10) || null;
    const data = await api("/api/book_apply", {{ rev: rev }});
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
    """Big call text; the MACRO line moves into the #macro-box banner when present."""
    txt = ctrl.call_text or ""
    if ctrl.macro_state():
        txt = "\n".join(ln for ln in txt.split("\n") if not ln.strip().startswith(("MACRO:", "ADJ:")))
    return txt


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
                    ))
                    return
                if path == "/api/undo":
                    self._json(200, ctrl.undo_last())
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
                        ),
                    )
                    return
                self._json(404, {"ok": False, "error": "not found"})
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc), "trace": traceback.format_exc()})

    return Handler


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
    server = ThreadingHTTPServer((host, port), handler)
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
    "LivePlayController",
    "LogRow",
    "make_handler",
    "pick_port",
    "render_live_html",
    "run_live_server",
]
