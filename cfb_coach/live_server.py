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
from cfb_coach.outcome import parse_outcome
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
    score: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

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
            "call_text": self.call_text,
            "heard": self.heard,
            "cpu_only": self.cpu_only,
            "default_side": self.default_side,
            "has_pending_call": self.last_call is not None and not self.ended,
            "ended": self.ended,
            "result_wl": self.result_wl,
            "score": self.score,
            "retrain_summary": self.retrain_summary,
            "log": [r.to_dict() for r in self.log],
        }

    def _macro_of(self, call: Any) -> str | None:
        if call is None:
            return None
        if getattr(call, "macro", None):
            return call.macro
        if getattr(call, "side", "") == "defense":
            return getattr(call, "adj_or_macro", None)
        return None

    def _log_pending_result(self, outcome_raw: str) -> dict[str, Any] | None:
        if not self.last_call or not self.last_sit:
            return None
        parsed = parse_outcome(outcome_raw)
        result_text = parsed.to_result_text()
        sit = self.last_sit
        call = self.last_call
        cov = getattr(sit, "coverage_hint", None) or self.last_coverage
        concept = getattr(sit, "concept_hint", None) or self.last_concept
        # Prefer coverage for offense / concept for defense in look column
        look = (cov if call.side == "offense" else concept) or cov or concept or ""

        self.db.log_snap(
            opponent_id=self.opponent_id,
            side=call.side,
            situation_raw=getattr(sit, "raw", "") or "",
            our_call=call.format().split("\n")[0],
            formation=call.formation,
            play=call.play,
            macro=self._macro_of(call),
            down=getattr(sit, "down", None),
            distance=getattr(sit, "distance", None),
            yardline=getattr(sit, "yardline", None),
            result=result_text,
            coverage_seen=cov,
            concept_seen=concept,
            session_id=self.session_id or None,
        )
        # Mild live bumps (same spirit as terminal loop)
        try:
            from cfb_coach.tendency import mild_bump_concept, mild_bump_coverage

            success = parsed.success_for(call.side)
            if concept and call.side == "defense":
                mild_bump_concept(
                    self.db, self.opponent_id, concept, sit, success=bool(success)
                )
                self.last_concept = concept
            elif cov and call.side == "offense":
                mild_bump_coverage(self.db, self.opponent_id, cov, sit)
                self.last_coverage = cov
        except Exception:
            pass

        row = LogRow(
            formation=call.formation or "?",
            play=call.play or "?",
            result=result_text,
            look=look or "",
            call_text=call.format(),
            side=call.side,
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

    def call_only(self, sit_raw: str, *, side: str | None = None) -> dict[str, Any]:
        with self.lock:
            if self.ended:
                return {"ok": False, "error": "game already ended"}
            side_use = side or self.default_side
            if self.cpu_only:
                side_use = "offense"
            sit = self.parse_situation(sit_raw, default_side=side_use)
            from cfb_coach.situation import format_heard

            try:
                heard = format_heard(sit)
            except Exception:
                heard = getattr(sit, "label", sit_raw)
            call = self._make(sit)
            self.last_call, self.last_sit = call, sit
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
    ) -> dict[str, Any]:
        with self.lock:
            if self.ended:
                return {"ok": False, "error": "game already ended"}
            logged = None
            if self.last_call is not None and (outcome or "").strip():
                logged = self._log_pending_result(outcome)
            elif self.last_call is not None and not (outcome or "").strip():
                # Allow first snap without prior outcome
                pass
            elif (outcome or "").strip() and self.last_call is None:
                return {"ok": False, "error": "no prior call to attach outcome to"}

            side_use = side or self.default_side
            if self.cpu_only:
                side_use = "offense"
            sit = self.parse_situation(sit_raw, default_side=side_use)
            from cfb_coach.situation import format_heard

            try:
                heard = format_heard(sit)
            except Exception:
                heard = getattr(sit, "label", sit_raw)
            call = self._make(sit)
            self.last_call, self.last_sit = call, sit
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
  footer {{ margin-top: 1rem; font-size: .72rem; color: var(--muted); line-height: 1.4; }}
</style>
</head>
<body>
<main>
  <h1>Live Play <span class="badge" id="badge">{brand}</span> · keep sticks · type less</h1>
  <div class="heard" id="heard">vs {_esc(ctrl.opponent_id)}</div>
  <div class="call-label">PLAY</div>
  <div class="call" id="call">waiting for first situation…</div>
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
      <button type="button" class="outcome" data-out="stop">stop</button>
      <button type="button" class="outcome" data-out="convert">convert</button>
    </div>
    <div class="row">
      <label>or free text</label>
      <input class="wide" id="outcome-text" placeholder="+13 / gain 13 / incomplete / sack …"/>
    </div>

    <h2 style="margin-top:1rem">Next situation</h2>
    <div class="row">
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
      <button type="button" id="btn-call-only">Call only (no log)</button>
    </div>
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
  $("call").textContent = st.call_text || "";
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
  if (st.cpu_only) {{
    $("side").style.display = "none";
    $("side-label").style.display = "none";
  }} else {{
    $("side").style.display = "";
    $("side-label").style.display = "";
  }}
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
    const body = {{ outcome: out, sit: sit, side: $("side").value }};
    const data = await api("/api/result_call", body);
    outcomeChoice = "";
    $("outcome-text").value = "";
    document.querySelectorAll("#outcome-btns button.outcome").forEach(b => b.style.outline = "");
    renderState(data.state);
  }} catch (e) {{ setErr(String(e.message || e)); }}
}});

$("btn-call-only").addEventListener("click", async () => {{
  setErr("");
  try {{
    const data = await api("/api/call", {{ sit: buildSit(), side: $("side").value }});
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

fetch("/api/state").then(r => r.json()).then(st => renderState(st)).catch(() => {{}});
</script>
</body>
</html>
"""


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
                    self._json(200, ctrl.call_only(sit, side=body.get("side")))
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
                        ),
                    )
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
