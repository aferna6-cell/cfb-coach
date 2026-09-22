"""cfb-coach watch / copilot — SCREEN CO-PILOT live loop (v1.9.4).

Aidan plays on Xbox via HDMI monitor + controller. Laptop runs Remote Play
as the prototype video source. Coach is SIDE-CAR ONLY — never controls Xbox.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from cfb_coach.copilot import format_tips_block, tips_from_look, write_overlay_html
from cfb_coach.playcaller import Call, make_call
from cfb_coach.situation import Situation, parse_situation
from cfb_coach.vision import (
    CAPTURE_SETUP_NOTES,
    DefenseLook,
    ImageFileCapture,
    StubCapture,
    demo_sequence,
    naive_roi_heuristic,
    parse_look_tokens,
)


HELP = """
Screen Co-Pilot v1.9.4 — you stay on sticks; coach suggests only.

Typed commands (then Enter):
  f <front>      even|odd|nickel|dime|goal_line
  s <shell>      single_high|two_high|cover2|cover3|quarters|man
  p <pressure>   none|show|blitz_left|blitz_right|blitz_middle|all_out
  look <tokens>  e.g.  look nickel c3 blitz_left
  demo           advance one canned look
  tips           reprint last PLAY / adjusts
  hike           print HIKE — go (you snap; coach never presses)
  overlay        rewrite overlay HTML (browser strip; Xbox stays focused)
  correct …      correct last PlayRecord (see below)
  R/P/S/I/X      quick result: Run/Pass/Sack/Int/eXplosive on last play
  help           this text
  q / quit       exit

Manual corrections:
  correct family CROSSERS | correct result completion | correct yards 12
  R = run, P = pass/completion, S = sack, I = int, X = mark explosive

Live capture: --window Xbox --debug | --list-windows | --screen-region | --calibrate
Session: --opponent / --dynasty starts game_sessions + play_records
""".strip()



# --- live-loop UX helpers (heartbeat / status / non-blocking stdin) ---

_WAITING_FRAMES = (
    "waiting for frames… (is Remote Play visible? window title match?) "
    "capture={capture}"
)

_HEARTBEAT_INTERVAL_S = 5.0
_HEARTBEAT_AFTER_TROUBLE_S = 15.0

_TROUBLESHOOT_NO_FRAMES = """No frames for 5s — troubleshooting:
  · Leave Xbox Remote Play visible (not minimized / not covered)
  · List titles: cfb-coach watch --list-windows
  · Confirm --window title matches (case-insensitive substring; tries Xbox/Remote Play/Game Bar aliases)
  · Xbox app / Remote Play often cannot be captured via dxcam (UWP) — auto-fallback to mss should kick in ~2–3s
  · Recalibrate: cfb-coach watch --calibrate --window "Xbox"
  · Or: cfb-coach watch --screen-region   (uses crop from ~/.cfb-coach/vision_calib.json via mss)
"""

_CAPTURE_OK = "capture OK — PLAY suggestions below (Xbox stays focused)"

_LIVE_CMD_HINT = (
    "Commands: hike, tips, look …, help, q, R/P/S/I/X — keep Xbox focused; overlay is secondary"
)


def _waiting_frames_msg(capture_name: str) -> str:
    return _WAITING_FRAMES.format(capture=capture_name or "?")


def _heartbeat_interval(troub_printed: bool) -> float:
    """5s until first troubleshoot, then 15s only."""
    return _HEARTBEAT_AFTER_TROUBLE_S if troub_printed else _HEARTBEAT_INTERVAL_S


def _should_emit_heartbeat(
    *,
    now: float,
    loop_start: float,
    last_heartbeat: float,
    last_frame_wall: float,
    troub_printed: bool,
    typing: bool,
) -> bool:
    """Gate for waiting-for-frames heartbeat (no scroll flood, no mid-keystroke)."""
    if typing:
        return False
    if now - last_frame_wall < 1.0:
        return False
    interval = _heartbeat_interval(troub_printed)
    if last_heartbeat <= 0.0:
        return (now - loop_start) >= interval
    return (now - last_heartbeat) >= interval


def _status_line(
    short: str,
    *,
    fps: float | None = None,
    play_state: str = "",
    look_label: str = "",
) -> str:
    parts = [f"  … {short}"]
    if fps is not None:
        parts.append(f"fps={fps:.1f}")
    if play_state:
        parts.append(f"state={play_state}")
    if look_label:
        parts.append(f"look={look_label}")
    return " ".join(parts)


class StdinCommandQueue:
    """Non-blocking stdin for the live capture loop (Windows + Unix).

    A daemon thread reads complete lines so the capture while-loop never
    blocks on input(). Works alongside status/heartbeat prints.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = False
        self._partial: list[str] = []
        self._partial_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        if not sys.stdin.isatty():
            return
        self.enabled = True
        self._thread = threading.Thread(
            target=self._reader, name="cfb-stdin", daemon=True
        )
        self._thread.start()

    def _reader(self) -> None:
        # Prefer select on POSIX so we can wake on stop; msvcrt path for Windows
        # character accumulation; fallback to blocking readline in this thread.
        use_select = False
        try:
            import select

            use_select = hasattr(select, "select") and sys.platform != "win32"
        except ImportError:
            use_select = False

        if sys.platform == "win32":
            self._reader_msvcrt()
            return

        while not self._stop.is_set():
            try:
                if use_select:
                    import select

                    ready, _, _ = select.select([sys.stdin], [], [], 0.25)
                    if not ready:
                        continue
                    line = sys.stdin.readline()
                else:
                    line = sys.stdin.readline()
            except (EOFError, OSError, ValueError):
                break
            if line == "":
                break
            self._q.put(line.rstrip("\r\n"))

    def _reader_msvcrt(self) -> None:
        try:
            import msvcrt  # type: ignore
        except ImportError:
            # Fallback: blocking readline in this daemon thread
            while not self._stop.is_set():
                try:
                    line = sys.stdin.readline()
                except (EOFError, OSError, ValueError):
                    break
                if line == "":
                    break
                self._q.put(line.rstrip("\r\n"))
            return

        while not self._stop.is_set():
            try:
                if not msvcrt.kbhit():
                    time.sleep(0.05)
                    continue
                ch = msvcrt.getwch()
            except (EOFError, OSError):
                break
            if ch in ("\r", "\n"):
                # echo newline for Windows console
                try:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                except Exception:
                    pass
                with self._partial_lock:
                    line = "".join(self._partial)
                    self._partial.clear()
                self._q.put(line)
                continue
            if ch in ("\x08", "\x7f"):  # backspace
                with self._partial_lock:
                    if self._partial:
                        self._partial.pop()
                        try:
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                        except Exception:
                            pass
                continue
            if ch == "\x03":  # Ctrl+C — let main loop see KeyboardInterrupt via flag
                with self._partial_lock:
                    self._partial.clear()
                self._q.put("q")
                break
            with self._partial_lock:
                self._partial.append(ch)
            try:
                sys.stdout.write(ch)
                sys.stdout.flush()
            except Exception:
                pass

    def has_partial_input(self) -> bool:
        """True when user is mid-keystroke (Windows msvcrt path)."""
        with self._partial_lock:
            return bool(self._partial)

    def partial_text(self) -> str:
        with self._partial_lock:
            return "".join(self._partial)

    def poll(self) -> str | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def push(self, line: str) -> None:
        """Test / inject helper."""
        self._q.put(line)

    def close(self) -> None:
        self._stop.set()


def _dispatch_live_command(
    raw_in: str,
    *,
    emit: Callable[..., None],
    look: Any,
    tips: list[str],
    play_tracker: Any,
    live_engine: Any,
    db: Any,
    overlay_path: Path | None,
    inventory: dict[str, Any] | None,
    opponent: str | None,
) -> tuple[str, Path | None]:
    """Handle one typed command during live capture. Returns (action, overlay_path).

    action: "quit" | "ok"
    """
    if not raw_in.strip():
        return "ok", overlay_path
    low = raw_in.lower().strip()
    if low in ("q", "quit", "exit"):
        return "quit", overlay_path
    if low in ("h", "help", "?"):
        print(HELP)
        return "ok", overlay_path
    if low == "setup":
        print(CAPTURE_SETUP_NOTES.strip())
        return "ok", overlay_path
    if low == "tips":
        play_text = ""
        # live_state not passed; reprint tips under last look + playcaller
        try:
            call, _ = compute_live_call(look, opponent=opponent, db=db, live_engine=live_engine)
            play_text = call.format()
        except Exception:
            play_text = ""
        if play_text:
            print(format_play_block(play_text, adjusts=adjust_lines_from_look(look)[:2]))
        else:
            print(format_tips_block(look, tips))
        return "ok", overlay_path
    if low in ("hike", "go", "snap"):
        play_text = ""
        try:
            call, _ = compute_live_call(look, opponent=opponent, db=db, live_engine=live_engine)
            play_text = call.format()
        except Exception:
            play_text = ""
        print(format_play_block(play_text or "(hold call)", adjusts=[], hike=True))
        return "ok", overlay_path
    if low == "overlay":
        path_obj = overlay_path or _overlay_default_path()
        play_text = ""
        try:
            call, _ = compute_live_call(look, opponent=opponent, db=db, live_engine=live_engine)
            play_text = call.format()
        except Exception:
            play_text = ""
        path = write_overlay_html(
            str(path_obj), look, tips[:2], call_text=play_text, short_line=""
        )
        print(f"  overlay → {path}")
        return "ok", path_obj
    if low in ("r", "p", "s", "i", "x") or low.startswith("correct "):
        _handle_correction(raw_in, play_tracker, live_engine, db)
        return "ok", overlay_path
    if low.startswith("f ") or low.startswith("front "):
        token = raw_in.split(None, 1)[1]
        emit(
            DefenseLook(
                front=token,
                shell=look.shell,
                pressure=look.pressure,
                confidence=0.85,
                source="hotkey",
            )
        )
        return "ok", overlay_path
    if low.startswith("s ") or low.startswith("shell "):
        token = raw_in.split(None, 1)[1]
        emit(
            DefenseLook(
                front=look.front,
                shell=token,
                pressure=look.pressure,
                confidence=0.85,
                source="hotkey",
            )
        )
        return "ok", overlay_path
    if low.startswith("p ") or low.startswith("press ") or low.startswith("pressure "):
        token = raw_in.split(None, 1)[1]
        emit(
            DefenseLook(
                front=look.front,
                shell=look.shell,
                pressure=token,
                confidence=0.85,
                source="hotkey",
            )
        )
        return "ok", overlay_path
    if low.startswith("look "):
        emit(parse_look_tokens(raw_in.split(None, 1)[1]))
        return "ok", overlay_path
    if any(c.isalpha() for c in raw_in):
        emit(parse_look_tokens(raw_in))
        return "ok", overlay_path
    print("  unknown — type help")
    return "ok", overlay_path



def _overlay_default_path() -> Path:
    home = Path.home() / ".cfb-coach"
    home.mkdir(parents=True, exist_ok=True)
    return home / "copilot_overlay.html"




# Confidence at/above this → inject DefenseLook as soft live coverage evidence.
# Below: still emit a base situational playcaller Call (never tips-only).
LIVE_LOOK_CONF_THRESHOLD = 0.40

_SHELL_TO_COV = {
    "cover3": "Cover 3 Sky",
    "cover2": "Cover 2",
    "quarters": "Cover 4 Quarters",
    "two_high": "two-high",
    "single_high": "Cover 1",
    "man": "Cover 1",
}

_STATUS_INTERVAL_S = 2.0
_STATUS_INTERVAL_CALL_STABLE_S = 8.0


def build_live_situation(
    look: DefenseLook,
    *,
    obs: Any | None = None,
    short_line: str | None = None,
    conf_threshold: float = LIVE_LOOK_CONF_THRESHOLD,
) -> Situation:
    """Situation for live make_call: HUD when available, else default 1&10 pre-snap."""
    down: int | None = None
    distance: int | None = None
    side = "offense"
    if obs is not None:
        sit_hud = getattr(obs, "situation", None)
        if sit_hud is not None:
            down = getattr(sit_hud, "down", None)
            distance = getattr(sit_hud, "distance", None)
            hud_side = getattr(sit_hud, "side", None)
            if isinstance(hud_side, str):
                hs = hud_side.strip().upper()
                if hs in ("D", "DEF", "DEFENSE"):
                    side = "defense"
                elif hs in ("O", "OFF", "OFFENSE"):
                    side = "offense"

    if down is None or distance is None:
        head = (short_line or "").split("|", 1)[0].strip()
        if head and head not in ("?&?", "?"):
            parsed = parse_situation(head, default_side=side)
            if down is None and parsed.down is not None:
                down = parsed.down
            if distance is None and parsed.distance is not None:
                distance = parsed.distance
    if down is None:
        down = 1
    if distance is None:
        distance = 10

    sit = Situation(
        raw=f"{down}&{distance} pre-snap",
        side=side,
        down=int(down),
        distance=int(distance),
        short_yardage=int(distance) <= 3,
        long_yardage=(
            (int(down) in (2, 3, 4) and int(distance) >= 8)
            or (int(down) == 1 and int(distance) >= 15)
        ),
    )

    n = look.normalized()
    if float(n.confidence) >= conf_threshold:
        cov = None
        if n.pressure == "all_out":
            cov = "Cover 0"
        elif n.shell in _SHELL_TO_COV:
            cov = _SHELL_TO_COV[n.shell]
        if cov:
            sit.coverage_hint = cov
            sit.coverage_source = "live"
            sit.raw = f"{sit.raw} live {cov}"
        elif n.pressure in ("blitz_left", "blitz_right", "blitz_middle"):
            sit.coverage_hint = "pressure"
            sit.coverage_source = "live"
            sit.raw = f"{sit.raw} live pressure"
    return sit


def compute_live_call(
    look: DefenseLook,
    *,
    opponent: str | None,
    db: Any = None,
    live_engine: Any = None,
    obs: Any | None = None,
    short_line: str | None = None,
    rng: Any = None,
) -> tuple[Call, Situation]:
    """Always return a playcaller Call — even when DefenseLook is unknown/low-conf."""
    sit = build_live_situation(look, obs=obs, short_line=short_line)
    oid = (opponent or "cpu").strip() or "cpu"
    if rng is None:
        import random

        # Stable-ish across rapid frames so CALL does not thrash every tick
        seed_key = (
            f"{oid}|{sit.down}&{sit.distance}|{sit.coverage_hint or ''}|"
            f"{look.shell}|{look.pressure}|{int(float(look.confidence) * 10)}"
        )
        rng = random.Random(seed_key)
    call = make_call(
        sit,
        oid,
        db,
        rng=rng,
        live_engine=live_engine,
    )
    return call, sit



def look_signature(look: DefenseLook) -> tuple[str, str, str]:
    n = look.normalized()
    return (n.front, n.shell, n.pressure)


def adjust_lines_from_look(look: DefenseLook) -> list[str]:
    """Short ADJUST lines from live shell/pressure — not a full play replace."""
    n = look.normalized()
    lines: list[str] = []
    p = n.pressure
    s = n.shell
    if p == "all_out":
        lines.append("Cover 0 heat — hot X / max protect, quick game")
    elif p == "blitz_left":
        lines.append("pressure left — slide L, hot ready")
    elif p == "blitz_right":
        lines.append("pressure right — slide R, hot ready")
    elif p == "blitz_middle":
        lines.append("pressure mid — slide MID, hot ready")
    elif p == "show":
        lines.append("pressure show — confirm rush; hot ready if they come")
    if s == "cover3":
        lines.append("shell C3 — flood / crossers alive")
    elif s == "cover2":
        lines.append("shell C2 — soft underneath OK")
    elif s == "quarters":
        lines.append("shell quarters — under/crossers; seams careful")
    elif s == "two_high" and p in ("none", "unknown", "show"):
        lines.append("two-high — run first still good")
    elif s == "man":
        lines.append("man — rubs / mesh alive")
    elif s == "single_high" and p in ("none", "unknown", "show"):
        lines.append("single-high — seams / PA alive")
    # de-dup + cap 2
    seen: set[str] = set()
    out: list[str] = []
    for t in lines:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= 2:
            break
    return out


def audible_warranted(look: DefenseLook) -> bool:
    """Only replace the whole PLAY when heat clearly demands it."""
    n = look.normalized()
    if float(n.confidence) < LIVE_LOOK_CONF_THRESHOLD:
        return False
    return n.pressure == "all_out" or (
        n.shell == "man" and n.pressure in ("blitz_left", "blitz_right", "blitz_middle", "all_out")
    )


def format_play_block(
    play_text: str,
    *,
    short_line: str = "",
    adjusts: list[str] | None = None,
    hike: bool = False,
) -> str:
    """Terminal cadence: PLAY → optional ADJUST → optional HIKE."""
    lines: list[str] = []
    if short_line:
        lines.append(short_line)
    lines.append("PLAY")
    lines.append(play_text.strip() or "(no call)")
    for a in adjusts or []:
        lines.append(f"ADJUST  {a}")
    if hike:
        lines.append("HIKE — go")
    return "\n".join(lines)


def _format_live_overlay(
    short_line: str,
    tips: list[str],
    *,
    play_text: str = "",
    adjusts: list[str] | None = None,
    hike: bool = False,
    tendency_lines: list[str] | None = None,
    counter: str = "",
    fps: float | None = None,
    latency_ms: float | None = None,
) -> str:
    """Prefer PLAY/ADJUST/HIKE; tendency dump is secondary / quiet."""
    if play_text:
        block = format_play_block(
            play_text, short_line=short_line, adjusts=adjusts, hike=hike
        )
        # Optional one-liner counter only (no big tendency dump)
        if counter:
            block += f"\nCOUNTER  {counter}"
        return block
    # Fallback (no playcaller yet)
    lines = ["LIVE GAME", short_line]
    if tips:
        lines.append("TIPS")
        for i, tip in enumerate(tips[:2], 1):
            lines.append(f"  {i}. {tip}")
    return "\n".join(lines)


def _load_inventory(opponent_id: str | None) -> dict[str, Any] | None:
    if not opponent_id:
        return None
    try:
        from cfb_coach.db import CoachDB, resolve_db_path_from_env
        from cfb_coach.opponents import resolve_opponent
        from cfb_coach.prep import load_opponent_profile

        oid = resolve_opponent(opponent_id) or opponent_id
        db = CoachDB(resolve_db_path_from_env())
        try:
            opp = load_opponent_profile(oid, db)
            return (opp or {}).get("inventory") or opp
        finally:
            db.close()
    except Exception:
        return None


def _want_live(args: Any) -> bool:
    return bool(
        getattr(args, "window", None)
        or getattr(args, "screen_region", None)
        or getattr(args, "video", None)
        or getattr(args, "device", None)
        or getattr(args, "live", False)
    )


def run_watch(args: Any) -> int:
    """Entry used by CLI cmd_watch."""
    demo = bool(getattr(args, "demo", False))
    once = bool(getattr(args, "once", False))
    image = getattr(args, "image", None)
    video = getattr(args, "video", None)
    overlay = getattr(args, "overlay", None)
    opponent = getattr(args, "opponent", None)
    show_setup = bool(getattr(args, "setup", False))
    interval = float(getattr(args, "interval", 1.5) or 1.5)
    calibrate = bool(getattr(args, "calibrate", False))
    debug = bool(getattr(args, "debug", False))
    tts_on = bool(getattr(args, "tts", False))
    target_fps = float(getattr(args, "fps", 8.0) or 8.0)
    dynasty = getattr(args, "dynasty", None) or "alabama"
    record_plays = bool(getattr(args, "record_plays", False))
    post_one_liner = bool(getattr(args, "post_play_line", True))

    if show_setup:
        print(CAPTURE_SETUP_NOTES.strip())
        return 0

    if getattr(args, "list_windows", False):
        from cfb_coach.vision.capture import list_visible_windows

        titles = list_visible_windows()
        if titles is None:
            print(
                "list-windows needs Windows + pywin32 (pip install pywin32).",
                file=sys.stderr,
            )
            return 2
        if not titles:
            print("(no visible windows with titles)")
            return 0
        print(f"{len(titles)} visible window title(s):")
        for t in titles:
            print(f"  {t}")
        print('\nPick one substring for: cfb-coach watch --window "…"')
        return 0

    if calibrate:
        from cfb_coach.vision.calibrate import run_calibrate

        reg = getattr(args, "screen_region", None)
        if reg in ("calib", True, ""):
            reg = None
        return run_calibrate(
            window=getattr(args, "window", None),
            region=reg,
        )

    inventory = _load_inventory(opponent)
    overlay_path: Path | None = None
    no_overlay = bool(getattr(args, "no_overlay", False))
    # Default ON for live window/screen-region/video/device modes
    if no_overlay or overlay in ("off", "none", False):
        overlay_path = None
    elif overlay is True or overlay == "auto":
        overlay_path = _overlay_default_path()
    elif isinstance(overlay, str) and overlay:
        overlay_path = Path(overlay)
    elif overlay is None and _want_live(args):
        overlay_path = _overlay_default_path()

    capture: Any = StubCapture()
    look = DefenseLook(source="stub", confidence=0.0, notes="waiting for look")
    tips = tips_from_look(look, inventory=inventory, opponent_id=opponent)
    demo_idx = 0
    demo_looks = demo_sequence()

    tts = None
    if tts_on:
        from cfb_coach.vision.tts import TipTTS

        tts = TipTTS(enabled=True)
        if not tts.available:
            print("TTS requested but Windows SAPI unavailable — continuing silent.")

    db = None
    log_obs = None
    session = None
    live_engine = None
    try:
        from cfb_coach.db import CoachDB, resolve_db_path_from_env
        from cfb_coach.live_tendency import LiveTendencyEngine
        from cfb_coach.session import start_session

        db = CoachDB(resolve_db_path_from_env())
        log_obs = lambda obs: db.log_vision_observation(obs.to_dict())  # noqa: E731
        live_engine = LiveTendencyEngine()
        if opponent:
            session = start_session(opponent, dynasty=dynasty, db=db)
    except Exception:
        db = None
        log_obs = None
        live_engine = None
        session = None

    print("SCREEN CO-PILOT v1.9.4 — Xbox stays in your hands (sidecar only)")
    print("Prototype source: Xbox Remote Play on laptop · play on HDMI monitor")
    print("Future: capture-card drop-in backend. See: watch --setup")
    print("Now: demo + hotkeys + optional live/--image/--video. Type 'help'.")
    if opponent:
        print(f"Opponent context: {opponent}")
    if session:
        print(f"Session: {session.session_id} (dynasty={session.dynasty})")
    print("-" * 60)

    overlay_browser_opened = False

    def _open_overlay_browser_once(path: Path) -> None:
        nonlocal overlay_browser_opened
        if overlay_browser_opened:
            return
        try:
            import webbrowser

            webbrowser.open(path.resolve().as_uri())
            overlay_browser_opened = True
            print(f"  overlay opened → {path}  (keep Xbox Remote Play focused; glance here)")
        except Exception as e:
            print(f"  overlay write ok but browser open failed: {e}")

    live_state: dict[str, Any] = {
        "look": look,
        "tips": tips,
        "overlay_path": overlay_path,
        "play_text": "",
        "adjusts": [],
        "hike": False,
    }

    def emit(
        current: DefenseLook,
        *,
        short_line: str | None = None,
        obs: Any | None = None,
        play_text: str | None = None,
        adjusts: list[str] | None = None,
        hike: bool = False,
        quiet_overlay: bool = False,
    ) -> str:
        """Emit PLAY (primary) + optional ADJUST + optional HIKE. Returns play text."""
        nonlocal tips, look, overlay_path
        look = current.normalized()
        tips = tips_from_look(look, inventory=inventory, opponent_id=opponent)
        # Always compute a real playcaller Call when play_text not supplied
        if play_text is None:
            try:
                call, _sit = compute_live_call(
                    look,
                    opponent=opponent,
                    db=db,
                    live_engine=live_engine,
                    obs=obs,
                    short_line=short_line,
                )
                play_text = call.format()
            except Exception:
                play_text = live_state.get("play_text") or ""
        play_text = (play_text or "").strip()
        # ADJUST only when caller passes them (look change) — not on every PLAY
        adj = list(adjusts) if adjusts is not None else []

        live_state["look"] = look
        live_state["tips"] = tips
        live_state["overlay_path"] = overlay_path
        live_state["play_text"] = play_text
        live_state["adjusts"] = adj
        live_state["hike"] = hike

        counter = ""
        try:
            counter = live_engine.current_counter() if live_engine else ""
        except Exception:
            counter = ""
        # Don't spam trivial/base counters — PLAY/ADJUST/HIKE first
        if counter and counter.strip().lower() in ("base situational", "base", "none", ""):
            counter = ""

        print()
        if short_line or play_text:
            print(
                _format_live_overlay(
                    short_line or "",
                    tips,
                    play_text=play_text,
                    adjusts=adj,
                    hike=hike,
                    counter=counter,
                )
            )
        else:
            # Interactive typed looks without short_line: still lead with PLAY
            if play_text:
                print(format_play_block(play_text, adjusts=adj, hike=hike))
            else:
                print(format_tips_block(look, tips))

        if overlay_path is not None:
            path = write_overlay_html(
                str(overlay_path),
                look,
                adj[:2] if adj else tips[:2],
                call_text=play_text,
                short_line=short_line or "",
            )
            if not quiet_overlay and not overlay_browser_opened:
                _open_overlay_browser_once(Path(path))
            elif not quiet_overlay and not live_state.get("_overlay_announced"):
                print(f"  overlay → {path}")
                live_state["_overlay_announced"] = True
        if tts is not None and play_text:
            try:
                tts.speak_tips([play_text] + adj[:1])
            except Exception:
                pass
        return play_text

    use_pipeline = bool(
        video
        or getattr(args, "window", None)
        or getattr(args, "screen_region", None)
        or getattr(args, "device", None)
        or (image and debug)
    )

    if use_pipeline:
        return _run_pipeline_loop(
            args,
            emit=emit,
            once=once,
            debug=debug,
            target_fps=target_fps,
            log_obs=log_obs,
            db=db,
            image=image,
            video=video,
            session=session,
            live_engine=live_engine,
            record_plays=record_plays,
            post_one_liner=post_one_liner,
            live_state=live_state,
            inventory=inventory,
            opponent=opponent,
        )

    if image:
        cap = ImageFileCapture(image)
        raw = cap.grab()
        raw_bytes = None
        if raw is None:
            print(f"Image not found: {image}", file=sys.stderr)
            _close_db(db)
            return 2
        if isinstance(raw, (bytes, bytearray)):
            raw_bytes = bytes(raw)
        else:
            raw_bytes = getattr(raw, "png_bytes", None)
            if raw_bytes is None and getattr(raw, "bgr", None) is not None:
                try:
                    from cfb_coach.vision.capture import frame_to_bytes

                    raw_bytes = frame_to_bytes(raw)
                except Exception:
                    raw_bytes = None
        if not raw_bytes:
            raw_bytes = Path(image).read_bytes()
        emit(naive_roi_heuristic(raw_bytes))
        if once or not demo:
            if once or not sys.stdin.isatty():
                _close_session(session, db)
                _close_db(db)
                return 0

    if demo and (once or not sys.stdin.isatty()):
        for i, sample in enumerate(demo_looks):
            emit(sample)
            if once:
                _close_session(session, db)
                _close_db(db)
                return 0
            if i < len(demo_looks) - 1:
                time.sleep(max(0.2, interval))
        _close_session(session, db)
        _close_db(db)
        return 0

    if demo:
        emit(demo_looks[0])
        demo_idx = 1

    play_tracker = None
    if session is not None:
        from cfb_coach.vision.play_tracker import PlayTracker

        play_tracker = PlayTracker()
        play_tracker.configure(
            session_id=session.session_id,
            game_id=session.session_id,
            opponent_id=session.opponent_id,
            on_play_end=lambda rec: _on_play_end(rec, db, live_engine, session, post_one_liner),
        )

    while True:
        try:
            raw_in = input("look> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not raw_in:
            continue
        low = raw_in.lower()
        if low in ("q", "quit", "exit"):
            break
        if low in ("h", "help", "?"):
            print(HELP)
            continue
        if low == "setup":
            print(CAPTURE_SETUP_NOTES.strip())
            continue
        if low == "tips":
            try:
                call, _ = compute_live_call(look, opponent=opponent, db=db, live_engine=live_engine)
                print(format_play_block(call.format(), adjusts=adjust_lines_from_look(look)[:2]))
            except Exception:
                print(format_tips_block(look, tips))
            continue
        if low in ("hike", "go", "snap"):
            try:
                call, _ = compute_live_call(look, opponent=opponent, db=db, live_engine=live_engine)
                pt = call.format()
            except Exception:
                pt = live_state.get("play_text") or ""
            print(format_play_block(pt or "(hold call)", hike=True))
            continue
        if low == "overlay":
            if overlay_path is None:
                overlay_path = _overlay_default_path()
            try:
                call, _ = compute_live_call(look, opponent=opponent, db=db, live_engine=live_engine)
                pt = call.format()
            except Exception:
                pt = ""
            path = write_overlay_html(
                str(overlay_path), look, tips[:2], call_text=pt, short_line=""
            )
            print(f"  overlay → {path}")
            continue
        if low == "demo" or low.startswith("demo "):
            sample = demo_looks[demo_idx % len(demo_looks)]
            demo_idx += 1
            emit(sample)
            continue
        # Quick result corrections
        if low in ("r", "p", "s", "i", "x") or low.startswith("correct "):
            _handle_correction(raw_in, play_tracker, live_engine, db)
            continue
        if low.startswith("f ") or low.startswith("front "):
            token = raw_in.split(None, 1)[1]
            new = DefenseLook(
                front=token,
                shell=look.shell,
                pressure=look.pressure,
                confidence=0.85,
                source="hotkey",
            )
            emit(new)
            continue
        if low.startswith("s ") or low.startswith("shell "):
            token = raw_in.split(None, 1)[1]
            new = DefenseLook(
                front=look.front,
                shell=token,
                pressure=look.pressure,
                confidence=0.85,
                source="hotkey",
            )
            emit(new)
            continue
        if low.startswith("p ") or low.startswith("press ") or low.startswith("pressure "):
            token = raw_in.split(None, 1)[1]
            new = DefenseLook(
                front=look.front,
                shell=look.shell,
                pressure=token,
                confidence=0.85,
                source="hotkey",
            )
            emit(new)
            continue
        if low.startswith("look "):
            emit(parse_look_tokens(raw_in.split(None, 1)[1]))
            continue
        if any(c.isalpha() for c in raw_in):
            emit(parse_look_tokens(raw_in))
            continue
        print("  unknown — type help")

    try:
        capture.close()
    except Exception:
        pass
    _close_session(session, db)
    _close_db(db)
    return 0


def _on_play_end(
    rec: Any,
    db: Any,
    live_engine: Any,
    session: Any,
    post_one_liner: bool,
) -> None:
    if session is not None:
        session.play_count = getattr(session, "play_count", 0) + 1
    if db is not None:
        try:
            db.log_play_record(rec)
        except Exception:
            pass
    alerts = []
    if live_engine is not None:
        try:
            alerts = live_engine.add_play(rec)
        except Exception:
            alerts = []
        if db is not None:
            for a in alerts:
                try:
                    db.log_live_tendency_event(
                        session_id=getattr(session, "session_id", "") or rec.session_id,
                        kind=a.kind,
                        signal=a.signal,
                        message=a.message,
                    )
                except Exception:
                    pass
    if post_one_liner:
        tags = ",".join(rec.concept_tags[:2]) if rec.concept_tags else ""
        y = f" {rec.yards:+d}" if rec.yards is not None else ""
        print(
            f"  ▶ play {rec.play_id}: {rec.play_family}"
            f"{(' ' + tags) if tags else ''} → {rec.result_type}{y}"
        )
    for a in alerts:
        print(f"  ⚠ {a.message}")


def _handle_correction(
    raw_in: str,
    play_tracker: Any,
    live_engine: Any,
    db: Any,
) -> None:
    low = raw_in.lower().strip()
    fields: dict[str, Any] = {}
    if low == "r":
        fields = {"result_type": "run", "play_family": "RUN_INSIDE", "family_hint": "inside zone"}
    elif low == "p":
        fields = {"result_type": "completion", "family_hint": "dropback"}
    elif low == "s":
        fields = {"result_type": "sack", "play_family": "SACK"}
    elif low == "i":
        fields = {"result_type": "int", "play_family": "DROPBACK_PASS"}
    elif low == "x":
        fields = {"explosive": True}
    elif low.startswith("correct "):
        parts = raw_in.split(None, 2)
        if len(parts) < 3:
            print("  usage: correct family CROSSERS | correct result completion | correct yards 12")
            return
        key, val = parts[1].lower(), parts[2]
        if key in ("family", "play_family"):
            fields = {"play_family": val.upper(), "family_hint": val}
        elif key in ("result", "result_type"):
            fields = {"result_type": val.lower()}
        elif key == "yards":
            try:
                fields = {"yards": int(val)}
            except ValueError:
                print("  yards must be int")
                return
        elif key in ("concept", "tag"):
            fields = {"concept_tags": [val.upper()], "family_hint": val}
        else:
            fields = {key: val}
    else:
        return

    rec = None
    if play_tracker is not None:
        rec = play_tracker.correct_last(**fields)
    if rec is None and live_engine is not None and live_engine.plays:
        pid = live_engine.plays[-1].play_id
        live_engine.correct_play(pid, **{k: v for k, v in fields.items() if k != "family_hint"})
        rec = live_engine.plays[-1]
    if rec is None:
        print("  no PlayRecord to correct yet")
        return
    if "family_hint" in fields and play_tracker is not None:
        pass
    if db is not None:
        try:
            db.log_play_record(rec)
        except Exception:
            pass
    print(f"  corrected {rec.play_id}: family={rec.play_family} result={rec.result_type} yards={rec.yards}")


def _close_db(db: Any) -> None:
    if db is not None:
        try:
            db.close()
        except Exception:
            pass


def _close_session(session: Any, db: Any) -> None:
    if session is None:
        return
    try:
        from cfb_coach.session import end_session

        end_session(session, db=db)
    except Exception:
        pass


def _maybe_record_clip(
    *,
    record_plays: bool,
    session: Any,
    rec: Any,
    buffer: list[Any],
) -> None:
    """Optional rolling buffer → ~/.cfb-coach/games/<id>/plays/NNNN.mp4."""
    if not record_plays or session is None or rec is None:
        return
    low_conf = float(rec.conf("family", 1.0)) < 0.45 or float(rec.end_confidence) < 0.4
    if not (low_conf or rec.explosive or getattr(rec, "corrected", False)):
        return
    if not buffer:
        return
    try:
        import cv2  # type: ignore
    except ImportError:
        return
    out_dir = Path.home() / ".cfb-coach" / "games" / session.session_id / "plays"
    out_dir.mkdir(parents=True, exist_ok=True)
    # play_id like sess-0001 → 0001.mp4
    num = "".join(ch for ch in rec.play_id if ch.isdigit())[-4:] or "0000"
    path = out_dir / f"{num}.mp4"
    try:
        h, w = buffer[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            8.0,
            (w, h),
        )
        for fr in buffer[-64:]:
            writer.write(fr)
        writer.release()
        print(f"  recorded {path}")
    except Exception:
        pass


def _run_pipeline_loop(
    args: Any,
    *,
    emit: Any,
    once: bool,
    debug: bool,
    target_fps: float,
    log_obs: Any,
    db: Any,
    image: str | None,
    video: str | None,
    session: Any = None,
    live_engine: Any = None,
    record_plays: bool = False,
    post_one_liner: bool = True,
    live_state: dict[str, Any] | None = None,
    inventory: dict[str, Any] | None = None,
    opponent: str | None = None,
) -> int:
    from cfb_coach.vision.calibrate import load_calib
    from cfb_coach.vision.pipeline import VisionPipeline, build_capture_from_args
    from cfb_coach.vision.threaded_capture import ThreadedCapture

    live_state = live_state if live_state is not None else {}
    overlay_path: Path | None = live_state.get("overlay_path")

    try:
        cap = build_capture_from_args(args)
    except ImportError as e:
        print(f"Vision capture unavailable: {e}", file=sys.stderr)
        print("Install on native Windows: pip install -e \".[vision]\"", file=sys.stderr)
        _close_session(session, db)
        _close_db(db)
        return 2

    if getattr(cap, "name", "") == "stub" and (
        getattr(args, "window", None) or getattr(args, "screen_region", None)
    ):
        print(
            "Live capture deps missing (dxcam/mss). "
            "pip install -e \".[vision]\" on native Windows Python.",
            file=sys.stderr,
        )
        _close_session(session, db)
        _close_db(db)
        return 2

    # Threaded capture for live/window (overwrite latest; analysis drops stale)
    threaded = None
    want_thread = bool(
        getattr(args, "window", None)
        or getattr(args, "screen_region", None)
        or getattr(args, "device", None)
        or video
    )
    if want_thread and getattr(cap, "name", "") != "stub":
        threaded = ThreadedCapture(cap, target_fps=max(target_fps * 2, 15.0))
        threaded.start()
        pipe_cap: Any = threaded
    else:
        pipe_cap = cap

    pipe = VisionPipeline(pipe_cap, calib=load_calib())
    if session is not None:
        pipe.lifecycle.configure(
            session_id=session.session_id,
            game_id=session.session_id,
            opponent_id=session.opponent_id,
            on_play_end=lambda rec: _on_play_end(rec, db, live_engine, session, post_one_liner),
        )

    dbg = None
    if debug:
        from cfb_coach.vision.debug_view import DebugView

        dbg = DebugView()
        if not dbg.available:
            print("Debug view needs opencv-python — continuing without window.")

    print(f"Pipeline capture={getattr(pipe_cap, 'name', '?')} target_fps={target_fps:.0f}")
    matched = getattr(cap, "matched_title", None)
    if matched:
        print(f"Window match: {matched!r}")
    elif getattr(args, "window", None) and "dxcam" in str(getattr(cap, "name", "")):
        print(
            'No window title matched yet — try: cfb-coach watch --list-windows'
        )
    if session:
        print(f"Logging plays → session {session.session_id}")
    print(_LIVE_CMD_HINT)

    stdin_q = StdinCommandQueue()
    stdin_q.start()

    delay = 1.0 / max(1.0, target_fps)
    frames = 0
    last_short = ""
    roll_buf: list[Any] = []
    loop_start = time.time()
    last_frame_wall = loop_start
    last_heartbeat = 0.0
    last_status = 0.0
    saw_frame = False
    troub_printed = False
    last_obs: Any = None
    last_look_obj: Any = live_state.get("look")
    quit_requested = False

    # Pre-snap cadence: PLAY → ADJUST → HIKE
    HIKE_STABLE_S = 1.8
    HIKE_TIMEOUT_S = 4.5
    pre_play_text = str(live_state.get("play_text") or "")
    pre_sig: tuple[str, str, str] | None = None
    pre_adjusts: list[str] = []
    pre_hike_done = False
    pre_play_emitted = False
    pre_stable_since = 0.0
    pre_window_start = 0.0
    last_sit_key = ""
    overlay_refresh_at = 0.0

    def _current_capture_name() -> str:
        return str(getattr(pipe_cap, "name", "?") or "?")

    def _poll_commands() -> bool:
        """Drain stdin queue. Return True if quit requested."""
        nonlocal overlay_path, quit_requested, last_look_obj
        while True:
            cmd = stdin_q.poll()
            if cmd is None:
                break
            action, overlay_path = _dispatch_live_command(
                cmd,
                emit=emit,
                look=live_state.get("look") or last_look_obj,
                tips=list(live_state.get("tips") or []),
                play_tracker=getattr(pipe, "lifecycle", None),
                live_engine=live_engine,
                db=db,
                overlay_path=overlay_path,
                inventory=inventory,
                opponent=opponent,
            )
            live_state["overlay_path"] = overlay_path
            last_look_obj = live_state.get("look") or last_look_obj
            if action == "quit":
                quit_requested = True
                return True
        return False

    hb_line_len = 0

    def _end_heartbeat_line() -> None:
        nonlocal hb_line_len
        if hb_line_len > 0:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass
            hb_line_len = 0

    def _print_heartbeat_line(msg: str) -> None:
        """Single-line status via \\r — does not flood scrollback."""
        nonlocal hb_line_len
        pad = max(0, hb_line_len - len(msg))
        try:
            sys.stdout.write("\r" + msg + (" " * pad))
            sys.stdout.flush()
        except Exception:
            # Fallback if stdout odd
            print(msg)
        hb_line_len = len(msg)

    def _emit_heartbeat(now: float) -> None:
        nonlocal last_heartbeat
        typing = stdin_q.has_partial_input() if hasattr(stdin_q, "has_partial_input") else False
        if not _should_emit_heartbeat(
            now=now,
            loop_start=loop_start,
            last_heartbeat=last_heartbeat,
            last_frame_wall=last_frame_wall,
            troub_printed=troub_printed,
            typing=typing,
        ):
            return
        _print_heartbeat_line(_waiting_frames_msg(_current_capture_name()))
        last_heartbeat = now

    def _emit_troubleshoot(now: float) -> None:
        nonlocal troub_printed, last_heartbeat
        if troub_printed:
            return
        if saw_frame:
            return
        if now - loop_start < 5.0:
            return
        _end_heartbeat_line()
        print(_TROUBLESHOOT_NO_FRAMES.rstrip())
        troub_printed = True
        last_heartbeat = now  # next heartbeat in 15s

    def _emit_status(now: float, *, force: bool = False) -> None:
        nonlocal last_status
        if last_obs is None and not force:
            return
        interval = (
            _STATUS_INTERVAL_CALL_STABLE_S if pre_play_emitted and pre_hike_done
            else _STATUS_INTERVAL_S
        )
        if not force and now - last_status < interval:
            return
        short = last_obs.short_line() if last_obs is not None else last_short or "?"
        look_lbl = ""
        if last_look_obj is not None:
            try:
                look_lbl = last_look_obj.label()
            except Exception:
                look_lbl = ""
        play_state = getattr(last_obs, "play_state", "") if last_obs is not None else ""
        print(
            _status_line(
                short,
                fps=getattr(pipe, "fps", None),
                play_state=str(play_state or ""),
                look_label=look_lbl,
            )
        )
        last_status = now

    try:
        while True:
            if _poll_commands():
                break

            t0 = time.time()
            obs = None
            look = None
            try:
                # Prefer fresh frame when threaded
                if threaded is not None:
                    fresh = threaded.grab_fresh()
                    if fresh is None:
                        now = time.time()
                        _emit_troubleshoot(now)
                        _emit_heartbeat(now)
                        if saw_frame:
                            _emit_status(now)
                        time.sleep(min(0.02, delay))
                        if once and frames > 0:
                            break
                        if not sys.stdin.isatty() and frames >= 30:
                            break
                        continue
                    obs = pipe.process_frame(fresh)
                    look = obs.to_defense_look()
                    if record_plays and getattr(fresh, "bgr", None) is not None:
                        roll_buf.append(fresh.bgr)
                        if len(roll_buf) > 96:
                            roll_buf = roll_buf[-96:]
                else:
                    obs, look = pipe.step()
            except Exception as e:
                print(f"pipeline error: {e}", file=sys.stderr)
                break

            if obs is None or look is None:
                now = time.time()
                _emit_troubleshoot(now)
                _emit_heartbeat(now)
                time.sleep(min(0.02, delay))
                continue

            now = time.time()
            if not saw_frame:
                _end_heartbeat_line()
                print(_CAPTURE_OK)
                saw_frame = True
            last_frame_wall = now
            last_obs = obs
            last_look_obj = look
            frames += 1

            if log_obs is not None:
                try:
                    log_obs(obs)
                except Exception:
                    pass

            # If pipeline completed a play this tick
            last_play = getattr(pipe, "_last_play", None)
            if last_play is not None and record_plays:
                _maybe_record_clip(
                    record_plays=record_plays,
                    session=session,
                    rec=last_play,
                    buffer=roll_buf,
                )

            short = obs.short_line()
            extras = obs.extras or {}
            play_state = str(getattr(obs, "play_state", "") or "")
            sit_hud = getattr(obs, "situation", None)
            sit_key = ""
            if sit_hud is not None:
                sit_key = f"{getattr(sit_hud, 'down', None)}&{getattr(sit_hud, 'distance', None)}"
            else:
                sit_key = short.split("|", 1)[0].strip()

            # Reset pre-snap window on new down/distance or leaving active play
            new_presnap = False
            if sit_key and sit_key != last_sit_key:
                last_sit_key = sit_key
                new_presnap = True
            if play_state in ("PLAY_ACTIVE", "PLAY_ENDING", "POST_PLAY", "BETWEEN_PLAYS"):
                if pre_play_emitted and play_state != "PRE_SNAP":
                    # Snap happened — clear for next window
                    if play_state in ("PLAY_ACTIVE", "PLAY_ENDING", "POST_PLAY"):
                        pre_play_emitted = False
                        pre_hike_done = False
                        pre_sig = None
                        pre_adjusts = []
                        pre_play_text = ""
                        pre_stable_since = 0.0
                        pre_window_start = 0.0

            sig = look_signature(look)
            conf_ok = float(getattr(look, "confidence", 0.0) or 0.0) >= LIVE_LOOK_CONF_THRESHOLD

            # --- PRE-SNAP: emit PLAY once per window ---
            if (
                (not pre_play_emitted)
                and (
                    frames == 1
                    or new_presnap
                    or play_state in ("PRE_SNAP", "UNKNOWN", "BETWEEN_PLAYS", "")
                    or short != last_short
                )
            ):
                try:
                    call, _sit = compute_live_call(
                        look,
                        opponent=opponent,
                        db=db,
                        live_engine=live_engine,
                        obs=obs,
                        short_line=short,
                    )
                    pre_play_text = call.format()
                except Exception:
                    if not pre_play_text:
                        pre_play_text = "Gun Bunch X Nasty — Mesh Spot | No adj | Spot → Drag"
                pre_sig = sig
                pre_adjusts = []
                pre_hike_done = False
                pre_play_emitted = True
                pre_window_start = now
                pre_stable_since = now
                last_short = short
                emit(
                    look,
                    short_line=short,
                    obs=obs,
                    play_text=pre_play_text,
                    adjusts=[],
                    hike=False,
                )
                last_status = now
                # quiet overlay refresh stamp
                overlay_refresh_at = now

            # --- ADJUST: shell/pressure change before snap (do not replace PLAY) ---
            elif (
                pre_play_emitted
                and not pre_hike_done
                and pre_sig is not None
                and sig != pre_sig
                and conf_ok
            ):
                adj = adjust_lines_from_look(look)
                # Only print if adjusts are new vs last printed
                new_adj = [a for a in adj if a not in pre_adjusts]
                if audible_warranted(look):
                    # Rare: replace PLAY (Cover 0 / man+heat)
                    try:
                        call, _sit = compute_live_call(
                            look,
                            opponent=opponent,
                            db=db,
                            live_engine=live_engine,
                            obs=obs,
                            short_line=short,
                        )
                        pre_play_text = call.format()
                    except Exception:
                        pass
                    pre_adjusts = list(adj)
                    pre_sig = sig
                    pre_stable_since = now
                    emit(
                        look,
                        short_line=short,
                        obs=obs,
                        play_text=pre_play_text,
                        adjusts=pre_adjusts,
                        hike=False,
                    )
                    last_status = now
                elif new_adj:
                    pre_adjusts = list(dict.fromkeys(pre_adjusts + new_adj))[:2]
                    pre_sig = sig
                    pre_stable_since = now
                    # Print ADJUST lines only (keep PLAY above in scrollback)
                    print()
                    for a in new_adj[:2]:
                        print(f"ADJUST  {a}")
                    # Refresh overlay quietly with full block
                    if overlay_path is not None:
                        write_overlay_html(
                            str(overlay_path),
                            look,
                            pre_adjusts[:2],
                            call_text=pre_play_text,
                            short_line=short,
                        )
                    last_status = now
                else:
                    pre_sig = sig
                    pre_stable_since = now
            else:
                # Look unchanged — track stability for HIKE
                if pre_play_emitted and not pre_hike_done:
                    if pre_stable_since <= 0:
                        pre_stable_since = now
                    stable_for = now - pre_stable_since
                    timed_out = (
                        pre_window_start > 0 and (now - pre_window_start) >= HIKE_TIMEOUT_S
                    )
                    if stable_for >= HIKE_STABLE_S or timed_out:
                        pre_hike_done = True
                        print()
                        print("HIKE — go")
                        if overlay_path is not None:
                            write_overlay_html(
                                str(overlay_path),
                                look,
                                pre_adjusts[:2],
                                call_text=pre_play_text + "   · HIKE",
                                short_line=short,
                            )
                        last_status = now
                last_short = short

            # Quiet overlay refresh (1.5s) without console spam when PLAY unchanged
            if (
                overlay_path is not None
                and pre_play_text
                and now - overlay_refresh_at >= 1.5
            ):
                write_overlay_html(
                    str(overlay_path),
                    look,
                    pre_adjusts[:2],
                    call_text=pre_play_text + ("   · HIKE" if pre_hike_done else ""),
                    short_line=short,
                )
                overlay_refresh_at = now

            # Sparse status when nothing changed (longer after HIKE)
            status_gap = (
                _STATUS_INTERVAL_CALL_STABLE_S
                if pre_hike_done
                else _STATUS_INTERVAL_S
            )
            if now - last_status >= status_gap and pre_play_emitted:
                print(
                    _status_line(
                        short,
                        fps=pipe.fps,
                        play_state=play_state,
                        look_label=look.label(),
                    )
                )
                last_status = now
                if debug:
                    print(
                        f"      drop={extras.get('dropped', 0)} "
                        f"lat={extras.get('latency_ms', 0):.0f}ms "
                        f"q={extras.get('queue', extras.get('queue_depth', '?'))}"
                    )

            if dbg is not None:
                bgr = None
                try:
                    if threaded is not None:
                        fr = threaded.grab()
                        bgr = getattr(fr, "bgr", None) if fr is not None else None
                    else:
                        frame = cap.grab()
                        bgr = getattr(frame, "bgr", None) if frame is not None else None
                except Exception:
                    bgr = None
                counter = ""
                try:
                    counter = live_engine.current_counter() if live_engine else ""
                except Exception:
                    counter = ""
                lines = [
                    short,
                    f"state={play_state}",
                    f"PLAY: {pre_play_text}" if pre_play_text else look.label(),
                    "HIKE" if pre_hike_done else "",
                    f"COUNTER: {counter}" if counter else "",
                ]
                dbg.show(
                    bgr,
                    fps=pipe.fps,
                    lines=[x for x in lines if x],
                    rois=(pipe.calib or {}).get("rois"),
                )

            if _poll_commands():
                break

            if once:
                break
            if image and not video and not getattr(args, "window", None):
                if once or not sys.stdin.isatty():
                    break
                break

            elapsed = time.time() - t0
            time.sleep(max(0.0, delay - elapsed))

            if video and once:
                break
            if not sys.stdin.isatty() and not getattr(args, "window", None) and frames >= 3:
                if not video:
                    break
                if frames >= 30:
                    break
    except KeyboardInterrupt:
        print()
    finally:
        stdin_q.close()
        if dbg is not None:
            dbg.close()
        if threaded is not None:
            threaded.close()
        else:
            pipe.close()
        _close_session(session, db)
        _close_db(db)
    return 0
