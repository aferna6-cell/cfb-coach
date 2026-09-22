"""cfb-coach watch / copilot — SCREEN CO-PILOT live loop (v1.9.1).

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
Screen Co-Pilot v1.9.1 — you stay on sticks; coach suggests only.

Typed commands (then Enter):
  f <front>      even|odd|nickel|dime|goal_line
  s <shell>      single_high|two_high|cover2|cover3|quarters|man
  p <pressure>   none|show|blitz_left|blitz_right|blitz_middle|all_out
  look <tokens>  e.g.  look nickel c3 blitz_left
  demo           advance one canned look
  tips           reprint last tips
  overlay        rewrite overlay HTML (if --overlay path set)
  correct …      correct last PlayRecord (see below)
  R/P/S/I/X      quick result: Run/Pass/Sack/Int/eXplosive on last play
  help           this text
  q / quit       exit

Manual corrections:
  correct family CROSSERS | correct result completion | correct yards 12
  R = run, P = pass/completion, S = sack, I = int, X = mark explosive

Live capture: --window Xbox --debug | --image | --video | --calibrate
Session: --opponent / --dynasty starts game_sessions + play_records
""".strip()



# --- live-loop UX helpers (heartbeat / status / non-blocking stdin) ---

_WAITING_FRAMES = (
    "waiting for frames… (is Remote Play visible? window title match?) "
    "capture={capture}"
)

_TROUBLESHOOT_NO_FRAMES = """No frames for 5s — troubleshooting:
  · Leave Xbox Remote Play visible (not minimized / not covered)
  · Confirm --window title matches (e.g. "Xbox")
  · Recalibrate: cfb-coach watch --calibrate --window "Xbox"
  · Or try --screen-region from calib (watch --calibrate writes ~/.cfb-coach/calib.json)
"""

_CAPTURE_OK = "capture OK — LIVE tips below"

_LIVE_CMD_HINT = (
    "Type commands anytime (look nickel…, tips, help, q, R/P/S/I/X) — Enter to submit"
)


def _waiting_frames_msg(capture_name: str) -> str:
    return _WAITING_FRAMES.format(capture=capture_name or "?")


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

        buf: list[str] = []
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
                self._q.put("".join(buf))
                buf.clear()
                continue
            if ch in ("\x08", "\x7f"):  # backspace
                if buf:
                    buf.pop()
                    try:
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    except Exception:
                        pass
                continue
            if ch == "\x03":  # Ctrl+C — let main loop see KeyboardInterrupt via flag
                self._q.put("q")
                break
            buf.append(ch)
            try:
                sys.stdout.write(ch)
                sys.stdout.flush()
            except Exception:
                pass

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
        print(format_tips_block(look, tips))
        return "ok", overlay_path
    if low == "overlay":
        path_obj = overlay_path or _overlay_default_path()
        path = write_overlay_html(str(path_obj), look, tips)
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


def _format_live_overlay(
    short_line: str,
    tips: list[str],
    *,
    tendency_lines: list[str] | None = None,
    counter: str = "",
    fps: float | None = None,
    latency_ms: float | None = None,
) -> str:
    """Compact LIVE GAME overlay — top tendencies + CURRENT COUNTER (not 40 stats)."""
    lines = ["LIVE GAME", short_line]
    if tendency_lines:
        lines.append("TENDENCIES")
        for t in tendency_lines[:3]:
            lines.append(f"  · {t}")
    if counter:
        lines.append(f"CURRENT COUNTER: {counter}")
    if tips:
        lines.append("CALL")
        for i, tip in enumerate(tips[:3], 1):
            lines.append(f"  {i}. {tip}")
    dbg = []
    if fps is not None:
        dbg.append(f"fps={fps:.1f}")
    if latency_ms is not None:
        dbg.append(f"lat={latency_ms:.0f}ms")
    if dbg:
        lines.append("  [" + " ".join(dbg) + "]")
    return "\n".join(lines)


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

    if calibrate:
        from cfb_coach.vision.calibrate import run_calibrate

        return run_calibrate(
            window=getattr(args, "window", None),
            region=getattr(args, "screen_region", None),
        )

    inventory = _load_inventory(opponent)
    overlay_path: Path | None = None
    if overlay is True or overlay == "auto":
        overlay_path = _overlay_default_path()
    elif isinstance(overlay, str) and overlay:
        overlay_path = Path(overlay)

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

    print("SCREEN CO-PILOT v1.9.1 — Xbox stays in your hands (sidecar only)")
    print("Prototype source: Xbox Remote Play on laptop · play on HDMI monitor")
    print("Future: capture-card drop-in backend. See: watch --setup")
    print("Now: demo + hotkeys + optional live/--image/--video. Type 'help'.")
    if opponent:
        print(f"Opponent context: {opponent}")
    if session:
        print(f"Session: {session.session_id} (dynasty={session.dynasty})")
    print("-" * 60)

    live_state: dict[str, Any] = {
        "look": look,
        "tips": tips,
        "overlay_path": overlay_path,
    }

    def emit(current: DefenseLook, *, short_line: str | None = None) -> None:
        nonlocal tips, look, overlay_path
        look = current.normalized()
        tips = tips_from_look(look, inventory=inventory, opponent_id=opponent)
        live_state["look"] = look
        live_state["tips"] = tips
        live_state["overlay_path"] = overlay_path
        tend_lines = live_engine.top_lines(n=3) if live_engine else []
        counter = live_engine.current_counter() if live_engine else ""
        print()
        if short_line:
            print(
                _format_live_overlay(
                    short_line, tips, tendency_lines=tend_lines, counter=counter
                )
            )
        else:
            print(format_tips_block(look, tips))
            if tend_lines:
                print("TENDENCIES")
                for t in tend_lines:
                    print(f"  · {t}")
            if counter:
                print(f"CURRENT COUNTER: {counter}")
        if overlay_path is not None:
            path = write_overlay_html(str(overlay_path), look, tips)
            print(f"  overlay → {path}")
        if tts is not None:
            tts.speak_tips(tips)

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
            print(format_tips_block(look, tips))
            continue
        if low == "overlay":
            if overlay_path is None:
                overlay_path = _overlay_default_path()
            path = write_overlay_html(str(overlay_path), look, tips)
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
    capture_name = getattr(pipe_cap, "name", "?")

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

    def _emit_heartbeat(now: float) -> None:
        nonlocal last_heartbeat
        if now - last_frame_wall > 1.0 and now - last_heartbeat >= 1.0:
            print(_waiting_frames_msg(capture_name))
            last_heartbeat = now

    def _emit_troubleshoot(now: float) -> None:
        nonlocal troub_printed
        if troub_printed:
            return
        if not saw_frame and now - loop_start >= 5.0:
            print(_TROUBLESHOOT_NO_FRAMES.rstrip())
            troub_printed = True

    def _emit_status(now: float, *, force: bool = False) -> None:
        nonlocal last_status
        if last_obs is None and not force:
            return
        if not force and now - last_status < 2.0:
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
                        _emit_heartbeat(now)
                        _emit_troubleshoot(now)
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
                continue

            now = time.time()
            if not saw_frame:
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
            tend_lines = live_engine.top_lines(n=3) if live_engine else []
            counter = live_engine.current_counter() if live_engine else ""
            extras = obs.extras or {}
            if short != last_short or frames == 1:
                last_short = short
                emit(look, short_line=short)
                last_status = now
            elif now - last_status >= 2.0:
                # Always show life at least every 2s (even if short_line unchanged)
                print(
                    _status_line(
                        short,
                        fps=pipe.fps,
                        play_state=str(obs.play_state or ""),
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
                lines = [
                    short,
                    f"state={obs.play_state}",
                    look.label(),
                    f"COUNTER: {counter}" if counter else "",
                ]
                if tend_lines:
                    lines.append(" | ".join(tend_lines[:2]))
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
