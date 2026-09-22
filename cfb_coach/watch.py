"""cfb-coach watch / copilot — SCREEN CO-PILOT live loop (v1.8).

Aidan plays on Xbox via HDMI monitor + controller. Laptop runs Remote Play
as the prototype video source. Coach is SIDE-CAR ONLY — never controls Xbox.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

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
Screen Co-Pilot v1.8 — you stay on sticks; coach suggests only.

Typed commands (then Enter):
  f <front>      even|odd|nickel|dime|goal_line
  s <shell>      single_high|two_high|cover2|cover3|quarters|man
  p <pressure>   none|show|blitz_left|blitz_right|blitz_middle|all_out
  look <tokens>  e.g.  look nickel c3 blitz_left
  demo           advance one canned look
  tips           reprint last tips
  overlay        rewrite overlay HTML (if --overlay path set)
  help           this text
  q / quit       exit

Aliases work: c3, 2h, left, heat, zero, …
Live capture: --window Xbox --debug | --image | --video | --calibrate
""".strip()


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

    # Optional TTS / DB logger
    tts = None
    if tts_on:
        from cfb_coach.vision.tts import TipTTS

        tts = TipTTS(enabled=True)
        if not tts.available:
            print("TTS requested but Windows SAPI unavailable — continuing silent.")

    db = None
    log_obs = None
    try:
        from cfb_coach.db import CoachDB, resolve_db_path_from_env

        db = CoachDB(resolve_db_path_from_env())
        log_obs = lambda obs: db.log_vision_observation(obs.to_dict())  # noqa: E731
    except Exception:
        db = None
        log_obs = None

    print("SCREEN CO-PILOT v1.8 — Xbox stays in your hands (sidecar only)")
    print("Prototype source: Xbox Remote Play on laptop · play on HDMI monitor")
    print("Future: capture-card drop-in backend. See: watch --setup")
    print("Now: demo + hotkeys + optional live/--image/--video. Type 'help'.")
    if opponent:
        print(f"Opponent context: {opponent}")
    print("-" * 60)

    def emit(current: DefenseLook, *, short_line: str | None = None) -> None:
        nonlocal tips, look
        look = current.normalized()
        tips = tips_from_look(look, inventory=inventory, opponent_id=opponent)
        print()
        if short_line:
            print(short_line)
            print("CALL")
            for i, t in enumerate(tips, 1):
                print(f"  {i}. {t}")
        else:
            print(format_tips_block(look, tips))
        if overlay_path is not None:
            path = write_overlay_html(str(overlay_path), look, tips)
            print(f"  overlay → {path}")
        if tts is not None:
            tts.speak_tips(tips)

    # Live / video / image-via-pipeline path
    if _want_live(args) or (image and getattr(args, "pipeline", True) and debug):
        # For --image with --debug use pipeline; plain --image keeps naive heuristic below
        pass

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
        )

    # --image: one-shot naive heuristic (keeps Linux stdlib path)
    if image:
        cap = ImageFileCapture(image)
        raw = cap.grab()
        # New ImageFileCapture returns Frame; accept bytes too
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
            # Fallback read
            raw_bytes = Path(image).read_bytes()
        emit(naive_roi_heuristic(raw_bytes))
        if once or not demo:
            if once or not sys.stdin.isatty():
                _close_db(db)
                return 0

    # --demo auto-cycle (non-interactive if --once or not a tty)
    if demo and (once or not sys.stdin.isatty()):
        for i, sample in enumerate(demo_looks):
            emit(sample)
            if once:
                _close_db(db)
                return 0
            if i < len(demo_looks) - 1:
                time.sleep(max(0.2, interval))
        _close_db(db)
        return 0

    if demo:
        emit(demo_looks[0])
        demo_idx = 1

    if not sys.stdin.isatty() and not demo and not image:
        pass

    # Interactive loop
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
    _close_db(db)
    return 0


def _close_db(db: Any) -> None:
    if db is not None:
        try:
            db.close()
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
) -> int:
    from cfb_coach.vision.calibrate import load_calib
    from cfb_coach.vision.pipeline import VisionPipeline, build_capture_from_args

    try:
        cap = build_capture_from_args(args)
    except ImportError as e:
        print(f"Vision capture unavailable: {e}", file=sys.stderr)
        print("Install on native Windows: pip install -e \".[vision]\"", file=sys.stderr)
        _close_db(db)
        return 2

    # If user asked for window/live but we got stub, warn
    if getattr(cap, "name", "") == "stub" and (
        getattr(args, "window", None) or getattr(args, "screen_region", None)
    ):
        print(
            "Live capture deps missing (dxcam/mss). "
            "pip install -e \".[vision]\" on native Windows Python.",
            file=sys.stderr,
        )
        _close_db(db)
        return 2

    pipe = VisionPipeline(cap, calib=load_calib())
    dbg = None
    if debug:
        from cfb_coach.vision.debug_view import DebugView

        dbg = DebugView()
        if not dbg.available:
            print("Debug view needs opencv-python — continuing without window.")

    print(f"Pipeline capture={getattr(cap, 'name', '?')} target_fps={target_fps:.0f}")
    delay = 1.0 / max(1.0, target_fps)
    frames = 0
    last_short = ""

    try:
        while True:
            t0 = time.time()
            try:
                obs, look = pipe.step()
            except Exception as e:
                print(f"pipeline error: {e}", file=sys.stderr)
                break
            frames += 1
            if log_obs is not None:
                try:
                    log_obs(obs)
                except Exception:
                    pass
            short = obs.short_line()
            # Emit when look meaningfully changes or every ~1s
            if short != last_short or frames == 1:
                last_short = short
                emit(look, short_line=short)
            elif debug and frames % max(1, int(target_fps)) == 0:
                print(f"  … {short}  fps={pipe.fps:.1f}")

            if dbg is not None:
                # best-effort last frame from capture extras — re-grab not needed
                frame = None
                try:
                    frame = cap.grab()
                    bgr = getattr(frame, "bgr", None) if frame is not None else None
                except Exception:
                    bgr = None
                dbg.show(
                    bgr,
                    fps=pipe.fps,
                    lines=[short, f"state={obs.play_state}", look.label()],
                    rois=(pipe.calib or {}).get("rois"),
                )

            if once:
                break
            # --image / --video once without --once still can run limited
            if image and not video and not getattr(args, "window", None):
                # single image: one pass unless interactive
                if once or not sys.stdin.isatty():
                    break
                # if tty, fall into a short wait then exit unless they want more
                break

            elapsed = time.time() - t0
            time.sleep(max(0.0, delay - elapsed))

            # Non-tty without --window continuous: stop after a few frames for video test
            if video and once:
                break
            if not sys.stdin.isatty() and not getattr(args, "window", None) and frames >= 3:
                # safety for CI/automation
                if not video:
                    break
                if frames >= 30:
                    break
    except KeyboardInterrupt:
        print()
    finally:
        if dbg is not None:
            dbg.close()
        pipe.close()
        _close_db(db)
    return 0
