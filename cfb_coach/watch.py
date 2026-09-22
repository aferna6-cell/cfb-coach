"""cfb-coach watch / copilot — SCREEN CO-PILOT live loop (v0).

Aidan plays on Xbox + TV. Laptop is sidecar. Coach prints pre-snap tips from
DefenseLook (demo / hotkeys / optional --image). Vision is immature — hotkeys
are the real path until capture-card models improve.
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
Screen Co-Pilot v0 — you stay on sticks; coach suggests only.

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


def run_watch(args: Any) -> int:
    """Entry used by CLI cmd_watch."""
    demo = bool(getattr(args, "demo", False))
    once = bool(getattr(args, "once", False))
    image = getattr(args, "image", None)
    overlay = getattr(args, "overlay", None)
    opponent = getattr(args, "opponent", None)
    show_setup = bool(getattr(args, "setup", False))
    interval = float(getattr(args, "interval", 1.5) or 1.5)

    if show_setup:
        print(CAPTURE_SETUP_NOTES.strip())
        return 0

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

    print("SCREEN CO-PILOT v0 — Xbox stays in your hands")
    print("Primary capture later: HDMI capture card → laptop (see: watch --setup)")
    print("Now: demo + typed look injection. Type 'help' for keys.")
    if opponent:
        print(f"Opponent context: {opponent}")
    print("-" * 60)

    def emit(current: DefenseLook) -> None:
        nonlocal tips, look
        look = current.normalized()
        tips = tips_from_look(look, inventory=inventory, opponent_id=opponent)
        print()
        print(format_tips_block(look, tips))
        if overlay_path is not None:
            path = write_overlay_html(str(overlay_path), look, tips)
            print(f"  overlay → {path}")

    # --image: one-shot naive heuristic
    if image:
        cap = ImageFileCapture(image)
        raw = cap.grab()
        if raw is None:
            print(f"Image not found: {image}", file=sys.stderr)
            return 2
        emit(naive_roi_heuristic(raw))
        if once or not demo:
            # if only --image without interactive demo loop wanting more
            if once or not sys.stdin.isatty():
                return 0

    # --demo auto-cycle (non-interactive if --once or not a tty)
    if demo and (once or not sys.stdin.isatty()):
        for i, sample in enumerate(demo_looks):
            emit(sample)
            if once:
                return 0
            if i < len(demo_looks) - 1:
                time.sleep(max(0.2, interval))
        return 0

    if demo:
        emit(demo_looks[0])
        demo_idx = 1

    # Interactive / piped loop (TTY or stdin script). Non-TTY without demo/image
    # still works when the caller pipes commands (tests / automation).
    if not sys.stdin.isatty() and not demo and not image:
        # If stdin is closed/empty we cannot inject looks — warn but try loop.
        pass

    # Interactive loop
    while True:
        try:
            raw = input("look> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not raw:
            continue
        low = raw.lower()
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
            token = raw.split(None, 1)[1]
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
            token = raw.split(None, 1)[1]
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
            token = raw.split(None, 1)[1]
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
            emit(parse_look_tokens(raw.split(None, 1)[1]))
            continue
        # bare tokens: treat as look injection
        if any(c.isalpha() for c in raw):
            emit(parse_look_tokens(raw))
            continue
        print("  unknown — type help")

    try:
        capture.close()
    except Exception:
        pass
    return 0
