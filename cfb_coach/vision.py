"""Pluggable vision interface for SCREEN CO-PILOT (v0).

DefenseLook is the contract. Capture backends are stubs for v0:
  - Demo / hotkey injection (works now)
  - Optional PNG path + naive ROI heuristic
  - Capture-card / DirectShow / OBS hooks documented for Windows later

Aidan plays on Xbox + TV (controller). Laptop is a sidecar tip display.
Primary future capture path = HDMI capture card (Elgato etc.) into the laptop,
NOT Xbox Remote Play as the play method. Remote Play is an optional mirror only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


# Canonical vocab (keep coarse — Cover 3 vs Quarters is hard; start here)
FRONTS = ("even", "odd", "nickel", "dime", "goal_line", "unknown")
SHELLS = ("single_high", "two_high", "cover2", "cover3", "quarters", "man", "unknown")
PRESSURES = ("none", "show", "blitz_left", "blitz_right", "blitz_middle", "all_out", "unknown")


@dataclass
class DefenseLook:
    """Pre-snap opponent defense look — what coach sees / injects."""

    front: str = "unknown"
    shell: str = "unknown"
    pressure: str = "unknown"
    confidence: float = 0.0
    source: str = "stub"  # stub | demo | hotkey | image | capture
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> DefenseLook:
        front = (self.front or "unknown").lower().strip()
        shell = (self.shell or "unknown").lower().strip()
        pressure = (self.pressure or "unknown").lower().strip()
        # aliases
        aliases_front = {
            "4-3": "even",
            "43": "even",
            "3-4": "odd",
            "34": "odd",
            "ni": "nickel",
            "nick": "nickel",
            "gl": "goal_line",
        }
        aliases_shell = {
            "1h": "single_high",
            "one_high": "single_high",
            "single": "single_high",
            "c1": "single_high",
            "cover1": "single_high",
            "2h": "two_high",
            "two": "two_high",
            "c2": "cover2",
            "cover_2": "cover2",
            "invert": "cover2",
            "c3": "cover3",
            "cover_3": "cover3",
            "sky": "cover3",
            "c4": "quarters",
            "cover4": "quarters",
            "cover_4": "quarters",
            "palms": "quarters",
            "c6": "two_high",
            "cover6": "two_high",
            "c9": "two_high",
            "cover9": "two_high",
            "tampa": "cover2",
            "tampa2": "cover2",
            "zero": "man",
            "c0": "man",
            "cover0": "man",
        }
        aliases_pressure = {
            "off": "none",
            "0": "none",
            "clean": "none",
            "sim": "show",
            "show_blitz": "show",
            "left": "blitz_left",
            "l": "blitz_left",
            "right": "blitz_right",
            "r": "blitz_right",
            "mid": "blitz_middle",
            "a": "all_out",
            "zero_blitz": "all_out",
            "heat": "blitz_middle",
            "blitz": "blitz_middle",
        }
        front = aliases_front.get(front, front)
        shell = aliases_shell.get(shell, shell)
        pressure = aliases_pressure.get(pressure, pressure)
        if front not in FRONTS:
            front = "unknown"
        if shell not in SHELLS:
            shell = "unknown"
        if pressure not in PRESSURES:
            pressure = "unknown"
        conf = max(0.0, min(1.0, float(self.confidence)))
        return DefenseLook(
            front=front,
            shell=shell,
            pressure=pressure,
            confidence=conf,
            source=self.source,
            notes=self.notes,
            extras=dict(self.extras),
        )

    def label(self) -> str:
        n = self.normalized()
        return f"front={n.front} shell={n.shell} pressure={n.pressure} conf={n.confidence:.2f}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())


class CaptureBackend(Protocol):
    """Hook for future frame grabbers (capture card / OBS / window)."""

    name: str

    def grab(self) -> bytes | None:
        """Return raw frame bytes (PNG/JPEG) or None if unavailable."""
        ...

    def close(self) -> None:
        ...


class StubCapture:
    """No-op capture — hotkeys/demo drive looks until vision matures."""

    name = "stub"

    def grab(self) -> bytes | None:
        return None

    def close(self) -> None:
        return None


class ImageFileCapture:
    """One-shot PNG/JPG path for offline testing."""

    name = "image"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def grab(self) -> bytes | None:
        if not self.path.is_file():
            return None
        return self.path.read_bytes()

    def close(self) -> None:
        return None


# --- Windows capture-card path (documented hooks; not wired on Linux v0) ---
#
# Primary (Aidan plays on Xbox/TV; laptop is sidecar):
#   HDMI out (Xbox) → capture card (Elgato Cam Link / HD60 / 4K X etc.)
#   → Windows host sees DirectShow / UVC device
#   → OBS (optional) Virtual Camera, or OpenCV VideoCapture(index)
#   → cfb-coach watch --device N   (future) or --window "OBS"
#
# Suggested future backends (do NOT import heavy deps in v0):
#   - dxcam / mss: fast desktop ROI grab of the capture preview window
#   - cv2.VideoCapture(device_index): DirectShow index for the card
#   - FindWindow + BitBlt: grab OBS/Elgato preview HWND
#
# Optional alternate: Xbox Remote Play mirror on the laptop — only if he
# chooses to mirror; he still plays on the TV controller. Not the default.
#
# WSL note: USB/capture devices are awkward across the WSL boundary.
# Prefer running the watch loop on native Windows Python against the
# DirectShow device, or grab frames in Windows and pass --image / a shared
# folder PNG into WSL for the tip engine.


def parse_look_tokens(tokens: list[str] | str) -> DefenseLook:
    """Parse typed injection like: 'nickel two_high blitz_left' or 'c3 heat'."""
    if isinstance(tokens, str):
        parts = tokens.replace(",", " ").split()
    else:
        parts = list(tokens)
    look = DefenseLook(source="hotkey", confidence=0.85)
    for raw in parts:
        t = raw.lower().strip()
        if not t:
            continue
        # try each dimension
        trial = DefenseLook(front=t, shell="unknown", pressure="unknown", confidence=0.85, source="hotkey")
        n = trial.normalized()
        if n.front != "unknown":
            look.front = n.front
            continue
        trial = DefenseLook(front="unknown", shell=t, pressure="unknown", confidence=0.85, source="hotkey")
        n = trial.normalized()
        if n.shell != "unknown":
            look.shell = n.shell
            continue
        trial = DefenseLook(front="unknown", shell="unknown", pressure=t, confidence=0.85, source="hotkey")
        n = trial.normalized()
        if n.pressure != "unknown":
            look.pressure = n.pressure
            continue
        # key=value
        if "=" in t:
            k, _, v = t.partition("=")
            if k in ("front", "f"):
                look.front = v
            elif k in ("shell", "s"):
                look.shell = v
            elif k in ("pressure", "p", "press"):
                look.pressure = v
            elif k in ("conf", "confidence", "c"):
                try:
                    look.confidence = float(v)
                except ValueError:
                    pass
    return look.normalized()


def naive_roi_heuristic(image_bytes: bytes) -> DefenseLook:
    """Extremely naive stub: use file size / header entropy as fake signal.

    Real models later. For --image testing we guess:
      - tiny file → unknown
      - mid → two_high / none
      - larger → single_high / show
    Does not decode pixels (no PIL/cv2 dep). Optional improvement later.
    """
    n = len(image_bytes or b"")
    if n < 500:
        return DefenseLook(
            front="unknown",
            shell="unknown",
            pressure="unknown",
            confidence=0.1,
            source="image",
            notes="image too small / empty",
        )
    # PNG magic?
    is_png = image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    is_jpg = image_bytes[:2] == b"\xff\xd8"
    # coarse buckets by size only
    if n < 80_000:
        look = DefenseLook(
            front="nickel",
            shell="two_high",
            pressure="none",
            confidence=0.35,
            source="image",
            notes="naive size heuristic (small frame → two-high clean)",
        )
    elif n < 250_000:
        look = DefenseLook(
            front="nickel",
            shell="single_high",
            pressure="show",
            confidence=0.35,
            source="image",
            notes="naive size heuristic (mid frame → single-high show)",
        )
    else:
        look = DefenseLook(
            front="nickel",
            shell="cover3",
            pressure="blitz_middle",
            confidence=0.3,
            source="image",
            notes="naive size heuristic (large frame → C3 + middle heat guess)",
        )
    fmt = "png" if is_png else ("jpg" if is_jpg else "raw")
    look.extras["format"] = fmt
    look.extras["bytes"] = n
    return look.normalized()


def demo_sequence() -> list[DefenseLook]:
    """Canned looks for --demo mode."""
    samples = [
        DefenseLook("nickel", "two_high", "none", 0.9, "demo", "Cover 6/9 look"),
        DefenseLook("nickel", "cover3", "none", 0.85, "demo", "single-high sky"),
        DefenseLook("nickel", "quarters", "none", 0.8, "demo", "Cover 4"),
        DefenseLook("nickel", "cover2", "none", 0.8, "demo", "Invert / Tampa-ish"),
        DefenseLook("nickel", "single_high", "blitz_left", 0.9, "demo", "Sam/edge heat"),
        DefenseLook("nickel", "man", "all_out", 0.95, "demo", "Cover 0 heat"),
        DefenseLook("even", "two_high", "show", 0.7, "demo", "show blitz two-high"),
        DefenseLook("dime", "quarters", "blitz_middle", 0.75, "demo", "dime A-gap"),
    ]
    return [s.normalized() for s in samples]


CAPTURE_SETUP_NOTES = """
Capture-card path (PRIMARY — Aidan plays on Xbox/TV; laptop is sidecar)
-----------------------------------------------------------------------
1. Xbox HDMI → capture card (Elgato Cam Link / HD60 / 4K X / similar) → laptop USB.
2. Confirm Windows sees the device (Camera app, Elgato Wave Link / 4K Capture Utility, or OBS).
3. Optional: OBS → Start Virtual Camera (stable name for OpenCV/DirectShow).
4. Future CLI (not in v0):  cfb-coach watch --device 0
   or grab the preview window: cfb-coach watch --window "OBS" / "4K Capture Utility"
5. WSL: USB capture is painful across the boundary. Prefer native Windows Python for
   device grab, OR dump a frame to a shared folder and use:
     cfb-coach watch --image /mnt/c/Users/.../frame.png
6. v0 today: --demo and typed hotkeys (f/s/p) inject DefenseLook without any camera.

Optional alternate: Xbox Remote Play mirror on the laptop (only if you choose to
mirror). You still play on the TV controller — Remote Play is NOT the default
play method. If mirroring, mss/dxcam/FindWindow can grab the Remote Play HWND later.
"""
