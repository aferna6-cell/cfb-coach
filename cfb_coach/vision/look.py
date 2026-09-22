"""DefenseLook contract + demo/hotkey/image stubs (vision package core).

DefenseLook is the coach tip contract. Live vision (Milestone 1) lives in sibling
modules and maps GameObservation → DefenseLook.

Aidan plays CFB 27 on Xbox via DIRECT HDMI to monitor + Xbox controller.
Separately, Windows laptop runs Xbox Remote Play so the laptop sees the game.
CFB Coach captures Remote Play (prototype source) — SIDE-CAR ONLY, never controls Xbox.
Capture card is a FUTURE drop-in CaptureBackend.
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


# --- Windows capture path (Milestone 1) ---
#
# Active prototype: Xbox Remote Play window on the laptop (dxcam / mss).
#   cfb-coach watch --window "Xbox" --debug
# Play remains on direct HDMI monitor + Xbox controller (sidecar only).
#
# Future drop-in: HDMI capture card → DeviceCapture / window grab of preview.
# Heavy deps (dxcam, mss, cv2, win32) are LAZY — never imported at package init.


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
Xbox Remote Play vision (ACTIVE PROTOTYPE — play stays on HDMI monitor)
-----------------------------------------------------------------------
1. On Xbox: enable remote features (Settings → Devices & connections → Remote features).
2. On Windows laptop: open Xbox app → Remote Play → connect to console.
3. Keep PLAYING on the direct-HDMI monitor with the Xbox controller.
   Laptop is a sidecar tip display — coach NEVER presses buttons / auto-plays.
4. Native Windows Python (not WSL for live grab):
     pip install -e ".[vision]"
5. Calibrate once:
     cfb-coach watch --calibrate
   (saves window title + crop to ~/.cfb-coach/vision_calib.json)
6. List window titles (pick Remote Play exact title if needed):
     cfb-coach watch --list-windows
7. Live watch:
     cfb-coach watch --window "Xbox" --debug
   Or use calib crop via mss:
     cfb-coach watch --screen-region
8. Debug view: OpenCV window with frame, FPS, crop/ROIs, and state text.
9. Xbox control is SEPARATE from vision — sticks stay in your hands.
10. Remote Play = current prototype video source (laptop sees the game).
11. Capture card (Elgato etc.) = future drop-in CaptureBackend (same pipeline).

Also works offline:
  cfb-coach watch --demo / --image test.png / --video sample.mp4
  Typed hotkeys (f/s/p/look) still inject DefenseLook without any camera.

WSL note: prefer native Windows Python for dxcam/mss window grab, OR dump a
frame to a shared folder and tip from WSL via --image.
"""
