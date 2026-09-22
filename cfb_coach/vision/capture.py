"""CaptureBackend implementations — Windows-heavy deps are lazy / optional.

grab() returns a Frame (numpy BGR when available) or raw bytes for legacy paths.
Protocol stays flexible; document return types on each backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class Frame:
    """Captured frame. Prefer BGR ndarray when vision extras installed."""

    bgr: Any = None  # numpy.ndarray | None
    png_bytes: bytes | None = None
    width: int = 0
    height: int = 0
    source: str = ""
    ts: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.bgr is not None or bool(self.png_bytes)


@runtime_checkable
class CaptureBackend(Protocol):
    """Hook for frame grabbers (Remote Play window / region / file / future card)."""

    name: str

    def grab(self) -> Frame | bytes | None:
        """Return Frame (preferred), raw PNG/JPEG bytes, or None if unavailable."""
        ...

    def close(self) -> None:
        ...


class StubCapture:
    """No-op capture — hotkeys/demo drive looks until vision matures."""

    name = "stub"

    def grab(self) -> Frame | None:
        return None

    def close(self) -> None:
        return None


class ImageFileCapture:
    """One-shot / repeat PNG/JPG path for offline testing."""

    name = "image"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def grab(self) -> Frame | bytes | None:
        if not self.path.is_file():
            return None
        raw = self.path.read_bytes()
        bgr = None
        w = h = 0
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            arr = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                h, w = int(bgr.shape[0]), int(bgr.shape[1])
        except Exception:
            bgr = None
        return Frame(bgr=bgr, png_bytes=raw, width=w, height=h, source="image")

    def close(self) -> None:
        return None


class VideoFileCapture:
    """Offline video frames via OpenCV (optional dep)."""

    name = "video"

    def __init__(self, path: str | Path, *, loop: bool = True) -> None:
        self.path = Path(path)
        self.loop = loop
        self._cap: Any = None
        self._opened = False

    def _ensure(self) -> bool:
        if self._opened and self._cap is not None:
            return True
        try:
            import cv2  # type: ignore
        except ImportError as e:
            raise ImportError(
                "VideoFileCapture requires opencv-python — pip install -e '.[vision]'"
            ) from e
        if not self.path.is_file():
            return False
        self._cap = cv2.VideoCapture(str(self.path))
        self._opened = True
        return bool(self._cap.isOpened())

    def grab(self) -> Frame | None:
        if not self._ensure():
            return None
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            if self.loop:
                self._cap.set(0, 0)  # CAP_PROP_POS_FRAMES
                ok, bgr = self._cap.read()
            if not ok or bgr is None:
                return None
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        return Frame(bgr=bgr, width=w, height=h, source="video")

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._opened = False


class MssRegionCapture:
    """mss monitor/region grab — works on Windows/Linux when mss installed."""

    name = "mss"

    def __init__(self, region: dict[str, int] | None = None) -> None:
        """region: {left, top, width, height} in screen pixels."""
        self.region = region
        self._mss: Any = None

    def _ensure(self) -> Any:
        if self._mss is not None:
            return self._mss
        try:
            import mss  # type: ignore
        except ImportError as e:
            raise ImportError(
                "MssRegionCapture requires mss — pip install -e '.[vision]'"
            ) from e
        self._mss = mss.mss()
        return self._mss

    def grab(self) -> Frame | None:
        sct = self._ensure()
        try:
            import numpy as np  # type: ignore
        except ImportError as e:
            raise ImportError(
                "MssRegionCapture requires numpy — pip install -e '.[vision]'"
            ) from e
        mon = self.region
        if mon is None:
            # primary monitor
            mon = sct.monitors[1]
        shot = sct.grab(mon)
        # mss returns BGRA
        bgra = np.array(shot, dtype=np.uint8)
        bgr = bgra[:, :, :3].copy()
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        return Frame(bgr=bgr, width=w, height=h, source="mss")

    def close(self) -> None:
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
        self._mss = None


class DxcamWindowCapture:
    """dxcam window/region grab — Windows preferred path for Remote Play."""

    name = "dxcam"

    def __init__(
        self,
        *,
        window_substring: str | None = "Xbox",
        region: tuple[int, int, int, int] | None = None,
        output_color: str = "BGR",
    ) -> None:
        self.window_substring = window_substring
        self.region = region  # left, top, right, bottom
        self.output_color = output_color
        self._camera: Any = None
        self._resolved_region: tuple[int, int, int, int] | None = region
        self._grab_region: tuple[int, int, int, int] | None = None
        self.matched_title: str | None = None
        if self._resolved_region is None and self.window_substring:
            matched: list[str] = []
            self._resolved_region = find_window_region(
                self.window_substring, matched_title=matched
            )
            if matched:
                self.matched_title = matched[0]

    def _ensure(self) -> Any:
        if self._camera is not None:
            return self._camera
        try:
            import dxcam  # type: ignore
        except ImportError as e:
            raise ImportError(
                "DxcamWindowCapture requires dxcam (Windows) — pip install -e '.[vision]'"
            ) from e
        from cfb_coach.vision.winenum import ensure_dpi_aware, to_output_local

        ensure_dpi_aware()
        if self.window_substring and self.region is None:
            # Re-resolve AFTER DPI awareness: earlier rects may be logical px.
            matched: list[str] = []
            rect = find_window_region(self.window_substring, matched_title=matched)
            if rect is not None:
                self._resolved_region = rect
            if matched:
                self.matched_title = matched[0]
        # dxcam default output = primary monitor, origin (0,0) on the desktop.
        # Create full-output camera, then clamp: dxcam raises "Invalid Region"
        # for any rect outside 0..width/0..height (snapped windows are -7px).
        self._camera = dxcam.create(output_color=self.output_color)
        rect = self._resolved_region
        if rect is not None:
            local = to_output_local(
                rect, (0, 0), (int(self._camera.width), int(self._camera.height))
            )
            if local is None:
                raise RuntimeError(
                    f"window rect {rect} is not on the primary monitor "
                    f"({self._camera.width}x{self._camera.height}) — dxcam only "
                    "captures the primary output; use --capture wgc or mss"
                )
            self._grab_region = local
        return self._camera

    def grab(self) -> Frame | None:
        cam = self._ensure()
        frame = cam.grab(region=self._grab_region) if self._grab_region else cam.grab()
        if frame is None:
            return None
        h, w = int(frame.shape[0]), int(frame.shape[1])
        return Frame(bgr=frame, width=w, height=h, source="dxcam")

    def close(self) -> None:
        self._camera = None


class DeviceCapture:
    """Future drop-in: OpenCV VideoCapture device index (capture card). Stub-ready."""

    name = "device"

    def __init__(self, index: int = 0, *, width: int = 1920, height: int = 1080) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: Any = None

    def _ensure(self) -> bool:
        if self._cap is not None:
            return True
        try:
            import cv2  # type: ignore
        except ImportError as e:
            raise ImportError(
                "DeviceCapture requires opencv-python — pip install -e '.[vision]'"
            ) from e
        import sys as _sys

        # DirectShow opens UVC capture cards fast on Windows; MSMF can stall.
        api = getattr(cv2, "CAP_DSHOW", 0) if _sys.platform == "win32" else 0
        self._cap = cv2.VideoCapture(self.index, api) if api else cv2.VideoCapture(self.index)
        # Many UVC cards default to 640x480 — too small for HUD digits.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return bool(self._cap.isOpened())

    def grab(self) -> Frame | None:
        if not self._ensure():
            return None
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            return None
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        return Frame(bgr=bgr, width=w, height=h, source="device")

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None


# Window-title helpers live in winenum (ctypes, DPI-aware); re-exported here
# for backward compatibility with existing imports.
from cfb_coach.vision.winenum import (  # noqa: E402
    WINDOW_TITLE_ALIASES,
    _enum_windows_matching,
    find_window,
    list_visible_windows,
    window_search_needles,
)


def find_window_region(
    substring: str | None = "Xbox",
    *,
    matched_title: list[str] | None = None,
) -> tuple[int, int, int, int] | None:
    """Windows-only: find visible window region by case-insensitive substring.

    Tries primary needle then common Xbox/Remote Play aliases when appropriate.
    If matched_title is a list, appends the winning title for diagnostics.
    """
    for needle in window_search_needles(substring):
        hits = _enum_windows_matching(needle)
        if hits:
            title, rect = hits[0]
            if matched_title is not None:
                matched_title.append(title)
            return rect
    return None


def _find_window_region(substring: str) -> tuple[int, int, int, int] | None:
    """Backward-compatible wrapper."""
    return find_window_region(substring)


def frame_to_bytes(frame: Frame | bytes | None) -> bytes | None:
    """Best-effort PNG/JPEG bytes for legacy naive_roi_heuristic."""
    if frame is None:
        return None
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    if frame.png_bytes:
        return frame.png_bytes
    if frame.bgr is not None:
        try:
            import cv2  # type: ignore

            ok, buf = cv2.imencode(".png", frame.bgr)
            if ok:
                return bytes(buf.tobytes())
        except Exception:
            return None
    return None


# --- dxcam → mss auto-fallback (Xbox UWP / protected windows) ---

DXCAM_NO_FRAMES_FALLBACK_MSG = (
    "dxcam got no frames — falling back to mss screen region"
)

# Seconds of empty grabs after start before switching to mss (~2–3s).
DXCAM_FALLBACK_AFTER_S = 2.5


def ltrb_to_mss_region(rect: tuple[int, int, int, int]) -> dict[str, int]:
    """Convert win32 (left, top, right, bottom) → mss {left, top, width, height}."""
    left, top, right, bottom = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    return {
        "left": left,
        "top": top,
        "width": max(1, right - left),
        "height": max(1, bottom - top),
    }


def resolve_mss_region_for_window(
    window_substring: str | None,
    *,
    calib_region: dict[str, int] | None = None,
    matched_title: list[str] | None = None,
) -> dict[str, int] | None:
    """Prefer calib crop if set; else matched window GetWindowRect → mss dict."""
    if isinstance(calib_region, dict) and all(
        k in calib_region for k in ("left", "top", "width", "height")
    ):
        return {
            "left": int(calib_region["left"]),
            "top": int(calib_region["top"]),
            "width": int(calib_region["width"]),
            "height": int(calib_region["height"]),
        }
    rect = find_window_region(window_substring, matched_title=matched_title)
    if rect is None:
        return None
    return ltrb_to_mss_region(rect)


class DxcamMssFallbackCapture:
    """Window capture: try dxcam briefly; if no frames, mss on window rect / calib.

    Xbox PC app / Remote Play are often UWP/protected and yield empty dxcam
    grabs even when the window title matches. After ~2.5s with no frame,
    switches once to mss screen-region capture and prints
    DXCAM_NO_FRAMES_FALLBACK_MSG.
    """

    def __init__(
        self,
        *,
        window_substring: str | None = "Xbox",
        region: tuple[int, int, int, int] | None = None,
        calib_mss_region: dict[str, int] | None = None,
        fallback_after_s: float = DXCAM_FALLBACK_AFTER_S,
    ) -> None:
        self.window_substring = window_substring
        self._region_ltrb = region
        self._calib_mss_region = calib_mss_region
        self._fallback_after_s = float(fallback_after_s)
        self._dxcam: DxcamWindowCapture | None = None
        self._mss: MssRegionCapture | None = None
        self._backend: Any = None
        self._mode = "dxcam"
        self._t0: float | None = None
        self._got_frame = False
        self._fallback_done = False
        self._fallback_msg_printed = False
        self.matched_title: str | None = None
        self.last_error: str | None = None

        # Construct dxcam wrapper (does not import dxcam until grab)
        self._dxcam = DxcamWindowCapture(
            window_substring=window_substring,
            region=region,
        )
        self.matched_title = getattr(self._dxcam, "matched_title", None)
        self._backend = self._dxcam

        # If window already matched but no region was passed, stash ltrb for mss
        if self._region_ltrb is None and self._dxcam._resolved_region is not None:
            self._region_ltrb = self._dxcam._resolved_region

    @property
    def name(self) -> str:
        return self._mode

    def _print_fallback_once(self) -> None:
        if self._fallback_msg_printed:
            return
        print(DXCAM_NO_FRAMES_FALLBACK_MSG)
        self._fallback_msg_printed = True

    def _switch_to_mss(self) -> bool:
        """Build mss backend from calib crop or window rect. True if switched."""
        if self._fallback_done and self._mss is not None:
            return True
        titles: list[str] = []
        region = resolve_mss_region_for_window(
            self.window_substring,
            calib_region=self._calib_mss_region,
            matched_title=titles,
        )
        if region is None and self._region_ltrb is not None:
            region = ltrb_to_mss_region(self._region_ltrb)
        if titles and not self.matched_title:
            self.matched_title = titles[0]
        if region is None:
            return False
        try:
            self._mss = MssRegionCapture(region=region)
        except ImportError:
            return False
        if self._dxcam is not None:
            try:
                self._dxcam.close()
            except Exception:
                pass
        self._backend = self._mss
        self._mode = "mss"
        self._fallback_done = True
        self._print_fallback_once()
        return True

    def grab(self) -> Frame | None:
        import time as _time

        if self._t0 is None:
            self._t0 = _time.time()

        frame: Frame | None = None
        try:
            frame = self._backend.grab() if self._backend is not None else None
        except ImportError:
            # dxcam (or current backend) missing — fall back immediately
            if self._mode == "dxcam" and self._switch_to_mss():
                try:
                    return self._backend.grab() if self._backend is not None else None
                except Exception:
                    return None
            raise
        except Exception as e:
            self.last_error = f"{self._mode}: {type(e).__name__}: {e}"
            frame = None

        if frame is not None:
            self._got_frame = True
            return frame

        if (
            not self._got_frame
            and not self._fallback_done
            and self._mode == "dxcam"
            and (_time.time() - self._t0) >= self._fallback_after_s
        ):
            if self._switch_to_mss():
                try:
                    return self._backend.grab() if self._backend is not None else None
                except Exception:
                    return None
        return None

    def close(self) -> None:
        for b in (self._mss, self._dxcam):
            if b is not None:
                try:
                    b.close()
                except Exception:
                    pass
        self._backend = None
