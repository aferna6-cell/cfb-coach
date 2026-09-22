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

    def _ensure(self) -> Any:
        if self._camera is not None:
            return self._camera
        try:
            import dxcam  # type: ignore
        except ImportError as e:
            raise ImportError(
                "DxcamWindowCapture requires dxcam (Windows) — pip install -e '.[vision]'"
            ) from e
        if self._resolved_region is None and self.window_substring:
            self._resolved_region = _find_window_region(self.window_substring)
        region = self._resolved_region
        self._camera = dxcam.create(output_color=self.output_color, region=region)
        return self._camera

    def grab(self) -> Frame | None:
        cam = self._ensure()
        frame = cam.grab()
        if frame is None:
            return None
        h, w = int(frame.shape[0]), int(frame.shape[1])
        return Frame(bgr=frame, width=w, height=h, source="dxcam")

    def close(self) -> None:
        self._camera = None


class DeviceCapture:
    """Future drop-in: OpenCV VideoCapture device index (capture card). Stub-ready."""

    name = "device"

    def __init__(self, index: int = 0) -> None:
        self.index = index
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
        self._cap = cv2.VideoCapture(self.index)
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


def _find_window_region(substring: str) -> tuple[int, int, int, int] | None:
    """Windows-only: FindWindow-ish via win32gui. Returns (l,t,r,b) or None."""
    try:
        import win32gui  # type: ignore
    except ImportError:
        return None

    needle = substring.lower()
    found: list[tuple[int, int, int, int]] = []

    def _enum(hwnd: int, _: Any) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd) or ""
        if needle in title.lower():
            rect = win32gui.GetWindowRect(hwnd)
            found.append(rect)

    try:
        win32gui.EnumWindows(_enum, None)
    except Exception:
        return None
    if not found:
        return None
    return found[0]


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
