"""Windows Graphics Capture (WGC) backend — capture ONE window by HWND.

Why WGC for Remote Play (see docs/vision-capture-rca.md):
- Captures the target window's own surface, so a terminal / overlay on top of
  it does not pollute the frame (mss/dxcam copy whatever is on screen).
- Works per-window in physical px — no GetWindowRect / DPI / invisible-border
  math, no "Invalid Region".
- Same API OBS "Window Capture (Windows 10 1903+)" uses.

What WGC does NOT fix: if the Xbox app / browser stops *rendering* the stream
when it loses focus or is fully covered, WGC faithfully captures the paused
frame. Keep the stream window focused (tips go to the console / overlay).

Optional dep (native Windows Python only):  pip install windows-capture
(abi3 wheel — works on Python 3.14). Lazy import; never loaded on Linux.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from cfb_coach.vision.capture import Frame


class WgcWindowCapture:
    """Latest-frame WGC grabber. grab() returns newest Frame (BGR) or None."""

    name = "wgc"

    def __init__(self, *, hwnd: int | None = None, title: str | None = None) -> None:
        if hwnd is None and not title:
            raise ValueError("WgcWindowCapture needs hwnd or title")
        self.hwnd = hwnd
        self.title = title
        self.matched_title: str | None = title
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._last_seq = 0
        self._control: Any = None
        self._closed = threading.Event()

    def _ensure(self) -> None:
        if self._control is not None:
            return
        try:
            from windows_capture import WindowsCapture  # type: ignore
        except ImportError as e:
            raise ImportError(
                "WGC capture requires windows-capture — pip install windows-capture "
                "(native Windows Python)"
            ) from e

        kwargs: dict[str, Any] = {
            "cursor_capture": False,
            "draw_border": False,  # Win11: hide the yellow capture border
        }
        if self.hwnd is not None:
            kwargs["window_hwnd"] = int(self.hwnd)
        else:
            kwargs["window_name"] = self.title
        cap = WindowsCapture(**kwargs)

        def on_frame_arrived(frame: Any, _control: Any) -> None:
            try:
                # frame_buffer is a view into mapped GPU memory — copy now.
                bgr = frame.frame_buffer[:, :, :3].copy()
                f = Frame(
                    bgr=bgr,
                    width=int(frame.width),
                    height=int(frame.height),
                    source="wgc",
                    ts=time.time(),
                )
                with self._lock:
                    self._latest = f
                    self._seq += 1
            except Exception as e:  # keep capture thread alive
                self.last_error = f"{type(e).__name__}: {e}"

        def on_closed() -> None:
            self._closed.set()

        # Assign directly (WindowsCapture.event() keys off __name__).
        cap.frame_handler = on_frame_arrived
        cap.closed_handler = on_closed
        try:
            self._control = cap.start_free_threaded()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            raise

    def grab(self) -> Frame | None:
        self._ensure()
        with self._lock:
            f = self._latest
            seq = self._seq
        if f is None or seq == self._last_seq:
            # WGC only delivers on change; a frozen stream = no new frames.
            return None
        self._last_seq = seq
        return f

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def close(self) -> None:
        ctl = self._control
        self._control = None
        if ctl is not None:
            try:
                ctl.stop()
            except Exception:
                pass
