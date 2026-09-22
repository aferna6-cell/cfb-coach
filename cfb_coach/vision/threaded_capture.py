"""Threaded capture — overwrite latest frame; analysis drops stale."""

from __future__ import annotations

import threading
import time
from typing import Any

from cfb_coach.vision.capture import CaptureBackend, Frame


class ThreadedCapture:
    """Wrap a CaptureBackend: capture thread overwrites latest; readers get newest."""

    def __init__(self, backend: CaptureBackend, *, target_fps: float = 30.0) -> None:
        self.backend = backend
        self.target_fps = max(1.0, float(target_fps))
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._dropped = 0
        self._last_grab_seq = -1
        self._running = False
        self._thread: threading.Thread | None = None
        self._capture_fps = 0.0
        self._frames = 0
        self._fps_t = time.time()
        self._latency_ms = 0.0
        self._last_error: str | None = None
        self._errors = 0

    @property
    def name(self) -> str:
        return f"threaded:{getattr(self.backend, 'name', 'capture')}"

    @property
    def capture_fps(self) -> float:
        return self._capture_fps

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def queue_depth(self) -> int:
        # overwrite buffer ⇒ 0 or 1
        with self._lock:
            return 0 if self._latest is None else 1

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    @property
    def last_error(self) -> str | None:
        """Most recent capture exception (thread-swallowed) or backend-reported error."""
        return self._last_error or getattr(self.backend, "last_error", None)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="cfb-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None

    def close(self) -> None:
        self.stop()
        try:
            self.backend.close()
        except Exception:
            pass

    def grab(self) -> Frame | None:
        """Return latest frame; if same as last grab, still return it (overwrite model)."""
        with self._lock:
            frame = self._latest
            seq = self._seq
        if frame is None:
            # fallback sync grab if thread not started
            try:
                return self.backend.grab()
            except Exception:
                return None
        if seq == self._last_grab_seq:
            # stale for analysis — caller may drop
            self._dropped += 1
        self._last_grab_seq = seq
        if frame.ts:
            self._latency_ms = max(0.0, (time.time() - float(frame.ts)) * 1000.0)
        return frame

    def grab_fresh(self) -> Frame | None:
        """Return frame only if newer than last analysis grab; else None (drop stale)."""
        with self._lock:
            frame = self._latest
            seq = self._seq
        if frame is None or seq == self._last_grab_seq:
            if seq == self._last_grab_seq and frame is not None:
                self._dropped += 1
            return None
        self._last_grab_seq = seq
        if frame.ts:
            self._latency_ms = max(0.0, (time.time() - float(frame.ts)) * 1000.0)
        return frame

    def debug_stats(self) -> dict[str, Any]:
        return {
            "capture_fps": self._capture_fps,
            "dropped": self._dropped,
            "queue": self.queue_depth,
            "latency_ms": self._latency_ms,
            "seq": self._seq,
            "errors": self._errors,
            "last_error": self.last_error,
        }

    def _loop(self) -> None:
        delay = 1.0 / self.target_fps
        while self._running:
            t0 = time.time()
            try:
                frame = self.backend.grab()
            except Exception as e:
                # Never kill the thread, but never hide WHY there are no frames.
                self._last_error = f"{type(e).__name__}: {e}"
                self._errors += 1
                frame = None
            if frame is not None:
                if getattr(frame, "ts", None) is None:
                    try:
                        frame.ts = time.time()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                with self._lock:
                    self._latest = frame
                    self._seq += 1
                self._frames += 1
                now = time.time()
                if now - self._fps_t >= 0.5:
                    self._capture_fps = self._frames / (now - self._fps_t)
                    self._frames = 0
                    self._fps_t = now
            elapsed = time.time() - t0
            time.sleep(max(0.0, delay - elapsed))
