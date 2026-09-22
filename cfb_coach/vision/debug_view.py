"""Optional OpenCV debug window — frame, FPS, crop, ROIs, state text."""

from __future__ import annotations

from typing import Any


class DebugView:
    """Lazy OpenCV window. No-op if cv2 missing."""

    def __init__(self, title: str = "CFB Coach Vision") -> None:
        self.title = title
        self._enabled = False
        self._cv2: Any = None
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            self._enabled = True
        except ImportError:
            self._enabled = False

    @property
    def available(self) -> bool:
        return self._enabled

    def show(
        self,
        frame_bgr: Any,
        *,
        fps: float | None = None,
        lines: list[str] | None = None,
        rois: dict[str, dict[str, float]] | None = None,
    ) -> None:
        if not self._enabled or frame_bgr is None or self._cv2 is None:
            return
        cv2 = self._cv2
        img = frame_bgr.copy()
        h, w = img.shape[:2]
        if rois:
            for name, r in rois.items():
                x0 = int(r.get("x", 0) * w)
                y0 = int(r.get("y", 0) * h)
                x1 = int((r.get("x", 0) + r.get("w", 0)) * w)
                y1 = int((r.get("y", 0) + r.get("h", 0)) * h)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 1)
                cv2.putText(
                    img,
                    name,
                    (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )
        y = 18
        if fps is not None:
            cv2.putText(
                img,
                f"FPS {fps:.1f}",
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
            )
            y += 18
        for line in lines or []:
            cv2.putText(
                img,
                line[:80],
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += 16
        cv2.imshow(self.title, img)
        cv2.waitKey(1)

    def close(self) -> None:
        if self._enabled and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.title)
            except Exception:
                try:
                    self._cv2.destroyAllWindows()
                except Exception:
                    pass
