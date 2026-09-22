"""ROI-based HUD extraction stubs — OCR optional / modular."""

from __future__ import annotations

from typing import Any

from cfb_coach.vision.observation import SituationHUD


def extract_hud(
    frame_bgr: Any | None,
    *,
    rois: dict[str, dict[str, float]] | None = None,
    use_ocr: bool = False,
) -> SituationHUD:
    """Extract coarse situation from HUD ROIs.

    Without OCR this returns an empty SituationHUD (stubs ready for Tesseract etc.).
    Relative ROIs are fractions of frame width/height.
    """
    sit = SituationHUD()
    if frame_bgr is None:
        return sit
    if not use_ocr:
        # Stub: mark that HUD path ran; real OCR later
        return sit

    try:
        # Optional OCR path — only if pytesseract present; never hard-dep
        import pytesseract  # type: ignore
        import cv2  # type: ignore
    except ImportError:
        return sit

    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    rois = rois or {
        "down_distance": {"x": 0.40, "y": 0.88, "w": 0.20, "h": 0.08},
    }
    dd = rois.get("down_distance")
    if dd:
        x0 = int(dd["x"] * w)
        y0 = int(dd["y"] * h)
        x1 = int((dd["x"] + dd["w"]) * w)
        y1 = int((dd["y"] + dd["h"]) * h)
        crop = frame_bgr[y0:y1, x0:x1]
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray) or ""
            sit = _parse_down_distance(text, sit)
        except Exception:
            pass
    return sit


def _parse_down_distance(text: str, sit: SituationHUD) -> SituationHUD:
    import re

    m = re.search(r"([1-4])\s*[&and]+\s*(\d{1,2})", text, re.I)
    if m:
        sit.down = int(m.group(1))
        sit.distance = int(m.group(2))
    return sit


def crop_roi(frame_bgr: Any, roi: dict[str, float]) -> Any | None:
    """Crop relative ROI; returns None if frame missing."""
    if frame_bgr is None:
        return None
    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    x0 = max(0, int(roi.get("x", 0) * w))
    y0 = max(0, int(roi.get("y", 0) * h))
    x1 = min(w, int((roi.get("x", 0) + roi.get("w", 1)) * w))
    y1 = min(h, int((roi.get("y", 0) + roi.get("h", 1)) * h))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_bgr[y0:y1, x0:x1]
