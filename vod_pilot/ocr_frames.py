"""Frame sampling and RapidOCR. Tesseract was tried and could not read the HUD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    t: float
    texts: list[str]
    green: float


def green_ratio(frame_bgr: np.ndarray) -> float:
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    return float(mask.mean() / 255.0)


class FrameOCR:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def read(self, frame_bgr: np.ndarray) -> list[str]:
        result, _elapsed = self._engine(frame_bgr)
        if not result:
            return []
        texts: list[str] = []
        for item in result:
            text = str(item[1]).strip()
            score = float(item[2])
            if text and score >= 0.45:
                texts.append(text)
        return texts


def iter_samples(video_path: str, sample_fps: float = 1.0, max_seconds: float | None = None):
    """Yield (t, frame) at about ``sample_fps`` by decoding the file once."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            if max_seconds is not None and t > max_seconds:
                break
            if index % step == 0:
                yield t, frame
            index += 1
    finally:
        cap.release()
