"""Classical CV blob / centroid extraction — interface stays clean if weak."""

from __future__ import annotations

from typing import Any


def extract_player_centroids(
    frame_bgr: Any | None,
    *,
    field_roi: dict[str, float] | None = None,
    max_blobs: int = 22,
) -> list[tuple[float, float]]:
    """Return normalized (x, y) centroids from a crude field mask + components.

    Pure fallback: if cv2/numpy missing or frame empty → [].
    Coordinates normalized to the (optionally ROI-cropped) field: x,y in [0,1].
    """
    if frame_bgr is None:
        return []
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return []

    img = frame_bgr
    if field_roi:
        from cfb_coach.vision.hud import crop_roi

        cropped = crop_roi(frame_bgr, field_roi)
        if cropped is not None:
            img = cropped

    h, w = int(img.shape[0]), int(img.shape[1])
    if h < 8 or w < 8:
        return []

    # Crude green-ish field mask (HSV) — weak but interface-stable
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Grass: broad green; invert to find non-grass (players)
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    grass = cv2.inRange(hsv, lower, upper)
    players_mask = cv2.bitwise_not(grass)
    # Clean
    kernel = np.ones((3, 3), np.uint8)
    players_mask = cv2.morphologyEx(players_mask, cv2.MORPH_OPEN, kernel)
    players_mask = cv2.morphologyEx(players_mask, cv2.MORPH_CLOSE, kernel)

    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        players_mask, connectivity=8
    )
    out: list[tuple[float, float]] = []
    # skip label 0 (background)
    areas = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        areas.append((area, i))
    areas.sort(reverse=True)
    # Filter tiny / huge
    min_a = max(8, (h * w) // 8000)
    max_a = (h * w) // 20
    for area, i in areas:
        if area < min_a or area > max_a:
            continue
        cx, cy = centroids[i]
        out.append((float(cx) / w, float(cy) / h))
        if len(out) >= max_blobs:
            break
    return out


def split_offense_defense(
    centroids: list[tuple[float, float]],
    *,
    los_y: float = 0.5,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split by LOS proxy: y < los_y → offense (bottom), else defense."""
    offense = [p for p in centroids if p[1] < los_y]
    defense = [p for p in centroids if p[1] >= los_y]
    return offense, defense
