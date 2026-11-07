"""Geometric extraction from Madden play-art diagrams (AceMadden-style 556x258 panels) into PROPOSED assignments.

What this is: colour/shape segmentation (PIL + numpy only) that turns a diagram into players, route polylines, defensive zones and rush
arrows, each with measured geometry. What it is NOT: verification. Every record is ``tier = diagram_extracted_unreviewed`` and needs a human
decision before anything is promoted to ``diagram_verified``. Specifically it never claims:

* which positional slot (WR1, SLWR, HB1...) a marker is: the art does not say; markers are indexed left-to-right and mapped to slots only
  by a reviewer or by formation knowledge;
* what a line colour MEANS (e.g. a highlighted primary route): colours are recorded as measured, meaning unverified;
* yardage: depth is a fraction of the panel height (calibration to yards is not verified);
* a QB read order: a diagram does not provide one.

Anything ambiguous (route with no start marker, marker with no route, touching lines, unexpected colours) becomes a flag in ``flags``.
"""
from __future__ import annotations

import colorsys
import io
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

EXTRACTOR_VERSION = "play_art_extract.v1"
TIER = "diagram_extracted_unreviewed"
PANEL = (556, 258)

# hue buckets in degrees; measured from AceMadden panels, recorded in output as colour names only (no football meaning attached)
HUES = {"red": (330, 20), "yellow": (40, 75), "green": (90, 150), "cyan": (165, 199), "blue": (200, 255), "purple": (256, 310)}
BADGE_BY_COLOUR = {"offense": {"blue": "PASS", "red": "RUN"}, "defense": {"orange": "MAN", "purple": "MATCH", "blue": "ZONE", "red": "BLITZ"}}


def _hue_name(h: float) -> str | None:
    for name, (lo, hi) in HUES.items():
        if (lo <= h <= hi) if lo <= hi else (h >= lo or h <= hi):
            return name
    return None


def _hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = arr.astype(float) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    rc = np.where(m, (mx - r) / np.where(m, d, 1), 0)
    gc = np.where(m, (mx - g) / np.where(m, d, 1), 0)
    bc = np.where(m, (mx - b) / np.where(m, d, 1), 0)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0 * 360.0
    s = np.where(mx > 1e-9, d / np.where(mx > 1e-9, mx, 1), 0)
    return h, s, mx


def _components(mask: np.ndarray, min_px: int) -> list[np.ndarray]:
    """8-connected components as (n,2) arrays of (x, y)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        q = deque([(x0, y0)])
        seen[y0, x0] = True
        pts = []
        while q:
            x, y = q.popleft()
            pts.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
        if len(pts) >= min_px:
            out.append(np.array(pts))
    return out


def _dilate(mask: np.ndarray, r: int = 3) -> np.ndarray:
    """Bridge small gaps (a marker or watermark glyph sitting on a line) so one route is one component."""
    out = mask.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def _geodesic_path(pts: np.ndarray) -> list[tuple[float, float]]:
    """Longest shortest-path through a thick line: the line's centre-ish track, ordered end to end."""
    s = {(int(x), int(y)) for x, y in pts}

    def bfs(src):
        dist, par = {src: 0}, {src: None}
        q = deque([src])
        last = src
        while q:
            c = q.popleft()
            last = c
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in s and n not in dist:
                        dist[n], par[n] = dist[c] + 1, c
                        q.append(n)
        return last, par

    a, _ = bfs(next(iter(s)))
    b, par = bfs(a)
    path, c = [], b
    while c is not None:
        path.append((float(c[0]), float(c[1])))
        c = par[c]
    return path[::-1]


def _rdp(p: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if len(p) < 3:
        return p
    (x1, y1), (x2, y2) = p[0], p[-1]
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1.0
    dmax, idx = 0.0, 0
    for i in range(1, len(p) - 1):
        d = abs(dy * p[i][0] - dx * p[i][1] + x2 * y1 - y2 * x1) / n
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return _rdp(p[: idx + 1], eps)[:-1] + _rdp(p[idx:], eps)
    return [p[0], p[-1]]


def _simplify(path: list[tuple[float, float]], eps: float = 4.0, min_end: float = 14.0) -> list[tuple[float, float]]:
    """RDP, then drop end stubs shorter than ``min_end`` px (corner artefacts of the dilated mask / arrowheads)."""
    pts = _rdp(path, eps)
    while len(pts) > 2 and math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]) < min_end:
        pts = pts[1:]
    while len(pts) > 2 and math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]) < min_end:
        pts = pts[:-1]
    return pts


def _markers(white: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for c in _components(white, 6):
        x0, x1, y0, y1 = c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w > 40 or h > 40:
            continue                                  # text / hash marks / badges, not a player marker
        fill = len(c) / float(w * h)
        shape = "x" if fill < 0.55 else ("square" if abs(w - h) <= 3 and fill > 0.8 else "circle")
        out.append({"x": float(c[:, 0].mean()), "y": float(c[:, 1].mean()), "shape": shape, "px": int(len(c))})
    return out


def _badge(arr: np.ndarray, side: str) -> dict[str, Any]:
    """The coloured label chip at the bottom-left of the panel (e.g. MAN / MATCH / BLITZ / ZONE, PASS / RUN). Colour is measured; the word is the site's
    convention and is recorded as a hint, not OCR'd and not verified."""
    h, w = arr.shape[:2]
    patch = arr[int(h * 0.90):int(h * 0.99), 2:int(w * 0.11)].reshape(-1, 3).astype(float)
    px = patch[(patch.max(1) - patch.min(1)) > 60]                       # saturated chip pixels only
    if len(px) < 40:
        return {"colour": None, "hint": None}
    r, g, b = px.mean(0) / 255.0
    hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
    deg = hh * 360
    name = "orange" if 15 <= deg < 40 else _hue_name(deg)
    return {"colour": name, "hint": BADGE_BY_COLOUR[side].get(name), "basis": "chip colour; text not read; unverified"}


def _white_bars(white: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for c in _components(white, 20):
        x0, x1, y0, y1 = c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()
        w, hgt = x1 - x0 + 1, y1 - y0 + 1
        if max(w, hgt) >= 25 and max(w, hgt) / max(1, min(w, hgt)) >= 4:
            out.append({"x": round(float(c[:, 0].mean()), 1), "y_top": int(y0), "y_bottom": int(y1), "length_px": int(max(w, hgt)), "vertical": bool(hgt >= w)})
    return sorted(out, key=lambda b: b["x"])


def _family(path: list[tuple[float, float]], start_x: float) -> dict[str, Any]:
    """Coarse shape label from the simplified polyline. A SUGGESTION for the reviewer; never a taxonomy assignment."""
    pts = _simplify(path)
    segs = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        segs.append((x2 - x1, -(y2 - y1)))                     # dx right, dy UPFIELD
    mid = PANEL[0] / 2
    toward_sideline = lambda dx: (dx > 0) == (start_x >= mid)   # noqa: E731
    total_up = max(0.0, -(pts[-1][1] - pts[0][1]))
    first = segs[0]
    ang = math.degrees(math.atan2(abs(first[0]), max(first[1], 1e-6))) if first[1] > 0 else 90.0
    label = "unknown_complex"
    (cx1, cy1), (cx2, cy2) = pts[0], pts[-1]
    chord = math.hypot(cx2 - cx1, cy2 - cy1) or 1.0
    dev = max(abs((cy2 - cy1) * (x - cx1) - (cx2 - cx1) * (y - cy1)) / chord for x, y in pts)
    chord_ang = math.degrees(math.atan2(abs(cx2 - cx1), max(-(cy2 - cy1), 1e-6))) if cy2 < cy1 else 90.0
    if len(segs) > 1 and dev <= 16 and cy2 < cy1 and chord >= 40:
        label = "vertical" if chord_ang < 20 else ("diagonal_upfield" if chord_ang < 70 else "flat_or_horizontal")
    elif len(segs) == 1:
        label = "vertical" if ang < 20 else ("diagonal_upfield" if first[1] > 0 and ang < 70 else "flat_or_horizontal")
        if first[1] < 0:
            label = "backward_or_swing"
    elif len(segs) == 2:
        (dx1, dy1), (dx2, dy2) = segs
        if dy1 > 0 and abs(dx2) > abs(dy2) * 1.5:
            label = "break_out" if toward_sideline(dx2) else "break_in"
        elif dy1 > 0 and dy2 < 0:
            label = "curl_or_comeback"
        elif dy1 > 0 and dy2 > 0:
            label = "stem_then_diagonal"
    return {"suggested_shape": label, "n_segments": len(segs), "upfield_px": round(total_up, 1),
            "depth_fraction_of_panel": round(total_up / PANEL[1], 3), "first_segment_angle_from_vertical": round(ang, 1)}


def _chain(lines: list[dict[str, Any]], gap: float = 22.0) -> list[dict[str, Any]]:
    """Join same-colour fragments whose ends nearly touch (a glyph or marker sits on the line). Short stubs next to a longer line
    of the same colour are arrowheads and are absorbed."""
    lines = [dict(l) for l in lines]
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                if a["colour"] != b["colour"]:
                    continue
                best = None
                for ea, pa in ((0, a["path"]), (-1, a["path"][::-1])):
                    for eb, pb in ((0, b["path"]), (-1, b["path"][::-1])):
                        d = math.hypot(pa[-1][0] - pb[0][0], pa[-1][1] - pb[0][1])
                        if d <= gap and (best is None or d < best[0]):
                            best = (d, pa, pb)
                if best:
                    _, pa, pb = best
                    lines[i] = {"colour": a["colour"], "px": a["px"] + b["px"], "path": pa + pb}
                    del lines[j]
                    changed = True
                    break
            if changed:
                break
    return lines


def analyze(source: bytes | str | Path, side: str) -> dict[str, Any]:
    if side not in ("offense", "defense"):
        raise ValueError(side)
    img = Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source).convert("RGB")
    flags: list[str] = []
    if img.size != PANEL:
        flags.append(f"unexpected_panel_size_{img.size[0]}x{img.size[1]}")
    arr = np.array(img)
    h, s, v = _hsv(arr)
    # the bottom-left badge ("PASS"/"ZONE") and bottom-right counters are chrome, not diagram
    chrome = np.zeros(arr.shape[:2], bool)
    chrome[int(arr.shape[0] * 0.88):, :] = True
    white = (s < 0.18) & (v > 0.86) & ~chrome
    vivid = ((s > 0.42) & (v > 0.42) | (s > 0.18) & (v > 0.72) & (h >= 165) & (h <= 199)) & ~chrome      # second clause: pale cyan lines (backs / motion)
    dim = (s > 0.3) & (v > 0.22) & ~vivid & ~chrome        # translucent zone fills
    players = _markers(white)
    badge = _badge(arr, side)
    stubs = _white_bars(white)
    for i, p in enumerate(sorted(players, key=lambda m: (m["x"], m["y"]))):
        p["index_left_to_right"] = i
    colour_of = {}
    lines = []
    for name in HUES:
        hm = np.vectorize(lambda x: _hue_name(x) == name)(h) if False else None
    hue_name = np.full(h.shape, "", dtype=object)
    for name, (lo, hi) in HUES.items():
        m = ((h >= lo) & (h <= hi)) if lo <= hi else ((h >= lo) | (h <= hi))
        hue_name[m] = name
    for name in HUES:
        mask = _dilate(vivid & (hue_name == name))
        mask &= ~chrome
        for comp in _components(mask, 60):
            path = _geodesic_path(comp)
            if len(path) < 12:
                continue
            lines.append({"colour": name, "px": int(len(comp)), "path": path})
    lines = _chain(lines)
    out: dict[str, Any] = {"extractor": EXTRACTOR_VERSION, "tier": TIER, "side": side, "players": players, "flags": flags,
                           "colour_meaning": "UNVERIFIED (colours are recorded as measured)"}
    if side == "offense":
        routes = []
        for ln in lines:
            path = ln["path"]
            ends = [path[0], path[-1]]
            # start = the end nearer a player marker; otherwise the end lower on the panel (nearer the line of scrimmage)
            def near(pt):
                return min((math.hypot(pt[0] - m["x"], pt[1] - m["y"]) for m in players), default=1e9)
            start, end = (ends[0], ends[1]) if (near(ends[0]), -ends[0][1]) <= (near(ends[1]), -ends[1][1]) else (ends[1], ends[0])
            if end is ends[0]:
                path = path[::-1]
            owner = min(players, key=lambda m: math.hypot(start[0] - m["x"], start[1] - m["y"]), default=None)
            d_owner = math.hypot(start[0] - owner["x"], start[1] - owner["y"]) if owner else 1e9
            r = {"colour": ln["colour"], "start": [round(start[0], 1), round(start[1], 1)], "end": [round(end[0], 1), round(end[1], 1)],
                 "polyline": [[round(x, 1), round(y, 1)] for x, y in _simplify(path)],
                 "owner_marker_index": owner["index_left_to_right"] if owner and d_owner <= 22 else None,
                 "owner_distance_px": round(d_owner, 1) if owner else None, **_family(path, start[0])}
            if r["owner_marker_index"] is None:
                flags.append("route_without_start_marker")
            routes.append(r)
        owners = [r["owner_marker_index"] for r in routes if r["owner_marker_index"] is not None]
        if len(owners) != len(set(owners)):
            flags.append("two_routes_share_a_marker")
        out["routes"] = sorted(routes, key=lambda r: r["start"][0])
        out["counts"] = {"markers": len(players), "routes": len(routes), "markers_with_route": len(set(owners))}
    else:
        zones = []
        for name in ("blue", "purple", "yellow", "green", "red"):
            m = _dilate(dim & (hue_name == name), 3) & ~chrome                       # a link line crossing an oval must not split it
            for comp in _components(m, 400):
                x0, x1, y0, y1 = comp[:, 0].min(), comp[:, 0].max(), comp[:, 1].min(), comp[:, 1].max()
                if (x1 - x0) < 25 or (y1 - y0) < 12:
                    continue
                zones.append({"colour": name, "centroid": [round(float(comp[:, 0].mean()), 1), round(float(comp[:, 1].mean()), 1)],
                              "bbox": [int(x0), int(y0), int(x1), int(y1)], "area_px": int(len(comp)),
                              "centroid_fraction": [round(float(comp[:, 0].mean()) / PANEL[0], 3), round(float(comp[:, 1].mean()) / PANEL[1], 3)]})
        # light-yellow/olive hook zones are dim: also look at low-value olive fills the hue test calls yellow
        rush = []
        links = []
        for ln in lines:
            path = ln["path"]
            ends = [path[0], path[-1]]
            if ln["colour"] == "red":
                near_sq = lambda pt: min((math.hypot(pt[0] - m["x"], pt[1] - m["y"]) for m in players if m["shape"] == "square"), default=1e9)  # noqa: E731
                a, b = (ends[0], ends[1]) if near_sq(ends[0]) <= near_sq(ends[1]) else (ends[1], ends[0])
                rush.append({"start": [round(a[0], 1), round(a[1], 1)], "end": [round(b[0], 1), round(b[1], 1)],
                             "direction_dx_dy_upfield": [round(b[0] - a[0], 1), round(-(b[1] - a[1]), 1)], "starts_at_square_marker": near_sq(a) <= 14})
            else:
                links.append({"colour": ln["colour"], "start": [round(ends[0][0], 1), round(ends[0][1], 1)], "end": [round(ends[1][0], 1), round(ends[1][1], 1)],
                              "length_px": round(math.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1]), 1)})
        out.update({"zones": zones, "rush_arrows": rush, "defender_links": links,
                    "counts": {"markers": len(players), "zones": len(zones), "rush_arrows": len(rush), "links": len(links)}})
        if len(players) < 7:
            flags.append("fewer_than_7_defender_markers")
    out["badge"] = badge
    if side == "offense":
        out["blocker_stubs"] = stubs
    out["needs_review"] = True
    return out


def render_overlay(source: bytes | str | Path, extraction: dict[str, Any]) -> Image.Image:
    """Debug picture for the reviewer: what the extractor thinks it saw, drawn on the original panel."""
    from PIL import ImageDraw
    img = Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source).convert("RGB")
    d = ImageDraw.Draw(img)
    for m in extraction["players"]:
        d.ellipse([m["x"] - 7, m["y"] - 7, m["x"] + 7, m["y"] + 7], outline=(0, 255, 255))
        d.text((m["x"] + 6, m["y"] - 16), str(m["index_left_to_right"]), fill=(0, 255, 255))
    for r in extraction.get("routes", []):
        d.line([tuple(p) for p in r["polyline"]], fill=(255, 0, 255), width=2)
        d.text((r["end"][0] + 4, r["end"][1] - 4), r["suggested_shape"], fill=(255, 255, 255))
    for z in extraction.get("zones", []):
        d.rectangle(z["bbox"], outline=(255, 255, 0))
    return img
