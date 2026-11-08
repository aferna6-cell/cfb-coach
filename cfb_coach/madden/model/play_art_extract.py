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

import io
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

EXTRACTOR_VERSION = "play_art_extract.v2"
TIER = "diagram_extracted_unreviewed"
PANEL = (556, 258)

# hue buckets in degrees; measured from AceMadden panels, recorded in output as colour names only (no football meaning attached)
HUES = {"red": (330, 20), "yellow": (40, 89), "green": (90, 150), "cyan": (151, 199), "blue": (200, 255), "purple": (256, 310)}


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
    """Expand locally without wrapping opposite image borders together."""
    height, width = mask.shape
    padded = np.pad(mask, r)
    out = mask.copy()
    for dx in range(2 * r + 1):
        for dy in range(2 * r + 1):
            out |= padded[dy:dy + height, dx:dx + width]
    return out


def _skeleton(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, preserving junction topology without widening crossings."""
    a = np.pad(mask, 1).astype(bool)
    while True:
        changed = False
        for phase in (0, 1):
            p = [np.roll(np.roll(a, dy, 0), dx, 1) for dy, dx in
                 ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))]
            count = sum(x.astype(np.uint8) for x in p)
            turns = sum((~p[i] & p[(i + 1) % 8]).astype(np.uint8) for i in range(8))
            if phase == 0:
                guard = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                guard = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = a & (count >= 2) & (count <= 6) & (turns == 1) & guard
            if remove.any():
                a[remove] = False
                changed = True
        if not changed:
            return a[1:-1, 1:-1]


def _branch_paths(mask: np.ndarray, players: list[dict]) -> list[dict]:
    """Compress skeleton junctions, pair opposite arms, then trace all edges.

    Pairing uses tangents outside the junction, not arbitrary pixel BFS order.
    Odd forks/equal pairings are retained as explicitly ambiguous proposals.
    """
    skeleton = _skeleton(mask)
    ys, xs = np.nonzero(skeleton)
    points = set(zip(xs.tolist(), ys.tolist()))
    graph = {}
    for x, y in points:
        graph[x, y] = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                       if (dx or dy) and (x + dx, y + dy) in points
                       and not (dx and dy and ((x + dx, y) in points or (x, y + dy) in points))]
    junctions = np.zeros_like(mask)
    for pt, neighbors in graph.items():
        if len(neighbors) > 2:
            junctions[pt[1], pt[0]] = True
    nodes = _components(_dilate(junctions, 3), 1)
    node_at = {}
    centers = []
    for i, node in enumerate(nodes):
        for x, y in node:
            node_at[int(x), int(y)] = i
        centers.append(tuple(node.mean(axis=0)))
    for pt, neighbors in graph.items():
        if len(neighbors) <= 1 and pt not in node_at:
            node_at[pt] = len(centers)
            centers.append(pt)
    remaining = skeleton.copy()
    for x, y in node_at:
        remaining[y, x] = False
    edges = []
    def proximity(pt):
        return min((math.dist(pt, (m['x'], m['y'])) for m in players), default=1e9)
    for comp in _components(remaining, 1):
        path = _geodesic_path(comp)
        if len(path) < 2:
            continue
        ends = []
        for pt in (path[0], path[-1]):
            nearby = [(math.dist(pt, q), node_at[q]) for q in node_at
                      if abs(q[0] - pt[0]) <= 2 and abs(q[1] - pt[1]) <= 2]
            if nearby:
                ends.append(min(nearby)[1])
            else:
                ends.append(len(centers))
                centers.append(pt)
        full = [centers[ends[0]], *path, centers[ends[1]]]
        length = sum(math.dist(a, b) for a, b in zip(full, full[1:]))
        if length < 12 and min(proximity(full[0]), proximity(full[-1])) > 16:
            continue  # arrowhead / thinning spur, not another route
        edges.append({'nodes': ends, 'path': full})
    incident = {i: [] for i in range(len(centers))}
    for i, edge in enumerate(edges):
        for node in edge['nodes']:
            incident[node].append(i)
    pair, ambiguous = {}, set()
    for node, arms in incident.items():
        choices = list(arms)
        def tangent(edge):
            path = edges[edge]['path']
            if edges[edge]['nodes'][0] != node:
                path = path[::-1]
            tip = next((p for p in path[1:] if math.dist(p, path[0]) >= 12), path[-1])
            v = np.subtract(tip, path[0])
            return v / (np.linalg.norm(v) or 1)
        if len(choices) > 2 and len(choices) % 2:
            ambiguous.add(node)
        while len(choices) > 1:
            candidates = sorted((float(np.dot(tangent(a), tangent(b))), a, b)
                                for i, a in enumerate(choices) for b in choices[i + 1:])
            score, a, b = candidates[0]
            # A tie between different continuations cannot establish identity.
            if any(other_score - score < 0.12 and ({a, b} & {c, d})
                   for other_score, c, d in candidates[1:]):
                ambiguous.add(node)
            pair[node, a], pair[node, b] = b, a
            choices.remove(a)
            choices.remove(b)
    used, paths = set(), []
    starts = sorted(((node, edge) for node, arms in incident.items() for edge in arms
                     if (node, edge) not in pair), key=lambda ne: (proximity(centers[ne[0]]), ne))
    starts += [(edge['nodes'][0], i) for i, edge in enumerate(edges)]
    for node, edge in starts:
        if edge in used:
            continue
        path, uncertain = [], False
        while edge not in used:
            used.add(edge)
            item = edges[edge]
            piece = item['path'] if item['nodes'][0] == node else item['path'][::-1]
            path += piece if not path else piece[1:]
            uncertain |= node in ambiguous
            node = item['nodes'][1] if item['nodes'][0] == node else item['nodes'][0]
            uncertain |= node in ambiguous
            following = pair.get((node, edge))
            if following is None or node in ambiguous:
                break
            edge = following
        if sum(math.dist(a, b) for a, b in zip(path, path[1:])) >= 12:
            paths.append({'path': path, 'px': len(path), 'junction_ambiguous': uncertain})
    return paths


def _owner(point, players, limit, ambiguity=5.0):
    distances = sorted((math.dist(point, (m['x'], m['y'])), m['index_left_to_right']) for m in players)
    if not distances:
        return {'owner_marker_index': None, 'owner_distance_px': None, 'association_flags': ['no_marker']}
    distance, index = distances[0]
    flags = []
    if distance > limit:
        flags.append('owner_too_far')
    if len(distances) > 1 and distances[1][0] - distance <= ambiguity:
        flags.append('owner_ambiguous')
    return {'owner_marker_index': None if flags else index, 'owner_distance_px': round(distance, 1),
            'association_flags': flags}


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
            # A coverage connector can touch a marker: retain dense marker cores,
            # rather than discarding the entire white component as text/a line.
            component_mask = np.zeros_like(white)
            component_mask[c[:, 1], c[:, 0]] = True
            density = sum(np.roll(np.roll(component_mask, dy, 0), dx, 1).astype(np.uint8)
                          for dy in range(-2, 3) for dx in range(-2, 3))
            for core in _components(component_mask & (density >= 22), 5):
                if np.ptp(core[:, 0]) > 12 or np.ptp(core[:, 1]) > 12:
                    continue
                out.append({"x": float(core[:, 0].mean()), "y": float(core[:, 1].mean()),
                            "shape": "circle", "px": int(len(core)),
                            "connector_component_bbox": [int(x0), int(y0), int(x1), int(y1)],
                            "connector_polyline": _simplify(_geodesic_path(c), eps=2, min_end=0)})
            continue                                  # text / hash marks / badges, not a player marker
        fill = len(c) / float(w * h)
        shape = "x" if fill < 0.55 else ("square" if abs(w - h) <= 3 and fill > 0.8 else "circle")
        out.append({"x": float(c[:, 0].mean()), "y": float(c[:, 1].mean()), "shape": shape, "px": int(len(c))})
    return out


def _family(path: list[tuple[float, float]], start_x: float) -> dict[str, Any]:
    """Coarse shape label from the simplified polyline. A SUGGESTION for the reviewer; never a taxonomy assignment."""
    pts = _simplify(path, eps=7.0, min_end=9.0)
    if len(pts) < 2:
        return {"suggested_shape": "unknown_complex", "n_segments": 0, "upfield_px": 0}
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
    elif len(segs) >= 2:
        dx1, dy1 = segs[0]
        dx2, dy2 = segs[-1]
        if dy1 > 8 and dy2 < -8:
            label = "curl_or_comeback"
        elif dy1 > 0 and abs(dx2) > abs(dy2) * 1.5:
            label = "break_out" if toward_sideline(dx2) else "break_in"
        elif dy1 > 0 and dy2 < 0:
            label = "curl_or_comeback"
        elif dy1 > 0 and dy2 > 0:
            label = "stem_then_diagonal"
        elif all(dy <= 3 for _, dy in segs):
            label = "backward_or_swing" if cy2 > cy1 + 10 else "flat_or_horizontal"
        elif len(segs) >= 3:
            label = "multi_break"
    return {"suggested_shape": label, "n_segments": len(segs), "upfield_px": round(total_up, 1),
            "depth_fraction_of_panel": round(total_up / PANEL[1], 3), "first_segment_angle_from_vertical": round(ang, 1)}


def _chain(lines: list[dict[str, Any]], gap: float = 22.0, players: list[dict] | None = None) -> list[dict[str, Any]]:
    """Join same-colour fragments whose ends nearly touch (a glyph or marker sits on the line). Short stubs next to a longer line
    of the same colour are arrowheads and are absorbed."""
    lines = [dict(l) for l in lines]
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                if a["colour"] != b["colour"] or a.get("junction_ambiguous") or b.get("junction_ambiguous"):
                    continue
                best = None
                for ea, pa in ((0, a["path"]), (-1, a["path"][::-1])):
                    for eb, pb in ((0, b["path"]), (-1, b["path"][::-1])):
                        d = math.hypot(pa[-1][0] - pb[0][0], pa[-1][1] - pb[0][1])
                        va = np.subtract(pa[-1], pa[max(0, len(pa) - 8)])
                        vb = np.subtract(pb[min(7, len(pb) - 1)], pb[0])
                        cosine = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) or 1))
                        def near_marker(pt):
                            return any(math.dist(pt, (m['x'], m['y'])) < 12 for m in (players or []))
                        at_marker = near_marker(pa[0]) and near_marker(pb[-1])
                        if d <= gap and cosine > 0.8 and not at_marker and (best is None or d < best[0]):
                            best = (d, pa, pb)
                if best:
                    _, pa, pb = best
                    lines[i] = {"colour": a["colour"], "px": a["px"] + b["px"], "path": pa + pb, "junction_ambiguous": a.get("junction_ambiguous", False) or b.get("junction_ambiguous", False)}
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
    vivid = (s > 0.42) & (v > (0.28 if side == "offense" else 0.42)) & ~chrome
    dim = (s > 0.18) & (v > 0.20) & ~chrome        # translucent and saturated zone fills
    players = _markers(white)
    for i, p in enumerate(sorted(players, key=lambda m: (m["x"], m["y"]))):
        p["index_left_to_right"] = i
    lines = []
    hue_name = np.full(h.shape, "", dtype=object)
    for name, (lo, hi) in HUES.items():
        m = ((h >= lo) & (h <= hi)) if lo <= hi else ((h >= lo) | (h <= hi))
        hue_name[m] = name
    for name in HUES:
        mask = vivid & (hue_name == name)
        for branch in _branch_paths(mask, players):
            lines.append({'colour': name, **branch})
    lines = _chain(lines, players=players)
    out: dict[str, Any] = {"extractor": EXTRACTOR_VERSION, "tier": TIER, "side": side, "players": players, "flags": flags,
                           "colour_meaning": "UNVERIFIED (colours are recorded as measured)", "model_feature_eligible": False, "verified": 0}
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
            r = {"colour": ln["colour"], "start": [round(start[0], 1), round(start[1], 1)], "end": [round(end[0], 1), round(end[1], 1)],
                 "polyline": [[round(x, 1), round(y, 1)] for x, y in _simplify(path)], "trace_complete": True,
                 **_owner(start, players, 36), **_family(path, start[0])}
            if ln.get('junction_ambiguous'):
                r['association_flags'].append('junction_ambiguous')
                r['suggested_shape'] = 'unknown_complex'
                r['trace_complete'] = False
                flags.append('route_junction_ambiguous')
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
        for name in ("blue", "purple", "yellow", "green", "red", "cyan", "neutral"):
            m = (dim & (hue_name == name)) if name != 'neutral' else ((s < 0.18) & (v > 0.28) & (v < 0.8) & ~chrome)
            # Close only narrow holes left by field lines/labels, not separated ellipses.
            interior = m.copy()
            for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                interior &= np.roll(np.roll(m, dy, 0), dx, 1)
            m &= _dilate(interior, 2)
            joined = _dilate(m, 2)
            for region in _components(joined, 400):
                comp = region[m[region[:, 1], region[:, 0]]]
                if len(comp) < 400:
                    continue
                x0, x1, y0, y1 = comp[:, 0].min(), comp[:, 0].max(), comp[:, 1].min(), comp[:, 1].max()
                if (x1 - x0) < 25 or (y1 - y0) < 12 or (x1 - x0) > 300 or len(comp) / ((x1-x0+1)*(y1-y0+1)) < 0.35:
                    continue
                zones.append({"colour": name, "centroid": [round(float(comp[:, 0].mean()), 1), round(float(comp[:, 1].mean()), 1)],
                              "bbox": [int(x0), int(y0), int(x1), int(y1)], "area_px": int(len(comp)),
                              "centroid_fraction": [round(float(comp[:, 0].mean()) / PANEL[0], 3), round(float(comp[:, 1].mean()) / PANEL[1], 3)],
                              **_owner((float(comp[:, 0].mean()), float(comp[:, 1].mean())), players, 120)})
        for zone in zones:
            if zone['colour'] == 'neutral':
                cx, cy = zone['centroid']
                candidates = []
                for marker in players:
                    box = marker.get('connector_component_bbox')
                    connector = marker.get('connector_polyline', [])
                    reaches_center = any(math.dist((cx, cy), endpoint) < 14 for endpoint in connector)
                    if box and reaches_center and box[0] - 4 <= cx <= box[2] + 4 and box[1] - 4 <= cy <= box[3] + 4:
                        candidates.append(marker)
                if candidates:
                    zone.update(_owner(zone['centroid'], candidates, 120))
                    zone['association_basis'] = 'visible_white_connector_to_marker'
        zone_owners = [z['owner_marker_index'] for z in zones if z['owner_marker_index'] is not None]
        if len(zone_owners) != len(set(zone_owners)):
            flags.append('multiple_zones_share_marker_review')
        # light-yellow/olive hook zones are dim: also look at low-value olive fills the hue test calls yellow
        rush = []
        links = []
        for ln in lines:
            path = ln["path"]
            ends = [path[0], path[-1]]
            if ln["colour"] == "red":
                near_sq = lambda pt: min((math.hypot(pt[0] - m["x"], pt[1] - m["y"]) for m in players if m["shape"] == "square"), default=1e9)  # noqa: E731
                a, b = (ends[0], ends[1]) if near_sq(ends[0]) <= near_sq(ends[1]) else (ends[1], ends[0])
                rush.append({"colour": "red", **_owner(a, [m for m in players if m["shape"] == "square"], 36), "start": [round(a[0], 1), round(a[1], 1)], "end": [round(b[0], 1), round(b[1], 1)],
                             "direction_dx_dy_upfield": [round(b[0] - a[0], 1), round(-(b[1] - a[1]), 1)], "starts_at_square_marker": near_sq(a) <= 14})
            else:
                links.append({"colour": ln["colour"], "start": [round(ends[0][0], 1), round(ends[0][1], 1)], "end": [round(ends[1][0], 1), round(ends[1][1], 1)],
                              "length_px": round(math.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1]), 1)})
        out.update({"zones": zones, "rush_arrows": rush, "defender_links": links,
                    "counts": {"markers": len(players), "zones": len(zones), "rush_arrows": len(rush), "links": len(links)}})
        if len(players) < 7:
            flags.append("fewer_than_7_defender_markers")
    out["flags"] = list(dict.fromkeys(flags))
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
        d.text(tuple(z['centroid']), f"zone owner={z.get('owner_marker_index')}", fill=(255, 255, 0))
    for r in extraction.get('rush_arrows', []):
        d.line([tuple(r['start']), tuple(r['end'])], fill=(255, 0, 255), width=2)
        d.text(tuple(r['end']), f"rush owner={r.get('owner_marker_index')}", fill=(255, 255, 0))
    return img
