"""(v0.14) Result / formation consistency pass, inside vodx_refine.refine_rows after the side, score and possession
passes. Every correction needs evidence from the NEXT snap of the same drive (down & distance, scorebug ball spot);
without it the value is nulled and flagged, never guessed.

Next-snap evidence `nx(i)`: the following row, not a kickoff, scorebug clock consistent (vodx_refine.clocks_ok),
gap <= 40 s, and no possession-change signal (side flips with own-snap evidence on both sides).
  * sack_sign: a sack with yards > 0 (dropped minus sign) -> yards = -|y| when the next d&d (down+1, distance +|y|
    +-1) or the offense-perspective yardline (next - this = -|y| +-1) agrees; a next snap that shows a GAIN of |y|, or
    no evidence -> yards null, success null (`result_unverified`), flag.
  * convert/gain vs distance: `convert` with yards < distance, or a gain >= distance followed by down+1 (not a first
    down). Next snap down+1 -> gain/loss of (distance - next distance); next snap 1st down -> convert (yards nulled if
    they contradict); otherwise result null + flag.
  * results recovered from the next d&d (rows with no result): down+1 -> yards = distance - next distance
    (`next_snap_dd`, conf 0.5); 1st & X -> 1st & Y of the same drive with the ball spot moved >= X under BOTH spot
    readings (or 1st & Goal at the next spot) -> convert, yards null unless unique (`next_snap_first_down`, conf 0.5).
  * Hail Mary formation gate: formation / call_screen_formation "Hail Mary" kept only late in a half (Q2/Q4 <= 0:40)
    or on distance >= 25 (same rule as the play gate); else moved out (flag hail_mary_formation_dropped).
Rows changed carry `result_fix` = {kind, before, after} and a quality flag."""
from __future__ import annotations
import re
from collections import Counter

MAX_GAP = 40.0


def _n(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _clocks_ok(r, n) -> bool:
    """Use the box refine pass when it is installed. A missing module is not that pass."""
    try:
        import vodx_refine as VR
    except ImportError:
        VR = None
    if VR is not None and hasattr(VR, "clocks_ok"):
        return bool(VR.clocks_ok(r, n))
    left, right = _cs(r.get("clock")), _cs(n.get("clock"))
    if left is None or right is None:
        return True
    return -2 <= (left - right) <= 50


def _spot_deltas(s0, s1) -> set:
    try:
        import vodx_refine as VR
    except ImportError:
        VR = None
    if VR is not None and hasattr(VR, "spot_deltas"):
        return set(VR.spot_deltas(s0, s1))
    out: set[int] = set()
    for spot in (s0, s1):
        if not isinstance(spot, (int, float)) or isinstance(spot, bool):
            return set()
    return {int(s1) - int(s0)}


def _same_drive(r, n):
    if n is None or n.get("is_kickoff") or r.get("is_kickoff"):
        return False
    if r.get("t_start") is None or n.get("t_start") is None:
        return False
    if not _clocks_ok(r, n) or n["t_start"] - (r.get("t_end") or r["t_start"]) > MAX_GAP:
        return False
    if r.get("result_kind") in ("td", "int", "fumble"):
        return False
    a, b = r.get("side"), n.get("side")
    if a in ("offense", "defense") and b in ("offense", "defense") and a != b:
        return False  # possession changed (or side unknown on one): no next-snap evidence
    return True


def _set_result(r, kind, yards, text, src, conf, fix, st):
    before = {"result": r.get("result"), "result_kind": r.get("result_kind"), "yards": r.get("yards")}
    r["result_kind"] = kind; r["yards"] = yards; r["result"] = text
    r.setdefault("conf", {})["result"] = conf if kind else 0.0
    sources = r.setdefault("field_sources", {})
    if kind:
        sources["result"] = src
    else:
        sources.pop("result", None)
    r["result_fix"] = {"kind": fix, "before": before, "after": {"result": text, "result_kind": kind, "yards": yards}}
    r.setdefault("quality_flags", []).append(f"result_fix:{fix}")
    st[f"result_fix:{fix}"] += 1


def _text(y):
    return "+0" if y == 0 else (f"+{y}" if y > 0 else f"{y}")


def _play_distance(row: dict):
    """Distance for yards. A blank is missing, never subtracted.

    Goal-to-go keeps ``distance`` empty when the spot was not read. The yards
    still use the yard line, or the distance the HUD painted before that clear.
    """
    dist = row.get("distance")
    if isinstance(dist, bool):
        return None
    if isinstance(dist, (int, float)):
        return int(dist)
    if dist not in ("", None):
        return None
    if not row.get("goal_to_go"):
        return None
    yl = row.get("yardline")
    if isinstance(yl, (int, float)) and not isinstance(yl, bool):
        ytg = 100 - int(yl)
        # A spot past the 10 is not a goal-to-go distance.
        if 1 <= ytg <= 10:
            return ytg
    base = row.get("baseline_distance")
    if isinstance(base, (int, float)) and not isinstance(base, bool):
        return int(base)
    return None


def _gain_kind(y):
    return "incomplete_or_no_gain" if y == 0 else ("gain" if y > 0 else "loss")


def _clock_s(clock):
    try:
        import vodx_refine as VR
    except ImportError:
        VR = None
    if VR is not None and hasattr(VR, "clock_s"):
        return VR.clock_s(clock)
    return _cs(clock)


def apply_quality(rows: list[dict], frames=None) -> Counter:
    st = Counter()
    for r in rows:
        if not isinstance(r, dict):
            continue
        for key in ("down", "distance", "yardline", "yards", "baseline_distance"):
            if r.get(key) == "":
                r[key] = None
        r.setdefault("quality_flags", [])
        r.setdefault("conf", {})
        r.setdefault("field_sources", {})
    for i, r in enumerate(rows):
        if r.get("is_kickoff"):
            continue
        n = rows[i + 1] if i + 1 < len(rows) else None
        ev = _same_drive(r, n)
        d, dist = r.get("down"), r.get("distance")
        nd, ndist = (n.get("down"), n.get("distance")) if ev else (None, None)
        kind, y = r.get("result_kind") or "", r.get("yards")
        nulled = any(q.startswith(("result_nulled", "result_spot_mismatch")) for q in r.get("quality_flags") or [])
        # ---- (2) sack with positive yards
        if kind == "sack" and isinstance(y, (int, float)) and y > 0:
            st["sack_positive_yards"] += 1
            dd_loss = nd == (d or 0) + 1 and dist is not None and ndist is not None and abs(ndist - (dist + y)) <= 1
            dd_gain = nd == (d or 0) + 1 and dist is not None and ndist is not None and abs(ndist - (dist - y)) <= 1 and y > 1
            yl0, yl1 = r.get("yardline"), n.get("yardline") if ev else None
            sp_loss = yl0 is not None and yl1 is not None and abs((yl1 - yl0) + y) <= 1
            sp_gain = yl0 is not None and yl1 is not None and abs((yl1 - yl0) - y) <= 1 and y > 1
            if (dd_loss or sp_loss) and not (dd_gain or sp_gain):
                _set_result(r, "sack", -abs(y), f"-{abs(y)}", (r["field_sources"].get("result") or "dd") + "+sign_fixed_by_next_snap",
                            max(0.6, r["conf"].get("result", 0.0)), "sack_sign_corrected", st)
            else:
                before = {"result": r.get("result"), "result_kind": kind, "yards": y}
                r["yards"] = None; r["result_unverified"] = "sack_with_positive_yards_no_next_snap_agreement"
                r["result_fix"] = {"kind": "sack_yards_nulled", "before": before, "after": {"result_kind": "sack", "yards": None}}
                r["quality_flags"].append("result_fix:sack_yards_nulled" + (":next_snap_shows_gain" if (dd_gain or sp_gain) else ""))
                st["result_fix:sack_yards_nulled"] += 1
            continue
        # ---- (4) convert / gain vs distance consistency
        if kind in ("convert", "gain") and isinstance(y, (int, float)) and dist is not None:
            conflict = (kind == "convert" and y < dist) or (kind == "gain" and y >= dist and nd == (d or 0) + 1)
            if conflict:
                st["convert_conflicts"] += 1
                if nd == (d or 0) + 1 and ndist is not None:
                    yy = dist - ndist
                    _set_result(r, _gain_kind(yy), yy, _text(yy), "next_snap_dd", 0.6, f"{kind}_vs_distance:next_snap_down+1", st)
                elif nd == 1:
                    _set_result(r, "convert", y if y >= dist else None, "convert", "next_snap_first_down", 0.6,
                                f"{kind}_vs_distance:next_snap_first_down", st)
                else:
                    _set_result(r, "", None, "", "", 0.0, f"{kind}_vs_distance:no_next_snap_nulled", st)
                continue
        # ---- (5) recover null results from the next snap's down & distance
        # A missing goal distance uses the painted HUD distance. "" is not subtracted.
        dist_num = _play_distance(r)
        ndist_num = _play_distance(n) if n is not None else None
        if not kind and ev and not nulled and d in (1, 2, 3) and dist_num is not None and nd is not None:
            if nd == d + 1 and ndist_num is not None and abs(dist_num - ndist_num) <= 99:
                yy = dist_num - ndist_num
                if yy == 0 and not _plus0_has_evidence(r, n):
                    if _zero_contradicted(r, n):
                        flags = r.setdefault("quality_flags", [])
                        if "spot_contradicts_plus0" not in flags:
                            flags.append("spot_contradicts_plus0")
                        r["result_flag"] = "spot_contradicts_plus0"
                        st["spot_contradicts_plus0"] += 1
                    continue
                _set_result(r, _gain_kind(yy), yy, _text(yy), "next_snap_dd", 0.5, "recovered:next_snap_down+1", st)
                continue
            if nd == 1 and d == 1:
                own = (r.get("side_evidence") or 0) * (n.get("side_evidence") or 0) > 0 or \
                      (r.get("side") == n.get("side") and r.get("side") in ("offense", "defense")
                       and min(r["conf"].get("side", 0), n["conf"].get("side", 0)) >= 0.7)
                s0, s1 = r.get("field_spot"), n.get("field_spot")
                dl = _spot_deltas(s0, s1) if s0 is not None and s1 is not None else set()
                moved = bool(dl) and all(x >= dist_num - 1 for x in dl)
                goal = ndist is not None and ndist < 10 and n.get("yards_to_goal") is not None and n["yards_to_goal"] == ndist
                if own and (moved or goal):
                    uniq = sorted(x for x in dl if x >= dist - 1)
                    yy = uniq[0] if len(set(uniq)) == 1 else None
                    _set_result(r, "convert", yy, "convert", "next_snap_first_down", 0.5,
                                "recovered:next_snap_first_down" + (":goal" if goal and not moved else ""), st)
                    continue
    # ---- (3) Hail Mary formation gate
    for r in rows:
        for fld in ("formation", "call_screen_formation"):
            v = r.get(fld) or ""
            if _n(v) != "HAILMARY":
                continue
            q, cs, dist = r.get("quarter"), _clock_s(r.get("clock")), r.get("distance")
            late = q in (2, 4) and cs is not None and cs <= 40
            if late or (dist is not None and dist >= 25):
                continue
            r[fld] = ""; r["conf"][fld] = 0.0; r["field_sources"].pop(fld, None)
            r["quality_flags"].append(f"hail_mary_formation_dropped:{fld}:q{q} {r.get('clock') or '?'} {d_str(r)}")
            st[f"hail_mary_formation_dropped:{fld}"] += 1
    _withhold_contradicted_plus0(rows)
    return st


def _plus0_has_evidence(row: dict, nxt: dict | None) -> bool:
    try:
        import vodx_pr88
    except ImportError:
        vodx_pr88 = None
    if vodx_pr88 is not None and hasattr(vodx_pr88, "plus0_has_evidence"):
        return bool(vodx_pr88.plus0_has_evidence(row, nxt))
    return False


def _zero_contradicted(row: dict, nxt: dict | None) -> bool:
    """A distance +0 whose spots are not on the same line."""
    if not isinstance(nxt, dict):
        return False
    for key in ("yardline", "field_spot"):
        left, right = row.get(key), nxt.get(key)
        if isinstance(left, bool) or isinstance(right, bool):
            continue
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and abs(int(left) - int(right)) > 2:
            return True
    flags = row.get("quality_flags") or []
    return row.get("result_flag") == "spot_contradicts_plus0" or any(
        "spot_contradicts_plus0" in str(item) for item in flags
    )


def _withhold_contradicted_plus0(rows: list[dict]) -> None:
    try:
        import vodx_pr88
    except ImportError:
        vodx_pr88 = None
    if vodx_pr88 is not None and hasattr(vodx_pr88, "withhold_contradicted_plus0"):
        vodx_pr88.withhold_contradicted_plus0(rows)


def d_str(r):
    return f"{r.get('down')}&{r.get('distance')}"


def quality_metrics(labels: list[dict]) -> dict:
    """summary.json `quality`: the Analyst's checks, by game x opponent_type (and all)."""
    from collections import defaultdict
    g = defaultdict(list)
    for l in labels:
        g[f"{l.get('game')}|{l.get('opponent_type')}"].append(l); g["all|all"].append(l)
    out = {}
    for k, ls in sorted(g.items()):
        known = [l for l in ls if l.get("success_known")]
        by_down = {}
        for d in (1, 2, 3, 4):
            x = [l for l in known if l.get("down") == d]
            by_down[str(d)] = {"n": len(x), "success": round(sum(1 for l in x if l.get("success")) / len(x), 3) if x else None}
        hm = [l for l in ls if _n(l.get("formation")) == "HAILMARY" or _n(l.get("call_screen_formation")) == "HAILMARY"]
        out[k] = {
            "known_results": len(known),
            "plus0": sum(1 for l in known if l.get("result") == "+0"),
            "plus0_share": round(sum(1 for l in known if l.get("result") == "+0") / len(known), 3) if known else None,
            "success_by_down": by_down,
            "clean_pair": sum(1 for l in ls if l.get("clean_pair")),
            "call_pair": sum(1 for l in ls if l.get("call_pair", l.get("clean_pair"))),
            "clean_pair_success_null": sum(1 for l in ls if l.get("clean_pair") and not l.get("success_known")),
            "sacks": sum(1 for l in ls if l.get("result_kind") == "sack"),
            "sack_positive_yards": sum(1 for l in ls if l.get("result_kind") == "sack" and (l.get("yards") or 0) > 0),
            "sack_unverified": sum(1 for l in ls if l.get("result_kind") == "sack" and l.get("result_unverified")),
            "convert_short_of_distance": sum(1 for l in ls if l.get("result_kind") == "convert" and isinstance(l.get("yards"), (int, float))
                                             and l.get("distance") is not None and l["yards"] < l["distance"]),
            "hail_mary_formation_rows": len(hm),
            "hail_mary_formation_implausible": sum(1 for l in hm if not ((l.get("quarter") in (2, 4) and _cs(l.get("clock")) is not None
                                                                         and _cs(l.get("clock")) <= 40) or (l.get("distance") or 0) >= 25)),
            "result_fix": dict(Counter(((l.get("result_fix") or {}).get("kind") or "").split(":")[0] for l in ls if l.get("result_fix"))),
            "opponent_type_source": dict(Counter(l.get("opponent_type_source") for l in ls)),
        }
    return out


def _cs(c):
    m = re.match(r"(\d{1,2}):(\d{2})", c or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


apply_quality._blank_safe = True
