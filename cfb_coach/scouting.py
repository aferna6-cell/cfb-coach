"""Opponent scouting from Aidan's own logged snaps (every game vs that opponent).

When we're on defense the log says what THEY ran (``concept_seen``) and how it went;
when we're on offense it says what coverage THEY played (``coverage_seen``) and which
of our plays worked. Grouped by down-and-distance bucket, this is the per-opponent
film study that drives the prep page and nudges live calls (defense picks coverage
families that have answered their favorite concepts in this kind of down).
"""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from typing import Any, Callable

from cfb_coach.outcome import outcome_success, parse_outcome

MIN_BUCKET_SNAPS = 4  # below this a bucket borrows the opponent's overall tendencies
MIN_PLAY_SNAPS = 3

BUCKET_LABELS = {
    "1st": "1st down",
    "2nd_short": "2nd & short (1-3)",
    "2nd_long": "2nd & 4+",
    "3rd_short": "3rd/4th & short (1-3)",
    "3rd_medium": "3rd/4th & medium (4-6)",
    "3rd_long": "3rd/4th & long (7+)",
    "red_zone": "Red zone",
}


def down_bucket(down: int | None, distance: int | None, yardline: int | None = None) -> str:
    if yardline is not None and yardline >= 80:
        return "red_zone"
    d = down or 1
    dist = distance if distance is not None else 10
    if d == 1:
        return "1st"
    if d == 2:
        return "2nd_short" if dist <= 3 else "2nd_long"
    if dist <= 3:
        return "3rd_short"
    return "3rd_medium" if dist <= 6 else "3rd_long"


def _row(s: Any, key: str) -> Any:
    try:
        return s[key]
    except (KeyError, IndexError, TypeError):
        return getattr(s, key, None)


def _yards(result: str | None) -> int | None:
    p = parse_outcome(result)
    return p.yards if p.yards is not None else None


def _rate(wins: int, n: int) -> float:
    return round(wins / n, 2) if n else 0.0


def scout_opponent(
    db: Any,
    opponent_id: str,
    *,
    family_of: Callable[[str | None], str | None] | None = None,
    coverage_class_of: Callable[[str | None], str | None] | None = None,
    limit: int = 3000,
) -> dict[str, Any]:
    """Full scouting report for one opponent (all logged games)."""
    try:
        snaps = list(db.get_recent_snaps(opponent_id, limit=limit)) if db is not None else []
    except Exception:  # noqa: BLE001
        snaps = []
    games = {(_row(s, "session_id") or "") for s in snaps}
    rep: dict[str, Any] = {
        "opponent_id": opponent_id,
        "snaps": len(snaps),
        "games": len({g for g in games if g}),
        "their_offense": {"n": 0, "buckets": {}, "overall": {}, "our_calls": []},
        "their_defense": {"n": 0, "buckets": {}, "overall": {}, "our_plays": []},
    }

    off_by_bucket: dict[str, list[Any]] = defaultdict(list)
    def_by_bucket: dict[str, list[Any]] = defaultdict(list)
    for s in snaps:
        b = down_bucket(_row(s, "down"), _row(s, "distance"), _row(s, "yardline"))
        if (_row(s, "side") or "offense") == "defense":
            off_by_bucket[b].append(s)
        else:
            def_by_bucket[b].append(s)

    def offense_summary(rows: list[Any]) -> dict[str, Any]:
        concepts: Counter[str] = Counter()
        fams: Counter[str] = Counter()
        wins: Counter[str] = Counter()
        yds: list[int] = []
        for s in rows:
            c = (_row(s, "concept_seen") or "").strip()
            ok = outcome_success(_row(s, "result"), "offense")
            y = _yards(_row(s, "result"))
            if y is not None:
                yds.append(y)
            if c:
                concepts[c] += 1
                if ok:
                    wins[c] += 1
                f = family_of(c) if family_of else None
                if f:
                    fams[f] += 1
        n = len(rows)
        return {
            "n": n,
            "concepts": [{"concept": c, "n": k, "share": _rate(k, n), "their_success": _rate(wins[c], k)}
                         for c, k in concepts.most_common(6)],
            "families": {f: _rate(k, sum(fams.values())) for f, k in fams.most_common()},
            "ypp": round(sum(yds) / len(yds), 1) if yds else None,
            "their_success": _rate(sum(1 for s in rows if outcome_success(_row(s, "result"), "offense")), n),
        }

    def defense_summary(rows: list[Any]) -> dict[str, Any]:
        covs: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        for s in rows:
            c = (_row(s, "coverage_seen") or "").strip()
            if c:
                covs[c] += 1
                cls = coverage_class_of(c) if coverage_class_of else None
                if cls:
                    classes[cls] += 1
        seen = sum(covs.values())
        return {
            "n": len(rows),
            "coverages": [{"coverage": c, "n": k, "share": _rate(k, seen)} for c, k in covs.most_common(6)],
            "classes": {c: _rate(k, sum(classes.values())) for c, k in classes.most_common()},
            "pressure_rate": _rate(sum(k for c, k in covs.items() if "pressure" in c.lower() or "blitz" in c.lower()), seen),
        }

    all_off = [s for rows in off_by_bucket.values() for s in rows]
    all_def = [s for rows in def_by_bucket.values() for s in rows]
    rep["their_offense"]["n"] = len(all_off)
    rep["their_offense"]["overall"] = offense_summary(all_off)
    rep["their_offense"]["buckets"] = {b: offense_summary(r) for b, r in off_by_bucket.items()}
    rep["their_defense"]["n"] = len(all_def)
    rep["their_defense"]["overall"] = defense_summary(all_def)
    rep["their_defense"]["buckets"] = {b: defense_summary(r) for b, r in def_by_bucket.items()}

    # Which of OUR calls held up on defense / worked on offense vs this opponent
    def call_table(rows: list[Any], side: str) -> list[dict[str, Any]]:
        agg: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for s in rows:
            agg[(_row(s, "formation") or "?", _row(s, "play") or "?")].append(s)
        out = []
        for (f, p), rs in agg.items():
            if len(rs) < MIN_PLAY_SNAPS:
                continue
            ok = sum(1 for s in rs if outcome_success(_row(s, "result"), side))
            ys = [y for y in (_yards(_row(s, "result")) for s in rs) if y is not None]
            out.append({"formation": f, "play": p, "n": len(rs), "success": _rate(ok, len(rs)),
                        "ypp": round(sum(ys) / len(ys), 1) if ys else None})
        out.sort(key=lambda r: (-r["success"], -r["n"]))
        return out

    rep["their_offense"]["our_calls"] = call_table(all_off, "defense")
    rep["their_defense"]["our_plays"] = call_table(all_def, "offense")
    return rep


def _bucket_or_overall(section: dict[str, Any], bucket: str) -> tuple[dict[str, Any], str]:
    b = (section.get("buckets") or {}).get(bucket) or {}
    if b.get("n", 0) >= MIN_BUCKET_SNAPS:
        return b, BUCKET_LABELS.get(bucket, bucket)
    return section.get("overall") or {}, "all downs"


def expected_families(report: dict[str, Any] | None, sit: Any) -> tuple[dict[str, float], str, int]:
    """Their offensive concept-family mix for this down bucket: ({family: share}, scope, n)."""
    if not report:
        return {}, "", 0
    bucket = down_bucket(getattr(sit, "down", None), getattr(sit, "distance", None), getattr(sit, "yardline", None))
    data, scope = _bucket_or_overall(report.get("their_offense") or {}, bucket)
    return dict(data.get("families") or {}), scope, int(data.get("n") or 0)


def expected_coverage(report: dict[str, Any] | None, sit: Any) -> tuple[dict[str, float], str, int]:
    """Their coverage-class mix vs our offense for this down bucket: ({class: share}, scope, n)."""
    if not report:
        return {}, "", 0
    bucket = down_bucket(getattr(sit, "down", None), getattr(sit, "distance", None), getattr(sit, "yardline", None))
    data, scope = _bucket_or_overall(report.get("their_defense") or {}, bucket)
    return dict(data.get("classes") or {}), scope, int(data.get("n") or 0)


def _pct(x: float | None) -> str:
    return f"{round((x or 0) * 100)}%"


def scouting_lines(report: dict[str, Any]) -> list[str]:
    """Plain-text scouting report for the prep sheet."""
    oid = report.get("opponent_id", "")
    if not report.get("snaps"):
        return [f"No logged snaps vs {oid} yet — scouting fills in after your first game."]
    lines = [f"From your logs: {report['snaps']} snaps over {report['games']} game(s) vs {oid}."]
    to = report.get("their_offense") or {}
    if to.get("n"):
        ov = to["overall"]
        lines.append(f"THEIR OFFENSE ({to['n']} snaps, {ov.get('ypp')} yds/play, {_pct(ov.get('their_success'))} successful):")
        for b in BUCKET_LABELS:
            d = (to.get("buckets") or {}).get(b)
            if not d or not d.get("concepts"):
                continue
            top = ", ".join(f"{c['concept']} {_pct(c['share'])}" for c in d["concepts"][:3])
            lines.append(f"  {BUCKET_LABELS[b]} (n={d['n']}): {top}")
        good = [c for c in to.get("our_calls") or [] if c["success"] >= 0.5][:3]
        bad = [c for c in reversed(to.get("our_calls") or []) if c["success"] < 0.4][:3]
        if good:
            lines.append("  Our D calls that held: " + "; ".join(f"{c['formation']} {c['play']} ({_pct(c['success'])} stops, n={c['n']})" for c in good))
        if bad:
            lines.append("  Our D calls they beat: " + "; ".join(f"{c['formation']} {c['play']} ({_pct(c['success'])} stops, n={c['n']})" for c in bad))
    td = report.get("their_defense") or {}
    if td.get("n"):
        ov = td["overall"]
        covs = ", ".join(f"{c['coverage']} {_pct(c['share'])}" for c in (ov.get("coverages") or [])[:4])
        lines.append(f"THEIR DEFENSE ({td['n']} snaps, pressure {_pct(ov.get('pressure_rate'))}): {covs}")
        for b in ("3rd_medium", "3rd_long", "red_zone"):
            d = (td.get("buckets") or {}).get(b)
            if d and d.get("coverages"):
                lines.append(f"  {BUCKET_LABELS[b]} (n={d['n']}): " + ", ".join(f"{c['coverage']} {_pct(c['share'])}" for c in d["coverages"][:3]))
        best = [p for p in td.get("our_plays") or [] if p["success"] >= 0.5][:4]
        if best:
            lines.append("  Our plays that worked: " + "; ".join(f"{p['formation']} {p['play']} ({_pct(p['success'])}, n={p['n']})" for p in best))
    return lines


def _lines_html(lines: list[str]) -> str:
    return "".join(
        f"<div style='margin:.15rem 0;{'padding-left:1rem;color:#8b949e' if ln.startswith('  ') else ''}'>{html.escape(ln.strip())}</div>"
        for ln in lines
    )


def opponent_study_html(scouting: list[str], research: list[str]) -> str:
    """Prep-page section: scouting from your logs + the daily research's counters for him."""
    if not scouting and not research:
        return ""
    body = f"<h3>From your logs</h3>{_lines_html(scouting)}" if scouting else ""
    if research:
        body += f"<h3>Daily research — counters for him</h3>{_lines_html(research)}"
    return f"<section id='scouting'><h2>Scouting report</h2>{body}</section>"


__all__ = [
    "BUCKET_LABELS",
    "down_bucket",
    "expected_coverage",
    "expected_families",
    "opponent_study_html",
    "scout_opponent",
    "scouting_lines",
]
