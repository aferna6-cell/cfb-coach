"""Where could the offense get smarter? Descriptive, read-only, no live change.

Uses only executed offensive plays the dataset marks supervised-eligible
(verified or attested), split by opponent kind. For each area it reports rates
with Wilson intervals so small samples read as small:

* short yardage (3rd/4th and 1-2): conversion by run vs pass
* coverage: success by the coverage seen and our play family
* explosive vs turnover by play family
* game state: success when leading / tied / trailing; red-zone touchdowns
* repetition: success when the exact play was called in the previous 3 snaps

    python scripts/madden_ml/offense_opportunity_audit.py --db AUDIT_COPY.db --out DIR
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfb_coach.madden.defense_select import coverage_seen_family  # noqa: E402
from cfb_coach.madden.model import heldout_eval as he  # noqa: E402


def _scores(db: str) -> dict[str, tuple[Any, Any]]:
    conn = sqlite3.connect(f"file:{Path(db).expanduser().resolve()}?mode=ro", uri=True)
    try:
        return {str(r[0]): (r[1], r[2]) for r in conn.execute(
            "SELECT ml_snap_id, score_for, score_against FROM snaps WHERE ml_snap_id IS NOT NULL")}
    finally:
        conn.close()


def _rate(pairs: list[int]) -> dict[str, Any]:
    w = he.wilson(sum(pairs), len(pairs))
    return {"n": w["n"], "rate": None if w["rate"] is None else round(w["rate"], 3),
            "ci": None if w["lo"] is None else [round(w["lo"], 3), round(w["hi"], 3)]}


def audit(rows: list[dict[str, Any]], scores: dict[str, tuple[Any, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("human", "cpu"):
        rs = [r for r in rows if he.is_verified_offense(r) and str(r.get("opponent_type") or "") == kind]
        rs.sort(key=lambda r: (str(r.get("game_id") or ""), str(r.get("snap_id") or "")))
        area: dict[str, Any] = {"executed_plays": len(rs)}
        groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        last: dict[str, list[str]] = defaultdict(list)
        for r in rs:
            fam = he.run_or_pass(r.get("action_play"))
            succ = he.offense_label(r, "success")
            conv = he.offense_label(r, "conversion")
            expl = he.offense_label(r, "explosive")
            tov = he.offense_label(r, "turnover")
            down, dist = r.get("down"), r.get("distance")
            if conv is not None and dist is not None and int(dist) <= 2:
                groups["short_yardage_conversion"][fam].append(conv)
            if conv is not None and dist is not None and int(dist) >= 7:
                groups["third_long_conversion"][fam].append(conv)
            cov = coverage_seen_family(r.get("coverage_seen"))
            if succ is not None and cov:
                groups["success_by_coverage_seen"][f"{cov}/{fam}"].append(succ)
            if expl is not None:
                groups["explosive_by_family"][fam].append(expl)
            if tov is not None:
                groups["turnover_by_family"][fam].append(tov)
            sc = scores.get(str(r.get("snap_id") or ""))
            if succ is not None and sc and sc[0] is not None and sc[1] is not None:
                d = int(sc[0]) - int(sc[1])
                state = "leading_9+" if d >= 9 else "leading_1-8" if d > 0 else "tied" if d == 0 else \
                    "trailing_1-8" if d > -9 else "trailing_9+"
                groups["success_by_score_state"][state].append(succ)
            yl = r.get("yardline")
            if yl is not None and int(yl) >= 80 and r.get("result") not in (None, ""):
                groups["red_zone_td"][fam].append(int(str(r.get("result")).lower().startswith("td")))
            g = str(r.get("game_id") or "")
            play = str(r.get("action_play") or "")
            if succ is not None and play:
                groups["success_after_repeat"]["repeat_in_last_3" if play in last[g][-3:] else "fresh"].append(succ)
            last[g].append(play)
            _ = down
        for name, by in groups.items():
            area[name] = {k: _rate(v) for k, v in sorted(by.items())}
        plays = [str(r.get("action_play") or "") for r in rs]
        area["distinct_plays"] = len(set(plays))
        area["top_play_share"] = round(max((plays.count(p) for p in set(plays)), default=0) / max(1, len(plays)), 3)
        out[kind] = area
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data = he.load_eval_data(args.db)
    report = audit(data.rows, _scores(args.db))
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "offense_opportunity_audit.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
