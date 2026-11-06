#!/usr/bin/env python3
"""Repeatable play-art annotation workflow for the 93 installed offensive plays.

  template  -> CSV with one row per (play, slot) = 1023 rows, pre-filled with what is already known
  manifest  -> which plays have a screenshot/recording file for you to read (names: "<formation>__<play>.png")
  import    -> validate your filled CSV and write the overlay JSON (never edits the base database)
  audit     -> coverage by tier/position, remaining gaps, disputes (optionally writes the markdown audit)

Evidence rules (enforced on import):
  * leave `tier` EMPTY (or `unknown`) for any slot you cannot read from the art -- never guess;
  * `diagram_verified` / `in_game_verified` need `source` = user_screenshot:<file> | user_recording:<file> | user_ingame:<note>;
  * `source_documented` needs a source URL; `football_inference` needs a source/basis note;
  * `category`: route|block|run|option|play_action|decoy|motion|pass_block ; for `route`, `task` is a taxonomy family.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import offense_assignments as oa  # noqa: E402
from cfb_coach.madden.model import offense_taxonomy as tx  # noqa: E402

COLS = ["formation", "play", "slot", "current_tier", "current_task", "category", "task", "side", "depth",
        "published_route", "tier", "source", "motion", "conditional", "notes"]


def _fname(formation: str, play: str) -> str:
    return f"{formation}__{play}".replace("/", "-").replace(" ", "_")


def cmd_template(a: argparse.Namespace) -> int:
    db = oa.load_effective()
    out = Path(a.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for p in db["plays"]:
            for slot, s in p["slots"].items():
                w.writerow({"formation": p["formation"], "play": p["play"], "slot": slot,
                            "current_tier": s["tier"], "current_task": s["task"] if s["tier"] != "unknown" else ""})
    print(f"wrote {out} ({sum(len(p['slots']) for p in db['plays'])} rows). Fill category/task/side/depth/tier/source "
          "only for slots you can read from the art.")
    return 0


def cmd_manifest(a: argparse.Namespace) -> int:
    folder = Path(a.images)
    have = {p.stem for p in folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".mp4", ".mov", ".webp")}
    db = oa.load_effective()
    rows = [(p["formation"], p["play"], _fname(p["formation"], p["play"]) in have) for p in db["plays"]]
    missing = [r for r in rows if not r[2]]
    print(f"{len(rows) - len(missing)}/{len(rows)} plays have art in {folder}")
    for f, pl, _ in missing[: a.show]:
        print(f"  missing: {_fname(f, pl)}.png")
    return 0


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def cmd_import(a: argparse.Namespace) -> int:
    db = oa.load_database()
    idx = oa.index(db)
    by_play: dict[tuple[str, str], dict] = {}
    problems: list[str] = []
    for n, r in enumerate(_rows(Path(a.csv)), start=2):
        tier = (r.get("tier") or "").strip()
        if tier in ("", "unknown"):
            continue
        key = (r["formation"], r["play"])
        if key not in idx:
            problems.append(f"row {n}: {key} is not an installed play")
            continue
        if r["slot"] not in idx[key]["slots"]:
            problems.append(f"row {n}: slot {r['slot']!r} not in {key[0]}")
            continue
        cat, task = (r.get("category") or "").strip(), (r.get("task") or "").strip()
        if cat not in oa.SLOT_CATEGORIES or cat == "unknown" or not task:
            problems.append(f"row {n}: needs a valid category and task")
            continue
        if cat == "route" and task not in tx.ROUTE_FAMILIES:
            problems.append(f"row {n}: route {task!r} not in taxonomy (see offense_taxonomy.ROUTE_FAMILIES)")
            continue
        if tier not in tx.EVIDENCE_TIERS:
            problems.append(f"row {n}: bad tier {tier!r}")
            continue
        params = {k: v for k, v in (("side", r.get("side")), ("depth", r.get("depth")),
                                    ("published_route", r.get("published_route"))) if (v or "").strip()}
        by_play.setdefault(key, {"formation": key[0], "play": key[1], "slots": {}})["slots"][r["slot"]] = {
            "category": cat, "task": task, "params": params, "tier": tier, "source": (r.get("source") or "").strip(),
            "pre_snap_motion": (r.get("motion") or None), "conditional": (r.get("conditional") or None)}
    overlay = {"schema": oa.OVERLAY_SCHEMA, "annotations": list(by_play.values())}
    try:
        merged, log = oa.merge_overlay(db, overlay)       # full validation incl. verified-source rule
    except oa.AssignmentError as exc:
        problems.append(str(exc))
        merged, log = None, []
    if problems:
        print("IMPORT REJECTED:\n- " + "\n- ".join(problems))
        return 1
    Path(a.out).write_text(json.dumps(overlay, indent=1, sort_keys=True), encoding="utf-8")
    print(f"overlay -> {a.out}: {sum(len(v['slots']) for v in by_play.values())} slot annotations on {len(by_play)} plays")
    print("\n".join(log[:40]))
    s = oa.coverage_stats(merged)
    print(f"after merge: fully verified {s['fully_verified_plays']}, partial {s['partial_plays']}, unknown {s['unknown_plays']}")
    return 0


def cmd_audit(a: argparse.Namespace) -> int:
    db = oa.load_effective()
    s, g = oa.coverage_stats(db), oa.audit_gaps(db)
    lines = ["# Offensive assignment audit (generated)", "",
             f"Inventory `{s['fingerprint']}`, {s['plays']} plays x 11 slots = {s['slot_records']} slot records.", "",
             "## Counts", "",
             f"- fully verified plays (all 11 slots diagram/in-game verified): **{s['fully_verified_plays']}**",
             f"- partial plays (some sourced slots): **{s['partial_plays']}**",
             f"- unknown plays (no sourced slot): **{s['unknown_plays']}**",
             f"- plays with a sourced route combination (>=2 known routes): **{s['plays_with_sourced_route_combination']}**",
             f"- plays with a name-level type hint (football_inference only): {s['plays_with_name_level_type_hint']}",
             f"- slot records by tier: {s['slots_by_tier']}", "", "## Tier by position", ""]
    for k, v in s["tier_by_position"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Remaining gaps (exact)", "",
              f"- OL slots unknown: {g['unknown_ol_slots']}/{g['ol_slots_total']}",
              f"- QB slots unknown: {g['unknown_qb_slots']}/{s['plays']}",
              f"- skill slots unknown: {g['unknown_skill_slots']}/{g['skill_slots_total']}",
              f"- formations whose skill slots are unnamed: {g['formations_with_unnamed_skill_slots']}",
              f"- disputes: {len(g['disputes'])}", "", "### Plays with no sourced slot", ""]
    lines += [f"- {x}" for x in g["plays_without_any_sourced_slot"]]
    text = "\n".join(lines) + "\n"
    if a.write:
        Path(a.write).write_text(text, encoding="utf-8")
    print(text if not a.write else f"wrote {a.write}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("template"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_template)
    p = sub.add_parser("manifest"); p.add_argument("--images", required=True); p.add_argument("--show", type=int, default=15)
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("import"); p.add_argument("--csv", required=True)
    p.add_argument("--out", default=str(oa.OVERLAY_PATH)); p.set_defaults(fn=cmd_import)
    p = sub.add_parser("audit"); p.add_argument("--write"); p.set_defaults(fn=cmd_audit)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
