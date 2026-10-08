"""Madden ML coach CLI. Opt-in; does not change heuristic play-calling defaults."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfb_coach.db import CoachDB
from cfb_coach.games import data_dir, madden_db_path
from cfb_coach.madden.data import load_seed
from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model import evaluate as evaluate_mod
from cfb_coach.madden.model import inference as inference_mod
from cfb_coach.madden.model import registry as registry_mod
from cfb_coach.madden.model import train as train_mod
from cfb_coach.madden.model.schema import CoachingMode, MLStatus, to_dict


def open_madden_db() -> CoachDB:
    return CoachDB(madden_db_path(), seed=load_seed())


def default_registry() -> Path:
    return Path(data_dir()) / "madden_ml_registry"


def cmd_ml_find_db(args: argparse.Namespace) -> int:
    """Print the active Madden database path from application config."""
    del args
    path = madden_db_path()
    print(f"madden_db: {path}")
    print(f"exists: {path.is_file()}")
    print(f"data_dir: {data_dir()}")
    env = __import__("os").environ.get("CFB_COACH_MADDEN_DB")
    print(f"CFB_COACH_MADDEN_DB: {env or '(unset — using default)'}")
    if path.is_file():
        print(f"size_bytes: {path.stat().st_size}")
    return 0


def cmd_ml_backup(args: argparse.Namespace) -> int:
    """Copy the active Madden DB to a timestamped backup. Never modifies the source."""
    src = madden_db_path()
    if not src.is_file():
        print(f"no database at {src}", file=sys.stderr)
        return 1
    dest_dir = Path(args.out or (src.parent / "backups"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"{src.stem}.backup.{stamp}{src.suffix}"
    shutil.copy2(src, dest)
    print(f"source: {src}")
    print(f"backup: {dest}")
    print(f"size_bytes: {dest.stat().st_size}")
    return 0


def cmd_ml_inspect(args: argparse.Namespace) -> int:
    db = open_madden_db()
    try:
        paths = list(args.path or [])
        rows = dataset_mod.build_rows(db=db, extra_paths=paths)
        report = dataset_mod.quality_report(rows)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"db: {db.path}")
        print(
            f"rows: {report['total_snaps']}  games: {report['unique_games']}  "
            f"verified_exec: {report['verified_executions']}  "
            f"supervised: {report['supervised_training_rows']}  "
            f"outcome_only: {report['outcome_only_examples']}"
        )
    finally:
        db.close()
    return 0


def cmd_ml_export(args: argparse.Namespace) -> int:
    db = open_madden_db()
    try:
        paths = list(args.path or [])
        rows = dataset_mod.build_rows(db=db, extra_paths=paths)
        if getattr(args, "supervised_only", False):
            rows = dataset_mod.supervised_rows(rows)
        if getattr(args, "sanitized", False):
            dataset_mod.export_sanitized(rows, args.out)
        else:
            dataset_mod.export_jsonl(rows, args.out)
        report = dataset_mod.quality_report(rows)
        print(f"wrote {len(rows)} rows -> {args.out}")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "total_snaps",
                        "unique_games",
                        "verified_executions",
                        "supervised_training_rows",
                        "outcome_only_examples",
                        "duplicates",
                        "eligibility",
                    )
                },
                indent=2,
            )
        )
    finally:
        db.close()
    return 0


def cmd_ml_train(args: argparse.Namespace) -> int:
    rows = _load_rows(args)
    reg = Path(args.registry or default_registry())
    out_dir = Path(args.out or (reg / "_train_scratch"))
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = train_mod.train(rows, seed=int(args.seed), out_dir=str(out_dir))
    # Keep the artifact beside the registry entry for reload/sha checks.
    if entry.artifact_path:
        version_dir = reg / (entry.model_version or "anonymous")
        version_dir.mkdir(parents=True, exist_ok=True)
        target_art = version_dir / registry_mod.ARTIFACT_FILENAME
        src = Path(entry.artifact_path)
        if src.resolve() != target_art.resolve():
            target_art.write_bytes(src.read_bytes())
        from dataclasses import replace

        entry = replace(entry, artifact_path=str(target_art))
    path = registry_mod.write_entry(entry, str(reg))
    print(f"model_version: {entry.model_version}")
    print(f"gate: {entry.gate.value}")
    print(f"artifact: {entry.artifact_path}")
    print(f"registry_entry: {path}")
    return 0


def cmd_ml_evaluate(args: argparse.Namespace) -> int:
    rows = _load_rows(args)
    reg = Path(args.registry or default_registry())
    version = args.model
    entry = registry_mod.read_entry(str(reg / version) if version else str(reg))
    updated = evaluate_mod.evaluate(entry, rows, seed=int(args.seed))
    path = registry_mod.write_entry(updated, str(reg))
    print(f"model_version: {updated.model_version}")
    print(f"gate: {updated.gate.value}  gate_passed: {updated.gate_passed}")
    print(f"metrics: {json.dumps([to_dict(m) for m in updated.metrics], indent=2)}")
    if updated.gate_report_path:
        print(f"report: {updated.gate_report_path}")
    print(f"registry_entry: {path}")
    return 0


def cmd_ml_list(args: argparse.Namespace) -> int:
    reg = Path(args.registry or default_registry())
    entries = registry_mod.list_entries(str(reg))
    if not entries:
        print(f"no models under {reg}")
        return 0
    current = None
    cur_file = reg / registry_mod.CURRENT_LINK
    if cur_file.is_file():
        current = cur_file.read_text(encoding="utf-8").strip()
    for entry in entries:
        mark = "*" if entry.model_version == current else " "
        print(
            f"{mark} {entry.model_version}  gate={entry.gate.value}  "
            f"passed={entry.gate_passed}  side={entry.side.value}"
        )
    return 0


def cmd_ml_status(args: argparse.Namespace) -> int:
    db = open_madden_db()
    try:
        mode = inference_mod.resolve_mode(db)
        version = inference_mod.resolve_model_version(db)
        reg = inference_mod.resolve_registry_dir(db)
        print(f"mode: {mode.value}")
        print(f"model_version: {version or '(none)'}")
        print(f"registry_dir: {reg}")
        print(f"hybrid_promotion_ready: False  # hybrid stays inactive this sprint")
        if version and reg:
            try:
                entry = registry_mod.read_entry(str(Path(reg) / version))
                print(
                    f"gate: {entry.gate.value}  "
                    f"promotion_allowed: {registry_mod.promotion_allowed(entry)}"
                )
            except (OSError, ValueError, FileNotFoundError) as exc:
                print(f"registry_error: {exc}")
        decisions = db.list_ml_decisions(limit=5)
        print(f"recent_ml_decisions: {len(decisions)}")
        for row in decisions:
            print(
                f"  id={row['id']} mode={row['mode']} shadow={row['shadow_status']} "
                f"agree={row['agree']} heur={row['heuristic_play']} "
                f"shadow_play={row['shadow_play']}"
            )
    finally:
        db.close()
    return 0


def cmd_ml_shadow(args: argparse.Namespace) -> int:
    db = open_madden_db()
    try:
        if args.off:
            inference_mod.set_mode(db, CoachingMode.HEURISTIC)
            print("mode: heuristic")
            return 0
        reg = Path(args.registry or default_registry())
        inference_mod.set_registry_dir(db, str(reg))
        if args.model:
            inference_mod.set_model_version(db, args.model)
        elif not inference_mod.resolve_model_version(db):
            current = reg / registry_mod.CURRENT_LINK
            if current.is_file():
                inference_mod.set_model_version(
                    db, current.read_text(encoding="utf-8").strip()
                )
        inference_mod.set_mode(db, CoachingMode.SHADOW)
        print("mode: shadow")
        print(f"registry_dir: {reg}")
        print(f"model_version: {inference_mod.resolve_model_version(db) or '(CURRENT)'}")
        print("Live calls still show the heuristic/VOD recommendation.")
    finally:
        db.close()
    return 0


def cmd_ml_heuristic(args: argparse.Namespace) -> int:
    del args
    db = open_madden_db()
    try:
        inference_mod.set_mode(db, CoachingMode.HEURISTIC)
        print("mode: heuristic")
    finally:
        db.close()
    return 0


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def cmd_ml_report(args: argparse.Namespace) -> int:
    """Per-game shadow report with data-quality and counterfactual caveats."""
    db = open_madden_db()
    try:
        limit = int(args.limit or 500)
        decisions = list(
            db.conn.execute(
                "SELECT * FROM ml_decisions ORDER BY id ASC LIMIT ?",
                (limit,),
            )
        )
        outcomes: dict[str, Any] = {}
        try:
            for row in db.conn.execute("SELECT * FROM ml_outcomes"):
                outcomes[str(row["snap_id"])] = row
        except Exception:  # noqa: BLE001
            pass
        snaps_by_ml: dict[str, Any] = {}
        try:
            for row in db.conn.execute(
                "SELECT * FROM snaps WHERE ml_snap_id IS NOT NULL"
            ):
                snaps_by_ml[str(row["ml_snap_id"])] = row
        except Exception:  # noqa: BLE001
            pass

        by_game: dict[str, list[Any]] = defaultdict(list)
        for row in decisions:
            gid = str(row["game_id"] or row["session_id"] or "unknown")
            by_game[gid].append(row)

        games_out: list[dict[str, Any]] = []
        all_latencies: list[float] = []
        for gid, rows in by_game.items():
            ok = [r for r in rows if r["shadow_status"] == MLStatus.OK.value]
            failed = [
                r
                for r in rows
                if r["shadow_status"] and r["shadow_status"] != MLStatus.OK.value
            ]
            agree = [r for r in ok if r["agree"] == 1]
            disagree = [r for r in ok if r["agree"] == 0]
            lats = [
                float(r["latency_ms_model"])
                for r in ok
                if r["latency_ms_model"] is not None
            ]
            all_latencies.extend(lats)
            verified = 0
            labeled = 0
            offense = 0
            defense = 0
            executed_outcomes: list[dict[str, Any]] = []
            for r in rows:
                sid = str(r["snap_id"] or "")
                outc = outcomes.get(sid)
                snap = snaps_by_ml.get(sid)
                side = None
                if snap is not None:
                    side = snap["side"] if hasattr(snap, "keys") else None
                if side and str(side).startswith("d"):
                    defense += 1
                else:
                    offense += 1
                exec_status = None
                if outc is not None:
                    exec_status = outc["executed_status"]
                    if exec_status == "identified" and outc["executed_verification"] == "verified":
                        verified += 1
                    payload = outc["outcome_json"]
                    result = None
                    if payload:
                        try:
                            result = json.loads(payload).get("result")
                        except (TypeError, json.JSONDecodeError):
                            result = None
                    if result:
                        labeled += 1
                    if exec_status == "identified":
                        executed_outcomes.append(
                            {
                                "snap_id": sid,
                                "executed_play": outc["executed_play"],
                                "result": result,
                                "note": "observed on executed play only",
                            }
                        )
                elif snap is not None and snap["result"]:
                    labeled += 1
            models = sorted({r["model_version"] for r in rows if r["model_version"]})
            games_out.append(
                {
                    "game_id": gid,
                    "n_calls": len(rows),
                    "shadow_scored_ok": len(ok),
                    "shadow_failed_or_timeout": len(failed),
                    "agree": len(agree),
                    "disagree": len(disagree),
                    "verified_execution_rate": round(verified / len(rows), 3) if rows else 0.0,
                    "valid_outcome_label_rate": round(labeled / len(rows), 3) if rows else 0.0,
                    "offense_calls": offense,
                    "defense_calls": defense,
                    "model_versions": models,
                    "latency_ms": {
                        "p50": _percentile(lats, 0.50),
                        "p90": _percentile(lats, 0.90),
                        "p99": _percentile(lats, 0.99),
                        "max": max(lats) if lats else None,
                    },
                    "executed_outcomes": executed_outcomes[:20],
                    "disagreements": [
                        {
                            "snap_id": r["snap_id"],
                            "heuristic": f"{r['heuristic_formation']}/{r['heuristic_play']}",
                            "shadow": f"{r['shadow_formation']}/{r['shadow_play']}",
                            "counterfactual_result": "unknown",
                            "note": (
                                "Shadow alternative was not executed; "
                                "do not claim yardage advantage."
                            ),
                        }
                        for r in disagree[:20]
                    ],
                }
            )

        # Dataset quality trend (all rows, not inventing data).
        rows = dataset_mod.build_rows(db=db)
        quality = dataset_mod.quality_report(rows)
        report = {
            "n_decisions": len(decisions),
            "n_games": len(by_game),
            "latency_ms_overall": {
                "p50": _percentile(all_latencies, 0.50),
                "p90": _percentile(all_latencies, 0.90),
                "p99": _percentile(all_latencies, 0.99),
                "max": max(all_latencies) if all_latencies else None,
                "budget_ms": 150,
            },
            "data_quality": {
                "total_snaps": quality["total_snaps"],
                "unique_games": quality["unique_games"],
                "verified_executions": quality["verified_executions"],
                "supervised_training_rows": quality["supervised_training_rows"],
                "outcome_only_examples": quality["outcome_only_examples"],
                "unknown_executions": quality["unknown_executions"],
                "cpu_vs_human": quality["cpu_vs_human"],
                "eligibility": quality["eligibility"],
            },
            "games": games_out,
            "caveat": (
                "Counterfactual results for unchosen shadow plays are unknown. "
                "Do not infer yardage or win-rate gains from disagreements alone."
            ),
        }
        print(json.dumps(report, indent=2, default=str))
    finally:
        db.close()
    return 0


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = list(args.path or [])
    db = None
    try:
        if getattr(args, "db", True):
            db = open_madden_db()
            return dataset_mod.build_rows(db=db, extra_paths=paths)
        return dataset_mod.build_rows(db=None, extra_paths=paths)
    finally:
        if db is not None:
            db.close()


def build_ml_subparser(sub: Any) -> None:
    p = sub.add_parser(
        "ml",
        help="Madden ML coach: inspect/export/train/evaluate/shadow (opt-in; default heuristic)",
    )
    p.set_defaults(game="madden27")
    ml_sub = p.add_subparsers(dest="ml_cmd", required=True)

    p_find = ml_sub.add_parser(
        "find-db", help="Show the active Madden database path (read-only)"
    )
    p_find.set_defaults(func=cmd_ml_find_db)

    p_bak = ml_sub.add_parser(
        "backup-db", help="Copy the Madden DB to a timestamped backup (source untouched)"
    )
    p_bak.add_argument(
        "--out",
        default=None,
        help="Backup directory (default: <data_dir>/backups)",
    )
    p_bak.set_defaults(func=cmd_ml_backup)

    p_ins = ml_sub.add_parser(
        "inspect", help="Inspect training-row coverage, eligibility, and provenance"
    )
    p_ins.add_argument("--path", action="append", default=[], help="Extra CSV/JSON/JSONL/SQLite path")
    p_ins.set_defaults(func=cmd_ml_inspect)

    p_exp = ml_sub.add_parser("export", help="Export validated JSONL training rows")
    p_exp.add_argument("--out", required=True, help="Output .jsonl path")
    p_exp.add_argument("--path", action="append", default=[], help="Extra input paths")
    p_exp.add_argument(
        "--supervised-only",
        action="store_true",
        help="Export only verified/trusted rows eligible for play-specific supervised training",
    )
    p_exp.add_argument(
        "--sanitized",
        action="store_true",
        help="Scrub opponent/session ids and local paths for debugging shares",
    )
    p_exp.set_defaults(func=cmd_ml_export)

    p_tr = ml_sub.add_parser("train", help="Train a supervised baseline and register it")
    p_tr.add_argument("--seed", type=int, required=True)
    p_tr.add_argument("--path", action="append", default=[], help="Extra training paths")
    p_tr.add_argument("--out", default=None, help="Scratch dir for the artifact before registry write")
    p_tr.add_argument("--registry", default=None, help="Registry root (default ~/.cfb-coach/madden_ml_registry)")
    p_tr.set_defaults(func=cmd_ml_train)

    p_ev = ml_sub.add_parser("evaluate", help="Held-out evaluation + promotion gate")
    p_ev.add_argument("--seed", type=int, required=True)
    p_ev.add_argument("--model", default=None, help="Model version (default CURRENT)")
    p_ev.add_argument("--path", action="append", default=[], help="Extra eval paths")
    p_ev.add_argument("--registry", default=None)
    p_ev.set_defaults(func=cmd_ml_evaluate)

    p_ls = ml_sub.add_parser("list", help="List registered models")
    p_ls.add_argument("--registry", default=None)
    p_ls.set_defaults(func=cmd_ml_list)

    p_st = ml_sub.add_parser("status", help="Show ML mode, model, recent shadow decisions")
    p_st.set_defaults(func=cmd_ml_status)

    p_sh = ml_sub.add_parser("shadow", help="Enable shadow mode (heuristic still displayed)")
    p_sh.add_argument("--off", action="store_true", help="Return to heuristic mode")
    p_sh.add_argument("--model", default=None)
    p_sh.add_argument("--registry", default=None)
    p_sh.set_defaults(func=cmd_ml_shadow)

    p_he = ml_sub.add_parser("heuristic", help="Return to heuristic mode")
    p_he.set_defaults(func=cmd_ml_heuristic)

    p_rp = ml_sub.add_parser(
        "report",
        help="Per-game shadow report (agreements, latency, verified-exec rates)",
    )
    p_rp.add_argument("--limit", type=int, default=500)
    p_rp.set_defaults(func=cmd_ml_report)
