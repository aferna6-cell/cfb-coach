"""Madden ML coach CLI. Opt-in; does not change heuristic play-calling defaults."""

from __future__ import annotations

import argparse
import json
import sys
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


def cmd_ml_inspect(args: argparse.Namespace) -> int:
    db = open_madden_db()
    try:
        paths = list(args.path or [])
        rows = dataset_mod.build_rows(db=db, extra_paths=paths)
        report = dataset_mod.validate_rows(rows)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"db: {db.path}")
        print(f"rows: {report['n_rows']}  labeled: {report['labeled']}  "
              f"verified_exec: {report['verified_executions']}  "
              f"recommendation_only: {report['recommendation_only']}")
    finally:
        db.close()
    return 0


def cmd_ml_export(args: argparse.Namespace) -> int:
    db = open_madden_db()
    try:
        paths = list(args.path or [])
        rows = dataset_mod.build_rows(db=db, extra_paths=paths)
        dataset_mod.export_jsonl(rows, args.out)
        report = dataset_mod.validate_rows(rows)
        print(f"wrote {len(rows)} rows -> {args.out}")
        print(json.dumps({k: report[k] for k in ("labeled", "unlabeled", "verified_executions", "duplicates", "provenance")}, indent=2))
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
                print(f"gate: {entry.gate.value}  promotion_allowed: {registry_mod.promotion_allowed(entry)}")
            except (OSError, ValueError, FileNotFoundError) as exc:
                print(f"registry_error: {exc}")
        decisions = db.list_ml_decisions(limit=5)
        print(f"recent_ml_decisions: {len(decisions)}")
        for row in decisions:
            print(
                f"  id={row['id']} mode={row['mode']} shadow={row['shadow_status']} "
                f"agree={row['agree']} heur={row['heuristic_play']} shadow_play={row['shadow_play']}"
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
                inference_mod.set_model_version(db, current.read_text(encoding="utf-8").strip())
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


def cmd_ml_report(args: argparse.Namespace) -> int:
    db = open_madden_db()
    try:
        rows = db.list_ml_decisions(limit=int(args.limit or 200))
        ok = [r for r in rows if r["shadow_status"] == MLStatus.OK.value]
        agree = [r for r in ok if r["agree"] == 1]
        disagree = [r for r in ok if r["agree"] == 0]
        failed = [r for r in rows if r["shadow_status"] and r["shadow_status"] != MLStatus.OK.value]
        latencies = [float(r["latency_ms_model"]) for r in ok if r["latency_ms_model"] is not None]
        report = {
            "n_decisions": len(rows),
            "shadow_ok": len(ok),
            "agree": len(agree),
            "disagree": len(disagree),
            "failed_or_non_ok": len(failed),
            "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
            "max_latency_ms": max(latencies) if latencies else None,
            "budget_ms": 150,
        }
        print(json.dumps(report, indent=2))
        for r in disagree[:20]:
            print(
                f"disagree snap={r['snap_id']} heur={r['heuristic_formation']}/{r['heuristic_play']} "
                f"model={r['shadow_formation']}/{r['shadow_play']}"
            )
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

    p_ins = ml_sub.add_parser("inspect", help="Inspect training-row coverage and provenance")
    p_ins.add_argument("--path", action="append", default=[], help="Extra CSV/JSON/JSONL/SQLite path")
    p_ins.set_defaults(func=cmd_ml_inspect)

    p_exp = ml_sub.add_parser("export", help="Export validated JSONL training rows")
    p_exp.add_argument("--out", required=True, help="Output .jsonl path")
    p_exp.add_argument("--path", action="append", default=[], help="Extra input paths")
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

    p_rp = ml_sub.add_parser("report", help="Shadow vs heuristic comparison report")
    p_rp.add_argument("--limit", type=int, default=200)
    p_rp.set_defaults(func=cmd_ml_report)
