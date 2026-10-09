"""Madden ML coach CLI. Opt-in; does not change heuristic play-calling defaults."""

from __future__ import annotations

import argparse
import json
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


def open_madden_db(*, read_only: bool = False) -> CoachDB:
    path = madden_db_path()
    if read_only:
        return CoachDB.open_read_only(path)
    return CoachDB(path, seed=load_seed())


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
    """Consistent SQLite backup via the backup API. Never modifies the source."""
    src = madden_db_path()
    if not src.is_file():
        print(f"no database at {src}", file=sys.stderr)
        return 1
    dest_dir = Path(args.out or (src.parent / "backups"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"{src.stem}.backup.{stamp}{src.suffix}"
    # Prefer read-only source connection so inspect/backup never migrates.
    db = CoachDB.open_read_only(src)
    try:
        db.backup_to(dest)
    finally:
        db.close()
    print(f"source: {src}")
    print(f"backup: {dest}")
    print(f"method: sqlite3.Connection.backup")
    print(f"size_bytes: {dest.stat().st_size}")
    return 0


def cmd_ml_inspect(args: argparse.Namespace) -> int:
    """Read-only inspection — never migrates or writes the gameplay DB."""
    paths = list(args.path or [])
    db = None
    try:
        src = madden_db_path()
        if src.is_file():
            db = CoachDB.open_read_only(src)
            rows = dataset_mod.build_rows(db=db, extra_paths=paths)
            db_label = str(db.path)
        else:
            rows = dataset_mod.build_rows(db=None, extra_paths=paths)
            db_label = f"(missing) {src}"
        report = dataset_mod.quality_report(rows)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"db: {db_label}")
        print(f"read_only: True")
        print(
            f"rows: {report['total_snaps']}  games: {report['unique_games']}  "
            f"verified_exec: {report['verified_executions']}  "
            f"supervised: {report['supervised_training_rows']}  "
            f"outcome_only: {report['outcome_only_examples']}"
        )
    finally:
        if db is not None:
            db.close()
    return 0


def cmd_ml_export(args: argparse.Namespace) -> int:
    """Export training rows. Source gameplay DB is opened read-only."""
    paths = list(args.path or [])
    db = None
    try:
        src = madden_db_path()
        if src.is_file():
            db = CoachDB.open_read_only(src)
            rows = dataset_mod.build_rows(db=db, extra_paths=paths)
        else:
            rows = dataset_mod.build_rows(db=None, extra_paths=paths)
        if getattr(args, "supervised_only", False):
            rows = dataset_mod.supervised_rows(rows)
        if getattr(args, "sanitized", False):
            dataset_mod.export_sanitized(rows, args.out)
        else:
            dataset_mod.export_jsonl(rows, args.out)
        report = dataset_mod.quality_report(rows)
        print(f"wrote {len(rows)} rows -> {args.out}")
        print(f"source_read_only: True")
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
                        "ambiguous_matches",
                        "eligibility",
                    )
                    if k in report
                },
                indent=2,
            )
        )
    finally:
        if db is not None:
            db.close()
    return 0


def cmd_ml_migrate(args: argparse.Namespace) -> int:
    """Explicit migration of a Madden DB. Requires a prior verified backup path."""
    src = madden_db_path()
    backup = Path(args.backup) if args.backup else None
    if not src.is_file():
        print(f"no database at {src}", file=sys.stderr)
        return 1
    if backup is None or not backup.is_file():
        print(
            "Refusing to migrate without --backup pointing at an existing verified backup.",
            file=sys.stderr,
        )
        print(
            "Run: python -m cfb_coach ml backup-db && python -m cfb_coach ml migrate-db --backup <path>",
            file=sys.stderr,
        )
        return 2
    db = CoachDB(src, seed=load_seed(), read_only=False, migrate=True)
    try:
        print(f"migrated: {src}")
        print(f"backup_ref: {backup}")
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
        print("Experimental ML calling is OFF. Live calls use the heuristic coach.")
    finally:
        db.close()
    return 0


def cmd_ml_experimental(args: argparse.Namespace) -> int:
    """Enable opt-in experimental ML offense selection (not competitive hybrid)."""
    db = open_madden_db()
    try:
        if args.off:
            inference_mod.set_mode(db, CoachingMode.HEURISTIC)
            print("mode: heuristic")
            return 0
        from cfb_coach.madden.model import experimental_live as exp_live
        from cfb_coach.madden.model import experimental_model as exp_mod
        from cfb_coach.madden.model import dataset as ds

        # Ensure an artifact exists (prior-driven is fine with empty verified data).
        art_path = exp_live.resolve_artifact_path(db)
        if art_path is None or getattr(args, "retrain", False):
            rows = ds.build_rows(db=db)
            art = exp_mod.train_experimental(rows, side="offense")
            from cfb_coach.games import data_dir

            dest = Path(data_dir()) / "madden_ml_experimental" / "offense.json"
            exp_mod.save_artifact(art, dest)
            exp_live.set_artifact_path(db, dest)
            db.set_meta(exp_live.META_KNOWLEDGE, art.knowledge_version)
            db.set_meta(exp_live.META_DATA_VERSION, art.data_version)
            print(f"trained: {dest}")
            print(f"evidence_quality: {art.evidence_quality}")
            print(f"n_supervised: {art.n_supervised}")
            print(f"n_discounted_priors: {art.n_discounted_priors}")
            print(f"knowledge_version: {art.knowledge_version}")
            print(f"knowledge_origin: {art.knowledge_origin or '(none)'}")
            print(f"model_version: {art.model_version}")
            print(f"artifact_path: {dest}")
            print(f"note: {art.note}")
            # Keep learning adjustment evidence after every ordinary retrain,
            # but do NOT replace a deliberately promoted action model without
            # review. No cold/sparse action data can auto-promote.
            from cfb_coach.madden.model.offense_action_learning import (
                load_action_evidence, save_action_evidence,
                train_action_evidence,
            )
            try:
                existing_action = load_action_evidence(db)
                if (existing_action or {}).get("mode") == "bounded_active":
                    print("action_model: existing bounded-active artifact preserved; "
                          "run offense-action-learn --train to replace it with shadow")
                else:
                    shadow_action = train_action_evidence(db)
                    save_action_evidence(db, shadow_action)
                    print(f"action_model: shadow | verified_rows: "
                          f"{shadow_action['n_verified_action_rows']} | "
                          f"eligible_groups: {shadow_action['ready_groups']}")
            except Exception as action_exc:
                print(f"action_model: shadow refresh skipped ({action_exc})")
            if art.n_supervised == 0:
                print(
                    "WARNING: zero supervised rows in this DB — do not claim the "
                    "eight historical games trained the model until audit-history "
                    "on the laptop confirms their data was included."
                )
        inference_mod.set_mode(db, CoachingMode.EXPERIMENTAL)
        print("mode: experimental")
        print("EXPERIMENTAL PILOT — not a validated competitive model.")
        print("Offense-only ML selection within the full confirmed offensive playbook.")
        print("Model selects the offensive play; heuristic is a failure fallback only.")
        print("On timeout/error/illegal → heuristic fallback.")
        print("Restore heuristic: python -m cfb_coach ml heuristic")
    finally:
        db.close()
    return 0


def cmd_ml_experimental_preflight(args: argparse.Namespace) -> int:
    """Reproducible laptop preflight: locate DB, audit, train, print versions.

    Does not enable experimental mode unless ``--enable`` is passed.
    Never invents historical games.
    """
    del args
    print("=== Madden ML experimental preflight ===")
    print("--- find-db ---")
    cmd_ml_find_db(argparse.Namespace())
    src = madden_db_path()
    if not src.is_file():
        print("STOP: no madden DB found. Set CFB_COACH_MADDEN_DB or run Franchise logging first.")
        return 2
    print("--- backup-db ---")
    cmd_ml_backup(argparse.Namespace(out=None))
    print("--- audit-history ---")
    cmd_ml_audit_history(argparse.Namespace())
    print("--- inspect ---")
    cmd_ml_inspect(argparse.Namespace(path=[]))
    print("--- train-experimental --seed 7 --install ---")
    rc = cmd_ml_train_experimental(
        argparse.Namespace(
            seed=7,
            side="offense",
            path=[],
            out=None,
            install=True,
            db=True,
        )
    )
    if rc != 0:
        return rc
    from cfb_coach.madden.model import experimental_live as exp_live
    from cfb_coach.madden.model import experimental_model as exp_mod

    db = open_madden_db()
    try:
        # Preflight must not silently activate experimental mode.
        inference_mod.set_mode(db, CoachingMode.HEURISTIC)
        art_path = exp_live.resolve_artifact_path(db)
        print(f"active_artifact_path: {art_path}")
        if art_path and art_path.is_file():
            art = exp_mod.load_artifact(art_path)
            print(f"supervised_count: {art.n_supervised}")
            print(f"discounted_prior_count: {art.n_discounted_priors}")
            print(f"knowledge_version: {art.knowledge_version}")
            print(f"knowledge_origin: {art.knowledge_origin or '(none)'}")
            print(f"model_version: {art.model_version}")
            print(f"evidence_quality: {art.evidence_quality}")
            if art.n_supervised == 0:
                print(
                    "NO CLAIM: eight historical games did NOT train this artifact "
                    "from this DB (n_supervised=0). Re-run on the laptop DB that "
                    "contains those Franchise sessions."
                )
        print("mode remains: heuristic (use `ml experimental` to opt in)")
        print("mode_check:", inference_mod.resolve_mode(db).value)
    finally:
        db.close()
    print("=== preflight complete ===")
    print("Next (explicit): python -m cfb_coach ml experimental --retrain")
    return 0


def cmd_ml_audit_history(args: argparse.Namespace) -> int:
    """Read-only audit of historical Franchise snaps. Never invents games."""
    from cfb_coach.madden.model import historical as hist

    db = None
    try:
        src = madden_db_path()
        if src.is_file():
            db = CoachDB.open_read_only(src)
            report = hist.audit_database(db)
            report["db"] = str(src)
            report["read_only"] = True
        else:
            report = {
                "n_rows": 0,
                "n_games": 0,
                "db": f"(missing) {src}",
                "note": "No local madden27.db — cannot invent historical games.",
            }
        print(json.dumps(report, indent=2, default=str))
    finally:
        if db is not None:
            db.close()
    return 0


def cmd_ml_confirm_execution(args: argparse.Namespace) -> int:
    """Retrospectively verify an executed play from user evidence."""
    from cfb_coach.madden.model import historical as hist

    db = open_madden_db()
    try:
        out = hist.confirm_execution(
            db,
            ml_snap_id=args.snap_id,
            executed_formation=args.formation,
            executed_play=args.play,
            evidence=args.evidence,
            executed_macro=args.macro,
        )
        print(json.dumps(out, indent=2))
    except (ValueError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()
    return 0


def cmd_ml_recover_game_execution(args: argparse.Namespace) -> int:
    """One-time recovery for game 141d4a16b4184ee7 (dry-run unless --apply)."""
    from cfb_coach.madden.model import execution_recovery as recovery

    game_id = str(getattr(args, "game_id", None) or recovery.RECOVERY_GAME_ID)
    attestation = getattr(args, "attest", None)
    apply = bool(getattr(args, "apply", False))
    backup = getattr(args, "backup", None)

    db = open_madden_db()
    try:
        if apply:
            report = recovery.apply_recovery(
                db,
                game_id=game_id,
                attestation=str(attestation or ""),
                backup_path=backup,
                apply=True,
            )
        else:
            report = recovery.preview_recovery(
                db, game_id=game_id, attestation=attestation
            )
        print(json.dumps(report, indent=2, default=str))
        if apply and report.get("applied"):
            print(
                f"before_verified={report['before']['n_verified_executions']} "
                f"after_verified={report['after']['n_verified_executions']} "
                f"delta={report.get('delta_verified')}",
                file=sys.stderr,
            )
            print(
                f"before_supervised_game={report['before']['n_game_supervised_rows']} "
                f"after_supervised_game={report['after']['n_game_supervised_rows']}",
                file=sys.stderr,
            )
            print(f"backup={report.get('backup')}", file=sys.stderr)
        if not report.get("ok", True) and apply:
            return 2
        if apply and not report.get("applied"):
            return 2
    finally:
        db.close()
    return 0


def cmd_ml_train_experimental(args: argparse.Namespace) -> int:
    """Train the hierarchical shrinkage experimental model (no promotion gate)."""
    from cfb_coach.madden.model import experimental_live as exp_live
    from cfb_coach.madden.model import experimental_model as exp_mod

    rows = _load_rows(args)
    art = exp_mod.train_experimental(rows, side=str(args.side or "offense"))
    dest = Path(args.out or (Path(data_dir()) / "madden_ml_experimental" / f"{art.side}.json"))
    exp_mod.save_artifact(art, dest)
    # Compare against logistic baseline when enough data exists.
    comparison: dict[str, Any] = {
        "experimental": {
            "kind": art.kind,
            "evidence_quality": art.evidence_quality,
            "n_supervised": art.n_supervised,
            "n_discounted_priors": art.n_discounted_priors,
            "global_rate": art.global_rate,
            "knowledge_version": art.knowledge_version,
            "knowledge_origin": art.knowledge_origin,
            "model_version": art.model_version,
            "research_concept_boost": art.research_concept_boost,
            "research_family_boost": art.research_family_boost,
            "note": art.note,
        }
    }
    print(f"supervised_count: {art.n_supervised}")
    print(f"discounted_prior_count: {art.n_discounted_priors}")
    print(f"knowledge_version: {art.knowledge_version}")
    print(f"knowledge_origin: {art.knowledge_origin or '(none)'}")
    print(f"model_version: {art.model_version}")
    print(f"artifact_path: {dest}")
    try:
        from cfb_coach.madden.model import evaluate as evaluate_mod
        from cfb_coach.madden.model import train as train_mod

        if art.n_supervised >= 5:
            log_entry = train_mod.train(rows, seed=int(args.seed or 7), out_dir=str(dest.parent / "_log_scratch"))
            comparison["logistic_baseline"] = {
                "model_version": log_entry.model_version,
                "gate": log_entry.gate.value,
                "note": "Offline comparison only — does not activate hybrid.",
            }
        else:
            comparison["logistic_baseline"] = {
                "skipped": True,
                "reason": "insufficient supervised rows for a meaningful logistic fit",
            }
    except Exception as exc:  # noqa: BLE001
        comparison["logistic_baseline"] = {"error": str(exc)}

    comparison["heuristic_baseline"] = {
        "note": (
            "Heuristic remains the live default. Experimental mode must be "
            "explicitly enabled and shows both picks."
        )
    }
    print(f"artifact: {dest}")
    print(json.dumps(comparison, indent=2, default=str))
    if getattr(args, "install", False):
        db = open_madden_db()
        try:
            exp_live.set_artifact_path(db, dest)
            db.set_meta(exp_live.META_KNOWLEDGE, art.knowledge_version)
            db.set_meta(exp_live.META_DATA_VERSION, art.data_version)
            print(f"installed artifact path in meta: {dest}")
        finally:
            db.close()
    return 0


def cmd_ml_offense_action_learn(args: argparse.Namespace) -> int:
    """Train / inspect / gate observational pre-snap action evidence."""
    from cfb_coach.madden.model.offense_action_learning import (
        load_action_evidence, promote_action_evidence,
        rollback_action_evidence, save_action_evidence,
        train_action_evidence,
    )

    actions = sum(bool(x) for x in (args.train, args.promote, args.rollback))
    if actions > 1:
        print("Choose one of --train, --promote or --rollback", file=sys.stderr)
        return 2
    db = open_madden_db(read_only=not actions)
    try:
        try:
            if args.train:
                art = train_action_evidence(db)
                save_action_evidence(db, art)
            elif args.promote:
                art = promote_action_evidence(db)
            elif args.rollback:
                art = rollback_action_evidence(db)
            else:
                art = load_action_evidence(db)
        except ValueError as exc:
            print(f"Action learning refused: {exc}", file=sys.stderr)
            return 2
        if art is None:
            print("No saved action model. Run ml offense-action-learn --train first.")
            return 0
        view = dict(art)
        if not args.details:
            view["comparisons"] = {
                k: v for k, v in (art.get("comparisons") or {}).items()
                if v.get("ready_for_bounded_adjustment")
            }
        print(json.dumps(view, indent=2, default=str))
        return 0
    finally:
        db.close()


def cmd_ml_offense_macro_lab(args: argparse.Namespace) -> int:
    """Generate original source-backed, unarmed macro drafts for the installed book."""
    from cfb_coach.madden.model import offense_macro_lab as lab

    db = open_madden_db(read_only=not args.stage)
    try:
        try:
            result = (
                lab.stage_variants(db, limit=args.limit)
                if args.stage else lab.propose_variants(db, limit=args.limit)
            )
        except ValueError as exc:
            print(f"Macro lab refused: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        db.close()


def cmd_ml_offense_actions(args: argparse.Namespace) -> int:
    """Show researched, compatible and verified Custom Adjustment readiness."""
    from cfb_coach.madden.model.offense_action_inventory import offense_actions_report

    db = open_madden_db(read_only=True)
    try:
        report = offense_actions_report(db, args.opponent)
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["playbook_installed"] else 2
    finally:
        db.close()


def cmd_ml_offense_inventory(args: argparse.Namespace) -> int:
    """Audit every confirmed formation/play and its recommendation usage."""
    from cfb_coach.madden.model.offense_inventory import inventory_report

    db = open_madden_db(read_only=True)
    try:
        report = inventory_report(
            db, opponent_id=args.opponent, game_id=args.game_id
        )
        if args.summary:
            report.pop("play_usage", None)
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["installed"] else 2
    finally:
        db.close()


def cmd_ml_offense_design(args: argparse.Namespace) -> int:
    """Design/inspect the custom offense; open the Madden-style browser by default.

    The HTML page is read-only. Stage/confirm/macro-verify actions still require
    explicit local CLI commands and in-game installation confirmation.
    """
    from cfb_coach.madden.model import offense_designer as designer
    from cfb_coach.madden.model import offense_design_browser as browser

    if args.show and (args.stage or args.confirm_installed or args.verify_macro or args.unverify_macro or args.rollback_design):
        print("--show cannot be combined with mutation flags", file=sys.stderr)
        return 2
    if sum(bool(x) for x in (
        args.stage, args.confirm_installed, args.verify_macro, args.unverify_macro, args.rollback_design
    )) > 1:
        print("Stage/confirm/verify/rollback are separate deliberate actions", file=sys.stderr)
        return 2

    text_mode = bool(getattr(args, "text", False))
    open_browser = not (text_mode or getattr(args, "no_open", False))
    write_browser = not text_mode
    read_only = not bool(args.stage or args.confirm_installed or args.verify_macro or args.unverify_macro or args.rollback_design)
    db = open_madden_db(read_only=read_only)

    def display(payload: dict[str, Any] | None, mode: str) -> None:
        if text_mode or payload is None:
            print(json.dumps(payload or {"status": "no_staged_or_installed_design"}, indent=2))
            return
        path = browser.write_and_open(
            db, payload, mode=mode, opponent_id=args.opponent,
            open_browser=open_browser,
        )
        print(f"Madden ML Offensive Designer → {path}")
        print(f"design_status: {mode} | proposal_id: {payload.get('proposal_id')}")
        print("The browser checklist is read-only. Confirm actual editor changes using the displayed CLI command.")

    try:
        if args.show:
            data = designer.staged_design(db)
            mode = "staged"
            if data is None:
                data = browser.installed_design(db, args.opponent)
                mode = "installed"
            display(data, mode)
            return 0
        if args.rollback_design:
            try:
                result = designer.rollback_design(
                    db, proposal_id=args.rollback_design, attestation=args.attest or ""
                )
            except ValueError as exc:
                print(f"Rollback refused: {exc}", file=sys.stderr)
                return 2
            print(json.dumps(result, indent=2))
            display(browser.installed_design(db, args.opponent), "installed")
            return 0
        if args.unverify_macro:
            result = designer.unverify_created_macro(
                db, name=args.unverify_macro, opponent_id=args.opponent,
            )
            print(json.dumps(result, indent=2))
            display(browser.installed_design(db, args.opponent), "installed")
            return 0
        if args.verify_macro:
            try:
                result = designer.verify_created_macro(
                    db, name=args.verify_macro, opponent_id=args.opponent,
                    attestation=args.attest or "",
                    retire_existing=args.retire_existing,
                )
            except ValueError as exc:
                print(f"Macro verification refused: {exc}", file=sys.stderr)
                return 2
            print(json.dumps(result, indent=2))
            display(browser.installed_design(db, args.opponent), "installed")
            return 0
        if args.confirm_installed:
            try:
                result = designer.confirm_installed(
                    db, proposal_id=args.confirm_installed, attestation=args.attest or ""
                )
            except ValueError as exc:
                print(f"Installation refused: {exc}", file=sys.stderr)
                return 2
            print(json.dumps(result, indent=2))
            display(browser.installed_design(db, args.opponent), "installed")
            return 0
        try:
            design = designer.design_offense(
                db, opponent_id=args.opponent,
                max_formations=args.max_formations
            )
            if args.stage:
                design = designer.stage_design(db, design)
        except (ValueError, OSError) as exc:
            print(f"Offense design unavailable: {exc}", file=sys.stderr)
            return 2
        display(design, "staged" if args.stage else "preview")
        if args.stage:
            print("STAGED ONLY — active book unchanged until built in Madden and confirmed.")
        else:
            print("PREVIEW ONLY — no applied playbook or macro changes.")
        return 0
    finally:
        db.close()


def cmd_ml_postgame_experimental(args: argparse.Namespace) -> int:
    from cfb_coach.madden.model import experimental_live as exp_live

    db = open_madden_db()
    try:
        report = exp_live.postgame_experimental_compare(db, game_id=args.game_id)
        print(json.dumps(report, indent=2, default=str))
    finally:
        db.close()
    return 0


def cmd_ml_research_refresh(args: argparse.Namespace) -> int:
    """On-demand Madden research refresh (candidate update; does not auto-overwrite policy)."""
    from cfb_coach.madden.research_refresh import run_research_refresh

    result = run_research_refresh(
        dry_run=bool(getattr(args, "dry_run", False)),
        open_pr=bool(getattr(args, "open_pr", False)),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


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
    """Load training/eval rows without writing to the gameplay database."""
    paths = list(args.path or [])
    db = None
    try:
        if getattr(args, "db", True):
            src = madden_db_path()
            if src.is_file():
                db = CoachDB.open_read_only(src)
                return dataset_mod.build_rows(db=db, extra_paths=paths)
            return dataset_mod.build_rows(db=None, extra_paths=paths)
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
        "backup-db",
        help="Consistent SQLite backup via backup API (source untouched, WAL-safe)",
    )
    p_bak.add_argument(
        "--out",
        default=None,
        help="Backup directory (default: <data_dir>/backups)",
    )
    p_bak.set_defaults(func=cmd_ml_backup)

    p_mig = ml_sub.add_parser(
        "migrate-db",
        help="Explicit schema migration (requires --backup of a verified copy)",
    )
    p_mig.add_argument(
        "--backup",
        required=True,
        help="Path to an existing verified backup created by backup-db",
    )
    p_mig.set_defaults(func=cmd_ml_migrate)

    p_ins = ml_sub.add_parser(
        "inspect",
        help="Read-only inspect of training-row coverage (no schema migration)",
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

    p_ex = ml_sub.add_parser(
        "experimental",
        help="Enable EXPERIMENTAL ML offense selection (opt-in pilot; not hybrid)",
    )
    p_ex.add_argument("--off", action="store_true", help="Return to heuristic mode")
    p_ex.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the experimental artifact before enabling",
    )
    p_ex.set_defaults(func=cmd_ml_experimental)

    p_pf = ml_sub.add_parser(
        "experimental-preflight",
        help="Laptop preflight: find-db → backup → audit → inspect → train-experimental",
    )
    p_pf.set_defaults(func=cmd_ml_experimental_preflight)

    p_ah = ml_sub.add_parser(
        "audit-history",
        help="Read-only audit of historical Franchise games (never invents data)",
    )
    p_ah.set_defaults(func=cmd_ml_audit_history)

    p_ce = ml_sub.add_parser(
        "confirm-execution",
        help="Retrospectively verify an executed play with explicit user evidence",
    )
    p_ce.add_argument("--snap-id", required=True, help="ml_snap_id or snaps.id")
    p_ce.add_argument("--formation", required=True)
    p_ce.add_argument("--play", required=True)
    p_ce.add_argument("--evidence", required=True, help="Recording note / recollection")
    p_ce.add_argument("--macro", default=None)
    p_ce.set_defaults(func=cmd_ml_confirm_execution)

    p_rec = ml_sub.add_parser(
        "recover-game-execution",
        help=(
            "One-time recovery for experimental game 141d4a16b4184ee7 "
            "(dry-run by default; requires --attest + --apply to mutate)"
        ),
    )
    p_rec.add_argument(
        "--game-id",
        default=None,
        help="Must be 141d4a16b4184ee7 (locked); other games refused",
    )
    p_rec.add_argument(
        "--attest",
        default=None,
        help="Explicit confirmation that final displayed calls were executed",
    )
    p_rec.add_argument(
        "--apply",
        action="store_true",
        help="Apply recovery after backup (omit for dry-run preview)",
    )
    p_rec.add_argument(
        "--backup",
        default=None,
        help="Backup destination path (default: <db_dir>/backups/...)",
    )
    p_rec.set_defaults(func=cmd_ml_recover_game_execution)

    p_te = ml_sub.add_parser(
        "train-experimental",
        help="Train hierarchical shrinkage experimental model (no promotion gate)",
    )
    p_te.add_argument("--seed", type=int, default=7)
    p_te.add_argument("--side", default="offense", choices=("offense", "defense"))
    p_te.add_argument("--path", action="append", default=[], help="Extra training paths")
    p_te.add_argument("--out", default=None)
    p_te.add_argument(
        "--install",
        action="store_true",
        help="Write artifact path into DB meta (does not enable experimental mode)",
    )
    p_te.add_argument("--no-db", dest="db", action="store_false", default=True)
    p_te.set_defaults(func=cmd_ml_train_experimental)

    p_learn = ml_sub.add_parser(
        "offense-action-learn",
        help="Learn verified pre-snap adjustment outcomes; shadow until evidence-gated promotion",
    )
    p_learn.add_argument("--train", action="store_true",
                         help="Build a SHADOW model from verified, explicitly confirmed action outcomes")
    p_learn.add_argument("--promote", action="store_true",
                         help="Activate only adequately compared multi-game contexts (bounded ±0.04)")
    p_learn.add_argument("--rollback", action="store_true",
                         help="Immediately disable learned action-score shifts")
    p_learn.add_argument("--details", action="store_true",
                         help="Include all sparse action/context comparisons")
    p_learn.set_defaults(func=cmd_ml_offense_action_learn)

    p_lab = ml_sub.add_parser(
        "offense-macro-lab",
        help="Generate novel concept-targeted, source-backed offensive macro drafts (never armed)",
    )
    p_lab.add_argument("--limit", type=int, default=6)
    p_lab.add_argument("--stage", action="store_true",
                       help="Add drafts to the confirmed-book macro blueprint registry; no arming")
    p_lab.set_defaults(func=cmd_ml_offense_macro_lab)

    p_act = ml_sub.add_parser(
        "offense-actions",
        help="Audit current offensive macros, settings, action triggers and readiness (read-only)",
    )
    p_act.add_argument("-o", "--opponent", default="cpu")
    p_act.set_defaults(func=cmd_ml_offense_actions)

    p_inv = ml_sub.add_parser(
        "offense-inventory",
        help="Audit ALL installed offensive formations/plays and call coverage; no DB changes",
    )
    p_inv.add_argument("-o", "--opponent", default="cpu")
    p_inv.add_argument("--game-id", default=None, help="Count recommendations in one game only")
    p_inv.add_argument("--summary", action="store_true", help="Totals only; omit per-play rows")
    p_inv.set_defaults(func=cmd_ml_offense_inventory)

    p_od = ml_sub.add_parser(
        "offense-design",
        help="Model proposes whole offensive formations with ALL source plays and macro drafts",
    )
    p_od.add_argument("--opponent", "-o", default="cpu")
    p_od.add_argument("--max-formations", type=int, default=5)
    p_od.add_argument("--text", action="store_true",
                      help="Print the original JSON report instead of opening the HTML designer")
    p_od.add_argument("--no-open", action="store_true",
                      help="Write HTML locally but do not launch a browser")
    p_od.add_argument("--show", action="store_true", help="Show staged design without modifying DB")
    p_od.add_argument("--stage", action="store_true", help="Store proposal only; live locked book stays untouched")
    p_od.add_argument("--confirm-installed", metavar="PROPOSAL_ID", default=None,
                      help="Confirm ALL proposed plays/formations were built and checked inside Madden")
    p_od.add_argument("--attest", default=None, help="Explicit in-game installation/activation evidence")
    p_od.add_argument("--rollback-design", metavar="PROPOSAL_ID", default=None,
                      help="Restore prior offense only after reinstallation in Madden")
    p_od.add_argument("--unverify-macro", metavar="NAME", default=None,
                      help="Immediately stop live ML from offering this no-longer-armed macro")
    p_od.add_argument("--verify-macro", metavar="NAME", default=None,
                      help="Verify a generated macro was built and armed in Madden")
    p_od.add_argument("--retire-existing", metavar="ID", default=None,
                      help="Explicitly replace an existing active offense macro in the eight-slot loadout")
    p_od.set_defaults(func=cmd_ml_offense_design)

    p_pg = ml_sub.add_parser(
        "postgame-experimental",
        help="Heuristic vs ML comparison for experimental sessions",
    )
    p_pg.add_argument("--game-id", default=None)
    p_pg.set_defaults(func=cmd_ml_postgame_experimental)

    p_rr = ml_sub.add_parser(
        "research-refresh",
        help="On-demand Madden research refresh (candidate PR; never silent policy overwrite)",
    )
    p_rr.add_argument("--dry-run", action="store_true")
    p_rr.add_argument(
        "--open-pr",
        action="store_true",
        help="Attempt to open a reviewable PR when meaningful updates exist",
    )
    p_rr.set_defaults(func=cmd_ml_research_refresh)

    p_rp = ml_sub.add_parser(
        "report",
        help="Per-game shadow report (agreements, latency, verified-exec rates)",
    )
    p_rp.add_argument("--limit", type=int, default=500)
    p_rp.set_defaults(func=cmd_ml_report)
