#!/usr/bin/env python3
"""Retain the successful Lions COORDINATOR, retrain only its historical offense ML.

The original 428-row JSON has been superseded. This trainer calls the SAME
train_experimental() function from the actual 2026-11-04 Lions code revision,
now using all currently eligible offensive rows in a consistent copy of the
owner's existing Madden SQLite database.

Default: READ-ONLY preview. --apply writes ONLY a new isolated Lions folder.
Never rewrites the primary DB, the existing 655-row model, or Unified V1.

    python3 scripts/train_lions_legacy_offense.py
    python3 scripts/train_lions_legacy_offense.py --apply
    python3 scripts/train_lions_legacy_offense.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
from typing import Any

from scripts import restore_lions_baseline as old

DEFAULT_SOURCE = Path.home() / ".cfb-coach" / "madden27.db"
DEFAULT_WORKDIR = Path.home() / "lions-retrained-legacy"
MODEL_KIND = "hierarchical_shrinkage.v1"
MIN_SUPERVISED = 400
BASE_COMMIT = "0c4d1036372ad67be702ce0698f07de7857b7811"


def _protect(src: Path, output: Path) -> None:
    """Protect the real primary/Unified roots; allow isolated pytest temp dirs."""
    primary = (Path.home() / ".cfb-coach").resolve()
    unified = (Path.home() / "unified_v1_cpu1").resolve()
    if (output == src or output == src.parent or output.is_relative_to(primary)
            or output.is_relative_to(unified) or src.is_relative_to(output)
            or output == Path.home().resolve()):
        raise old.RestoreError("Output must be separate from the original franchise, Unified V1, and source database")
    if (output / "madden27-lions.db").resolve() == src:
        raise old.RestoreError("Cannot overwrite the original Madden database")


def _rows(db_path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Read same verified/provenance-filtered rows that old Lions model used."""
    from cfb_coach.db import CoachDB
    from cfb_coach.madden.model import dataset as dataset_mod

    database = CoachDB.open_read_only(db_path)
    try:
        all_rows = dataset_mod.build_rows(db=database)
    finally:
        database.close()
    verified = dataset_mod.supervised_rows(all_rows)
    offense = [r for r in verified
               if str(r.get("side") or "").lower().startswith("o")]
    return all_rows, len(offense), len(verified) - len(offense)


def _train(rows: list[dict[str, Any]], min_supervised: int):
    """No alternative model, no fallback, no synthetic training examples."""
    from cfb_coach.madden.model import experimental_model as old_ml

    artifact = old_ml.train_experimental(rows, side="offense", seed=7)
    if (artifact.kind != MODEL_KIND or artifact.side != "offense"
            or artifact.evidence_quality != "empirical"
            or artifact.n_supervised < min_supervised):
        raise old.RestoreError(
            f"Rejecting training: type={artifact.kind}, side={artifact.side}, "
            f"eligible_supervised={artifact.n_supervised}, "
            f"quality={artifact.evidence_quality}. Expected >= {min_supervised} "
            "verified offensive examples; no prior-only substitution."
        )
    return artifact


def _destination_paths(workdir: Path) -> tuple[Path, Path, Path]:
    return (workdir / "madden27-lions.db",
            workdir / "models" / "lions-hierarchical-shrinkage-new.json",
            workdir / "TRAIN_REPORT.json")


def _historical_books() -> tuple[dict[str, dict[str, Any]], str]:
    manifest = old.load_manifest()
    historical = Path.home() / "Downloads" / "madden27.backup.20261105T013541Z.db"
    return old.historical_records(manifest, historical)


def preview(args: argparse.Namespace) -> int:
    src = args.source_db.expanduser().resolve()
    output = args.workdir.expanduser().resolve()
    _protect(src, output)
    recs, provenance = _historical_books()
    print("LIONS COORDINATOR SOURCE:", BASE_COMMIT)
    print("SOURCE DATABASE (READ-ONLY):", src)
    print("MODEL ALGORITHM:", MODEL_KIND, "(the ORIGINAL Lions train_experimental)")
    print("PLAYBOOK RECORD:", provenance)
    print("O BOOK:", old._counts(recs["offense"]["formations"]))
    print("D BOOK:", old._counts(recs["defense"]["formations"]), "- legacy heuristic")
    with old._read_only_source(src) as conn:
        old._verify_db(conn)
    rows, n_off, n_def = _rows(src)
    print("DEDUPLICATED TOTAL ROWS:", len(rows))
    print("VERIFIED ELIGIBLE OFFENSE:", n_off)
    print("OTHER ELIGIBLE (NOT OFFENSE TRAINING):", n_def)
    print("EXPECTED MINIMUM:", args.min_supervised)
    print("FRESH ARTIFACT LABEL WILL BE: madden-ml.experimental.offense.<eligible>.empirical")
    if n_off < args.min_supervised:
        print("NO GO: Too few eligible OFFENSE rows. Training not attempted.")
        return 2
    print("PREVIEW PASS — no files were changed.")
    print("To train only inside", output, "use --apply.")
    return 0


def train(args: argparse.Namespace) -> int:
    src = args.source_db.expanduser().resolve()
    dest = args.workdir.expanduser().resolve()
    _protect(src, dest)
    if dest.exists():
        raise old.RestoreError(f"Refusing to overwrite existing Lions working folder: {dest}. "
                               "Reuse it with --check or choose another --workdir.")
    if not src.is_file():
        raise old.RestoreError(f"Missing source database: {src}")
    recs, provenance = _historical_books()

    # Stage an entire consistent copy in a sibling folder, then atomically
    # publish it only after the exact old algorithm fits the actual DB rows.
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".lions-training-stage-", dir=dest.parent))
    staged_db, staged_model, _ = _destination_paths(stage)
    final_db, final_model, final_report = _destination_paths(dest)
    try:
        with old._read_only_source(src) as conn:
            old._verify_db(conn)
            with sqlite3.connect(staged_db) as target:
                conn.backup(target)

        rows, n_off, n_other = _rows(staged_db)
        print("TRAINING ROWS:", len(rows), "deduplicated,",
              n_off, "verified offensive,", n_other, "other eligible")
        if n_off < args.min_supervised:
            raise old.RestoreError(f"Only {n_off} verified offensive examples; "
                                   f"require >= {args.min_supervised} for this baseline.")
        trained = _train(rows, args.min_supervised)
        from cfb_coach.madden.model import experimental_model as old_ml

        staged_model.parent.mkdir(parents=True, exist_ok=True)
        old_ml.save_artifact(trained, staged_model)
        loaded = old_ml.load_artifact(staged_model)
        if (loaded.model_version != trained.model_version or
                loaded.n_supervised != trained.n_supervised or
                loaded.kind != MODEL_KIND):
            raise old.RestoreError("Trained model failed round-trip serialization.")
        # The model file is at final_model after atomic rename.
        with sqlite3.connect(staged_db) as target:
            old._verify_db(target)
            state = old._read_state(target)
            state["applied"].update(recs)
            for side in ("offense", "defense"):
                state["pending"].pop(side, None)
            old._set_meta(target, "active_playbook_json", json.dumps(state))
            old._set_meta(target, "ml_mode", "experimental")
            old._set_meta(target, "ml_experimental_artifact", str(final_model))
            old._set_meta(target, "ml_knowledge_version", trained.knowledge_version)
            old._set_meta(target, "ml_data_version", trained.data_version)
            old._set_meta(target, old.MODE_DEFENSE, "off")
            target.commit()
            old._verify_db(target)
        report = {
            "schema": "lions.retrained.original-algorithm.v1",
            "source_code_commit": BASE_COMMIT,
            "source_database": str(src),
            "source_database_untouched": True,
            "snapshot_db_sha256": old.digest(staged_db),
            "restored_db": str(final_db),
            "model_artifact": str(final_model),
            "model_sha256": old.digest(staged_model),
            "model_algorithm": trained.kind,
            "model_version": trained.model_version,
            "trained_supervised_rows": trained.n_supervised,
            "trained_discounted_prior_rows": trained.n_discounted_priors,
            "training_eligible_other_side": n_other,
            "training_total_deduplicated": len(rows),
            "model_evidence_quality": trained.evidence_quality,
            "model_data_version": trained.data_version,
            "model_knowledge_version": trained.knowledge_version,
            "original_lions_reference": "madden-ml.experimental.offense.428.empirical",
            "original_model_recreated": False,
            "playbook_record": provenance,
            "offense_formations": list(recs["offense"]["formations"]),
            "offense_plays": 93,
            "defense_formations": list(recs["defense"]["formations"]),
            "defense_plays": 123,
            "defense_coordinator": "legacy_heuristic",
            "unified_v1_unchanged": True,
        }
        (stage / "TRAIN_REPORT.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, dest)  # dest does not exist; staging is sibling
        print("TRAINED:", trained.model_version,
              "(", trained.n_supervised, "supervised +",
              trained.n_discounted_priors, "discounted priors)")
        print("ISOLATED DB:", final_db)
        print("MODEL FILE:", final_model)
        print("UNIFIED + MAIN FRANCHISE + EXISTING 655 FILE: UNTOUCHED")
        print("Next: python3 scripts/train_lions_legacy_offense.py --check")
        return 0
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def check(args: argparse.Namespace) -> int:
    workdir = args.workdir.expanduser().resolve()
    db_path, model_path, report_path = _destination_paths(workdir)
    if not all(p.is_file() for p in (db_path, model_path, report_path)):
        raise old.RestoreError("Missing trained Lions DB/model/report; run --apply first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("restored_db") != str(db_path)
            or report.get("model_artifact") != str(model_path)
            or report.get("model_sha256") != old.digest(model_path)):
        raise old.RestoreError("Trained artifact does not match its report.")
    manifest = old.load_manifest()
    with old._read_only_source(db_path) as conn:
        old._verify_db(conn)
        state = old._read_state(conn)
        for side in ("offense", "defense"):
            if state["applied"][side]["formations"] != manifest["sides"][side]["formations"]:
                raise old.RestoreError(f"Restored {side} playbook changed.")
        if old._get_meta(conn, "ml_mode") != "experimental":
            raise old.RestoreError("Offensive experimental mode is OFF.")
        if old._get_meta(conn, old.MODE_DEFENSE) != "off":
            raise old.RestoreError("Defensive experimental caller is still ON.")
        if old._get_meta(conn, "ml_experimental_artifact") != str(model_path):
            raise old.RestoreError("The new offense artifact was not selected.")

    from cfb_coach.db import CoachDB
    from cfb_coach.madden.model.experimental_model import load_artifact
    from cfb_coach.madden.playcaller import make_call
    from cfb_coach.madden.situation import parse_madden_situation

    trained = load_artifact(model_path)
    if (trained.model_version != report.get("model_version")
            or trained.n_supervised < args.min_supervised
            or trained.kind != MODEL_KIND
            or trained.evidence_quality != "empirical"):
        raise old.RestoreError("New Lions model identity/integrity check failed.")

    # Test the *real* old playcaller, not a proxy. This is a read-only smoke
    # assessment and does not create game sessions.
    cases = [
        "o 1&10 my 25", "o 2&8 my 43", "o 3&6 opp 45",
        "o 3&2 opp 7", "o 1&10 opp 22", "o 2&15 my 7",
        "o 3&16 my 48", "o 1&10 my 42", "o 2&3 opp 34", "o 1&10 opp 15",
        "d 1&10 opp 25", "d 2&6 opp 38", "d 3&6 opp 45", "d 3&11 my 48",
    ]
    problems = []
    play_pairs = {"offense": set(), "defense": set()}
    db = CoachDB.open_read_only(db_path)
    try:
        for raw in cases:
            sit = parse_madden_situation(raw, default_side="offense")
            start = time.perf_counter()
            call = make_call(sit, "cpu", db, active_macros=[],
                             live_macros=False, coach_cpu_defense=True)
            elapsed = (time.perf_counter() - start) * 1000
            side = "offense" if raw.startswith("o ") else "defense"
            actual_book = manifest["sides"][side]["formations"]
            if (call.side != side or
                    call.play not in actual_book.get(call.formation, [])):
                problems.append(f"ILLEGAL/WRONG SIDE: {raw} -> {call.formation}/{call.play}/{call.side}")
            play_pairs[side].add((call.formation, call.play))
            if side == "offense":
                ml = getattr(call, "ml_experimental", None)
                if (not isinstance(ml, dict) or ml.get("fell_back")
                        or ml.get("model_version") != trained.model_version):
                    problems.append(f"ML FALLBACK: {raw}: {ml}")
            print(f"  {raw:22} → {call.side}: {call.formation} — {call.play} "
                  f"({elapsed:.1f}ms total)")
    finally:
        db.close()
    print("OFFENSE TRAINED:", trained.model_version,
          "verified rows:", trained.n_supervised,
          "discounted:", trained.n_discounted_priors)
    print("DEFENSE: ORIGINAL HEURISTIC | NO DEFENSIVE ML")
    print("DISTINCT PLAYS ON SMOKE:", {s: len(p) for s, p in play_pairs.items()})
    if problems:
        print("LIONS TRAINED VERDICT: NO GO")
        for problem in problems:
            print(" ", problem)
        return 2
    print("LIONS TRAINED VERDICT: GO (functional smoke; not a gameplay effectiveness guarantee)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE,
                        help="Source Madden franchise DB; never written.")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR,
                        help="New isolated destination, never auto-overwritten.")
    parser.add_argument("--min-supervised", type=int, default=MIN_SUPERVISED,
                        help="Minimum genuinely verified offensive examples before training (default 400).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Fit old Lions algorithm + copy DB into isolated folder.")
    mode.add_argument("--check", action="store_true", help="Smoke-test the trained copy without writes.")
    args = parser.parse_args()
    if args.min_supervised < 40:
        parser.error("min-supervised must be at least 40 (empirical model)")
    try:
        if args.check:
            return check(args)
        if args.apply:
            return train(args)
        return preview(args)
    except (old.RestoreError, sqlite3.Error, OSError, ValueError) as exc:
        print("LIONS LEGACY TRAIN: NO GO —", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
