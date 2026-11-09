#!/usr/bin/env python3
"""Restore the November 2026 Lions play setup to an ISOLATED database copy.

Neither the source Madden database nor the Unified V1 SQLite store is writable
through this command. No model is trained or invented. A missing/mismatched
historical 428-row model blocks restoration.

Usage:
    python3 scripts/restore_lions_baseline.py
    python3 scripts/restore_lions_baseline.py --apply
    python3 scripts/restore_lions_baseline.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "cfb_coach" / "madden" / "lions_baseline_snapshot.json"
EXPECTED_MODEL_VERSION = "madden-ml.experimental.offense.428.empirical"
EXPECTED_SUPERVISED = 428
EXPECTED_COUNTS = {"offense": (5, 93), "defense": (6, 123)}
MODE_DEFENSE = "ml_defense_coordinator_mode.v1"


class RestoreError(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))


def _read_state(conn: sqlite3.Connection) -> dict[str, Any]:
    raw = _get_meta(conn, "active_playbook_json") or "{}"
    state = json.loads(raw)
    if "applied" in state or "pending" in state:
        return {"applied": dict(state.get("applied") or {}),
                "pending": dict(state.get("pending") or {})}
    return {"applied": state, "pending": {}}


def _counts(forms: dict[str, list[str]]) -> tuple[int, int]:
    return len(forms), sum(len(plays) for plays in forms.values())


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for side, expected in EXPECTED_COUNTS.items():
        forms = manifest["sides"][side]["formations"]
        if _counts(forms) != expected:
            raise RestoreError(f"Historical {side} inventory no longer matches {expected}")
        if any(len(plays) != len(set(plays)) for plays in forms.values()):
            raise RestoreError(f"Duplicate play in historical {side} inventory")
    return manifest


def historical_records(manifest: dict[str, Any], historical_db: Path | None
                       ) -> tuple[dict[str, dict[str, Any]], str]:
    """Prefer the actual old applied DB record (preserving original audibles)."""
    recs = {}
    provenance = "recorded-installed-inventory-snapshot"
    if historical_db is not None and historical_db.is_file():
        try:
            with sqlite3.connect(f"file:{historical_db.resolve()}?mode=ro", uri=True) as c:
                previous = _read_state(c)["applied"]
            if all(
                previous.get(side, {}).get("formations") == manifest["sides"][side]["formations"]
                for side in ("offense", "defense")
            ):
                recs = {side: dict(previous[side]) for side in ("offense", "defense")}
                provenance = f"historical applied records: {historical_db}"
        except (sqlite3.Error, ValueError, KeyError):
            pass
    if not recs:
        for side in ("offense", "defense"):
            previous = manifest["sides"][side]
            forms = previous["formations"]
            recs[side] = {
                "side": side, "mode": "custom", "name": previous["recorded_playbook_name"],
                "formations": {key: list(val) for key, val in forms.items()},
                "formation_sources": dict(previous.get("formation_sources") or {}),
                "audibles": {}, "core": list(forms),
                "rev": 1, "locked_ts": previous.get("recorded_at", ""),
                "reason": "Pinned historical Lions 2026-11-04 baseline; restored to an isolated copy",
            }
    for side in ("offense", "defense"):
        got = _counts(recs[side]["formations"])
        if got != EXPECTED_COUNTS[side]:
            raise RestoreError(f"Old {side} book mismatch: {got}")
        if recs[side]["formations"] != manifest["sides"][side]["formations"]:
            raise RestoreError(f"Old {side} book differs from pinned historical manifest")
    return recs, provenance


def find_model(source: sqlite3.Connection, override: Path | None
               ) -> tuple[Path | None, list[str]]:
    """Never silently train or substitute a new artifact."""
    choices = []
    if override:
        choices.append(override)
    current = _get_meta(source, "ml_experimental_artifact")
    if current:
        choices.append(Path(current).expanduser())
    choices.append(Path.home() / ".cfb-coach" / "madden_ml_experimental" / "offense.json")
    seen = set()
    findings = []
    for path in choices:
        path = path.expanduser()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if not path.is_file():
            findings.append(f"{path} (missing)")
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            version = raw.get("model_version")
            n = int(raw.get("n_supervised") or 0)
            side = raw.get("side")
            quality = raw.get("evidence_quality")
            findings.append(f"{path} (version={version}, supervised={n}, quality={quality})")
            if (
                version == EXPECTED_MODEL_VERSION
                and n == EXPECTED_SUPERVISED
                and side == "offense"
                and raw.get("kind") == "hierarchical_shrinkage.v1"
                and quality == "empirical"
            ):
                return path, findings
        except (ValueError, OSError, TypeError) as exc:
            findings.append(f"{path} (invalid: {exc})")
    return None, findings


def _read_only_source(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise RestoreError(f"Source database not found: {path}")
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def _verify_db(conn: sqlite3.Connection) -> None:
    result = conn.execute("PRAGMA quick_check").fetchone()
    if result is None or result[0] != "ok":
        raise RestoreError(f"Database integrity check failed: {result}")


def restore(args: argparse.Namespace) -> int:
    src = args.source_db.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()
    out_db = workdir / "madden27-lions.db"
    manifest = load_manifest()
    with _read_only_source(src) as source:
        _verify_db(source)
        model_path, findings = find_model(source, args.model)
        recs, provenance = historical_records(manifest, args.historical_db)
        current = _read_state(source)
        counts = {}
        for table in ("snaps", "game_sessions", "ml_decisions", "ml_outcomes"):
            try:
                counts[table] = source.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = None
        print("SOURCE (read only):", src)
        print("Historical playbook origin:", provenance)
        print("Offense:", _counts(recs["offense"]["formations"]), "five-formation Lions book")
        print("Defense:", _counts(recs["defense"]["formations"]), "six-formation heuristic book")
        print("Currently applied offense:", _counts(
            (current["applied"].get("offense") or {}).get("formations") or {}))
        print("Current source rows:", counts)
        print("Expected offense artifact:", EXPECTED_MODEL_VERSION)
        for finding in findings:
            print("  artifact candidate:", finding)
        if model_path is None:
            raise RestoreError(
                "Original 428-row empirical artifact not found. Do not retrain or play: "
                "supply an actual historical artifact with --model /path/to/offense.json."
            )
        print("SELECTED ORIGINAL ARTIFACT:", model_path)
        print("MODEL SHA256:", digest(model_path))
        if not args.apply:
            print("DRY RUN ONLY: no database or model files changed.")
            print("To create isolated Lions copy: rerun with --apply.")
            return 0
        if out_db.exists():
            raise RestoreError(
                f"Refusing to overwrite existing restored game {out_db}. "
                "Use --check or choose a fresh --workdir."
            )
        workdir.mkdir(parents=True, exist_ok=True)
        dest_model = workdir / "models" / "offense_428_empirical.json"
        dest_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, dest_model)
        if digest(model_path) != digest(dest_model):
            raise RestoreError("Artifact copy checksum mismatch")
        # SQLite's online-backup API copies every table including WAL changes.
        # Never copy the live .db file with shutil.copy2.
        with sqlite3.connect(out_db) as target:
            source.backup(target)
        with sqlite3.connect(out_db) as target:
            _verify_db(target)
            state = _read_state(target)
            # Reset only selected active BOOK records on the isolated copy;
            # preserve all other tables, game logs and unrelated metadata.
            state["applied"].update(recs)
            for side in ("offense", "defense"):
                state["pending"].pop(side, None)
            _set_meta(target, "active_playbook_json", json.dumps(state))
            _set_meta(target, "ml_mode", "experimental")
            _set_meta(target, "ml_experimental_artifact", str(dest_model))
            _set_meta(target, MODE_DEFENSE, "off")
            # Keep legacy dataset/model metadata in original DB untouched.
            target.commit()
            _verify_db(target)
        report = {
            "source_db": str(src),
            "source_never_written": True,
            "source_db_sha256_at_restore": digest(src),
            "restored_db": str(out_db),
            "model_artifact": str(dest_model),
            "model_sha256": digest(dest_model),
            "model_version": EXPECTED_MODEL_VERSION,
            "offense_formations": list(recs["offense"]["formations"]),
            "offense_plays": EXPECTED_COUNTS["offense"][1],
            "defense_formations": list(recs["defense"]["formations"]),
            "defense_plays": EXPECTED_COUNTS["defense"][1],
            "offense_mode": "experimental",
            "defense_mode": "off (legacy heuristic)",
            "historical_record": provenance,
            "source_game_rows": counts,
        }
        (workdir / "RESTORE_REPORT.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("RESTORED ISOLATED DATABASE:", out_db)
        print("SOURCE UNCHANGED; UNIFIED V1 UNTOUCHED.")
        print("Next: python3 scripts/restore_lions_baseline.py --check --workdir", workdir)
        return 0


def check(args: argparse.Namespace) -> int:
    workdir = args.workdir.expanduser().resolve()
    db_path = workdir / "madden27-lions.db"
    if not db_path.is_file():
        raise RestoreError(f"No restored Lions DB found: {db_path}")
    manifest = load_manifest()
    with _read_only_source(db_path) as c:
        _verify_db(c)
        st = _read_state(c)
        for side in ("offense", "defense"):
            form = st["applied"][side]["formations"]
            if form != manifest["sides"][side]["formations"]:
                raise RestoreError(f"Restore inventory mismatch on {side}")
        if _get_meta(c, "ml_mode") != "experimental":
            raise RestoreError("Offense is not experimental")
        if _get_meta(c, MODE_DEFENSE) != "off":
            raise RestoreError("Defensive ML is still enabled")
        model_path = Path(_get_meta(c, "ml_experimental_artifact") or "")
    if not model_path.is_file():
        raise RestoreError("Restored model artifact is missing")
    from cfb_coach.madden.model.experimental_model import load_artifact
    from cfb_coach.db import CoachDB
    from cfb_coach.madden.playcaller import make_call
    from cfb_coach.madden.situation import parse_madden_situation

    model = load_artifact(model_path)
    if model.model_version != EXPECTED_MODEL_VERSION or model.n_supervised != EXPECTED_SUPERVISED:
        raise RestoreError("Restored offense artifact identity changed")
    # Check the actual old coordinator, not only loading a JSON file. The
    # user must not accidentally run an experimental mode that times out and
    # silently shows heuristic offense instead.
    scenarios = [
        "o 1&10 my 25", "o 2&8 my 43", "o 3&6 opp 45",
        "o 3&2 opp 7", "o 1&10 opp 22", "o 2&15 my 7",
        "o 3&16 my 48", "o 1&10 my 42", "o 2&3 opp 34", "o 1&10 opp 15",
        "d 1&10 opp 25", "d 2&6 opp 38", "d 3&6 opp 45", "d 3&11 my 48",
    ]
    delays = []
    fallback = []
    results = []
    db = CoachDB.open_read_only(db_path)
    try:
        for i, raw in enumerate(scenarios):
            sit = parse_madden_situation(raw, default_side="offense")
            started = time.perf_counter()
            call = make_call(sit, "cpu", db, live_macros=False, active_macros=[],
                             coach_cpu_defense=True)
            delays.append((time.perf_counter() - started) * 1000.0)
            if raw.startswith("o "):
                info = getattr(call, "ml_experimental", None)
                if not isinstance(info, dict) or info.get("fell_back") or not info.get("model_version"):
                    fallback.append({"scenario": raw, "info": info, "play": call.play})
            elif call.side != "defense" or not call.play:
                fallback.append({"scenario": raw, "unexpected_defense": call.side})
            results.append((raw, call.side, call.formation, call.play))
    finally:
        db.close()
    results = [(a, b, c, d) for a, b, c, d in results]
    for a, b, c, d in results:
        print(f"  {a:24} → {b}: {c} — {d}")
    print("OFFENSE ARTIFACT:", model.model_version, "| verified rows:", model.n_supervised)
    print("DEFENSE:", "legacy heuristic — 6 restored formations")
    print("SMOKE INFERENCE max_ms:", round(max(delays), 1))
    print("FALLBACKS:", len(fallback))
    if fallback:
        print(json.dumps(fallback, indent=2, default=str))
        print("NO GO: original ML offense fell back or defense did not remain heuristic.")
        return 2
    if max(delays) > 150:
        print("NO GO: one or more smoke requests exceeded 150ms; investigate before launch.")
        return 2
    print("LIONS RESTORE CHECK: PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-db", type=Path, default=Path.home() / ".cfb-coach" / "madden27.db",
                   help="Your existing franchise DB, read-only. Default: ~/.cfb-coach/madden27.db")
    p.add_argument("--workdir", type=Path, default=Path.home() / "lions-legacy-rollback",
                   help="Isolated restored game directory (never points into primary data dir).")
    p.add_argument("--historical-db", type=Path,
                   default=Path.home() / "Downloads" / "madden27.backup.20261105T013541Z.db",
                   help="Optional November 4 historical DB for original applied book records and audibles.")
    p.add_argument("--model", type=Path, default=None,
                   help="Exact 428-row empirical offense.json, only if not at its saved path.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true",
                      help="Clone the source DB using SQLite backup and modify only the copy.")
    mode.add_argument("--check", action="store_true",
                      help="Verify original ML vs legacy heuristic over representative game situations.")
    args = p.parse_args()
    try:
        if args.check:
            return check(args)
        return restore(args)
    except (RestoreError, sqlite3.Error, OSError, ValueError) as exc:
        print("LIONS RESTORE: NO GO —", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
