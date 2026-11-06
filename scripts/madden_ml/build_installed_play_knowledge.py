"""Build a NEW research-only database for the recorded installed offense/defense.

Example: python scripts/madden_ml/build_installed_play_knowledge.py \
  --output /tmp/madden27-installed-play-knowledge.db

Never modifies the live Madden game DB, installation or playcaller.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from cfb_coach.madden.model.installed_playbook_catalog import (
    annotation_queue, import_installed, installed_coverage, write_queue_csv,
)
from cfb_coach.madden.model.play_knowledge import (
    ConceptTag, KnowledgeError, PlayKnowledgeDB, PlayerJob, play_key,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "cfb_coach" / "data" / "madden27"


def _safe_output(file: Path, parser: argparse.ArgumentParser) -> Path:
    file = file.expanduser().resolve()
    live = (Path.home() / ".cfb-coach").resolve()
    if file == live or live in file.parents:
        parser.error("generated research files must be outside ~/.cfb-coach")
    if file.exists():
        parser.error(f"file exists; refusing to overwrite: {file}")
    return file


def _insert_annotations(db: PlayKnowledgeDB, annotations: dict) -> dict[str, int]:
    if annotations.get("schema") != "madden27.play_knowledge.seed.v1":
        raise KnowledgeError("unexpected source annotation schema")
    count = {"offense": 0, "defense": 0}
    for row in annotations.get("annotations") or []:
        side = row["side"]
        source, evidence = str(row["source"]), str(row.get("evidence") or "")
        if not source.startswith("https://") or evidence != "source_documented":
            raise KnowledgeError("installed annotations must have sourced, unverified Madden 27 page")
        key = play_key(side, row["book"], row["formation"], row["play"])
        if db.get(key) is None:
            raise KnowledgeError(f"source annotation not in installed snapshot: {side}/{row['formation']}/{row['play']}")
        jobs = [PlayerJob(j["position"], j["kind"], j["task"], evidence, source, dict(j.get("params") or {}))
                for j in row.get("jobs") or []]
        concepts = [ConceptTag(c, evidence, source) for c in row.get("concepts") or []]
        reads = row.get("read_order") or []
        db.annotate(key, jobs, read_order=reads, read_source=(source if reads else ""), concepts=concepts)
        count[side] += 1
    return count


def _history_counts(file: Path) -> dict[tuple[str,str,str],int]:
    """Optional *logged call frequency* is not executed-play verification."""
    uri = file.expanduser().resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        return {(side, form, play): n for side,form,play,n in con.execute("""
            SELECT side,formation,play,COUNT(*) AS n FROM snaps
            WHERE side IN ('offense','defense') AND formation IS NOT NULL AND play IS NOT NULL
            GROUP BY side,formation,play""")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DATA / "installed_playbook_snapshot.json")
    parser.add_argument("--source-catalog", type=Path, default=DATA / "playbooks_full.json",
                        help="existing full source-book catalog for checking source/formation membership")
    parser.add_argument("--annotations", type=Path, default=DATA / "installed_play_annotations.json")
    parser.add_argument("--no-annotations", action="store_true")
    parser.add_argument("--history-db", type=Path, help="OPTIONAL read-only game DB to prioritize review by logged frequency")
    parser.add_argument("--output", type=Path, required=True, help="NEW research SQLite database, not Madden live DB")
    parser.add_argument("--queue", type=Path, help="CSV to verify all installed plays and unknown assignments")
    parser.add_argument("--report", type=Path, help="machine-readable build summary JSON")
    args = parser.parse_args()
    out = _safe_output(args.output, parser)
    queue = _safe_output(args.queue or out.with_suffix(".review.csv"), parser)
    report = _safe_output(args.report or out.with_suffix(".report.json"), parser)
    if len({out,queue,report}) != 3:
        parser.error("output, queue and report files must be distinct")
    if not args.snapshot.is_file():
        parser.error(f"missing installed snapshot: {args.snapshot}")
    payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
    full = (json.loads(args.source_catalog.read_text(encoding="utf-8"))
            if args.source_catalog.is_file() else None)
    history = _history_counts(args.history_db) if args.history_db else {}
    if args.no_annotations:
        ann = None
    else:
        if not args.annotations.is_file():
            parser.error(f"missing source annotations: {args.annotations}")
        ann = json.loads(args.annotations.read_text(encoding="utf-8"))
    # Guarantee no overwriting the user's files. New DB is kept separate.
    out.parent.mkdir(parents=True, exist_ok=True)
    queue.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    try:
        with PlayKnowledgeDB(out) as db:
            status = import_installed(db, payload, full_catalog=full)
            sourced = _insert_annotations(db, ann) if ann else {"offense": 0, "defense": 0}
            rows = annotation_queue(db)
            for row in rows:
                row["historical_logged_calls_not_verified_executions"] = history.get(
                    (row["side"], row["formation"], row["play"]), 0)
            rows.sort(key=lambda x: (-x["historical_logged_calls_not_verified_executions"],
                                     x["side"], x["formation"], x["play"]))
            status["source_annotated_plays"] = sourced
            status["knowledge_coverage"] = installed_coverage(db)
            status["queue_rows"] = len(rows)
            status["source_catalog_found"] = full is not None
            status["annotation_notes"] = "Published route/coverage descriptions are not Xbox-verified; unannotated player assignments are unknown."
            status["recorded_execution_status"] = "Historical call counts are NOT proof of executed plays."
        write_queue_csv(rows, queue)
        report.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status, indent=2))
        print(f"Research DB: {out}\nReview queue: {queue}\nReport: {report}")
    except Exception:
        # Failure must not leave an apparently complete DB or report.
        out.unlink(missing_ok=True)
        queue.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
        raise

if __name__ == "__main__":
    main()