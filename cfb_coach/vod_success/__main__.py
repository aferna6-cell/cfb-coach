"""Train the VOD success model into the box output layout.

Example::

    python -m cfb_coach.vod_success train snaps/ --models-dir models --label-version v0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cfb_coach.vod_success.concepts.taxonomy import build_taxonomy, write_taxonomy
from cfb_coach.vod_success.concepts.train import train_and_write as train_concepts
from cfb_coach.vod_success.train import load_rows, train_and_write


def _read_label_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for raw in paths:
        path = Path(raw)
        files = sorted(p for p in path.rglob("*") if p.suffix in {".json", ".jsonl"}) if path.is_dir() else [path]
        for file in files:
            text = file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            if file.suffix == ".jsonl":
                rows.extend(json.loads(line) for line in text.splitlines() if line.strip())
                continue
            payload = json.loads(text)
            if isinstance(payload, list):
                rows.extend(item for item in payload if isinstance(item, dict))
            elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                rows.extend(item for item in payload["rows"] if isinstance(item, dict))
            elif isinstance(payload, dict):
                rows.append(payload)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfb_coach.vod_success",
        description=(
            "Train a play-call success model from VOD snaps. "
            "Writes model_params.json, metrics.json, beaters.json, metrics.md, "
            "and manifest.json under <models-dir>/<label version>/, "
            "and updates <models-dir>/LAST_TRAINED.json. "
            "Does not change prep or live calls."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train", help="Fit strata and write the v0.7 output files")
    train.add_argument("paths", nargs="+", help="CSV, JSON, or JSONL batches, or a directory of them")
    train.add_argument("--models-dir", required=True, help="Directory that holds LAST_TRAINED.json and version folders")
    train.add_argument("--label-version", required=True, help="Label version, for example v0.7")
    train.add_argument("--manifest", help="JSONL manifest with file, opponent_type, and game_id")
    train.add_argument("--labels-dir", help="Label directory recorded in manifest.json")
    train.add_argument("--latest-json", help="Optional LATEST json to hash into the manifest")
    train.add_argument("--snaps-notes", help="Optional notes file to hash into the manifest")
    concepts = sub.add_parser("concepts", help="Build a concept taxonomy or train the concept model")
    concept_cmds = concepts.add_subparsers(dest="concept_cmd", required=True)
    build = concept_cmds.add_parser("build", help="Map call names to concepts and write vod_concepts.v1")
    build.add_argument("--version", required=True, help="Taxonomy version, for example v1")
    build.add_argument("--out", required=True, help="Directory for <version>.json, the unmapped list, and LATEST.json")
    build.add_argument("--labels", nargs="+", required=True, help="JSON or JSONL snaps")
    build.add_argument("--label-version", default="v0", help="Label version recorded in the taxonomy")
    fit = concept_cmds.add_parser("train", help="Fit P(success | situation, concept) and write concept/c<N>/")
    fit.add_argument("--models-dir", required=True, help="Directory that will hold concept/LAST_TRAINED.json and concept/c<N>/")
    fit.add_argument("--concepts", required=True, help="Taxonomy JSON (vod_concepts.v1)")
    fit.add_argument("--labels", nargs="+", required=True, help="JSON or JSONL snaps with a known result")
    fit.add_argument("--label-version", required=True, help="Label version recorded in the model")
    fit.add_argument("--version", help="Force the output name, for example c1")
    fit.add_argument("--force", action="store_true", help="Rebuild even when labels, concepts, and script match")
    args = parser.parse_args(argv)
    if args.cmd == "concepts":
        return _concepts(args)
    rows = load_rows(args.paths, manifest=args.manifest)
    written = train_and_write(
        rows,
        args.models_dir,
        label_version=args.label_version,
        input_paths=args.paths,
        labels_dir=args.labels_dir,
        latest_json_path=args.latest_json,
        snaps_notes_path=args.snaps_notes,
    )
    metrics = written["metrics"]
    print(f"Read {metrics['counts'] and sum(c['rows'] for c in metrics['counts'].values())} snaps.")
    for name, counts in metrics["counts"].items():
        trained = "trained" if counts["trained"] else "not trained"
        print(
            f"{name}: {counts['rows']} rows, {counts['known_success_rows']} known, {trained}."
        )
    print(f"Wrote {written['model_dir']}")
    print(f"Updated {args.models_dir}/LAST_TRAINED.json")
    return 0


def _concepts(args: argparse.Namespace) -> int:
    rows = _read_label_rows(args.labels)
    if args.concept_cmd == "build":
        document, unmapped = build_taxonomy(rows, version=args.version, label_version=args.label_version)
        path = write_taxonomy(args.out, document, unmapped)
        print(f"Wrote {path} ({len(unmapped)} unmapped names)")
        return 0
    taxonomy = json.loads(Path(args.concepts).read_text(encoding="utf-8"))
    written = train_concepts(
        rows,
        args.models_dir,
        taxonomy,
        label_version=args.label_version,
        concepts_path=args.concepts,
        label_paths=args.labels,
        force=args.force,
        version=args.version,
    )
    if written.get("unchanged"):
        print(f"UNCHANGED {written['version']}")
        return 0
    print(f"TRAINED {written['version']} -> {written['model_dir']}")
    print(f"Updated {args.models_dir}/concept/LAST_TRAINED.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
