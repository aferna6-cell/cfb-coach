"""Train the VOD success model into the box output layout.

Example::

    python -m cfb_coach.vod_success train snaps/ --models-dir models --label-version v0.7
"""

from __future__ import annotations

import argparse
import sys

from cfb_coach.vod_success.train import load_rows, train_and_write


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
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
