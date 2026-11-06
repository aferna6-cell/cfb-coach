"""Train a defense logistic v2 candidate and gate it against the current artifact.

Read-only on the database. Writes only ``--out`` and ``--report``; never the
current artifact. Promotion is a separate, manual step.

Example::

    python scripts/madden_ml/train_defense_logistic.py --db ~/.cfb-coach/madden27.db \\
      --current ~/.cfb-coach/models/defense_v2.json \\
      --out ~/.cfb-coach/models/defense_v3_candidate.json \\
      --unattested-after 2026-11-03T00:00:00
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

# An editable install may point at a different checkout; train with this one.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cfb_coach.madden.model import defense_logistic_train as train
from cfb_coach.madden.model import defense_v1


def _attestations(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        session, sep, note = value.partition("=")
        if not sep or not session.strip() or not note.strip():
            raise SystemExit(f"--attest needs SESSION=NOTE, got {value!r}")
        out[session.strip()] = note.strip()
    return out


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--current", type=Path, required=True, help="live artifact to beat (read only)")
    p.add_argument("--out", type=Path, required=True, help="candidate path; must not exist")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--attest", action="append", default=[], metavar="SESSION=NOTE",
                   help="record the user's confirmation of logged outcomes for audit; "
                        "does NOT verify the executed defensive call or admit it for training")
    p.add_argument("--unattested-after", default=None, metavar="ISO",
                   help="legacy audit cutoff for logged calls; training always requires verified execution")
    p.add_argument("--current-trained-before", default=None, metavar="ISO",
                   help="sessions started before this are flagged in-sample for --current "
                        "(default: --current file mtime)")
    args = p.parse_args(argv)

    out = args.out.expanduser()
    current_path = args.current.expanduser()
    if out.resolve() == current_path.resolve():
        raise SystemExit("--out must not be the current artifact")
    if out.exists():
        raise SystemExit(f"--out already exists: {out}")
    attested = _attestations(args.attest)
    current = defense_v1.load_artifact(current_path)
    trained_before = args.current_trained_before or dt.datetime.fromtimestamp(
        current_path.stat().st_mtime, tz=dt.timezone.utc).isoformat()

    conn = sqlite3.connect(f"file:{args.db.expanduser()}?mode=ro", uri=True)
    try:
        rows = train.collect_rows(conn, attested=attested, unattested_after=args.unattested_after)
    finally:
        conn.close()
    missing = sorted(set(attested) - {r.session_id for r in rows})
    if missing:
        raise SystemExit(f"attested sessions have no scorable defensive snaps: {missing}")
    labels = train.label_calls(rows, current)
    unlabeled = [r for r in rows if (r.formation, r.play) not in labels]
    rows = [r for r in rows if (r.formation, r.play) in labels]
    admitted = [r for r in rows if r.admitted]
    games = {r.session_id for r in admitted}
    if len(games) < 2:
        raise SystemExit(
            f"Only {len(admitted)} verified defensive executions across {len(games)} games. "
            "Need verified calls from at least two games for safe, game-separated training. "
            "Logged recommendations and score attestations cannot replace execution proof; "
            "no model was written."
        )
    # Neither training nor play-specific holdout evaluation may attribute
    # results to an unverified displayed recommendation.
    in_sample = {r.session_id for r in admitted if r.started_ts < trained_before}
    report = train.leave_one_game_out(admitted, labels, current, current, current_in_sample=in_sample)
    verdict = train.gate(report)
    lam = train.choose_lambda(admitted, labels, current)
    candidate = train.fit_artifact(admitted, labels, current, lam)
    defense_v1.validate_artifact(candidate)
    candidate["training"] = {
        **train.training_summary(rows, attested, args.unattested_after, lam),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_commit": _commit(),
        "trainer": "cfb_coach.madden.model.defense_logistic_train",
        "label_source_artifact": str(current_path),
        "unlabeled_snaps_dropped": len(unlabeled),
        "evaluation": {
            "split": "leave_one_game_out",
            "current_artifact": str(current_path),
            "current_in_sample_sessions": sorted(in_sample),
            "gate": verdict,
            "all_games": report["all_games"],
            "current_out_of_sample": report["current_out_of_sample"],
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    full = {"candidate": str(out), "gate": verdict, "evaluation": report,
            "training": candidate["training"]}
    if args.report:
        args.report.expanduser().write_text(json.dumps(full, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(full, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
