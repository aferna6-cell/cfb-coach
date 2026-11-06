#!/usr/bin/env python3
"""Offline A/B/C/D evaluation of the structure-aware offensive coordinator.

Read-only on the game DB (open a COPY). Writes the report OUTSIDE ~/.cfb-coach.
    python scripts/madden_ml/offense_structural_eval.py --db /path/to/coach_copy.db --out /tmp/offense-eval
With fewer than 8 verified offense games the report states that no claim is possible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import heldout_eval as he  # noqa: E402
from cfb_coach.madden.model import offense_structural_eval as ev  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to a COPY of the Madden game database (opened read-only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-train-games", type=int, default=he.DEFAULT_MIN_TRAIN_GAMES)
    ap.add_argument("--bootstrap", type=int, default=he.DEFAULT_BOOTSTRAP)
    ap.add_argument("--no-behavior", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).expanduser().resolve()
    live = (Path.home() / ".cfb-coach").resolve()
    if out == live or live in out.parents:
        print("refusing to write inside the live data directory", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    data = he.load_eval_data(a.db)
    rep = ev.evaluate(data, min_train_games=a.min_train_games, n_boot=a.bootstrap, run_behavior=not a.no_behavior)
    (out / "offense_structural_eval.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (out / "offense_structural_eval.md").write_text(ev.render_markdown(rep), encoding="utf-8")
    print(ev.render_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
