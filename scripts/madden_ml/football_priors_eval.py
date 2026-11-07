#!/usr/bin/env python3
"""Held-out, whole-game, chronological comparison: name-only (current) vs structured football priors vs priors + learned cells.

  python scripts/madden_ml/football_priors_eval.py --db ~/madden27_copy.db --out OUT      # your database COPY (opened read-only)
Prints a champion/challenger gate. KEEP_CHAMPION is the default: promotion needs the whole-game interval to clear zero."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import football_priors_eval as fe  # noqa: E402
from cfb_coach.madden.model import offense_readiness as rd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    live = (Path.home() / ".cfb-coach").resolve()
    out = Path(a.out).expanduser().resolve()
    if out == live or live in out.parents or Path(a.db).expanduser().resolve().is_relative_to(live):
        print("refusing: use a COPY of the database and an output directory outside ~/.cfb-coach", file=sys.stderr)
        return 2
    data = rd.load_rows(a.db)
    rep = fe.evaluate(data.rows, data.game_order, n_boot=a.boot)
    rep["source"] = f"real database copy: {a.db}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "football_priors_eval.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("games", "rows", "name_only", "structured_prior", "structured_learned", "gate")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
