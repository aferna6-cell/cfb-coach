"""Read-only human-game execution audit with the owner's attestation as a separate tier.

    python scripts/madden_ml/audit_human_attestation.py --db AUDIT_COPY.db --out ~/.cfb-research/attestation

Writes ``audit.json`` (summary with real opponent ids), ``rows.jsonl`` (one row per
snap) and ``summary_anonymized.json`` (opponents as human_1..N, safe to share).
Refuses to write inside the live data directory. Never writes the database.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import attestation_audit as aa  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from cfb_coach.games import data_dir

    out_root = Path(args.out).expanduser().resolve()
    live = Path(data_dir()).expanduser().resolve()
    if out_root == live or live in out_root.parents:
        raise SystemExit(f"refusing to write inside the live data directory {live}")
    conn = aa.connect_readonly(str(Path(args.db).expanduser().resolve()))
    try:
        rows = aa.audit_rows(conn)
    finally:
        conn.close()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / f"run-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "audit.json").write_text(json.dumps(aa.summarize(rows), indent=2), encoding="utf-8")
    alias = aa.opponent_aliases(rows)
    (out / "summary_anonymized.json").write_text(
        json.dumps(aa.summarize(rows, alias=alias), indent=2), encoding="utf-8")
    with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in aa.as_dicts(rows):
            fh.write(json.dumps(r, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
