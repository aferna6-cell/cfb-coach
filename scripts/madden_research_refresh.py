#!/usr/bin/env python3
"""Standalone Madden research-refresh entry (works on main without ``ml`` CLI).

Usage:
  PYTHONPATH=. python scripts/madden_research_refresh.py
  PYTHONPATH=. python scripts/madden_research_refresh.py --dry-run

Prints one JSON object to stdout for GitHub Actions / operators.
Never overwrites active research or gameplay settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as scripts/... without install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Madden 27 research candidate refresh")
    p.add_argument("--dry-run", action="store_true", help="Fetch/extract but write nothing")
    args = p.parse_args(argv)
    from cfb_coach.madden.research_refresh import run_research_refresh

    result = run_research_refresh(dry_run=bool(args.dry_run))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
