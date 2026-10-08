#!/usr/bin/env python3
"""Write GitHub Actions outputs from a research-refresh JSON result file.

Usage:
  PYTHONPATH=. python -m cfb_coach ml research-refresh > /tmp/research_refresh.json
  python scripts/research_refresh_gha_outputs.py /tmp/research_refresh.json >> "$GITHUB_OUTPUT"

Avoids fragile inline JSON parsing in shell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    path = Path(argv[0] if argv else "/tmp/research_refresh.json")
    try:
        raw = path.read_text(encoding="utf-8")
        # Allow tee'd stdout that may contain non-JSON prefixes: take last JSON object.
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0:
            raise ValueError("no JSON object in file")
        data = json.loads(raw[start : end + 1])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ok=False")
        print(f"candidate=")
        print(f"report=")
        print(f"error={type(exc).__name__}:{exc}", file=sys.stderr)
        return 1
    ok = bool(data.get("ok"))
    print(f"ok={ok}")
    print(f"candidate={data.get('candidate_path') or ''}")
    print(f"report={data.get('change_report_path') or ''}")
    print(f"extracted={len(data.get('extracted_findings') or [])}")
    print(f"diffs={len(data.get('contradictions_or_changes') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
