#!/usr/bin/env python3
"""Read-only defensive V2 testing-franchise preflight.

Does not change database, install a playbook, enable the model or promote
weights. A PASS is a source/configuration check, NOT proof of Xbox installation
or 150ms end-to-end runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfb_coach.db import CoachDB
from cfb_coach.madden import playbook
from cfb_coach.madden.model import defense_v2_live as v2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-db", type=Path, required=True)
    p.add_argument("--franchise", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--installed-fingerprint", required=True)
    p.add_argument("--roles-json", type=Path)
    p.add_argument("--confirmed-actions-json", type=Path)
    args = p.parse_args()
    errors = []
    src = args.test_db.expanduser().resolve()
    home = (Path.home() / ".cfb-coach").resolve()
    if not src.is_file() or src == home or home in src.parents:
        p.error("explicit existing isolated test DB outside ~/.cfb-coach required")
    db = CoachDB.open_read_only(src)
    try:
        sessions = db.conn.execute("SELECT COUNT(*) FROM game_sessions WHERE dynasty=?",
                                   (args.franchise,)).fetchone()[0]
        if not sessions:
            errors.append("no recorded test-franchise session")
        try:
            applied = playbook.active_books(db, ("defense",))["defense"]
            book = applied["formations"]
            stock = sum(len(plays) for plays in book.values())
            hash_now = v2.installed_fingerprint(book)
            if hash_now != args.installed_fingerprint:
                errors.append("confirmed installed-book fingerprint differs from applied book")
            if len(book) > 15 or not book:
                errors.append("no installed defensive book or formation cap exceeded")
        except Exception as exc:
            errors.append("recorded defensive applied book unavailable: " + str(exc))
            book = {}
            hash_now = None
            stock = 0
    finally:
        db.close()
    try:
        model = v2.load_model(args.model)
        if model.get("franchise") != args.franchise or int(model.get("n_verified") or 0) < 1:
            errors.append("V2 model is not trained on a verified execution from this franchise")
        if not model.get("trained_sessions") or not model.get("trained_through_started_ts"):
            errors.append("V2 model lacks training chronology")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append("model not ready: " + str(exc))
    if args.roles_json:
        from cfb_coach.madden.model.defense_v2_roles import verify_document
        try:
            verify_document(json.loads(args.roles_json.read_text(encoding="utf-8")))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append("verified role ledger invalid: " + str(exc))
    if args.confirmed_actions_json:
        from cfb_coach.madden.model.defense_v2_actions import read_confirmed_options
        try:
            read_confirmed_options(args.confirmed_actions_json)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append("verified Xbox controls invalid: " + str(exc))
    report = {
        "schema": "madden.defense.v2.preflight.v1",
        "status": "LOCAL_CONFIG_PASS_ONLY" if not errors else "NO_GO",
        "franchise": args.franchise,
        "isolated_test_db": str(src),
        "recorded_formations": len(book),
        "recorded_stock_plays": stock,
        "recorded_inventory_fingerprint": hash_now,
        "owner_confirmed_fingerprint": args.installed_fingerprint,
        "errors": errors,
        "xbox_installation_independently_verified": False,
        "150ms_live_loaded_benchmark_complete": False,
        "policy_value_validated_in_games": False,
        "does_not_activate": True,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
