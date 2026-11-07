#!/usr/bin/env python3
"""Madden catalogue ingestion (AceMadden-compatible). Private store only; nothing third-party is written into the repository.

  discover   walk the PUBLIC listing pages (robots-respecting, rate-limited, no login) and summarise playbooks/formations. Writes nothing to the catalogue.
  sync       reconcile discovered public metadata into the store (never deletes; marks `not_seen`)   [--details adds playbook memberships]
  import     ingest an AUTHORIZED export (jsonl/csv) into the store                                 [needs --permission-note]
  art        download play art for catalogued plays (only with --permission-note)
  validate   identities, relationships, evidence rules, duplicate handling
  audit      coverage by side + discrepancies against the repository's own catalogue + permission status

Examples
  python scripts/madden_ml/ace_catalog.py discover --max-pages 2
  python scripts/madden_ml/ace_catalog.py sync --permission-note "AceMadden support email 2026-11-07: <paste>"
  python scripts/madden_ml/ace_catalog.py import --file ace_export.jsonl --permission-note "licence ref ..."
  python scripts/madden_ml/ace_catalog.py audit
Permission requests: support@acemadden.com. If the terms or the answer change, stop and re-check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfb_coach.madden import catalog as repo_catalog  # noqa: E402
from cfb_coach.madden.model import ace_catalog as ac  # noqa: E402

DEFAULT_STORE = Path.home() / "cfb_acemadden"


def cmd_authorize(a) -> int:
    store = ac.Store(a.store)
    try:
        rec = ac.record_authorization(store, sender=a.sender, date=a.date, scopes=[x for x in a.scopes.split(",") if x], restrictions=a.restrictions,
                                      private_reference=a.reference, summary=a.summary)
    except ValueError as exc:
        print(f"not recorded: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: rec[k] for k in ("sender", "date", "scopes", "restrictions")}, indent=1))
    return 0


def _permission(a, store=None) -> dict:
    auth = ac.load_authorization(store) if store is not None else None
    if auth:
        return {"status": "authorized", "scopes": auth["scopes"], "sender": auth["sender"], "date": auth["date"], "restrictions": auth["restrictions"],
                "contact": ac.PERMISSION_CONTACT}
    note = (getattr(a, "permission_note", None) or "").strip()
    return {"status": "owner_attested" if note else "public_metadata_only", "note": note, "contact": ac.PERMISSION_CONTACT,
            "terms_checked": "https://acemadden.com/terms (rev. 2026-06-03): copyright on play art; silent on automated collection/ML training"}


def _fetcher(a, store):
    auth = ac.load_authorization(store)
    return ac.Fetcher(store, delay=a.delay, permission_note=(a.permission_note or (auth["summary"] if auth else None)))


def cmd_discover(a) -> int:
    store = ac.Store(a.store)
    f = _fetcher(a, store)
    recs, quarantine, stats = ac.discover(f, game=a.game, max_pages=a.max_pages, permission=_permission(a, store))
    forms = {}
    for r in recs.values():
        forms.setdefault((r["side"], r["formation_slug"]), 0)
        forms[(r["side"], r["formation_slug"])] += 1
    print(json.dumps({"requests": f.requests, **stats, "formations": {s: sum(1 for k in forms if k[0] == s) for s in ac.SIDES}}, indent=2))
    (store.root / "discovered.json").write_text(json.dumps({"stats": stats, "ids": sorted(recs)}), encoding="utf-8")
    print("catalogue NOT modified (use `sync`).")
    return 0


def cmd_sync(a) -> int:
    store = ac.Store(a.store)
    f = _fetcher(a, store)
    ac.require_scope(store, "metadata_bulk") if a.max_pages is None else None   # bulk sync needs the scope; small samples do not
    perm = _permission(a, store)
    recs, quarantine, stats = ac.discover(f, game=a.game, max_pages=a.max_pages, permission=perm)
    if a.details:
        for r in recs.values():
            try:
                d = ac.parse_detail(f.detail(r))
            except (FileNotFoundError, PermissionError):
                continue
            if d.get("formation_label"):
                r["formation_label"] = d["formation_label"]
            if d.get("playbooks_text_unverified"):
                r["source"]["detail_hint_playbooks_unverified"] = d["playbooks_text_unverified"]
            r["content_hash"] = ac.content_hash(r)
    errs = [e for e in ac.validate(recs) if not e.startswith("info:")]
    if errs:
        print("validation failed; nothing written:\n  " + "\n  ".join(errs[:20]), file=sys.stderr)
        return 1
    for q in quarantine:
        store.append("quarantine.jsonl", q)
    print(json.dumps(ac.sync(store, recs, scope_sides=ac.SIDES), indent=2), f"quarantined={len(quarantine)}")
    return 0


def cmd_import(a) -> int:
    if not a.permission_note and not ac.load_authorization(ac.Store(a.store)):
        print("import needs --permission-note (the licence/authorisation this export came under)", file=sys.stderr)
        return 2
    store = ac.Store(a.store)
    recs, bad = ac.import_export(a.file, game=a.game, version=a.version, permission=_permission(a, store))
    errs = [e for e in ac.validate(recs) if not e.startswith("info:")]
    if errs:
        print("validation failed; nothing written:\n  " + "\n  ".join(errs[:20]), file=sys.stderr)
        return 1
    for q in bad:
        store.append("quarantine.jsonl", q)
    sides = {r["side"] for r in recs.values()}
    print(json.dumps(ac.sync(store, recs, scope_sides=sides), indent=2), f"rejected_rows={len(bad)}")
    return 0


def cmd_art(a) -> int:
    store = ac.Store(a.store)
    ac.require_scope(store, "images_download")
    ac.require_scope(store, "images_local_storage")
    f = _fetcher(a, store)
    recs = store.read()
    done = 0
    for k, r in sorted(recs.items()):
        if r.get("art_sha256") or not r.get("art_url") or (a.side and r["side"] != a.side):
            continue
        try:
            png = f.art(r)
        except PermissionError:
            store.write_atomic(recs)          # keep progress, then stop: 401/403 is never bypassed
            raise
        except Exception as e:                 # one bad image must not lose the run; record the reason, retry on rerun
            r["art_error"] = f"{type(e).__name__}: {e}"[:200]
            failed = locals().get("failed", 0) + 1
            continue
        r.pop("art_error", None)
        h = hashlib.sha256(png).hexdigest()
        (store.root / "raw").mkdir(exist_ok=True)
        (store.root / "raw" / f"{r['side']}__{r['formation_slug']}__{r['play_slug']}.png").write_bytes(png)
        r["art_sha256"], r["permitted_use"] = h, _permission(a, store)
        r["content_hash"] = ac.content_hash(r)
        done += 1
        if done % 200 == 0:                    # checkpoint so an interruption loses at most 200 hashes
            store.write_atomic(recs)
            print(f"checkpoint {done} downloaded", flush=True)
        if a.limit and done >= a.limit:
            break
    store.write_atomic(recs)
    print(f"downloaded {done} images into {store.root / 'raw'} (private; not for redistribution)")
    return 0


def cmd_validate(a) -> int:
    errs = ac.validate(ac.Store(a.store).read())
    for e in errs:
        print(e)
    print(f"{sum(not e.startswith('info:') for e in errs)} errors, {sum(e.startswith('info:') for e in errs)} notes")
    return 1 if any(not e.startswith("info:") for e in errs) else 0


def _reference():
    return {s: [(f, p) for b in repo_catalog.book_names(s) for f, ps in repo_catalog.book_formations(s, b).items() for p in ps] for s in ac.SIDES}


def cmd_audit(a) -> int:
    store = ac.Store(a.store)
    recs = store.read()
    perm = next((r["permitted_use"] for r in recs.values()), {"status": "no data imported yet"})
    rep = ac.audit(recs, reference=_reference(), permission=perm)
    if not recs:
        rep["NOTE"] = "the store is empty: full extraction has NOT occurred. Run `sync` (public metadata) or `import` (authorized export)."
    print(json.dumps(rep, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=str(DEFAULT_STORE), help="private data directory (refused inside a git work tree)")
    ap.add_argument("--game", default="madden27")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests (be gentle)")
    ap.add_argument("--permission-note", default="", help="record of the written permission you hold (stored with every record)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, fn in (("discover", cmd_discover), ("sync", cmd_sync), ("import", cmd_import), ("art", cmd_art), ("validate", cmd_validate), ("audit", cmd_audit), ("authorize", cmd_authorize)):
        p = sub.add_parser(n)
        p.set_defaults(fn=fn)
        if n in ("discover", "sync"):
            p.add_argument("--max-pages", type=int, default=None)
        if n == "sync":
            p.add_argument("--details", action="store_true")
        if n == "import":
            p.add_argument("--file", required=True)
            p.add_argument("--version", default="")
        if n == "authorize":
            p.add_argument("--sender", required=True); p.add_argument("--date", required=True)
            p.add_argument("--scopes", required=True, help="comma list of what the email grants: " + ",".join(ac.SCOPES))
            p.add_argument("--restrictions", default=""); p.add_argument("--reference", required=True, help="private pointer to the email (not the email text)")
            p.add_argument("--summary", required=True, help="your own truthful summary of the grant")
        if n == "art":
            p.add_argument("--side", choices=ac.SIDES)
            p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
