"""The ONE legal-candidate filter for civil_scheme_only. Runs before scoring and again before display.

A variant is LIVE-legal only if every check passes, in this order:
  1. catalog      it is one of the 26 catalog variant IDs (red-zone packages / the unnamed 5WR setup never are)
  2. stock        its formation and base play exist, by exact normalized name, in the in-repo Buccaneers stock book
  3. book         the formation and base play are in the APPLIED (installed) offense book of record
  4. executable   the owner confirmed the variant's adjustments are executable. Aidan confirmed ALL catalog
                  adjustments for this catalog version on 2026-11-10 (t333u), recorded in
                  ``adjustments.OWNER_CONFIRMED_ADJUSTMENTS``; a different catalog version needs a new
                  confirmation (per variant: ``ml civil-scheme verify``).
Shadow evaluation (offline only) may skip 4 and is always labeled SHADOW.
Nothing here substitutes a different play, drops an adjustment, or offers an unadjusted version
(``unadjusted_allowed`` is False for all 26).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from cfb_coach.madden import catalog as madden_catalog
from cfb_coach.madden.scheme.catalog import SchemeCatalog, SchemeVariant

VERIFIED_META_KEY = "civil_scheme_verified_json"
STOCK_BOOK = "Buccaneers"
NO_LEGAL = "NO_LEGAL_SCHEME_PLAY"


@dataclass
class Verdict:
    variant_id: str
    legal: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"variant_id": self.variant_id, "legal": self.legal, "reasons": list(self.reasons),
                "checks": dict(self.checks)}


def stock_has(formation: str, play: str, book: str = STOCK_BOOK) -> tuple[bool, bool]:
    """(formation exists, play exists in that formation) by exact normalized name. No aliases."""
    forms = madden_catalog.book_formations("offense", book)
    fkey = {madden_catalog.norm(f): f for f in forms}
    f = fkey.get(madden_catalog.norm(formation))
    if f is None:
        return False, False
    return True, madden_catalog.norm(play) in {madden_catalog.norm(p) for p in forms[f]}


def book_names(book: Mapping[str, Any] | None, formation: str, play: str) -> tuple[str | None, str | None]:
    """The applied book's own spelling of (formation, play), matched by exact normalized name (no aliases)."""
    n = madden_catalog.norm
    forms = dict((book or {}).get("formations") or {})
    f = next((k for k in forms if n(k) == n(formation)), None)
    if f is None:
        return None, None
    return f, next((q for q in forms[f] or [] if n(q) == n(play)), None)


def _book_has(book: Mapping[str, Any] | None, formation: str, play: str) -> tuple[bool, bool]:
    f, q = book_names(book, formation, play)
    return f is not None, q is not None


def load_verified(db: Any) -> dict[str, dict[str, Any]]:
    if db is None:
        return {}
    raw = db.get_meta(VERIFIED_META_KEY)
    if not raw:
        return {}
    try:
        return {str(k): dict(v) for k, v in dict(json.loads(raw)).items()}
    except (ValueError, TypeError):
        return {}


def record_verified(db: Any, cat: SchemeCatalog, variant_id: str, *, note: str = "", revoke: bool = False,
                    book_play: str | None = None) -> dict[str, Any]:
    """Owner-only (CLI). Marks one exact variant executable; requires it to pass checks 1-3 first.

    ``book_play``: the owner's confirmation that the guide's base-play name appears in the in-game Buccaneers
    book under a DIFFERENT name (e.g. guide "Z Spot" = book "L Spot"). Only an owner confirmation can create
    such an alias; it is stored with the verification and shown on every call."""
    v = cat.get(variant_id)
    if v is None:
        raise ValueError(f"{variant_id} is not one of the {len(cat.variants)} catalog variants")
    led = load_verified(db)
    if revoke:
        led.pop(variant_id, None)
    else:
        verdict = check(v, cat, db=db, book=None, require_verified=False, alias=book_play)
        if not verdict.legal:
            raise ValueError(f"cannot verify {variant_id}: " + "; ".join(verdict.reasons))
        led[variant_id] = {"book_play": book_play or None,"verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "by": "owner_cli",
                           "catalog_version": cat.version(), "note": note,
                           "formation": v.formation, "base_play": v.base_play,
                           "adjustments": [a.as_dict() for a in v.adjustments]}
    db.set_meta(VERIFIED_META_KEY, json.dumps(led, sort_keys=True))
    return led


def _applied_offense_book(db: Any) -> dict[str, Any] | None:
    if db is None:
        return None
    from cfb_coach.madden import playbook
    return playbook.load_books(db).get("offense")


def check(v: SchemeVariant | None, cat: SchemeCatalog, *, db: Any = None, book: Mapping[str, Any] | None = None,
          require_verified: bool = True, verified: Mapping[str, Any] | None = None,
          variant_id: str | None = None, alias: str | None = None) -> Verdict:
    vid = v.variant_id if v is not None else str(variant_id or "")
    out = Verdict(vid, False)
    if v is None or cat.get(vid) is None or vid in cat.blocked_ids:
        out.checks["catalog"] = False
        out.reasons.append(f"{vid or '?'} is not one of the catalog's scheme variants (red-zone packages and the "
                           "unnamed 5WR setup are never callable)")
        return out
    out.checks["catalog"] = True
    if v != cat.get(vid):
        out.reasons.append("variant content differs from the catalog entry (no edited / substituted variants)")
        out.checks["catalog"] = False
        return out
    led = verified if verified is not None else (load_verified(db) if (require_verified or alias is None) else {})
    alias = alias or (led.get(vid) or {}).get("book_play")
    play = alias or v.base_play
    if alias:
        out.checks["owner_confirmed_name_alias"] = True
    f_ok, p_ok = stock_has(v.formation, play)
    out.checks["stock_formation"], out.checks["stock_play"] = f_ok, p_ok
    if not f_ok:
        out.reasons.append(f"formation '{v.formation}' is not in the in-repo {STOCK_BOOK} stock book")
    elif not p_ok:
        out.reasons.append(f"base play '{play}' is not in {STOCK_BOOK} '{v.formation}' in the repo's stock data "
                           "(exact name; only an owner-confirmed book name via `verify --book-play` is accepted)")
    if book is None:
        book = _applied_offense_book(db)
    if book is None:
        out.checks["book"] = False
        out.reasons.append("no applied offense book of record (run prep / prepare-book, install it, then prep --mark-applied)")
    else:
        bf, bp = _book_has(book, v.formation, play)
        out.checks["book_formation"], out.checks["book_play"] = bf, bp
        out.checks["book"] = bf and bp
        if not bf:
            out.reasons.append(f"formation '{v.formation}' is not in the applied offense book")
        elif not bp:
            out.reasons.append(f"'{play}' is not in the applied book's '{v.formation}'")
    if require_verified:
        rec = led.get(vid)
        per_variant = bool(rec) and rec.get("catalog_version") == cat.version()
        blanket = owner_confirmation_applies(cat)
        out.checks["owner_verified"] = per_variant or blanket
        out.checks["owner_confirmed_adjustments"] = blanket
        if not (per_variant or blanket):
            out.reasons.append("adjustments not owner-confirmed for this catalog version (ml civil-scheme verify --variant ...)")
    out.legal = not out.reasons
    return out


def owner_confirmation_applies(cat: SchemeCatalog) -> bool:
    from cfb_coach.madden.scheme.adjustments import OWNER_CONFIRMED_ADJUSTMENTS as oc
    return cat.catalog_id == oc["catalog_id"] and cat.version() == oc["catalog_version"]


def legal_set(cat: SchemeCatalog, *, db: Any = None, book: Mapping[str, Any] | None = None,
              require_verified: bool = True) -> tuple[list[SchemeVariant], dict[str, Verdict]]:
    verified = load_verified(db) if require_verified else {}
    if book is None:
        book = _applied_offense_book(db)
    verdicts = {v.variant_id: check(v, cat, db=db, book=book, require_verified=require_verified, verified=verified)
                for v in cat.variants}
    return [v for v in cat.variants if verdicts[v.variant_id].legal], verdicts


def revalidate_call(db: Any, call: Any, cat: SchemeCatalog) -> None:
    """Called by the live display gate. Raises ValueError when a displayed civil call is not exactly legal."""
    sv = getattr(call, "scheme_variant", None)
    if not isinstance(sv, Mapping):
        raise ValueError("civil_scheme_only call has no scheme variant record")
    if sv.get("status") == NO_LEGAL:
        if call.play != NO_LEGAL:
            raise ValueError("NO_LEGAL_SCHEME_PLAY must not carry a play")
        return
    vid = str(sv.get("variant_id") or "")
    v = cat.get(vid)
    verdict = check(v, cat, db=db, require_verified=not sv.get("shadow"), variant_id=vid)
    if not verdict.legal:
        raise ValueError("civil_scheme_only call failed re-validation: " + "; ".join(verdict.reasons))
    assert v is not None
    n = madden_catalog.norm
    want_play = sv.get("book_play") or v.base_play
    if sv.get("book_play") and (load_verified(db).get(vid) or {}).get("book_play") != sv.get("book_play"):
        raise ValueError("book-name alias is not the owner-confirmed one")
    if (n(call.formation), n(call.play)) != (n(v.formation), n(want_play)):
        raise ValueError("displayed formation/play differs from the variant's")
    shown = [tuple(a) for a in sv.get("adjustments") or []]
    want = [(a.target, a.action) for a in v.adjustments]
    if shown != want:
        raise ValueError("displayed adjustments differ from the variant's exact adjustments")
    if (getattr(call, "adjustment", None) or {}).get("id") != f"scheme:{vid}":
        raise ValueError("adjustment id does not carry the variant id")
