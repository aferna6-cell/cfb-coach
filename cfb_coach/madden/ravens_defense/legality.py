"""The ONE legal filter. Runs before scoring and again on the chosen candidate before display.

A candidate is live-legal only if every item holds; otherwise it is rejected with reasons
(never repaired by dropping a macro or substituting a similarly named play):
  * it is one of the 14 catalog candidates built from the 12 approved call IDs;
  * formation is Nickel 2-4 Wide, or Dime 2-3 Odd for the DB Fire 2 blitz only;
  * the exact base play is in the in-repo Ravens stock inventory AND in the installed book;
  * the installed two-formation fingerprint matches Aidan's confirmation;
  * the required macro (24 Base / DB Fire) and any optional macro (Zone It) are owner-confirmed;
  * the call is COMPATIBLE under the one persistent coaching-adjustment profile;
  * dataset blocks (Cover 9 Show 2 Match alias) are cleared by owner confirmation.
``assume_setup`` (shadow / offline evaluation ONLY) skips the install/macro/coaching items and is recorded;
dataset blocks still apply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense.confirmations import Confirmations


@dataclass(frozen=True)
class Verdict:
    candidate_id: str
    legal: bool
    reasons: tuple[str, ...]
    setup_assumed: bool = False


def check(cat: rc.RavensCatalog, cand: rc.Candidate, *, installed: Mapping[str, Sequence[str]] | None,
          conf: Confirmations, assume_setup: bool = False) -> Verdict:
    reasons: list[str] = []
    known = cat.by_id().get(cand.candidate_id)
    if known is None or known != cand or cand.call_id not in rc.APPROVED_CALL_IDS:
        return Verdict(cand.candidate_id, False, ("NOT_AN_APPROVED_CANDIDATE",))
    if cand.formation not in rc.FORMATIONS:
        reasons.append("FORMATION_NOT_ALLOWED")
    if cand.formation == rc.DIME and not (cand.is_blitz and cand.call_id == "dime_dbfire2"):
        reasons.append("DIME_ONLY_FOR_DB_FIRE_2_BLITZ")
    expected = rc.REQUIRED_MACRO_BY_PACKAGE.get(cand.package)
    if cand.required_macro != expected:
        reasons.append("REQUIRED_MACRO_MISSING_OR_WRONG")
    if cand.optional_macro and cand.optional_macro not in (cand.raw.get("optional_macros") or []):
        reasons.append("OPTIONAL_MACRO_NOT_ALLOWED_ON_CALL")
    if cand.base_play not in rc.ravens_stock_book().get(cand.formation, ()):
        reasons.append("NOT_IN_RAVENS_STOCK_INVENTORY")
    if not assume_setup:
        inst = installed or {}
        if cand.formation not in inst:
            reasons.append("FORMATION_NOT_INSTALLED")
        elif cand.base_play not in inst[cand.formation]:
            reasons.append("PLAY_NOT_INSTALLED")
        fp = rc.installed_fingerprint(inst)
        if not fp or conf.installed_book_fingerprint != fp:
            reasons.append("INSTALLED_BOOK_FINGERPRINT_UNCONFIRMED")
        for m in cand.macros:
            if not conf.macro_confirmed(m):
                reasons.append(f"MACRO_UNCONFIRMED:{m}")
        status = conf.coaching_status(cand.call_id)
        if status != "COMPATIBLE":
            reasons.append(f"COACHING_PROFILE:{status}")
    # dataset blocks hold even in shadow: the Guide's "Cover 9" is not proven to be Ravens "Cover 9 Show 2"
    if cand.raw.get("live_eligible") is False and not (
            cand.call_id == "match_cover9_show2" and conf.cover9_show2_alias_confirmed):
        reasons.append("DATASET_BLOCK:" + str(cand.raw.get("coverage_alias_status") or "not_live_eligible"))
    return Verdict(cand.candidate_id, not reasons, tuple(reasons), assume_setup)


def legal_set(cat: rc.RavensCatalog, *, installed: Mapping[str, Sequence[str]] | None, conf: Confirmations,
              assume_setup: bool = False) -> tuple[list[rc.Candidate], dict[str, Verdict]]:
    verdicts = {c.candidate_id: check(cat, c, installed=installed, conf=conf, assume_setup=assume_setup)
                for c in cat.candidates}
    return [c for c in cat.candidates if verdicts[c.candidate_id].legal], verdicts


def book_report(cat: rc.RavensCatalog, *, installed: Mapping[str, Sequence[str]] | None,
                conf: Confirmations) -> dict[str, Any]:
    legal, verdicts = legal_set(cat, installed=installed, conf=conf)
    stock = rc.stock_catalog_check(cat)
    return {
        "catalog": cat.version(), "dataset_sha256": cat.sha256,
        "contract_errors": rc.contract_errors(cat.raw),
        "installed_two_formation_fingerprint": rc.installed_fingerprint(installed or {}),
        "owner_confirmed_fingerprint": conf.installed_book_fingerprint,
        "owner_confirmations_file": conf.path, "owner_confirmation_errors": list(conf.errors),
        "installed_formations": sorted((installed or {}).keys()),
        "candidates": [{**c.as_dict(), "in_ravens_stock_inventory": stock[c.candidate_id],
                        "installed": bool(installed and c.base_play in (installed.get(c.formation) or ())),
                        "live_legal": verdicts[c.candidate_id].legal,
                        "reasons": list(verdicts[c.candidate_id].reasons)} for c in cat.candidates],
        "live_legal_count": len(legal),
    }
