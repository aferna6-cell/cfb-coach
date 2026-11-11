"""Owner-only, private setup confirmations for the Ravens defense (never committed, never inferred).

Everything the code cannot see on the Xbox screen must come from this file, written by Aidan:
installed-book fingerprint, macros that physically exist and execute, the ONE persistent
coaching-adjustment profile and which calls work under it, the Cover 9 -> Cover 9 Show 2 alias,
and in-game USER verifications. A missing file means: nothing is confirmed (fail closed).

Schema ``madden27.ravens_defense.owner_confirmations.v1``::

    {"schema": "...v1",
     "installed_book_fingerprint": "RAV-DEF-XXXXXXXXXXXX",
     "macros": {"24_BASE": {"exists": true, "executes": true, "verified_on": "YYYY-MM-DD"}, ...},
     "coaching_profile": {"profile_id": "one persistent profile", "verified_on": "YYYY-MM-DD",
                          "compatible_call_ids": [...], "incompatible_call_ids": [...],
                          "advanced_nickel_adjustments_active": false},
     "cover9_show2_alias_confirmed": false,
     "standing_macro_attestation": false,
     "user_verifications": [{"call_id": "match_cover3", "zone_it": false, "offense_shape": "trips",
                             "formation": "Nickel 2-4 Wide", "check": "Skinny MEG", "user": "Hook Curl",
                             "verified_on": "YYYY-MM-DD", "source": "owner_in_game"}]}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "madden27.ravens_defense.owner_confirmations.v1"
ENV = "CFB_COACH_RAVENS_DEFENSE_CONFIRMATIONS"
SHAPES = ("trips", "bunch", "2x2", "stacks", "5wr_tight", "4x1", "3_plus_2_uncertain")


def default_path() -> Path:
    env = (os.environ.get(ENV) or "").strip()
    return Path(env).expanduser() if env else Path.home() / ".cfb-coach" / "ravens_defense" / "owner_confirmations.json"


@dataclass(frozen=True)
class Confirmations:
    path: str | None = None
    installed_book_fingerprint: str | None = None
    macros: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    coaching_profile: Mapping[str, Any] = field(default_factory=dict)
    cover9_show2_alias_confirmed: bool = False
    standing_macro_attestation: bool = False
    user_verifications: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def present(self) -> bool:
        return self.path is not None and not self.errors

    def macro_confirmed(self, macro_id: str) -> bool:
        m = self.macros.get(macro_id) or {}
        return bool(m.get("exists") is True and m.get("executes") is True and str(m.get("verified_on") or "").strip())

    def coaching_status(self, call_id: str) -> str:
        prof = self.coaching_profile or {}
        if not prof.get("profile_id") or not str(prof.get("verified_on") or "").strip():
            return "PROFILE_UNCONFIRMED"
        if prof.get("advanced_nickel_adjustments_active") is not False:
            return "ADVANCED_NICKEL_ADJUSTMENTS_NOT_RULED_OUT"
        if call_id in (prof.get("incompatible_call_ids") or []):
            return "INCOMPATIBLE"
        if call_id in (prof.get("compatible_call_ids") or []):
            return "COMPATIBLE"
        return "CALL_NOT_AUDITED_UNDER_PROFILE"

    def user_verification(self, call_id: str, zone_it: bool, shape: str | None) -> Mapping[str, Any] | None:
        hits = [u for u in self.user_verifications if u.get("call_id") == call_id
                and bool(u.get("zone_it")) == bool(zone_it) and u.get("offense_shape") == shape]
        return hits[0] if len(hits) == 1 else None


def _validate(raw: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        return ["unrecognized owner confirmation schema"]
    seen = set()
    for u in raw.get("user_verifications") or []:
        if not isinstance(u, dict):
            errs.append("user verification must be an object")
            continue
        need = ("call_id", "offense_shape", "formation", "user", "verified_on", "source")
        if any(not isinstance(u.get(k), str) or not u.get(k).strip() for k in need):
            errs.append("user verification missing " + ",".join(k for k in need if not u.get(k)))
        if u.get("source") != "owner_in_game":
            errs.append("user verification source must be owner_in_game")
        if u.get("offense_shape") not in SHAPES:
            errs.append(f"unknown offense_shape {u.get('offense_shape')}")
        key = (u.get("call_id"), bool(u.get("zone_it")), u.get("offense_shape"))
        if key in seen:
            errs.append(f"duplicate user verification {key}")
        seen.add(key)
    return errs


def load(path: str | Path | None = None) -> Confirmations:
    p = Path(path).expanduser() if path else default_path()
    if not p.is_file():
        return Confirmations()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Confirmations(path=str(p), errors=(f"unreadable: {exc}",))
    errs = _validate(raw)
    if errs:  # a malformed file confirms nothing
        return Confirmations(path=str(p), errors=tuple(errs))
    return Confirmations(
        path=str(p),
        installed_book_fingerprint=raw.get("installed_book_fingerprint"),
        macros=dict(raw.get("macros") or {}),
        coaching_profile=dict(raw.get("coaching_profile") or {}),
        cover9_show2_alias_confirmed=raw.get("cover9_show2_alias_confirmed") is True,
        standing_macro_attestation=raw.get("standing_macro_attestation") is True,
        user_verifications=tuple(raw.get("user_verifications") or ()),
    )


def coaching_audit(cat: Any, conf: Confirmations) -> dict[str, Any]:
    """Actual coaching-adjustment facts available to the code, and the per-call status."""
    ref = json.loads((cat_path := Path(__file__).resolve().parents[3] /
                      "datasets/madden27/civil_match_nickel_combined.json").read_text(encoding="utf-8"))
    profiles = ref.get("coaching_adjustment_profiles") or {}
    conflicts = [
        "Match profile values: " + str((profiles.get("match") or {}).get("values")),
        "Nickel profile values: " + str((profiles.get("nickel") or {}).get("values")),
        "The original Match and Nickel guides use different global coaching adjustments; neither set was "
        "transcribed, so option-by-option compatibility cannot be computed from repo data.",
        "Formation switches do not change global coaching adjustments (dataset switching_policy).",
    ]
    return {
        "reference_file": str(cat_path.relative_to(cat_path.parents[2])) + " (reference only)",
        "switching_policy": profiles.get("switching_policy"),
        "conflicts": conflicts,
        "owner_profile": dict(conf.coaching_profile or {}) or None,
        "per_call": {c: conf.coaching_status(c) for c in cat.calls},
    }
