"""Structured, auditable reading of each variant's presnap adjustments.

Every catalog adjustment is kept verbatim (target + action, in order) for the live card.
This module only adds a machine-readable interpretation used to build the adjusted design:

* the commanded route family (``offense_taxonomy`` ids) for a route change,
* stem changes (deeper / shallower / all the way down) and motion,
* which earlier instruction a modifier refers to ("HB corner route -> Stem down").

Nothing is invented: an action outside the small vocabulary below is ``unknown`` and the
variant carries a verification warning. Controller inputs are never generated; when the
source text itself mentions one (e.g. "one press on D-pad") it is shown verbatim only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from cfb_coach.madden.model import offense_taxonomy as tx
from cfb_coach.madden.scheme.catalog import Adjustment, SchemeVariant

# Catalog wording -> taxonomy route family. Exact phrase matches only.
ROUTE_WORDS: tuple[tuple[str, str], ...] = (
    ("slot fade", "fade"), ("short cross", "shallow_cross"), ("drag route", "drag"), ("drag", "drag"),
    ("streak", "streak"), ("post route", "post"), ("post", "post"), ("in route", "in"),
    ("flat route", "flat"), ("corner route", "corner"), ("wheel route", "wheel"), ("out route", "out"),
    ("slant route", "slant"), ("slant", "slant"),
)
# Hot-route options present in the repo's Madden vocabulary (offense_taxonomy.HOT_ROUTE_OPTIONS).
REPO_HOT_ROUTE_FAMILIES = frozenset(tx.HOT_ROUTE_TO_FAMILY.values())
BAND_ORDER = ("shallow", "short", "intermediate", "deep")


@dataclass(frozen=True)
class AdjustmentSpec:
    index: int
    target: str
    action: str
    role: str                      # normalized player role (the catalog target, without a route suffix)
    route_family: str | None       # commanded route family, None when this line only modifies a route/motion
    stem_shift: int                # -1 one step shallower, -9 all the way down, +1 one step deeper, 0 none
    motion: str | None             # verbatim motion instruction, if any
    refers_to: int | str | None    # index of the earlier instruction it modifies, "base_route" (unknown owner) or None
    kind: str                      # hot_route | stem | route_with_stem | route_with_motion | unknown
    representable: bool            # can the adjusted design represent this instruction exactly?
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "target": self.target, "action": self.action, "role": self.role,
                "route_family": self.route_family, "stem_shift": self.stem_shift, "motion": self.motion,
                "refers_to": self.refers_to, "kind": self.kind, "representable": self.representable,
                "warnings": list(self.warnings)}


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def _route_in(text: str) -> str | None:
    t = f" {_norm(text)} "
    for word, fam in ROUTE_WORDS:
        if f" {word} " in t:
            return fam
    return None


def _stem(text: str) -> int:
    t = _norm(text)
    if "stem" not in t:
        return 0
    if "down" in t:
        return -9 if "all the way" in t else -1
    if "up" in t:
        return 1
    return 0


def _motion(text: str) -> str | None:
    raw = text.strip()
    m = re.search(r"(motion[^;]*?)(?:;|$)", raw, flags=re.I)
    if not m:
        return None
    tail = raw[m.start():].strip()
    return tail


def _split_role(target: str) -> tuple[str, str | None]:
    """'HB corner route' -> ('HB', 'corner'); 'Outside bunch WR' -> ('Outside bunch WR', None)."""
    fam = _route_in(target)
    if fam is None:
        return target.strip(), None
    words = target.strip().split()
    # Drop the trailing route words ("corner route", "post route", "in route", "post").
    for word, f in ROUTE_WORDS:
        if f == fam and _norm(target).endswith(word):
            n = len(word.split())
            return " ".join(words[:-n]).strip(), fam
    return target.strip(), fam


def parse_variant(v: SchemeVariant) -> tuple[AdjustmentSpec, ...]:
    out: list[AdjustmentSpec] = []
    for a in v.adjustments:
        out.append(_parse_one(a, out))
    return tuple(out)


def _parse_one(a: Adjustment, earlier: Sequence[AdjustmentSpec]) -> AdjustmentSpec:
    warnings: list[str] = []
    role, target_route = _split_role(a.target)
    route = _route_in(a.action)
    stem = _stem(a.action)
    motion = _motion(a.action)
    refers: int | str | None = None
    if "d pad" in _norm(a.action) or "d-pad" in a.action.lower():
        warnings.append("source text names a controller input; shown verbatim, not generated")
    if role.lower().startswith("existing"):
        warnings.append("modifies an existing stock route whose owner is not verified; recorded, not applied to geometry")
        return AdjustmentSpec(a.index, a.target, a.action, a.target, None, stem, motion, "base_route",
                              "stem" if stem else "unknown", False, tuple(warnings))
    if route is None and stem != 0 and target_route is None:
        # A stem on this player's route: the route commanded earlier in this variant, else the stock route.
        prior = next((e for e in reversed(earlier) if _norm(e.role) == _norm(role) and e.route_family), None)
        if prior is not None:
            return AdjustmentSpec(a.index, a.target, a.action, role, None, stem, motion, prior.index, "stem", True,
                                  tuple(warnings))
        warnings.append(f"stem on {role}'s stock route; stock route ownership is not verified, so the change is "
                        "recorded but not applied to geometry")
        return AdjustmentSpec(a.index, a.target, a.action, role, None, stem, motion, "base_route", "stem", False,
                              tuple(warnings))
    if target_route is not None and route is None:
        # A modifier on an existing route: "HB corner route" -> "Stem down".
        prior = next((e for e in reversed(earlier) if _norm(e.role) == _norm(role) and e.route_family == target_route), None)
        if prior is not None:
            refers = prior.index
        else:
            refers = "base_route"
            warnings.append(f"modifies the stock {target_route} route of {role or 'an unnamed player'}; "
                            "stock route ownership is not verified, so the change is recorded but not applied to geometry")
        kind = "stem" if stem else "unknown"
        representable = refers != "base_route" and stem != 0
        if not stem:
            warnings.append("modifier action not recognized; verify in game")
        return AdjustmentSpec(a.index, a.target, a.action, role, None, stem, motion, refers, kind,
                              representable, tuple(warnings))
    if role.lower().startswith("existing") or (target_route is not None and _norm(role) == "existing"):
        refers = "base_route"
        warnings.append("modifies an existing stock route whose owner is not verified; recorded, not applied to geometry")
        return AdjustmentSpec(a.index, a.target, a.action, role, None, stem, motion, refers,
                              "stem" if stem else "unknown", False, tuple(warnings))
    if route is None:
        warnings.append("action not recognized as a route, stem or motion; verify in game")
        return AdjustmentSpec(a.index, a.target, a.action, role, None, stem, motion, None, "unknown", False,
                              tuple(warnings))
    if route not in REPO_HOT_ROUTE_FAMILIES:
        warnings.append(f"'{route}' is not in the repo's hot-route vocabulary; confirm it is a selectable option for {role}")
    if motion:
        warnings.append("motion/return timing is not modeled beyond a motion flag; verify execution in game")
    kind = "route_with_motion" if motion else ("route_with_stem" if stem else "hot_route")
    return AdjustmentSpec(a.index, a.target, a.action, role, route, stem, motion, None, kind, True, tuple(warnings))


def shift_band(band: str, shift: int) -> str:
    if shift == 0 or band not in BAND_ORDER:
        return band
    if shift <= -9:
        return BAND_ORDER[0]
    i = max(0, min(len(BAND_ORDER) - 1, BAND_ORDER.index(band) + shift))
    return BAND_ORDER[i]


def checklist_lines(v: SchemeVariant) -> list[str]:
    """Human-readable, verbatim adjustment list for the live card / pregame sheet (no controller buttons)."""
    lines = []
    for s in parse_variant(v):
        line = f"{s.index + 1}. {s.target}: {s.action}"
        if s.warnings:
            line += "  [verify: " + "; ".join(s.warnings) + "]"
        lines.append(line)
    return lines
