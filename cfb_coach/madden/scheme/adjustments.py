"""Structured, auditable reading of each variant's presnap adjustments.

Every catalog adjustment is kept verbatim (target + action, in order) for the live card.
This module only adds a machine-readable interpretation used to build the adjusted design:

* the commanded route family (``offense_taxonomy`` ids) for a route change,
* stem changes (deeper / shallower / all the way down) and motion,
* which earlier instruction a modifier refers to ("HB corner route -> Stem down").

Owner confirmation (Aidan, 2026-11-10, t333u): "all these adjustments exist in game and i am aware of
how to do them". Every catalog adjustment kind (hot routes incl. wheel and short cross, stems up/down incl.
"all the way", stems on stock routes, motion / return motion, D-pad stem text) is therefore treated as
executable; nothing here blocks a call. ``notes`` only describe how the adjustment is represented on the
provisional geometry (e.g. a stock-route stem applied directionally because the exact depth is unknown).
Controller inputs are never generated; when the source text mentions one it is shown verbatim.
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
    # red-zone package setup (Spot Y Option 1), owner instruction t335u
    ("swing", "swing"), ("crosser", "cross"),
)
# Hot-route options present in the repo's Madden vocabulary (offense_taxonomy.HOT_ROUTE_OPTIONS) ...
REPO_HOT_ROUTE_FAMILIES = frozenset(tx.HOT_ROUTE_TO_FAMILY.values())
# ... plus the catalog adjustment kinds the owner confirmed exist in game and are executable.
OWNER_CONFIRMED_ADJUSTMENTS = {
    "owner_confirmed_adjustments": "2026-11-10",
    "source": "t333u",
    "statement": "all these adjustments exist in game and i am aware of how to do them",
    "kinds": ["hot_route", "wheel", "short_cross", "stem_up", "stem_down", "stem_all_the_way_down",
              "stem_on_stock_route", "motion", "return_motion", "d_pad_stem_text"],
    "catalog_id": "lions_civil_bucs_5wr_v4",
    "catalog_version": "lions_civil_bucs_5wr_v4@15c120decc4b",
}
# Owner instruction t337u: every formation and play in the scheme dataset (26 variants + red-zone package plays)
# is in Aidan's installed Madden book, so no applied-book or stock-name check gates them.
OWNER_CONFIRMED_BOOK = {
    "owner_confirmed_installed": "2026-11-10",
    "source": "t337u",
    "statement": "we can assume all formations in the dataset are in my book because they are so all plays are "
                 "legal to call",
    "scope": "every catalog variant and red-zone package play: formation and play installed under the dataset's name",
    # t338u: the owner sent the two I Form Close Flex red-zone plays (5WR Tight sheet) as plays from his book.
    "t338u_addition": "I Form Close Flex: Stretch, Mesh (owner screenshots 2026-11-10; in the Buccaneers stock data)",
    "catalog_id": "lions_civil_bucs_5wr_v4",
    "catalog_version": "lions_civil_bucs_5wr_v4@15c120decc4b",
}
# t333u covers every adjustment in the catalog, including the red-zone package setup (drag, swing, crosser)
# that t335u made callable.
CIVIL_ROUTE_FAMILIES = REPO_HOT_ROUTE_FAMILIES | frozenset({"wheel", "shallow_cross", "swing", "cross"})
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
    representable: bool            # True for every catalog kind (stock-route stems are applied directionally if needed)
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
    notes: list[str] = []
    role, target_route = _split_role(a.target)
    route = _route_in(a.action)
    stem = _stem(a.action)
    motion = _motion(a.action)
    if role.lower().startswith("existing"):
        # "Existing corner route: Stem down all the way" -> the stock route of that family.
        return AdjustmentSpec(a.index, a.target, a.action, a.target, None, stem, motion, "base_route",
                              "stem_on_stock_route" if stem else "stock_route_modifier", True, tuple(notes))
    if route is None and stem != 0 and target_route is None:
        # A stem on this player's route: the route commanded earlier in this variant, else the stock route.
        prior = next((e for e in reversed(earlier) if _norm(e.role) == _norm(role) and e.route_family), None)
        if prior is not None:
            return AdjustmentSpec(a.index, a.target, a.action, role, None, stem, motion, prior.index, "stem", True,
                                  tuple(notes))
        return AdjustmentSpec(a.index, a.target, a.action, role, None, stem, motion, "base_route",
                              "stem_on_stock_route", True, tuple(notes))
    if target_route is not None and route is None:
        # A modifier on an existing route: "HB corner route" -> "Stem down".
        prior = next((e for e in reversed(earlier) if _norm(e.role) == _norm(role) and e.route_family == target_route), None)
        refers: int | str = prior.index if prior is not None else "base_route"
        kind = ("stem" if refers != "base_route" else "stem_on_stock_route") if stem else "route_modifier"
        return AdjustmentSpec(a.index, a.target, a.action, role, None, stem, motion, refers, kind, True, tuple(notes))
    if route is None:
        notes.append("action has no route/stem/motion keyword the model reads; shown verbatim, not modeled")
        return AdjustmentSpec(a.index, a.target, a.action, role, None, stem, motion, None, "unmodeled", False,
                              tuple(notes))
    kind = "route_with_motion" if motion else ("route_with_stem" if stem else "hot_route")
    return AdjustmentSpec(a.index, a.target, a.action, role, route, stem, motion, None, kind, True, tuple(notes))


def stem_label(shift: int) -> str | None:
    if shift <= -9:
        return "STEM DOWN (all the way)"
    if shift < 0:
        return "STEM DOWN"
    if shift > 0:
        return "STEM UP"
    return None


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
        guide = v.adjustments[s.index].guide_text if s.index < len(v.adjustments) else ""
        if guide:
            line += f'  (guide wording: "{guide}")'
        tag = stem_label(s.stem_shift)
        if tag:
            line += f"   [{tag}]"
        lines.append(line)
    if not lines:
        lines.append("(no adjustments listed in the catalog for this play: stock play)")
    return lines
