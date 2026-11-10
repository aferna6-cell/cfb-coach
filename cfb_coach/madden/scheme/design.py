"""Base-play geometry and the ADJUSTED design of each scheme variant.

Base routes, in order of trust:

1. first-party routes at a verified tier (``diagram_verified`` / ``in_game_verified``). The
   repo's ``source_documented`` routes come from a provider that does not permit ML use
   (see ``offense_play_features.full_vector``) and are never used here;
2. provisional, anonymous AceMadden-derived geometry decoded by the existing
   ``offense_geometry_decode`` (owner-permitted, confidence-weighted prior only; never a
   verified assignment and never a player identity). The export is a LOCAL file that is not
   committed to git; without it the base geometry is honestly UNKNOWN;
3. nothing: name-only hint (play type / concept from the play NAME, low trust).

Adjusted design (variant): every commanded route is attached to its catalog player role
(identity comes from the catalog instruction, not from geometry). Because provisional
geometry is anonymous we cannot know which stock route each adjusted player was running,
so each stock route is kept with weight ``(1 - k / n_base)`` (the expected share of stock
routes NOT overridden when ``k`` of ``n_base`` routes are re-assigned). This is an explicit,
documented approximation; the design records ``ownership_unresolved`` and the card asks for
verification. Stems shift the commanded route's depth band; modifiers that target a stock
route of unknown ownership are recorded as ``unrepresented`` (never silently dropped).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden.model import football_priors as fpri
from cfb_coach.madden.model import offense_assignments as oa
from cfb_coach.madden.model import offense_taxonomy as tx
from cfb_coach.madden.scheme import adjustments as adj
from cfb_coach.madden.scheme.catalog import SchemeVariant

ENV_EXPORT = "CFB_COACH_PROVISIONAL_EXPORT"
DEFAULT_EXPORTS = (
    Path("~/.cfb-coach/catalog/provisional_catalog_offense.v1.jsonl"),
    Path("/workspace/madden_ml/issue_132_catalog_track/provisional_catalog_offense.v1.jsonl"),
)
VERIFIED_ROUTE_TIERS = frozenset({"diagram_verified", "in_game_verified"})
TRUST = {"verified": 0.9, "high_confidence_extracted": fpri.TIER_TRUST["football_inference"],
         "partial": fpri.NAME_TRUST, "commanded": fpri.TIER_TRUST["football_inference"]}
NOMINAL_ELIGIBLE = 5.0


def norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


@dataclass(frozen=True)
class DesignRoute:
    family: str
    band: str
    side: str          # left | right | middle | unknown
    weight: float      # trust x ownership weight, 0..1
    source: str        # base_verified | base_provisional | commanded
    owner: str | None  # catalog role for commanded routes; None for anonymous stock routes

    @property
    def meta(self) -> tx.RouteFamily:
        return tx.ROUTE_FAMILIES.get(self.family) or tx.ROUTE_FAMILIES["other"]

    def spec(self) -> tx.RouteSpec:
        return tx.RouteSpec(self.family, self.side if self.side != "unknown" else "middle", self.band)

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "band": self.band, "side": self.side, "weight": round(self.weight, 4),
                "source": self.source, "owner": self.owner}


@dataclass(frozen=True)
class Design:
    formation: str
    play: str
    variant_id: str | None
    routes: tuple[DesignRoute, ...]
    geometry_status: str            # verified | high_confidence_extracted | partial | name_only | unknown
    base_route_count: int
    adjusted_roles: tuple[str, ...]
    motion: bool
    unrepresented: tuple[int, ...]  # adjustment indices recorded but not representable on geometry
    ownership_unresolved: bool
    hint: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def concept_trust(self) -> dict[str, float]:
        """concept -> trust: detected on the routes with weight > 0, trusted by their weakest weight; name hint at NAME_TRUST."""
        out: dict[str, float] = {}
        live = [r for r in self.routes if r.weight > 0.05 and r.meta.category != "blocking"]
        if live:
            specs = [r.spec() for r in live]
            by_label = {}
            for r, sp in zip(live, specs):
                by_label.setdefault(sp.label, []).append(r.weight)
            for c, hit in tx.detect_concepts(specs).items():
                ws = [w for lab in hit.get("routes", ()) for w in by_label.get(lab, [])] or [r.weight for r in live]
                cid = fpri.CONCEPT_ALIASES.get(c, c)
                out[cid] = max(out.get(cid, 0.0), min(ws))
        from cfb_coach.madden.model import football_knowledge as fk
        hint_c = fk.profile_for_play(self.play).get("concept_id")
        if hint_c and hint_c not in out:
            out[hint_c] = fpri.NAME_TRUST * (0.5 if self.adjusted_roles else 1.0)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"formation": self.formation, "play": self.play, "variant_id": self.variant_id,
                "routes": [r.as_dict() for r in self.routes], "geometry_status": self.geometry_status,
                "base_route_count": self.base_route_count, "adjusted_roles": list(self.adjusted_roles),
                "motion": self.motion, "unrepresented": list(self.unrepresented),
                "ownership_unresolved": self.ownership_unresolved, "warnings": list(self.warnings),
                "concepts": {k: round(v, 3) for k, v in sorted(self.concept_trust().items())}}


# ------------------------------------------------------------------ provisional geometry (local, not in git)

def resolve_export_path(explicit: str | Path | None = None) -> Path | None:
    for raw in (explicit, os.environ.get(ENV_EXPORT), *DEFAULT_EXPORTS):
        if raw:
            p = Path(raw).expanduser()
            if p.is_file():
                return p
    return None


@lru_cache(maxsize=4)
def _load_export(path: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        header = json.loads(fh.readline() or "{}")
        if header.get("schema") != "madden27.catalog_semantics.provisional.v1":
            raise ValueError("unexpected provisional export schema")
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("model_use_permitted") or r.get("verified"):
                continue
            key = (norm(r.get("formation_slug")), norm(r.get("play_slug") or r.get("play_name")))
            # Keep only what decoding needs: no art hashes, URLs or listing ids leave this function.
            out.setdefault(key, {"feature_tier": r.get("feature_tier"), "geometry": r.get("geometry")})
    return out


class GeometrySource:
    """Read-only lookup of decoded base routes for (formation, play)."""

    def __init__(self, export_path: str | Path | None = None, *, enabled: bool = True):
        self.path = resolve_export_path(export_path) if enabled else None
        self._rows = _load_export(str(self.path)) if self.path else {}

    @property
    def available(self) -> bool:
        return bool(self._rows)

    def base_routes(self, formation: str, play: str) -> tuple[tuple[DesignRoute, ...], str]:
        from cfb_coach.madden.model import offense_play_features as pf
        fp = pf.structure_lookup(formation, play)
        if fp is not None:
            ver = [(sl, sp, t) for sl, sp, t in fp.routes if t in VERIFIED_ROUTE_TIERS]
            if ver:
                return tuple(DesignRoute(sp.family, sp.band if sp.band in adj.BAND_ORDER else "intermediate",
                                         sp.side, TRUST["verified"], "base_verified", None)
                             for _, sp, _ in ver if sp.meta.category != "blocking"), "verified"
        row = self._rows.get((norm(formation), norm(play)))
        if row and row.get("geometry"):
            from cfb_coach.madden.model.offense_geometry_decode import decode_provisional_geometry
            tier = "high_confidence_extracted" if row.get("feature_tier") == "high_confidence_extracted" else "partial"
            try:
                st = decode_provisional_geometry(row["geometry"], formation=formation, play=play, feature_tier=tier)
            except Exception:  # noqa: BLE001 - undecodable geometry stays unknown
                st = None
            if st is not None and st.n_known:
                return tuple(DesignRoute(sp.family, sp.band, sp.side, TRUST[tier], "base_provisional", None)
                             for _, sp, _ in st.routes), tier
        return (), "name_only" if oa.name_hint(play).get("play_type") != "unknown" else "unknown"


# ------------------------------------------------------------------ designs

def stock_design(formation: str, play: str, geo: GeometrySource) -> Design:
    routes, status = geo.base_routes(formation, play)
    return Design(formation=formation, play=play, variant_id=None, routes=routes, geometry_status=status,
                  base_route_count=len(routes), adjusted_roles=(), motion=bool(oa.name_hint(play).get("motion")),
                  unrepresented=(), ownership_unresolved=False, hint=oa.name_hint(play))


def adjusted_design(v: SchemeVariant, geo: GeometrySource, applied: Sequence[int] | None = None) -> Design:
    """``applied``: 0-based adjustment indices actually executed (None = all). Skipped ones are left out, so a
    partially executed variant is represented as what was really run."""
    base, status = geo.base_routes(v.formation, v.base_play)
    specs = adj.parse_variant(v)
    if applied is not None:
        keep_idx = set(applied)
        specs = tuple(s for s in specs if s.index in keep_idx)
    commanded: dict[int, DesignRoute] = {}
    roles: list[str] = []
    unrep: list[int] = []
    warnings: list[str] = []
    motion = bool(oa.name_hint(v.base_play).get("motion"))
    for s in specs:
        warnings.extend(f"#{s.index + 1} {w}" for w in s.warnings)
        if s.motion:
            motion = True
        if s.route_family:
            band = adj.shift_band(tx.ROUTE_FAMILIES[s.route_family].default_band, s.stem_shift)
            # The newest instruction for a role wins (a player runs one route).
            for i, r in list(commanded.items()):
                if r.owner and norm(r.owner) == norm(s.role):
                    del commanded[i]
            commanded[s.index] = DesignRoute(s.route_family, band, "unknown", TRUST["commanded"], "commanded", s.role)
            if s.role not in roles:
                roles.append(s.role)
        elif isinstance(s.refers_to, int) and s.refers_to in commanded and s.stem_shift:
            r = commanded[s.refers_to]
            commanded[s.refers_to] = DesignRoute(r.family, adj.shift_band(r.band, s.stem_shift), r.side, r.weight,
                                                 r.source, r.owner)
        else:
            unrep.append(s.index)
    k = len(roles)
    n_base = len(base)
    keep = max(0.0, 1.0 - k / max(n_base, NOMINAL_ELIGIBLE)) if n_base else 0.0
    scaled = tuple(DesignRoute(r.family, r.band, r.side, r.weight * keep, r.source, None) for r in base)
    if n_base and k:
        warnings.append("stock route ownership is unresolved on provisional geometry: stock routes are down-weighted "
                        f"by {keep:.2f} instead of replaced; verify the adjusted routes in game")
    if not n_base:
        warnings.append(f"base play geometry {status.upper()}: only the commanded routes are represented")
    return Design(formation=v.formation, play=v.base_play, variant_id=v.variant_id,
                  routes=scaled + tuple(commanded[i] for i in sorted(commanded)), geometry_status=status,
                  base_route_count=n_base, adjusted_roles=tuple(roles), motion=motion, unrepresented=tuple(unrep),
                  ownership_unresolved=bool(n_base and k), hint=oa.name_hint(v.base_play),
                  warnings=tuple(dict.fromkeys(warnings)))


# ------------------------------------------------------------------ route features (identical for stock and adjusted designs)

ROUTE_FEATURES = (
    "g_known", "g_commanded", "g_adjusted_frac", "g_unrepresented", "g_motion", "g_status_provisional",
    "r_vertical", "r_breaking", "r_underneath", "r_crossing", "r_curl_sit", "r_mean_depth", "r_depth_spread",
    "r_levels", "r_side_imbalance", "r_man_beater", "r_zone_beater", "r_quick", "r_deep", "r_inside_break",
    "r_outside_break", "r_deep_middle", "r_flat_short", "r_horizontal",
)


def route_features(d: Design) -> dict[str, float]:
    f = {n: 0.0 for n in ROUTE_FEATURES}
    rs = [r for r in d.routes if r.weight > 0 and r.meta.category != "blocking"]
    W = sum(r.weight for r in rs)
    f["g_known"] = min(1.0, W / (NOMINAL_ELIGIBLE * TRUST["high_confidence_extracted"]))
    f["g_commanded"] = sum(1 for r in rs if r.source == "commanded") / NOMINAL_ELIGIBLE
    f["g_adjusted_frac"] = len(d.adjusted_roles) / NOMINAL_ELIGIBLE
    f["g_unrepresented"] = len(d.unrepresented) / NOMINAL_ELIGIBLE
    f["g_motion"] = 1.0 if d.motion else 0.0
    f["g_status_provisional"] = 1.0 if d.geometry_status in ("high_confidence_extracted", "partial") else 0.0
    if W <= 0:
        return f

    def share(pred) -> float:
        return sum(r.weight for r in rs if pred(r)) / W

    for c in ("vertical", "breaking", "underneath", "crossing", "curl_sit"):
        f["r_" + c] = share(lambda r, c=c: r.meta.category == c)
    ys = [tx.BAND_YARDS.get(r.band, 10.0) for r in rs]
    f["r_mean_depth"] = sum(y * r.weight for y, r in zip(ys, rs)) / W / 20.0
    f["r_depth_spread"] = (max(ys) - min(ys)) / 20.0
    f["r_levels"] = len({r.band for r in rs if r.weight >= 0.15}) / 4.0
    sided = [r for r in rs if r.side in ("left", "right")]
    if sided:
        ws = sum(r.weight for r in sided)
        f["r_side_imbalance"] = abs(sum(r.weight for r in sided if r.side == "left")
                                    - sum(r.weight for r in sided if r.side == "right")) / ws
    f["r_man_beater"] = sum(r.meta.man_beater * r.weight for r in rs) / W
    f["r_zone_beater"] = sum(r.meta.zone_beater * r.weight for r in rs) / W
    f["r_quick"] = share(lambda r: r.meta.quick)
    f["r_deep"] = sum(r.weight for r in rs if r.band == "deep") / 3.0
    f["r_inside_break"] = share(lambda r: r.meta.break_dir == "inside")
    f["r_outside_break"] = share(lambda r: r.meta.break_dir == "outside")
    f["r_deep_middle"] = sum(r.weight for r in rs if r.family in ("seam", "post", "deep_cross")
                             or (r.family == "in" and r.band == "deep")) / 2.0
    f["r_flat_short"] = share(lambda r: r.family in ("flat", "swing", "out", "angle", "texas")
                              and r.band in ("shallow", "short"))
    f["r_horizontal"] = share(lambda r: r.family in ("drag", "shallow_cross", "cross", "deep_cross", "flat", "swing"))
    return f


# ------------------------------------------------------------------ formation / personnel tokens (from the NAME, never invented)

FORMATION_TOKENS = ("gun", "pistol", "singleback", "iform", "5wr", "bunch", "tight", "stack", "clamp", "trips",
                    "doubles", "nasty", "te", "empty", "deuce", "close")


def formation_features(formation: str) -> dict[str, float]:
    words = set(re.findall(r"[a-z0-9]+", formation.lower().replace("i form", "iform")))
    return {"f_" + t: 1.0 if t in words else 0.0 for t in FORMATION_TOKENS}


PLAY_TYPES = ("pass", "run", "play_action", "screen", "rpo", "option")


def play_type_features(play: str) -> dict[str, float]:
    t = oa.name_hint(play).get("play_type", "unknown")
    return {"t_" + p: 1.0 if t == p else 0.0 for p in PLAY_TYPES}


def matchup_edge(d: Design, label_mix: Mapping[str, float], pressure: float) -> tuple[float, list[dict[str, Any]]]:
    """Existing football_priors knowledge: expected logit lift of the design's concepts under the belief mixture."""
    concepts = d.concept_trust()
    total, cells = 0.0, []
    if not concepts:
        return 0.0, []
    items = list(label_mix.items())
    if pressure > 0:
        items.append(("__pressure__", pressure))
    for key, w in items:
        desc = {"shell": "unk", "type": "unk", "variants": set(), "press": "man_blitz"} if key == "__pressure__" \
            else fpri.descriptor(key)
        if not desc or w <= 0:
            continue
        for c, trust in concepts.items():
            lift, _cw, ids = fpri.lift_for(desc, c)
            if ids:
                total += w * trust * lift
                cells.append({"coverage": "pressure" if key == "__pressure__" else key, "concept": c,
                              "lift": round(lift, 4), "weight": round(w * trust, 4), "priors": ids})
    cells.sort(key=lambda c: -abs(c["lift"] * c["weight"]))
    return max(-fpri.EDGE_CAP, min(fpri.EDGE_CAP, total)), cells[:6]
