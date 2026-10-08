"""Defensive Custom Adjustment compatibility for the Stage 3 shadow advisor.

Never invents settings. Never recommends a macro that is not armed and valid.
Detects pairwise setting conflicts so incompatible adjustments are not combined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from cfb_coach.madden.macros import (
    EXPERIMENTAL_MACROS,
    activate_buttons,
    display_name,
    is_experimental_macro,
    settings_rows,
)
from cfb_coach.madden.macro_pool import families, pool_macro

# Editor fields that cannot take two different non-Default values at once.
_EXCLUSIVE_SETTINGS = frozenset(
    {
        "Coverage",
        "Coverage Shell",
        "Man Coverage",
        "Zone Drops",
        "Safety Depth",
        "Safety Width",
        "Safety Midpoint",
        "LB Depth",
        "LB Width",
        "QB Contain",
        "Pass Rush",
        "Defensive Style",
    }
)


@dataclass
class ArmedMacroView:
    mid: str
    xbox_name: str
    families: list[str]
    when_to_arm: str
    settings: list[dict[str, Any]]
    researched_settings: dict[str, str]
    buttons: str
    shell_pair: str
    valid: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.mid,
            "xbox_name": self.xbox_name,
            "families": list(self.families),
            "when_to_arm": self.when_to_arm,
            "researched_settings": dict(self.researched_settings),
            "buttons": self.buttons,
            "shell_pair": self.shell_pair,
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


def inspect_armed_macro(mid: str) -> ArmedMacroView:
    """Validate one armed defense macro against the research DB editor settings."""
    key = (mid or "").strip().upper()
    meta = pool_macro(key) or {}
    reasons: list[str] = []
    if not meta:
        return ArmedMacroView(
            mid=key,
            xbox_name=key,
            families=[],
            when_to_arm="",
            settings=[],
            researched_settings={},
            buttons="",
            shell_pair="",
            valid=False,
            reasons=["not in research macro pool"],
        )
    if is_experimental_macro(key) or key in EXPERIMENTAL_MACROS:
        reasons.append("experimental macro — not auto-recommended without explicit config")
    rows = settings_rows(key)
    researched = {
        str(r.get("setting") or ""): str(r.get("value") or "")
        for r in rows
        if r.get("source") and r.get("source") != "default" and r.get("setting")
    }
    # Conflict markers inside a single macro's researched rows.
    by_setting: dict[str, set[str]] = {}
    for r in rows:
        if r.get("source") == "default":
            continue
        setting = str(r.get("setting") or "")
        val = str(r.get("value") or "")
        if setting and val and val.lower() != "default":
            by_setting.setdefault(setting, set()).add(val)
    for setting, vals in by_setting.items():
        if len(vals) > 1:
            reasons.append(f"internal setting conflict on {setting}: {sorted(vals)}")
    base = meta.get("base") or {}
    shell = meta.get("shell_pair") or " — ".join(
        x for x in (base.get("formation"), base.get("play")) if x
    )
    valid = not any("conflict" in r.lower() for r in reasons) and bool(meta)
    return ArmedMacroView(
        mid=key,
        xbox_name=display_name(key) if key else key,
        families=list(families(key)),
        when_to_arm=str(meta.get("when_to_arm") or meta.get("when") or ""),
        settings=rows,
        researched_settings=researched,
        buttons=activate_buttons(key) if key else "",
        shell_pair=str(shell or ""),
        valid=valid,
        reasons=reasons,
    )


def settings_conflict(a: ArmedMacroView, b: ArmedMacroView) -> list[str]:
    """Return human-readable conflicts if two macros cannot be combined."""
    conflicts: list[str] = []
    for setting in _EXCLUSIVE_SETTINGS:
        va = a.researched_settings.get(setting)
        vb = b.researched_settings.get(setting)
        if not va or not vb:
            continue
        if va.lower() == "default" or vb.lower() == "default":
            continue
        if va.strip().lower() != vb.strip().lower():
            conflicts.append(
                f"{setting}: {a.mid}={va!r} vs {b.mid}={vb!r}"
            )
    # Family-level mutual exclusion examples: SPY vs HEAT pressure doctrine.
    fa, fb = set(a.families), set(b.families)
    if "scram" in fa and "pressure" in fb and a.mid != b.mid:
        if "SPY" in a.mid and "HEAT" in b.mid:
            conflicts.append("SPY contain vs HEAT pressure — do not stack")
        if "SPY" in b.mid and "HEAT" in a.mid:
            conflicts.append("SPY contain vs HEAT pressure — do not stack")
    return conflicts


def compatible_macros(
    armed: Sequence[str],
    *,
    preferred_families: Sequence[str] | None = None,
    already_armed: str | None = None,
) -> list[dict[str, Any]]:
    """Rank valid armed macros; skip invalid and those conflicting with ``already_armed``."""
    preferred = set(preferred_families or [])
    current = inspect_armed_macro(already_armed) if already_armed else None
    out: list[dict[str, Any]] = []
    for mid in armed:
        view = inspect_armed_macro(mid)
        if not view.valid:
            out.append(
                {
                    **view.to_dict(),
                    "eligible": False,
                    "score": -1.0,
                    "why": "; ".join(view.reasons) or "invalid",
                }
            )
            continue
        conflicts = settings_conflict(current, view) if current and current.mid != view.mid else []
        if conflicts:
            out.append(
                {
                    **view.to_dict(),
                    "eligible": False,
                    "score": -1.0,
                    "why": "conflicts with current: " + "; ".join(conflicts),
                    "conflicts": conflicts,
                }
            )
            continue
        fam_hit = len(preferred & set(view.families))
        score = float(fam_hit) + (0.25 if view.researched_settings else 0.0)
        why_bits = []
        if fam_hit:
            why_bits.append(f"answers {sorted(preferred & set(view.families))}")
        if view.researched_settings:
            why_bits.append(f"{len(view.researched_settings)} researched fields")
        if view.when_to_arm:
            why_bits.append(view.when_to_arm[:80])
        out.append(
            {
                **view.to_dict(),
                "eligible": True,
                "score": score,
                "why": "; ".join(why_bits) or "armed valid macro",
            }
        )
    out.sort(key=lambda r: (-float(r.get("score") or 0.0), r.get("id") or ""))
    return out


def recommend_adjustment(
    armed: Sequence[str],
    *,
    concept_families: Mapping[str, float] | None = None,
    heuristic_macro: str | None = None,
    essential_only: bool = True,
) -> dict[str, Any] | None:
    """Pick at most one armed CA. Returns None when none is essential/compatible.

    ``essential_only`` mirrors the live-call convention: adj only when essential
    (repeated concept / situation), not every snap.
    """
    fam_weights = dict(concept_families or {})
    preferred = [f for f, w in sorted(fam_weights.items(), key=lambda kv: -kv[1]) if w > 0.08][:4]
    ranked = compatible_macros(armed, preferred_families=preferred, already_armed=None)
    eligible = [r for r in ranked if r.get("eligible")]
    if not eligible:
        return {
            "macro": None,
            "reason": "no valid armed Custom Adjustment",
            "incompatibility": "none armed or all conflicted/invalid",
            "candidates": ranked[:5],
        }
    # Prefer heuristic macro when it is eligible and near the top.
    if heuristic_macro:
        for row in eligible:
            if row["id"] == heuristic_macro.upper():
                if essential_only and float(row.get("score") or 0) < 0.5:
                    break
                return {
                    "macro": row["id"],
                    "xbox_name": row.get("xbox_name"),
                    "reason": f"keep heuristic {row['id']} ({row.get('why')})",
                    "incompatibility": None,
                    "candidates": ranked[:5],
                    "settings_ok": True,
                }
    top = eligible[0]
    if essential_only and float(top.get("score") or 0) < 0.5 and not preferred:
        return {
            "macro": None,
            "reason": "no essential adjustment — leave heuristic play clean",
            "incompatibility": None,
            "candidates": ranked[:5],
        }
    return {
        "macro": top["id"],
        "xbox_name": top.get("xbox_name"),
        "reason": top.get("why") or "best armed match",
        "incompatibility": None,
        "candidates": ranked[:5],
        "settings_ok": True,
        "buttons": top.get("buttons"),
    }
