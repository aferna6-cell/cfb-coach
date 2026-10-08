"""Defensive Custom Adjustment compatibility for the Stage 3 shadow advisor.

Never invents settings. Never recommends a macro that is not armed and valid.
Validates the complete Madden defensive editor field inventory. Checks
play↔adjustment compatibility (coverage family / shell / researched base).
Detects pairwise setting conflicts so incompatible adjustments are not stacked.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from cfb_coach.madden.defense_select import call_family
from cfb_coach.madden.macros import (
    EXPERIMENTAL_MACROS,
    activate_buttons,
    display_name,
    is_experimental_macro,
    settings_rows,
)
from cfb_coach.madden.macro_pool import families, pool_macro
from cfb_coach.madden import research_db as rdb

# Editor fields that cannot take two different non-Default values at once.
_EXCLUSIVE_SETTINGS = frozenset(
    {
        "Coverage",
        "Coverage Shell",
        "Coverage Shading",
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
        "Alignment",
    }
)

# Env / meta flag that authorizes experimental macros (HEAT, …). Default: off.
_ENV_ALLOW_EXPERIMENTAL = "CFB_COACH_ALLOW_EXPERIMENTAL_D_MACROS"


def experimental_macros_authorized() -> bool:
    raw = (os.environ.get(_ENV_ALLOW_EXPERIMENTAL) or "").strip().lower()
    return raw in ("1", "true", "on", "yes", "authorized")


def required_editor_fields() -> list[tuple[str, str]]:
    """Complete Madden defensive editor inventory: ``(section, setting)``."""
    fields = rdb.editor_fields().get("defense") or {}
    out: list[tuple[str, str]] = []
    for section, names in fields.items():
        for name in names or []:
            out.append((str(section), str(name)))
    return out


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
    base_formation: str | None = None
    base_play: str | None = None
    coverage_family_hint: str | None = None
    valid: bool = False
    complete: bool = False
    experimental: bool = False
    missing_fields: list[str] = field(default_factory=list)
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
            "base_formation": self.base_formation,
            "base_play": self.base_play,
            "coverage_family_hint": self.coverage_family_hint,
            "valid": self.valid,
            "complete": self.complete,
            "experimental": self.experimental,
            "missing_fields": list(self.missing_fields),
            "reasons": list(self.reasons),
        }


def _coverage_hint_from_settings(researched: Mapping[str, str], shell_pair: str, base_play: str | None) -> str | None:
    """Infer the coverage family this adjustment is built around."""
    for key in ("Coverage Shell", "Coverage", "Coverage Shading", "Alignment"):
        val = researched.get(key)
        if val and str(val).lower() != "default":
            fam = call_family(str(val))
            if fam:
                return fam
    # Fall back to researched base / shell play name.
    for text in (base_play, shell_pair):
        fam = call_family(text)
        if fam:
            return fam
    return None


def inspect_armed_macro(mid: str) -> ArmedMacroView:
    """Validate one armed defense macro against the complete editor inventory.

    Pool membership alone is not enough. Experimental / benched macros are
    ineligible unless explicitly authorized. Every required editor field must
    have an explicit researched value or Default.
    """
    key = (mid or "").strip().upper()
    meta = pool_macro(key) or {}
    reasons: list[str] = []
    missing_fields: list[str] = []
    experimental = bool(is_experimental_macro(key) or key in EXPERIMENTAL_MACROS)

    if not meta:
        return ArmedMacroView(
            mid=key,
            xbox_name=key or "?",
            families=[],
            when_to_arm="",
            settings=[],
            researched_settings={},
            buttons="",
            shell_pair="",
            valid=False,
            complete=False,
            experimental=experimental,
            reasons=["not in research macro pool — not usable"],
        )

    if experimental and not experimental_macros_authorized():
        reasons.append(
            "experimental/benched macro — ineligible without explicit authorized configuration"
        )

    rows = settings_rows(key)
    # Index settings by (section, setting) lower-case.
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        sec = str(r.get("section") or "").strip()
        name = str(r.get("setting") or "").strip()
        if not name:
            continue
        by_key[(sec.lower(), name.lower())] = r

    required = required_editor_fields()
    if not required:
        reasons.append("editor field inventory unavailable — cannot validate completeness")
    else:
        for sec, name in required:
            row = by_key.get((sec.lower(), name.lower()))
            if row is None:
                missing_fields.append(f"{sec}/{name}")
                continue
            val = str(row.get("value") or "").strip()
            if not val:
                missing_fields.append(f"{sec}/{name} (empty value)")
            # Explicit Default is OK; researched non-default is OK.
        if missing_fields:
            reasons.append(
                f"incomplete editor settings: {len(missing_fields)} required field(s) missing"
            )

    researched = {
        str(r.get("setting") or ""): str(r.get("value") or "")
        for r in rows
        if r.get("source") and r.get("source") != "default" and r.get("setting")
    }

    # Internal conflicts / unsupported markers.
    by_setting: dict[str, set[str]] = {}
    for r in rows:
        if r.get("source") == "default":
            continue
        setting = str(r.get("setting") or "")
        val = str(r.get("value") or "")
        if setting and val and val.lower() != "default":
            by_setting.setdefault(setting, set()).add(val)
        conf = str(r.get("confidence") or "").lower()
        if conf == "conflict":
            reasons.append(f"research conflict on {setting or '?'}")
    for setting, vals in by_setting.items():
        if len(vals) > 1:
            reasons.append(f"internal setting conflict on {setting}: {sorted(vals)}")

    base = meta.get("base") or {}
    base_form = str(base.get("formation") or "").strip() or None
    base_play = str(base.get("play") or "").strip() or None
    if base_play and base_play.lower() == "any":
        base_play = None
    shell = meta.get("shell_pair") or " — ".join(x for x in (base_form, base.get("play")) if x)
    cov_hint = _coverage_hint_from_settings(researched, str(shell or ""), base_play)

    complete = not missing_fields and bool(required)
    # valid requires: in pool, not experimental (unless authorized), complete inventory, no conflicts.
    blocking = [
        r
        for r in reasons
        if "experimental" in r.lower()
        or "conflict" in r.lower()
        or "incomplete" in r.lower()
        or "not in research" in r.lower()
        or "inventory unavailable" in r.lower()
    ]
    valid = bool(meta) and complete and not blocking

    return ArmedMacroView(
        mid=key,
        xbox_name=display_name(key) if key else key,
        families=list(families(key)),
        when_to_arm=str(meta.get("when_to_arm") or meta.get("when") or ""),
        settings=rows,
        researched_settings=researched,
        buttons=activate_buttons(key) if key else "",
        shell_pair=str(shell or ""),
        base_formation=base_form,
        base_play=base_play,
        coverage_family_hint=cov_hint,
        valid=valid,
        complete=complete,
        experimental=experimental,
        missing_fields=missing_fields[:12],
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
            conflicts.append(f"{setting}: {a.mid}={va!r} vs {b.mid}={vb!r}")
    fa, fb = set(a.families), set(b.families)
    if "scram" in fa and "pressure" in fb and a.mid != b.mid:
        if "SPY" in a.mid and "HEAT" in b.mid:
            conflicts.append("SPY contain vs HEAT pressure — do not stack")
        if "SPY" in b.mid and "HEAT" in a.mid:
            conflicts.append("SPY contain vs HEAT pressure — do not stack")
    return conflicts


def play_adjustment_compatible(
    view: ArmedMacroView,
    *,
    formation: str | None,
    play: str | None,
    book: Mapping[str, Sequence[str]] | None = None,
) -> tuple[bool, str | None]:
    """True when the adjustment can ride with this specific formation/play.

    Uses researched base, shell pairing, coverage family, and book membership.
    When compatibility cannot be established, returns False so the caller can
    recommend the play without the adjustment.
    """
    if not view.valid:
        return False, "; ".join(view.reasons) or "adjustment invalid"
    if not formation or not play:
        return False, "missing formation/play for compatibility check"

    play_fam = call_family(play)
    # Coverage family mismatch: Cover 6 play + Cover 3-specific adjustment.
    if view.coverage_family_hint and play_fam:
        if view.coverage_family_hint != play_fam:
            # Soft allow when adjustment families are concept counters, not coverage.
            # Hard reject when researched Coverage Shell / base play names a different family.
            shell_names_coverage = bool(
                view.researched_settings.get("Coverage Shell")
                or view.researched_settings.get("Coverage")
                or view.base_play
            )
            if shell_names_coverage:
                return (
                    False,
                    f"coverage mismatch: play={play_fam} adj={view.coverage_family_hint}",
                )

    # Researched base play must match when it names a specific call (not "any").
    if view.base_play:
        bp = view.base_play.lower()
        if bp not in ("any", "default") and bp not in play.lower() and play.lower() not in bp:
            # Token match (e.g. "Tampa 2" vs "Tampa 2 Sink")
            tokens = [t for t in re.split(r"[/|,]", view.base_play) if t.strip()]
            if tokens and not any(t.strip().lower() in play.lower() for t in tokens if len(t.strip()) >= 4):
                return False, f"researched base play {view.base_play!r} ≠ selected {play!r}"

    # Researched base formation: require same front family when named.
    if view.base_formation:
        bf = view.base_formation.lower()
        if bf not in ("any", "4-man", "default") and bf not in formation.lower():
            # Allow nickel↔nickel style partials.
            if not any(
                tok in formation.lower()
                for tok in ("nickel", "dime", "4-3", "3-4", "46")
                if tok in bf
            ):
                return False, f"researched base formation {view.base_formation!r} ≠ {formation!r}"

    # Play must be in the applied book when a book is provided.
    if book is not None:
        plays = book.get(formation)
        if plays is None or play not in list(plays):
            return False, f"{formation}/{play} not in applied defensive book"

    return True, None


def compatible_macros(
    armed: Sequence[str],
    *,
    preferred_families: Sequence[str] | None = None,
    already_armed: str | None = None,
    formation: str | None = None,
    play: str | None = None,
    book: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Rank valid armed macros; skip invalid and play-incompatible ones."""
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
        if formation and play:
            ok, why = play_adjustment_compatible(
                view, formation=formation, play=play, book=book
            )
            if not ok:
                out.append(
                    {
                        **view.to_dict(),
                        "eligible": False,
                        "score": -1.0,
                        "why": why or "incompatible with selected play",
                        "play_incompatible": True,
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
    formation: str | None = None,
    play: str | None = None,
    book: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Pick at most one armed CA compatible with the selected play.

    If no compatible adjustment exists, returns macro=None so the play can
    still be recommended alone.
    """
    fam_weights = dict(concept_families or {})
    preferred = [f for f, w in sorted(fam_weights.items(), key=lambda kv: -kv[1]) if w > 0.08][:4]
    ranked = compatible_macros(
        armed,
        preferred_families=preferred,
        already_armed=None,
        formation=formation,
        play=play,
        book=book,
    )
    eligible = [r for r in ranked if r.get("eligible")]
    if not eligible:
        incompat = [r for r in ranked if r.get("play_incompatible")]
        return {
            "macro": None,
            "reason": (
                "no play-compatible armed Custom Adjustment"
                if incompat
                else "no valid armed Custom Adjustment"
            ),
            "incompatibility": (
                "; ".join((r.get("why") or "") for r in incompat[:3])
                if incompat
                else "none armed or all conflicted/invalid/incomplete"
            ),
            "candidates": ranked[:5],
        }
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
                    "complete": bool(row.get("complete")),
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
        "complete": bool(top.get("complete")),
        "buttons": top.get("buttons"),
    }
