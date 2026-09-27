"""Playbook delta engine — inventory (seed books) vs opponent-aware proposed tweaks.

Assumes BAMA META O/D are already fully stocked with every formation/play in seed.json
and the 8 active + 2 benched macros. Prep never asks to CREATE books or re-ADD all macros.
Only opponent-specific ADD/REMOVE/EDIT/BENCH/UNBENCH deltas are emitted,
and only when at least meta_grounded (CFB27 meta + film) — no ungrounded invention.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

from cfb_coach.db import CoachDB
from cfb_coach.gameplan import effective_gameplan, load_baseline
from cfb_coach.seed import load_seed
from cfb_coach.macros import (
    PROVEN,
    META_GROUNDED,
    USER_ACTIVE_CAP,
    active_loadout_cards,
    count_active,
    enrich_macro_delta,
    get_macro,
    is_prep_eligible,
    load_macro_catalog,
    normalize_status,
    validation_status,
)
from cfb_coach.opponents import is_cpu_opponent


# ---------------------------------------------------------------------------
# Inventory (current books — already stocked)
# ---------------------------------------------------------------------------

def build_inventory(seed: dict[str, Any] | None = None, bl: dict[str, Any] | None = None) -> dict[str, Any]:
    """Current inventory = seed playbooks + baseline macro set."""
    seed = seed or load_seed()
    bl = bl or load_baseline()
    pb = seed["playbooks"]
    league = seed["league"]["online_baseline"]
    mb = bl["macros_baseline"]

    offense: dict[str, Any] = {}
    for name, meta in (pb.get("offense_formations") or {}).items():
        offense[name] = {
            "role": meta.get("role", ""),
            "plays": list(meta.get("core") or []),
            "audibles": list(meta.get("audibles") or []),
        }

    defense: dict[str, Any] = {}
    for name, meta in (pb.get("defense_packages") or {}).items():
        defense[name] = {
            "role": meta.get("role", ""),
            "calls": list(meta.get("calls") or []),
        }

    macros_active = list(league.get("defensive_macros_active") or mb.get("keep") or [])
    macros_benched = list(league.get("defensive_macros_benched") or mb.get("bench") or [])
    o_macros = list(league.get("offensive_macros_active") or [])
    recipes = copy.deepcopy(mb.get("recipes") or {})

    return {
        "offense_book": league.get("custom_offense", "BAMA META O"),
        "defense_book": league.get("custom_defense", "BAMA META D"),
        "offense": offense,
        "defense": defense,
        "macros_active": macros_active,
        "macros_benched": macros_benched,
        "offensive_macros": o_macros,
        "macro_recipes": recipes,
        "note": league.get("note", ""),
    }


# ---------------------------------------------------------------------------
# Delta helpers
# ---------------------------------------------------------------------------

def _delta(
    action: str,
    target: str,
    detail: str,
    *,
    kind: str = "playbook",
    field: str = "",
    before: str = "",
    after: str = "",
    why: str = "",
) -> dict[str, str]:
    return {
        "action": action.upper(),
        "kind": kind,  # playbook | macro
        "target": target,
        "field": field,
        "before": before,
        "after": after,
        "detail": detail,
        "why": why,
    }


def _clean_purpose(text: str) -> str:
    """Strip leftover 'CREATE candidate —' wording from baseline recipe blurbs."""
    t = (text or "").strip()
    for prefix in ("CREATE candidate — ", "CREATE candidate - ", "CREATE candidate —", "CREATE — "):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    return t


def _delta_key(d: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        d.get("action", ""),
        d.get("target", ""),
        d.get("field", ""),
        d.get("detail", ""),
    )


# ---------------------------------------------------------------------------
# Opponent-aware proposed tweaks (diffs only — never recreate)
# ---------------------------------------------------------------------------

def propose_deltas(
    opponent_id: str,
    opp: dict[str, Any] | None = None,
    *,
    inventory: dict[str, Any] | None = None,
    dynasty: str | None = None,
) -> list[dict[str, str]]:
    """Opponent-specific adjustments relative to fully-stocked inventory."""
    inv = inventory or build_inventory()
    opp = opp or {}
    oid = (opponent_id or "").lower()
    dvs = opp.get("defense_vs_us") or {}
    arch = (dvs.get("archetype") or "").lower()
    offense = opp.get("offense") or {}
    deltas: list[dict[str, str]] = []

    bunch_aud = list((inv["offense"].get("Gun Bunch X Nasty") or {}).get("audibles") or [])
    cluster_aud = list((inv["offense"].get("Gun Cluster") or {}).get("audibles") or [])
    heat = (inv["macro_recipes"].get("HEAT") or {}).copy()

    # --- Gavin / split-field / Cover 6 users ---
    if oid == "gavin" or ("split_field" in arch) or ("cover_6" in arch and oid != "cpu"):
        # Elevate run audibles on Bunch
        new_bunch = ["Inside Zone", "HB Base", "Mesh Spot", "Counter Y"]
        if bunch_aud != new_bunch:
            deltas.append(
                _delta(
                    "EDIT",
                    "Gun Bunch X Nasty",
                    f"Audibles → {', '.join(new_bunch)} (run elevated; flood/cross demoted early)",
                    kind="playbook",
                    field="audibles",
                    before=", ".join(bunch_aud) or "(none)",
                    after=", ".join(new_bunch),
                    why="Gavin/split-field — RUN FIRST vs C6/C9",
                )
            )
        # Demote Mesh Post from early Cluster audibles
        new_cluster = [a for a in cluster_aud if a != "Mesh Post"]
        if "Outside Zone" not in new_cluster:
            new_cluster = ["Outside Zone"] + new_cluster
        if "Z Spot Shake" not in new_cluster:
            new_cluster.append("Z Spot Shake")
        # keep unique, max ~4
        seen: list[str] = []
        for a in new_cluster:
            if a not in seen:
                seen.append(a)
        new_cluster = seen[:4]
        if "Mesh Post" in cluster_aud or cluster_aud != new_cluster:
            deltas.append(
                _delta(
                    "EDIT",
                    "Gun Cluster",
                    f"Audibles → {', '.join(new_cluster)} (Mesh Post demoted early vs Cover 6)",
                    kind="playbook",
                    field="audibles",
                    before=", ".join(cluster_aud) or "(none)",
                    after=", ".join(new_cluster),
                    why="avoid C6 INT film on Mesh Post",
                )
            )
        # EDIT HEAT purpose / when for patience
        new_heat_when = (
            "Rare vs Gavin — only 3rd-short after clear pressure-worthy tell; "
            "never sell out early (he PA/scrambles the heat)"
        )
        if heat.get("when_to_arm") != new_heat_when:
            deltas.append(
                _delta(
                    "EDIT",
                    "HEAT",
                    "Tighten when-to-arm: selective only, never early-down chase",
                    kind="macro",
                    field="when_to_arm",
                    before=heat.get("when_to_arm", ""),
                    after=new_heat_when,
                    why="split-field patience on D",
                )
            )
        # ADD one new macro recipe (not re-adding the 8)
        if "CONTAIN-SCRAM" not in inv["macros_active"] and "CONTAIN-SCRAM" not in inv["macros_benched"]:
            rec = inv["macro_recipes"].get("CONTAIN-SCRAM") or {}
            deltas.append(
                _delta(
                    "ADD",
                    "CONTAIN-SCRAM",
                    f"New recipe — PURPOSE={_clean_purpose(rec.get('purpose', 'contain + spy lean'))}. "
                    f"WHEN={rec.get('when_to_arm', 'vs escape artist')}. "
                    f"Shell={rec.get('shell_pair', 'Nickel Over')}; User={rec.get('user_job', 'contain')}",
                    kind="macro",
                    field="recipe",
                    after="active custom adj",
                    why="Gavin escape threat (patch 1.012 contain fix)",
                )
            )

    # --- Quen / pressure ---
    if oid == "quen" or "pressure" in arch:
        new_bunch = ["Mesh Spot", "Return Whip Trail", "HB Base", "Inside Zone"]
        # Return Whip Trail may not be in audibles — that's an EDIT of audible slots
        if bunch_aud != new_bunch:
            deltas.append(
                _delta(
                    "EDIT",
                    "Gun Bunch X Nasty",
                    f"Audibles → {', '.join(new_bunch)} (protection/hot answers elevated)",
                    kind="playbook",
                    field="audibles",
                    before=", ".join(bunch_aud) or "(none)",
                    after=", ".join(new_bunch),
                    why="Quen/pressure — Mesh Spot / Whip / HB Base",
                )
            )
        if "PROT" not in inv.get("offensive_macros", []):
            rec = inv["macro_recipes"].get("PROT") or {}
            deltas.append(
                _delta(
                    "ADD",
                    "PROT",
                    f"O-side protection/hot cue — PURPOSE={_clean_purpose(rec.get('purpose', 'protection/hot'))}. "
                    f"WHEN={rec.get('when_to_arm', 'vs pressure')}. Pair Mesh Spot/Whip/HB Base.",
                    kind="macro",
                    field="recipe",
                    after="offensive macro",
                    why="protection macro lean",
                )
            )
        new_heat_when = (
            "3rd-short selective Cub/Mug vs Quen — not every snap; "
            "elevate CROSS/RPO/SCRAM readiness first"
        )
        if heat.get("when_to_arm") != new_heat_when:
            deltas.append(
                _delta(
                    "EDIT",
                    "HEAT",
                    "Selective Cub/Mug only; prioritize CROSS/RPO/SCRAM readiness",
                    kind="macro",
                    field="when_to_arm",
                    before=heat.get("when_to_arm", ""),
                    after=new_heat_when,
                    why="Cross Wheels + bubbles + scramble",
                )
            )

    # --- Tiano / C2-C3 ---
    if oid == "tiano" or "c2_c3" in arch:
        deltas.append(
            _delta(
                "EDIT",
                "Gun Bunch X Nasty",
                "RZ audible lean: prefer Mesh Spot / Z Spot GoalLine over auto-whip near scoring",
                kind="playbook",
                field="audibles",
                before=", ".join(bunch_aud) or "(none)",
                after="Mesh Spot, Inside Zone, Z Spot GoalLine, Deep Flood",
                why="C2-heavy near scoring — possession > hero",
            )
        )
        deltas.append(
            _delta(
                "EDIT",
                "HEAT",
                "GL: Even 6-1 OK; do NOT sell out run (Z Smash lesson). BUNCH/CROSS situational only",
                kind="macro",
                field="when_to_arm",
                before=heat.get("when_to_arm", ""),
                after="Avoid HEAT sellout at GL vs Tiano; situational BUNCH/CROSS only",
                why="Temple/Tiano GL lesson",
            )
        )

    # --- CPU ---
    if oid == "cpu" or "two_high_money" in arch:
        new_bunch = ["Mesh Spot", "Drive HB Under", "Inside Zone", "Deep Flood"]
        # Drive HB Under may not be audible — still an EDIT instruction
        if bunch_aud != new_bunch:
            deltas.append(
                _delta(
                    "EDIT",
                    "Gun Bunch X Nasty",
                    f"Audibles → {', '.join(new_bunch)} (no forced Mesh Post/Whip on money)",
                    kind="playbook",
                    field="audibles",
                    before=", ".join(bunch_aud) or "(none)",
                    after=", ".join(new_bunch),
                    why="CPU sticks coverage — take free underneath",
                )
            )

    # Film-driven macro field tweaks (EDIT only — macros already exist)
    blob = " ".join(
        str(v)
        for k, v in offense.items()
        if isinstance(v, (str, list))
        for v in (v if isinstance(v, list) else [v])
    ).lower()

    if "vertical" in blob or "four vert" in blob:
        vert = inv["macro_recipes"].get("VERT") or {}
        deltas.append(
            _delta(
                "EDIT",
                "VERT",
                "Arm readiness elevated — after 2+ live vertical tells → Quarters + VERT | User #3 seam",
                kind="macro",
                field="when_to_arm",
                before=vert.get("when_to_arm", ""),
                after="After 2+ live vertical tells this game — prioritize Quarters + VERT",
                why="film verticals",
            )
        )
    if "cross" in blob or "wheel" in blob:
        cross = inv["macro_recipes"].get("CROSS") or {}
        deltas.append(
            _delta(
                "EDIT",
                "CROSS",
                "Arm readiness elevated — after 2+ live crosser/wheel → Tampa/Quarters + CROSS",
                kind="macro",
                field="when_to_arm",
                before=cross.get("when_to_arm", ""),
                after="After 2+ live Cross Wheels / crosser tells — prioritize CROSS",
                why="film crossers",
            )
        )
    if "rpo" in blob or "bubble" in blob:
        rpo = inv["macro_recipes"].get("RPO") or {}
        deltas.append(
            _delta(
                "EDIT",
                "RPO",
                "Arm readiness elevated — after repeated bubble → C3 Sky + RPO | User flat",
                kind="macro",
                field="when_to_arm",
                before=rpo.get("when_to_arm", ""),
                after="After repeated bubble/RPO wins — prioritize RPO",
                why="film RPO/bubble",
            )
        )
    if "scram" in blob or "escape" in (offense.get("escape") or "").lower():
        if not any(d["target"] == "CONTAIN-SCRAM" and d["action"] == "ADD" for d in deltas):
            if "CONTAIN-SCRAM" not in inv["macros_active"]:
                rec = inv["macro_recipes"].get("CONTAIN-SCRAM") or {}
                deltas.append(
                    _delta(
                        "ADD",
                        "CONTAIN-SCRAM",
                        f"New recipe — PURPOSE={_clean_purpose(rec.get('purpose', 'contain integrity'))}. "
                        f"Patch 1.012 contain adj fix.",
                        kind="macro",
                        field="recipe",
                        after="active custom adj",
                        why="escape film",
                    )
                )

    # Deduplicate by key while preserving order
    seen_keys: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for d in deltas:
        k = _delta_key(d)
        if k not in seen_keys:
            seen_keys.add(k)
            unique.append(d)

    # Attach validation + copy blocks + swap plans for macro deltas
    oid = (opponent_id or "").lower()
    enriched: list[dict[str, Any]] = []
    for d in unique:
        if d.get("kind") == "macro":
            # _delta uses "target"; enrich expects that
            payload = {
                "action": d.get("action", ""),
                "target": d.get("target", ""),
                "detail": d.get("detail", ""),
            }
            e = enrich_macro_delta(payload, inv, opponent_id=oid)
            d = dict(d)
            d["validated_status"] = e.get("validated_status") or validation_status(d.get("target", ""))
            if e.get("copy_block"):
                d["copy_block"] = e["copy_block"]
            if e.get("full_settings"):
                d["full_settings"] = e["full_settings"]
            if e.get("swap_plan"):
                d["swap_plan"] = e["swap_plan"]
            if e.get("xbox_steps"):
                d["xbox_steps"] = e["xbox_steps"]
            if e.get("purpose"):
                d["purpose"] = e["purpose"]
            # Keep detail (may include swap hint)
            if e.get("detail") and d.get("action") == "ADD":
                d["detail"] = e["detail"]
        else:
            # Playbook deltas — opponent film/meta overlays are at least meta_grounded
            d = dict(d)
            d["validated_status"] = normalize_status(
                d.get("validated_status") or META_GROUNDED
            )
        # Prep must ONLY suggest ≥ meta_grounded (no ungrounded invention)
        if is_prep_eligible(str(d.get("validated_status") or "")):
            enriched.append(d)

    # Alabama (serious): drop ADD of non-proven / experimental macros unless already active.
    # Ohio State (experimental): keep meta_grounded CREATE candidates in prep deltas.
    try:
        from cfb_coach.dynasty import allow_experimental, normalize_dynasty, DEFAULT_DYNASTY

        mode = normalize_dynasty(dynasty or DEFAULT_DYNASTY)
        if not allow_experimental(mode):
            filtered: list[dict[str, Any]] = []
            for d in enriched:
                if (d.get("action") or "").upper() != "ADD":
                    filtered.append(d)
                    continue
                st = normalize_status(str(d.get("validated_status") or ""))
                if st == PROVEN:
                    filtered.append(d)
                else:
                    # keep as tip-only? skip from actionable prep deltas
                    continue
            enriched = filtered
    except Exception:
        pass

    # CPU games: offense-only — drop defense macro deltas (keep O playbook edits)
    if is_cpu_opponent(opponent_id):
        kept: list[dict[str, Any]] = []
        for d in enriched:
            if d.get("kind") == "macro":
                side = (d.get("side") or "").lower()
                if not side:
                    meta = get_macro(d.get("target") or "") or {}
                    side = (meta.get("side") or "defense").lower()
                if side == "defense":
                    continue
            # Drop D-package playbook edits if any slip through targeting defense packages
            tgt = (d.get("target") or "").lower()
            if d.get("kind") == "playbook" and any(
                x in tgt for x in ("nickel", "dime", "4-3", "cover ", "even 6")
            ):
                continue
            kept.append(d)
        enriched = kept
    return enriched


def call_emphasis_tips(opponent_id: str, opp: dict[str, Any] | None = None) -> list[str]:
    """Short call-emphasis tips — not a recreate list."""
    opp = opp or {}
    oid = (opponent_id or "").lower()
    dvs = opp.get("defense_vs_us") or {}
    arch = (dvs.get("archetype") or "").lower()
    tips: list[str] = []

    if oid == "gavin" or "split_field" in arch:
        tips.extend(
            [
                "Early downs: RUN FIRST (IZ / HB Base / Counter) vs C6/C9",
                "Pass: Mesh Spot easy; Cluster Z Spot Shake when Bunch overplayed",
                "Avoid Mesh Post spam into Cover 6",
                "D: base Nickel Over zones; VERT only after repeated 4-verts; SCRAM ready",
                "Cover 2 Invert early → Mesh Spot / IZ — not Deep Flood",
            ]
        )
    elif oid == "quen" or "pressure" in arch:
        tips.extend(
            [
                "Protection + hot: Mesh Spot, Return Whip Trail, HB Base",
                "Don't hero into Sam/Will/WS blitz",
                "D: CROSS / RPO / SCRAM elevated; HEAT selective on 3rd-short only",
            ]
        )
    elif oid == "tiano" or "c2_c3" in arch:
        tips.extend(
            [
                "Near scoring: Mesh Spot / Z Spot Shake / Deuce Duo — possession first",
                "Tag whip by leverage, not auto",
                "GL D: Even 6-1 OK; do not sell out run",
            ]
        )
    elif oid == "cpu" or "two_high_money" in arch:
        tips.extend(
            [
                "CPU game = offense-only coaching (no D calls / no D macros)",
                "Take free underneath — Mesh Spot / Drive HB Under / Inside Zone",
                "No forced Mesh Post/Whip into sticks coverage",
            ]
        )
    else:
        tips.extend(
            [
                "Thin film: establish run + Mesh Spot easy + Cluster mix",
                "Two-high → run; pressure → quick; C3 → flood once confirmed",
                "Macros situational — default = none until repeated tendency",
            ]
        )
    return tips


# ---------------------------------------------------------------------------
# Applied-diff persistence
# ---------------------------------------------------------------------------

def filter_new_deltas(
    proposed: list[dict[str, str]],
    applied: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Return proposed deltas not yet in the last-applied set."""
    if not applied:
        return list(proposed)
    applied_keys = {_delta_key(d) for d in applied}
    return [d for d in proposed if _delta_key(d) not in applied_keys]


def get_applied_deltas(db: CoachDB | None, opponent_id: str) -> list[dict[str, str]]:
    if db is None:
        return []
    sheet = db.get_install_sheet(opponent_id)
    if not sheet:
        return []
    return list(sheet.get("applied_deltas") or [])


def mark_prep_applied(
    db: CoachDB,
    opponent_id: str,
    deltas: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Mark deltas as applied so next prep only shows NEW changes."""
    sheet = db.get_install_sheet(opponent_id) or {}
    if deltas is None:
        deltas = list(sheet.get("proposed_deltas") or sheet.get("steps") or [])
    sheet["applied_deltas"] = deltas
    sheet["applied_ts"] = datetime.now(timezone.utc).isoformat()
    sheet["version"] = load_baseline().get("version", "?")
    db.save_install_sheet(opponent_id, sheet)
    return sheet


def save_prep_deltas(
    db: CoachDB,
    opponent_id: str,
    proposed: list[dict[str, str]],
    shown: list[dict[str, str]],
) -> dict[str, Any]:
    """Persist proposed + shown deltas; log newly shown rows."""
    prev = db.get_install_sheet(opponent_id) or {}
    applied = list(prev.get("applied_deltas") or [])
    payload = {
        "version": load_baseline().get("version", "?"),
        "ts": datetime.now(timezone.utc).isoformat(),
        "proposed_deltas": proposed,
        "shown_deltas": shown,
        "applied_deltas": applied,
        "new_since_last": shown,
        # keep legacy key for any old readers
        "steps": proposed,
    }
    db.save_install_sheet(opponent_id, payload)
    for d in shown:
        db.log_install_diff(
            opponent_id=opponent_id,
            action=d["action"],
            target=d["target"],
            detail=d["detail"],
            why=d.get("why", ""),
        )
    return payload


# ---------------------------------------------------------------------------
# Public build API
# ---------------------------------------------------------------------------

def build_prep_plan(
    opponent_id: str,
    opp: dict[str, Any] | None = None,
    *,
    db: CoachDB | None = None,
    persist: bool = True,
    dynasty: str | None = None,
    offline: bool = False,
    refresh_meta: bool = False,
) -> dict[str, Any]:
    """Build inventory + proposed deltas + tips; optionally persist.

    Runs live meta scout by default (unless offline) to enrich tips/deltas.
    """
    from cfb_coach.dynasty import (
        DEFAULT_DYNASTY,
        allow_experimental,
        doctrine_line,
        dynasty_config,
        get_session_dynasty,
        normalize_dynasty,
    )

    seed = load_seed()
    bl = load_baseline()
    inv = build_inventory(seed, bl)
    opp = opp or {}
    if dynasty is None and db is not None:
        dynasty = get_session_dynasty(db)
    dynasty = normalize_dynasty(dynasty or DEFAULT_DYNASTY)
    dcfg = dynasty_config(dynasty)
    proposed = propose_deltas(
        opponent_id, opp, inventory=inv, dynasty=dynasty
    )
    applied = get_applied_deltas(db, opponent_id)
    shown = filter_new_deltas(proposed, applied)
    tips = call_emphasis_tips(opponent_id, opp)
    eg = effective_gameplan(opponent_id, db)

    # Live meta scout — rich prep edge (Alabama users ~once/season)
    scout_dict: dict[str, Any] = {}
    try:
        from cfb_coach.meta_scout import apply_scout_bias, run_meta_scout

        scout = run_meta_scout(offline=offline, refresh=refresh_meta)
        proposed, tips, affect = apply_scout_bias(
            proposed, scout, dynasty=dynasty, tips=tips
        )
        scout.affect_this_prep = list(affect)
        scout_dict = scout.to_dict() if hasattr(scout, "to_dict") else dict(
            getattr(scout, "__dict__", {})
        )
        shown = filter_new_deltas(proposed, applied)
    except Exception as exc:  # noqa: BLE001 — never break prep
        scout_dict = {
            "available": False,
            "offline": offline,
            "message": f"Scout unavailable — using cached/baseline cfb27-2026-09 ({type(exc).__name__})",
            "baseline_fallback": "cfb27-2026-09",
            "patch_notes": [],
            "meta_offense": [],
            "meta_defense": [],
            "suggestions": [],
            "sources": [],
            "affect_this_prep": [],
            "confidence": "low",
        }

    # Dynasty default Active-8 / benched for loadout display
    if dcfg.get("default_active"):
        inv["macros_active"] = list(dcfg["default_active"])
    if dcfg.get("default_benched"):
        inv["macros_benched"] = list(dcfg["default_benched"])

    offense_only = is_cpu_opponent(opponent_id)
    if offense_only:
        # Strip D emphasis tips already handled; tag plan
        tips = [t for t in tips if not t.strip().lower().startswith("d:")]
        if not any("offense-only" in t.lower() for t in tips):
            tips.insert(0, "CPU game = offense-only coaching (no D calls / no D macros)")

    # Recompute proposed with dynasty-aware inventory already done; filter shown for CPU
    if offense_only:
        proposed = [d for d in proposed if not (
            d.get("kind") == "macro"
            and ((d.get("side") or (get_macro(d.get("target") or "") or {}).get("side") or "defense") == "defense")
        )]
        shown = filter_new_deltas(proposed, applied)

    budget = count_active(inv)
    macro_cards, loadout = active_loadout_cards(
        inv, shown, offense_only=offense_only
    )
    # Slot budget reflects post-swap loadout meter
    if loadout.get("meter"):
        budget = dict(budget)
        budget["meter"] = loadout["meter"]
        budget["total"] = loadout.get("total", budget.get("total"))
        if offense_only:
            budget["defense_count"] = 0
            budget["offense_count"] = len(loadout.get("offense") or [])
            budget["at_cap"] = budget["total"] >= budget.get("cap", USER_ACTIVE_CAP)

    # Surface any ADD-at-cap swap plan at plan level for banner
    swap_banners = [
        d["swap_plan"]
        for d in shown
        if d.get("kind") == "macro" and d.get("swap_plan")
    ]
    replacing_lines = [
        f"replacing {r['bench']} with {r['add']}"
        for r in (loadout.get("replacing") or [])
    ]

    plan = {
        "opponent_id": opponent_id,
        "display_name": opp.get("display_name", opponent_id),
        "team": opp.get("team_now", "?"),
        "version": bl.get("version", "?"),
        "game": bl.get("game", "CFB 27"),
        "patch": bl.get("patch", ""),
        "overlay_depth": eg.overlay.depth,
        "inventory": inv,
        "proposed_deltas": proposed,
        "shown_deltas": shown,
        "applied_count": len(applied),
        "tips": tips,
        "ts": datetime.now(timezone.utc).isoformat(),
        "active_cap": USER_ACTIVE_CAP,
        "slot_budget": budget,
        "macro_cards": macro_cards,
        "loadout": loadout,
        "replacing_lines": replacing_lines,
        "offense_only": offense_only,
        "swap_banners": swap_banners,
        "macro_catalog_version": (load_macro_catalog().get("version") or "?"),
        "dynasty": dynasty,
        "dynasty_config": dcfg,
        "doctrine": doctrine_line(),
        "meta_scout": scout_dict,
    }
    # v1.12: learned zone plan (open / red zone / goal line) aligned to the meta
    try:
        from cfb_coach.meta_align import build_prep_alignment

        plan["zone_alignment"] = build_prep_alignment(
            db, opponent_id, scout_dict or None, dynasty=dynasty
        )
    except Exception as exc:  # noqa: BLE001 — never break prep
        plan["zone_alignment"] = {"error": f"{type(exc).__name__}: {exc}"}
    # v1.13: autonomous custom playbook of record (per dynasty), decided from the
    # live meta research + his own results; emits an edit list only when it changes.
    if db is not None:
        try:
            from cfb_coach.cfb_playbook import plan_book

            plan["cfb_book"] = plan_book(db, dynasty, scout_dict or None, persist=persist)
        except Exception as exc:  # noqa: BLE001 — never break prep
            plan["cfb_book"] = {"error": f"{type(exc).__name__}: {exc}"}
    # v1.14: CPU games are offense-only; when the loadout carries no offensive macros,
    # the coach picks up to 8 offense custom adjustments that fit the custom book + research.
    if offense_only and not plan["macro_cards"]:
        try:
            from cfb_coach.macros import offense_book_loadout

            tp = (plan.get("cfb_book") or {}).get("target_plays") or {}
            if tp:
                sd = scout_dict or {}
                live_ok = (sd.get("mode") or "") in ("live", "cache")
                cards, lo = offense_book_loadout(tp, concept_signals=sd.get("concept_signals") if live_ok else None,
                                                 rz_signals=sd.get("rz_signals") if live_ok else None)
                plan["macro_cards"], plan["loadout"] = cards, lo
                plan["slot_budget"] = dict(plan["slot_budget"], meter=lo["meter"], total=lo["total"],
                                           offense_count=lo["total"], defense_count=0, at_cap=lo["total"] >= USER_ACTIVE_CAP)
        except Exception:  # noqa: BLE001 — never break prep
            pass
    if db is not None and persist:
        save_prep_deltas(db, opponent_id, proposed, shown)
    return plan


def format_book_section_text(book: dict[str, Any] | None) -> list[str]:
    """Terminal version of the custom playbook panel (details)."""
    book = book or {}
    if not book:
        return []
    if book.get("error"):
        return [f"## Custom playbook unavailable ({book['error']})"]
    lines = [f"## Custom playbook of record — {book.get('name')} ({book.get('dynasty')}) — formation level"]
    cur = book.get("current") or {}
    pend = book.get("pending") or {}
    if book.get("seeded_now"):
        lines.append(f"  Seeded rev {cur.get('rev')}: {book.get('seed_summary')}")
    if pend:
        lines.append(
            f"  STATUS: {len(book.get('edits') or [])} PENDING FORMATION CHANGE(S) (rev {pend.get('rev')}). Live calls use "
            + (f"rev {cur.get('rev')} until you run `book apply`." if cur else "the pending book (unconfirmed) until you run `book apply`.")
        )
        lines.append("  EDIT LIST (CFB 27 > Create & Share > Custom Playbooks):")
        lines += ["    " + ln for ln in (book.get("edit_text") or "").splitlines()]
    else:
        lines.append(f"  STATUS: no changes this prep — rev {cur.get('rev')} stands (no churn).")
    lines.append("  FORMATIONS (why):")
    for r in book.get("formation_table") or []:
        lines.append(f"    [{r['flag'] or '-'}] {r['formation']} ({r.get('source_book') or 'any'}, {r['n_plays']} plays) "
                     f"meta {r['meta']:+.2f}; {r['stats']} — {(r.get('why') or '')[:160]}")
    lines.append("  RESULTING BOOK:")
    lines += ["    " + ln for ln in (book.get("book_text") or "").splitlines()]
    flagged = [(k, f) for k, f in (book.get("flags") or {}).items() if f.get("flag") in ("demoted", "on_notice", "fading")]
    for k, f in flagged:
        lines.append(f"  [{f['flag']}] {k.replace('::', ' — ')}: {f['why']}")
    return lines


def format_prep_minimal_text(plan: dict[str, Any]) -> str:
    """v1.14 terminal/phone twin of the minimal prep page: formations, audibles, macros."""
    from cfb_coach.prep_book_html import research_failed, research_status_line

    scout = plan.get("meta_scout") or {}
    book = plan.get("cfb_book") or {}
    lines = [f"# PREP — vs {plan.get('display_name')} | {plan.get('game', 'CFB 27')} | dynasty {plan.get('dynasty')}"]
    if research_failed(scout):
        lines.append(f"!! LIVE META RESEARCH DID NOT RUN THIS PREP — {scout.get('message') or ''}")
    lines.append(research_status_line(scout))
    lines.append("")
    if book.get("error"):
        lines.append(f"## Formations: unavailable ({book['error']})")
    elif book:
        lines.append(f"## Formations — {book.get('name')}")
        label = {"new": "NEW    ", "applied": "in book", "remove": "REMOVE ", "changed": "RE-ADD "}
        for r in book.get("formation_list") or []:
            lines.append(f"  [{label.get(r['status'], r['status'])}] {r['formation']}  ({r.get('source_book') or 'any'} playbook, {r['n_plays']} plays)")
            if r.get("note"):
                lines.append(f"            {r['note']}")
        if book.get("pending") and book.get("apply_cmd"):
            lines.append(f"  After making the changes in CFB 27: {book['apply_cmd']}")
        lines.append("")
        lines.append("## Audibles (4 per formation)")
        for f, ps in (book.get("audibles") or {}).items():
            lines.append(f"  {f}: " + " | ".join(ps))
        lines.append("")
    lo = plan.get("loadout") or {}
    lines.append(f"## Macros — {lo.get('meter') or (plan.get('slot_budget') or {}).get('meter') or ''}")
    for c in plan.get("macro_cards") or []:
        lines.append(f"  {c.get('name')} ({c.get('side')})" + (f" — {c['why']}" if c.get("why") else ""))
        for ln in (c.get("copy_block") or "").splitlines():
            lines.append(f"      {ln}")
    if not plan.get("macro_cards"):
        lines.append("  (none)")
    if plan.get("details_path"):
        lines.append("")
        lines.append(f"Details: {plan['details_path']}")
    return "\n".join(lines)


def format_zone_alignment_text(za: dict[str, Any] | None) -> list[str]:
    """Terminal version of the learned-zone + meta alignment panels."""
    za = za or {}
    if not za or za.get("error"):
        return [f"## Zone plan unavailable ({za.get('error', 'no data')})"] if za else []
    labels = {"open": "Open field", "rz": "Red zone (inside the 20)", "gl": "Goal line / goal-to-go"}
    lines = ["## Zone plan (your learned results + current meta)"]
    for zone in ("rz", "gl", "open"):
        rows = (za.get("zone_plan") or {}).get(zone) or []
        lines.append(f"  {labels[zone]}:")
        for r in rows:
            lines.append(
                f"    {r['p']:.0%}  {r['formation']} — {r['play']}  (yours {r['learned']:+.2f}, n={r['n_zone']}; meta {r['meta']:+.2f})"
            )
    if za.get("conflicts"):
        lines.append("## Meta vs your data — conflicts")
        lines += [f"  ! {c['message']}" for c in za["conflicts"]]
    if za.get("support"):
        lines += [f"  + {c['message']}" for c in za["support"][:4]]
    if za.get("lab_candidates"):
        lines.append("## Ohio State lab candidates (verified in the OSU book, untested by you)")
        lines += [f"  ? {c['formation']} — {c['play']}: {c.get('note', '')}" for c in za["lab_candidates"]]
    li = za.get("learning") or {}
    if li:
        lines.append(f"Retrain rules {li.get('rules_version')} (DB rebuilt {li.get('rebuilt_at') or 'n/a'})")
    return lines


def format_delta_text(plan: dict[str, Any]) -> str:
    """Compact terminal dump of deltas only (for --text)."""
    shown = plan["shown_deltas"]
    pb = [d for d in shown if d.get("kind") == "playbook"]
    mac = [d for d in shown if d.get("kind") == "macro"]
    budget = plan.get("slot_budget") or count_active(plan.get("inventory") or {})
    lines = [
        f"# PREP — vs {plan['display_name']} ({plan['team']})  |  {plan['game']} / {plan['version']}",
        f"Dynasty: {plan.get('dynasty', 'alabama')}"
        + (
            " [experimental]"
            if (plan.get("dynasty_config") or {}).get("experimental_badge")
            else ""
        ),
        plan.get("doctrine")
        or "Doctrine: do NOT auto-use macros from one concept appearance — most snaps Cover 3 Sky / Quarters / Tampa 2 with no macro.",
        f"Books assumed stocked: {plan['inventory']['offense_book']} / {plan['inventory']['defense_book']}",
        f"Active loadout: {', '.join((plan.get('loadout') or {}).get('defense') or plan['inventory'].get('macros_active') or []) or '(none)'}"
        + (
            "  |  O: " + ", ".join((plan.get("loadout") or {}).get("offense") or [])
            if (plan.get("loadout") or {}).get("offense")
            else ""
        ),
        (
            "D macros: N/A — offense only (CPU)"
            if plan.get("offense_only")
            else f"Active slot budget (USER): {budget.get('meter', '?')}  — hard cap {plan.get('active_cap', USER_ACTIVE_CAP)} O+D (Aidan rule; EA may show 10)"
        ),
        "",
    ]
    for line in plan.get("replacing_lines") or []:
        lines.append(f"  {line}")
    if plan.get("replacing_lines"):
        lines.append("")
    for sp in plan.get("swap_banners") or []:
        lines.append(
            f"!! SWAP PLAN: ADD {sp.get('add')} → deactivate {sp.get('bench')} ({sp.get('bench_side')}) first"
        )
        for step in sp.get("xbox_steps") or []:
            lines.append(f"   {step}")
        lines.append("")
    if not shown:
        lines.append("## Playbook adjustments")
        lines.append("  No playbook changes — run baseline as-is")
        lines.append("")
        lines.append("## Macro adjustments")
        lines.append("  No macro changes — keep current 8 active / 2 benched")
    else:
        lines.append("## Playbook adjustments (deltas only)")
        if not pb:
            lines.append("  (none)")
        for d in pb:
            lines.append(f"  [{d['action']}] {d['target']}" + (f" · {d['field']}" if d.get("field") else ""))
            lines.append(f"       {d['detail']}")
            if d.get("before") or d.get("after"):
                lines.append(f"       {d.get('before', '')}  →  {d.get('after', '')}")
            if d.get("why"):
                lines.append(f"       why: {d['why']}")
        lines.append("")
        lines.append("## Macro adjustments (deltas only)")
        if not mac:
            lines.append("  (none)")
        for d in mac:
            badge = normalize_status(d.get("validated_status") or validation_status(d.get("target", "")))
            lines.append(
                f"  [{d['action']}] {d['target']}"
                + (f" · {d['field']}" if d.get("field") else "")
                + f"  [{badge}]"
            )
            lines.append(f"       {d['detail']}")
            if d.get("before") or d.get("after"):
                lines.append(f"       {d.get('before', '')}  →  {d.get('after', '')}")
            if d.get("why"):
                lines.append(f"       why: {d['why']}")
            if d.get("swap_plan"):
                sp = d["swap_plan"]
                lines.append(
                    f"       SWAP: deactivate {sp.get('bench')} ({sp.get('bench_side')}) "
                    f"before Activating {sp.get('add')}"
                )
                for step in (sp.get("xbox_steps") or [])[:8]:
                    lines.append(f"         {step}")

    lines.append("")
    try:
        from cfb_coach.meta_scout import MetaScoutResult, format_scout_text

        raw = plan.get("meta_scout") or {}
        if raw:
            scout_obj = MetaScoutResult(**{
                k: raw[k] for k in MetaScoutResult.__dataclass_fields__ if k in raw
            })
            lines.append(format_scout_text(scout_obj))
            lines.append("")
    except Exception:
        pass
    bk_lines = format_book_section_text(plan.get("cfb_book"))
    if bk_lines:
        lines.extend(bk_lines)
        lines.append("")
    za_lines = format_zone_alignment_text(plan.get("zone_alignment"))
    if za_lines:
        lines.extend(za_lines)
        lines.append("")
    lines.append("## Call emphasis")
    for t in plan["tips"]:
        lines.append(f"  - {t}")
    lines.append("")
    lines.append("Live caller ranks each call with these zone weights + meta priors. Inventory (formations→plays) is in the browser view.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Back-compat shims (old CREATE sheet removed from default path)
# ---------------------------------------------------------------------------

def build_install_steps(
    opponent_id: str,
    opp: dict[str, Any] | None = None,
    *,
    db: CoachDB | None = None,
) -> list[dict[str, str]]:
    """Deprecated name — returns opponent deltas only (no full recreate)."""
    return propose_deltas(opponent_id, opp)


def format_install_sheet(
    opponent_id: str,
    opp: dict[str, Any] | None = None,
    *,
    db: CoachDB | None = None,
    persist: bool = True,
) -> str:
    """Deprecated — text delta sheet (not CREATE-everything)."""
    plan = build_prep_plan(opponent_id, opp, db=db, persist=persist)
    return format_delta_text(plan)


def save_install_diff(
    db: CoachDB,
    opponent_id: str,
    steps: list[dict[str, str]],
) -> dict[str, Any]:
    """Deprecated wrapper — persist proposed deltas."""
    return save_prep_deltas(db, opponent_id, steps, steps)
