"""Madden macro loadout — 8-cap Active (O+D), validation tags, swap plans.

Status starts from the catalog (all meta_grounded) and is overridden per
Madden DB by postgame (proven / failed) via meta key `macro_status_json`.
"""

from __future__ import annotations

import json
from typing import Any

from cfb_coach.macros import FAILED, META_GROUNDED, PROVEN, UNVALIDATED, normalize_status
from cfb_coach.madden.data import get_macro, load_macro_catalog

USER_ACTIVE_CAP = 8
STATUS_META_KEY = "macro_status_json"


def status_overrides(db: Any) -> dict[str, str]:
    if db is None:
        return {}
    raw = db.get_meta(STATUS_META_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return {str(k).upper(): normalize_status(v) for k, v in dict(data).items()}


def set_status(db: Any, name: str, status: str) -> None:
    cur = status_overrides(db)
    cur[name.upper()] = normalize_status(status)
    db.set_meta(STATUS_META_KEY, json.dumps(cur))


def macro_status(name: str | None, db: Any = None) -> str:
    key = (name or "").upper()
    over = status_overrides(db)
    if key in over:
        return over[key]
    m = get_macro(key)
    return normalize_status((m or {}).get("validated_status")) if m else UNVALIDATED


def tag_live(name: str | None, db: Any = None) -> str:
    if not name or name.lower() == "none":
        return "none"
    s = macro_status(name, db)
    return name if s == PROVEN else f"{name} [{s}]"


def macro_side(name: str) -> str:
    return (get_macro(name) or {}).get("side") or "defense"


def split_loadout(active: list[str]) -> dict[str, list[str]]:
    return {
        "defense": [m for m in active if macro_side(m) == "defense"],
        "offense": [m for m in active if macro_side(m) == "offense"],
    }


def swap_plan(add: str, active: list[str], archetype: str | None) -> dict[str, Any]:
    """ADD at 8/8 → pick which Active macro to deactivate first."""
    hints = load_macro_catalog().get("swap_priority") or {}
    pref = hints.get((archetype or "").lower()) or hints.get("default") or {}
    same_side = [m for m in active if macro_side(m) == macro_side(add)]
    bench = next((m for m in pref.get("prefer_bench") or [] if m in active), None)
    bench = bench or (same_side[-1] if same_side else active[-1])
    return {
        "add": add,
        "bench": bench,
        "bench_side": macro_side(bench),
        "why": pref.get("why") or "keep coverage answers for this persona",
        "xbox_steps": [
            f"Deactivate {bench} (keeps Active at 8/8 O+D)",
            f"Build {add} from its copy block (pre-snap recipe; menu names approx)",
            f"Set {add} Active — confirm Active count is 8, not 9",
        ],
    }


def loadout_cards(
    active: list[str],
    db: Any = None,
    *,
    offense_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cards for ONLY the Active loadout (≤8). CPU → offense macros only."""
    names = [m for m in active if not offense_only or macro_side(m) == "offense"]
    cards: list[dict[str, Any]] = []
    for name in names:
        m = get_macro(name) or {"name": name, "side": "?"}
        m["id"] = name
        m["slot"] = "active"
        m["validated_status"] = macro_status(name, db)
        cards.append(m)
    split = split_loadout(active)
    total = len(names)
    cap = USER_ACTIVE_CAP
    meter = (
        f"Active {total}/{cap} (offense only — CPU)"
        if offense_only
        else f"Active {total}/{cap} (D {len(split['defense'])} + O {len(split['offense'])})"
    )
    loadout = {
        "meter": meter,
        "total": total,
        "defense": [] if offense_only else split["defense"],
        "offense": split["offense"],
    }
    return cards, loadout


# ---------------------------------------------------------------------------
# Aidan's exact settings (CFB parity: his settings verbatim, gaps flagged, never invented)
# ---------------------------------------------------------------------------

SETTINGS_FILENAME = "madden27_macros.json"
ACTIVE_META_KEY = "active_macros:{opp}"
EDITOR_PATH = {
    "offense": ["Create & Share", "Custom Adjustments", "Offense", "Create / Edit"],
    "defense": ["Create & Share", "Custom Adjustments", "Defense", "Create / Edit"],
}
IN_GAME = "At the line: LB → pick the custom adjustment (Aidan cap 8 Active, O+D)"


def settings_path():
    from cfb_coach.games import data_dir

    return data_dir() / SETTINGS_FILENAME


def load_user_settings() -> dict[str, Any]:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"macros": {}}


def _split_setting(raw: str) -> dict[str, str]:
    """'Coverage: Shading = Over the top' | 'Shading = Over the top' | 'Coverage / Shading: x'."""
    txt = raw.strip()
    sec = ""
    if "=" in txt:
        left, val = txt.split("=", 1)
        if ":" in left:
            sec, name = left.split(":", 1)
        elif "/" in left:
            sec, name = left.split("/", 1)
        else:
            name = left
    elif ":" in txt:
        left, val = txt.split(":", 1)
        if "/" in left:
            sec, name = left.split("/", 1)
        else:
            sec, name = "", left
    else:
        raise ValueError(f"setting needs a value: use \"Section: Setting = value\" (got {raw!r})")
    name, val, sec = name.strip(), val.strip(), sec.strip()
    if not name or not val:
        raise ValueError(f"setting needs a name and a value (got {raw!r})")
    return {"section": sec or name, "setting": name, "value": val}


def save_user_settings(mid: str, settings: list[str], *, replace: bool = False,
                       xbox_name: str | None = None, clear: bool = False) -> dict[str, Any]:
    """Store Aidan's exact in-game settings for one macro, verbatim (his wording)."""
    from datetime import datetime, timezone

    key = mid.upper()
    if not get_macro(key):
        raise ValueError(f"unknown Madden macro {mid!r} (known: {', '.join(sorted(load_macro_catalog().get('macros') or {}))})")
    data = load_user_settings()
    macros = data.setdefault("macros", {})
    if clear:
        macros.pop(key, None)
    else:
        rec = macros.setdefault(key, {"settings": []})
        rows = [] if replace else list(rec.get("settings") or [])
        for raw in settings:
            row = _split_setting(raw)
            rows = [r for r in rows if not (r["section"].lower() == row["section"].lower()
                                             and r["setting"].lower() == row["setting"].lower())]
            rows.append(row)
        rec["settings"] = rows
        if xbox_name:
            rec["xbox_name"] = xbox_name
        rec["updated"] = datetime.now(timezone.utc).isoformat()
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return macros.get(key) or {}


def aidan_settings(mid: str) -> list[dict[str, Any]]:
    """His settings for this macro, verbatim: the user file, then any catalog rows marked
    confirmed. Nothing else — approximations are never shown as settings."""
    key = (mid or "").upper()
    rows = [dict(r, source="yours") for r in (load_user_settings().get("macros", {}).get(key) or {}).get("settings") or []]
    have = {(r["section"].lower(), r["setting"].lower()) for r in rows}
    for k, ent in ((get_macro(key) or {}).get("full_settings") or {}).items():
        if not isinstance(ent, dict) or (ent.get("status") or "").lower() != "confirmed":
            continue
        sec, _, name = k.partition(" / ")
        if (sec.lower(), (name or sec).lower()) in have:
            continue
        rows.append({"section": ent.get("section") or sec, "setting": name or sec,
                     "value": str(ent.get("value") or ""), "source": "catalog (confirmed)"})
    return rows


def settings_gaps(mid: str) -> list[dict[str, str]]:
    """Settings the research says this macro needs that Aidan has NOT given: flagged, never filled."""
    key = (mid or "").upper()
    have = {r["setting"].lower() for r in aidan_settings(key)} | {f"{r['section']} / {r['setting']}".lower() for r in aidan_settings(key)}
    out = []
    for k, ent in ((get_macro(key) or {}).get("full_settings") or {}).items():
        if not isinstance(ent, dict) or (ent.get("status") or "").lower() == "confirmed":
            continue
        sec, _, name = k.partition(" / ")
        if k.lower() in have or (name or sec).lower() in have:
            continue
        out.append({"section": sec, "setting": name or sec, "research_value": str(ent.get("value") or "")})
    return out


def key_settings(mid: str) -> str:
    rows = aidan_settings(mid)
    return " · ".join(f"{r['setting']} {r['value']}" if r["setting"] != r["section"] else f"{r['section']}: {r['value']}"
                      for r in rows)


def _pairs_in_book(mid: str, book: dict[str, list[str]] | None, *, cap: int = 6) -> list[str]:
    import re

    m = get_macro(mid) or {}
    shell = str(m.get("shell_pair") or "")
    out: list[str] = []
    if not book:
        return [shell] if shell else []
    for f, ps in book.items():
        for p in ps:
            if (f in shell and p in shell) or (p in shell and len(p) > 5):
                out.append(f"{p} ({f})")
    if len(out) < cap:
        fam = (m.get("concept_family") or "")
        rx = {"vert": r"cover 4|quarters", "flood": r"cover 3 match|cover 3", "cross": r"tampa|cover 2",
              "stack": r"cover 3 match|cover 1", "scram": r"cover 4|spy", "run": r"cover 3 sky|cover 1|cover 3"}.get(fam)
        if mid == "O-PROT":
            rx = r"slant|stick|mesh|stutter|quick|screen"
        elif mid == "O-MAN":
            rx = r"mesh|slant|drive|cross|wheel|switch"
        elif mid == "O-RPO":
            rx = r"rpo|alert|inside zone"
        if rx:
            for f, ps in book.items():
                for p in ps:
                    lab = f"{p} ({f})"
                    if lab not in out and re.search(rx, p, re.I):
                        out.append(lab)
                    if len(out) >= cap:
                        break
    return out[:cap]


def macro_detail(mid: str, book: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Drill-down for one macro (CFB `offense_macro_detail` shape, both sides): exact settings
    = Aidan's only; missing ones listed as gaps with the research guess on the details page."""
    key = (mid or "").upper()
    meta = get_macro(key) or {}
    side = meta.get("side") or "defense"
    rows = aidan_settings(key)
    gaps = settings_gaps(key)
    user = (load_user_settings().get("macros", {}).get(key) or {})
    return {
        "id": key,
        "side": side,
        "xbox_name": user.get("xbox_name") or meta.get("xbox_name") or key,
        "editor_path": list(EDITOR_PATH.get(side) or []),
        "in_game": IN_GAME,
        "settings": rows,
        "has_settings": bool(rows),
        "gaps": [f"{g['section']} / {g['setting']} — NOT ON FILE (research guess: {g['research_value']})" for g in gaps],
        "gap_rows": gaps,
        "settings_source": "Aidan's settings (~/.cfb-coach/madden27_macros.json via `macro-settings --game madden27`)",
        "fire_when": meta.get("when_to_arm") or "",
        "pairs_with": _pairs_in_book(key, book),
        "shell_pair": meta.get("shell_pair") or "",
        "key": key_settings(key),
        "purpose": meta.get("purpose") or "",
        "sources": ([{"id": "research", "title": meta.get("source"), "url": "",
                      "supports": "purpose / when to arm / research guesses only — not your settings"}]
                    if meta.get("source") else []),
        "hot_route_menu_source": "",
        "limits": ("EA documents the Custom Adjustment categories but not every option label; "
                   "only settings you entered are shown as exact."),
        "set_cmd": f'PYTHONPATH=. python3 -m cfb_coach macro-settings --game madden27 {key} --set "Section: Setting = value"',
    }


def copy_block(detail: dict[str, Any]) -> str:
    """Tick-by-tick: only Aidan's settings, a loud flag for missing ones, then everything else at Default."""
    name = detail.get("xbox_name") or detail.get("id")
    lines = [f"MACRO: {name} ({detail.get('side')})", "Path: " + " > ".join(detail.get("editor_path") or []),
             f"[ ] Name: {name}"]
    cur = None
    for r in detail.get("settings") or []:
        if r["setting"] == r["section"]:
            lines.append(f"[ ] {r['section']}: {r['value']}")
            cur = None
            continue
        if r["section"] != cur:
            cur = r["section"]
            lines.append(cur)
        lines.append(f"  [ ] {r['setting']}: {r['value']}")
    if not detail.get("settings"):
        lines.append("[ ] (no exact settings from Aidan on file for this macro — enter them: "
                     f"{detail.get('set_cmd')})")
    elif detail.get("gap_rows"):
        lines.append("[!] Missing from your settings: " + ", ".join(
            f"{g['section']} / {g['setting']}" for g in detail["gap_rows"]) + " — enter them, nothing is guessed")
    lines.append("[ ] Everything else: Default")
    lines.append("[ ] Save -> set Active (Aidan cap 8 O+D)")
    lines.append(f"In game: LB -> {name}")
    if detail.get("fire_when"):
        lines.append(f"Fire when: {detail['fire_when']}")
    if detail.get("pairs_with"):
        lines.append("Pairs with: " + ", ".join(detail["pairs_with"]))
    return "\n".join(lines)


def attach_detail(card: dict[str, Any], book: dict[str, list[str]] | None = None) -> dict[str, Any]:
    det = macro_detail(str(card.get("id") or ""), book)
    card["ingame"] = det
    card["copy_block"] = copy_block(det)
    card["book_plays"] = det["pairs_with"]
    card["missing_settings"] = not det["has_settings"] or bool(det["gap_rows"])
    return card


def missing_settings_report(active: list[str]) -> list[str]:
    out = []
    for mid in active:
        d = macro_detail(mid)
        if not d["has_settings"]:
            out.append(f"{mid}: no exact settings on file — enter yours (nothing invented)")
        elif d["gap_rows"]:
            out.append(f"{mid}: missing {', '.join(g['setting'] for g in d['gap_rows'])}")
    return out


# ---------------------------------------------------------------------------
# Active 8 per prep (what live `play` may fire) + live suggestion
# ---------------------------------------------------------------------------

def store_active(db: Any, opponent_id: str, ids: list[str]) -> None:
    from datetime import datetime, timezone

    if db is None:
        return
    db.set_meta(ACTIVE_META_KEY.format(opp=opponent_id), json.dumps(
        {"active": list(ids)[:USER_ACTIVE_CAP], "ts": datetime.now(timezone.utc).isoformat()}))


def load_active(db: Any, opponent_id: str) -> list[str] | None:
    if db is None:
        return None
    raw = db.get_meta(ACTIVE_META_KEY.format(opp=opponent_id))
    if not raw:
        return None
    try:
        return [str(x).upper() for x in json.loads(raw).get("active") or [] if get_macro(str(x))][:USER_ACTIVE_CAP]
    except ValueError:
        return None


LEARNED_SUPPRESS = -0.15


def macro_info(mid: str, why: str, *, weight: float | None = None) -> dict[str, Any]:
    meta = get_macro(mid) or {}
    det = macro_detail(mid)
    return {
        "id": mid,
        "name": det["xbox_name"],
        "side": meta.get("side") or "defense",
        "why": why,
        "key": det["key"] or "no exact settings on file — enter yours",
        "settings": det["settings"],
        "missing_settings": not det["has_settings"],
        "fire_when": meta.get("when_to_arm") or "",
        "learned_weight": weight,
    }


def suggest_offense_macro(
    *,
    play: str,
    coverage_class: str | None,
    coverage_source: str,
    repeated: bool,
    active: list[str],
    archetype: str = "",
    passing_down: bool = False,
    weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """At most one Active-8 offense custom adjustment for this snap (CFB `suggest_offense_macro`
    shape; Madden catalog arm rules: coverage macros need a REPEATED look)."""
    import re

    from cfb_coach.madden.catalog import is_run

    act = [a for a in active if macro_side(a) == "offense"]
    if not act or not play:
        return None
    rpo = bool(re.search(r"rpo|alert", play, re.I))
    is_pass = not is_run(play) and not rpo
    # Madden arm rules ("after 2+ live man / pressure looks"): the look must be REPEATED this
    # game — one live look is a soft lean only (same doctrine as the D macros).
    look_ok = bool(coverage_class) and repeated and coverage_source in ("live", "last")

    def ok(mid: str) -> bool:
        w = (weights or {}).get(mid)
        return mid in act and not (w is not None and w <= LEARNED_SUPPRESS)

    if is_pass and look_ok and coverage_class == "pressure" and ok("O-PROT"):
        return macro_info("O-PROT", f"repeated pressure look on {play} — protection + hot",
                          weight=(weights or {}).get("O-PROT"))
    if is_pass and look_ok and coverage_class == "man" and ok("O-MAN"):
        return macro_info("O-MAN", f"repeated man look on {play} — stack / rub hot routes",
                          weight=(weights or {}).get("O-MAN"))
    if rpo and ok("O-RPO"):
        return macro_info("O-RPO", f"RPO {play} — give/keep read", weight=(weights or {}).get("O-RPO"))
    if is_pass and passing_down and archetype == "pressure_heavy" and ok("O-PROT"):
        return macro_info("O-PROT", "pressure persona on a passing down — protection first",
                          weight=(weights or {}).get("O-PROT"))
    return None


__all__ = [
    "FAILED",
    "META_GROUNDED",
    "PROVEN",
    "USER_ACTIVE_CAP",
    "aidan_settings",
    "attach_detail",
    "copy_block",
    "load_active",
    "loadout_cards",
    "macro_detail",
    "save_user_settings",
    "settings_gaps",
    "store_active",
    "suggest_offense_macro",
    "macro_side",
    "macro_status",
    "set_status",
    "split_loadout",
    "status_overrides",
    "swap_plan",
    "tag_live",
]
