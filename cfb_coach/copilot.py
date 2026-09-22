"""Pre-snap tip engine — DefenseLook + macros/gameplan → ≤3 short tips.

Doctrine: Aidan keeps full Xbox control. Coach suggests audible / hot /
protection / macro tip — never auto-plays. Coarse shell/front/pressure only;
Cover 3 vs Quarters discrimination is honestly hard.
"""

from __future__ import annotations

from typing import Any

from cfb_coach.macros import get_macro, load_macro_catalog
from cfb_coach.vision import DefenseLook


# Max tips printed / shown in overlay
MAX_TIPS = 3


def _active_offense_macros(inventory: dict[str, Any] | None = None) -> list[str]:
    """Names Aidan might flick — prefer catalog O macros + Active inventory."""
    names: list[str] = []
    if inventory:
        for key in ("offensive_macros", "offensive_macros_active"):
            for m in inventory.get(key) or []:
                if m and m not in names:
                    names.append(str(m).upper())
    # Always know these (meta_grounded O tags from catalog)
    for mid in ("PROT", "ZERO", "MAN", "C3", "C2", "O-HEAT", "O-RUN", "O-RPO"):
        if mid not in names and get_macro(mid):
            names.append(mid)
    return names


def _active_defense_macros(inventory: dict[str, Any] | None = None) -> list[str]:
    names: list[str] = []
    if inventory:
        for m in inventory.get("macros_active") or []:
            if m and m not in names:
                names.append(str(m).upper())
    if not names:
        # Temple proven Active-8 default
        names = ["CROSS", "VERT", "BUNCH", "RPO", "SCRAM", "RUN-IN", "RUN-OUT", "HEAT"]
    return names


def tips_from_look(
    look: DefenseLook,
    *,
    inventory: dict[str, Any] | None = None,
    opponent_id: str | None = None,
) -> list[str]:
    """Map a DefenseLook to ≤3 short pre-snap tips."""
    n = look.normalized()
    tips: list[str] = []
    o_macros = set(_active_offense_macros(inventory))
    d_macros = set(_active_defense_macros(inventory))

    pressure = n.pressure
    shell = n.shell
    front = n.front

    # --- Pressure first (protection / hot) ---
    if pressure in ("all_out",):
        tips.append("hot X / max protect — Cover 0 heat")
        if "ZERO" in o_macros or "PROT" in o_macros:
            tips.append("tag ZERO/PROT — no deep developing shot")
        tips.append("audible quick: Mesh Spot or Whip Trail")
    elif pressure in ("blitz_left",):
        tips.append("slide protect LEFT")
        tips.append("hot X ready (Sam/edge)")
        if "O-HEAT" in o_macros or "PROT" in o_macros:
            tips.append("PROT/HEAT reminder — don't hero into heat")
    elif pressure in ("blitz_right",):
        tips.append("slide protect RIGHT")
        tips.append("hot X ready (Will/edge)")
        if "O-HEAT" in o_macros or "PROT" in o_macros:
            tips.append("PROT/HEAT reminder — don't hero into heat")
    elif pressure in ("blitz_middle",):
        tips.append("slide protect MID / HB check-release")
        tips.append("hot X — A-gap answer")
        tips.append("quick game: Mesh Spot > deep shot")
    elif pressure == "show":
        tips.append("show-blitz — stay patient; confirm rush")
        tips.append("if they come: hot X; if drop: base read")

    # --- Shell (coverage family) ---
    if shell in ("two_high", "quarters"):
        if pressure in ("none", "unknown", "show"):
            tips.append("two-high → run first (Inside Zone / HB Base)")
        if "O-RUN" in o_macros:
            tips.append("flick RUN macro if stocked")
        if shell == "quarters" and pressure in ("none", "unknown"):
            tips.append("Quarters: under/crossers OK; seams careful")
    elif shell == "cover3":
        tips.append("C3 look → flood / crossers (Deep Flood, Mtn Cross)")
        if "C3" in o_macros:
            tips.append("flick C3 tag if stocked")
        tips.append("high-low flats; don't sit under forever")
    elif shell == "cover2":
        tips.append("C2/Invert → soft underneath (Mesh Spot / easy)")
        if "C2" in o_macros:
            tips.append("flick C2 tag if stocked")
        tips.append("avoid forced Deep Flood into invert flats")
    elif shell == "single_high":
        if pressure in ("none", "unknown", "show"):
            tips.append("single-high — seams / play-action alive")
        tips.append("confirm man vs zone before hero ball")
    elif shell == "man":
        tips.append("man → rubs / mesh / stacks")
        if "MAN" in o_macros:
            tips.append("flick MAN tag if stocked")
        tips.append("Whip Trail / Mesh Traffic beaters")

    # --- Front flavor ---
    if front == "dime" and "VERT" in d_macros:
        # mostly relevant when thinking D later; for O tips keep light
        if len(tips) < MAX_TIPS:
            tips.append("dime look — expect pass shell; check pressure")
    elif front == "goal_line":
        tips.append("GL front — short yd / sneak or fade alert")

    # --- Opponent light lean (optional) ---
    oid = (opponent_id or "").lower()
    if oid == "quen" and pressure in ("none", "unknown") and len(tips) < MAX_TIPS:
        tips.append("vs Quen prior: protection + hot loaded")
    if oid == "gavin" and shell in ("two_high", "quarters", "unknown") and len(tips) < MAX_TIPS:
        tips.append("vs Gavin prior: elevate run audibles")

    # Dedup preserve order, cap
    seen: set[str] = set()
    out: list[str] = []
    for t in tips:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= MAX_TIPS:
            break

    if not out:
        out = ["base look — no auto-counter; trust D&D call"]
        if n.confidence < 0.4:
            out.append("low confidence — wait for clearer shell/pressure")
        out.append("hotkeys: f=front s=shell p=pressure")

    return out[:MAX_TIPS]


def format_tips_block(look: DefenseLook, tips: list[str]) -> str:
    n = look.normalized()
    lines = [
        f"LOOK  {n.label()}  [{n.source}]",
    ]
    if n.notes:
        lines.append(f"      {n.notes}")
    lines.append("TIPS  (you stay on sticks — coach only)")
    for i, t in enumerate(tips, 1):
        lines.append(f"  {i}. {t}")
    return "\n".join(lines)


def default_overlay_path(filename: str = "copilot_overlay.html") -> "Path":
    """~/.cfb-coach/copilot_overlay.html (shared by watch + typed play)."""
    from pathlib import Path

    home = Path.home() / ".cfb-coach"
    home.mkdir(parents=True, exist_ok=True)
    return home / filename


def write_overlay_html(
    path: str,
    look: DefenseLook | None = None,
    tips: list[str] | None = None,
    *,
    call_text: str = "",
    short_line: str = "",
    mode: str = "watch",
) -> str:
    """Tiny auto-refresh overlay — big PLAY/CALL so Xbox can stay focused.

    Open on a second strip / half-screen browser while Remote Play keeps focus.
    Refresh every 1.5s. Works without vision deps (look/tips optional for typed play).
    """
    from pathlib import Path

    tips = tips or []
    tip_lis = "\n".join(f"<li>{_esc(t)}</li>" for t in tips[:2])
    # Prefer PLAY label for typed live play; CALL for watch/copilot
    label = "PLAY" if (mode == "play" or (call_text and look is None)) else "CALL"
    call_block = ""
    if call_text:
        # Keep multi-line SUGGEST readable
        call_html = "<br/>".join(_esc(line) for line in str(call_text).splitlines())
        call_block = f'''<div class="call-label">{label}</div>
  <div class="call">{call_html}</div>'''
    short_block = (
        f'<div class="short">{_esc(short_line)}</div>' if short_line else ""
    )
    look_block = ""
    if look is not None:
        n = look.normalized()
        look_block = f'''<div class="look">
    front=<b>{_esc(n.front)}</b> shell=<b>{_esc(n.shell)}</b>
    pressure=<b>{_esc(n.pressure)}</b>
    · conf={n.confidence:.2f} · {_esc(n.source)}
  </div>'''
    tips_block = f"<ol>\n    {tip_lis}\n  </ol>" if tip_lis else ""
    title = "CFB Coach — PLAY" if mode == "play" or look is None else "CFB Coach — CALL"
    heading = (
        "Live Play (typed)"
        if mode == "play" or look is None
        else "Screen Co-Pilot"
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="1.5"/>
<title>{title}</title>
<style>
  :root {{ --bg:#0e1117; --fg:#e6edf3; --muted:#8b949e; --accent:#3fb950; --call:#58a6ff; }}
  html, body {{ margin:0; height:100%; background:var(--bg); color:var(--fg);
    font-family: ui-sans-serif, system-ui, Segoe UI, sans-serif; }}
  main {{ padding: .75rem 1rem 1.25rem; max-width: 900px; }}
  h1 {{ font-size: .7rem; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; margin: 0 0 .35rem; }}
  .short {{ font-family: ui-monospace, Consolas, monospace; font-size: .95rem;
    color: var(--muted); margin-bottom: .5rem; }}
  .call-label {{ font-size: .7rem; letter-spacing: .1em; color: var(--accent);
    text-transform: uppercase; font-weight: 700; margin-top: .25rem; }}
  .call {{ font-size: clamp(1.35rem, 3.2vw, 2.15rem); font-weight: 800; line-height: 1.25;
    color: var(--call); margin: .2rem 0 .85rem; word-break: break-word; }}
  .look {{ font-family: ui-monospace, Consolas, monospace; font-size: .8rem;
    margin: .4rem 0 .75rem; padding: .5rem .65rem; border: 1px solid #30363d;
    border-radius: 6px; color: var(--muted); }}
  ol {{ margin: 0; padding-left: 1.2rem; }}
  li {{ margin: .35rem 0; font-size: 1.05rem; font-weight: 600; }}
  footer {{ margin-top: 1rem; font-size: .72rem; color: var(--muted); line-height: 1.4; }}
  .badge {{ display:inline-block; padding:.1rem .4rem; border-radius:4px;
    background:#21262d; color:var(--accent); font-size:.7rem; }}
</style>
</head>
<body>
<main>
  <h1>{heading} <span class="badge">v1.9.6</span> · keep Xbox focused</h1>
  {short_block}
  {call_block}
  {look_block}
  {tips_block}
  <footer>
    Aidan keeps sticks · glance here for the PLAY call<br/>
    Pin this strip on the other half of the screen. Auto-refresh 1.5s.
    Typed live = <code>cfb-coach play</code> · vision watch on hold.
  </footer>
</main>
</body>
</html>
"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p)


def _esc(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def catalog_macro_hint(name: str) -> str | None:
    """One-line purpose from macro catalog if present."""
    m = get_macro(name)
    if not m:
        return None
    purpose = m.get("purpose") or m.get("summary")
    xbox = m.get("xbox_name") or m.get("name") or name
    if purpose:
        return f"{xbox}: {purpose}"
    return str(xbox)


# Silence unused import lint if catalog only used indirectly
_ = load_macro_catalog
