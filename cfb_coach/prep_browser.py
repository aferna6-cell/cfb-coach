"""Polished dark HTML prep sheet (deltas only) — open in browser.

v1.6.0: Active loadout only (≤8 clickable macros) — never dump benched
(FLOOD/SCREEN) catalog. Swap → show 8 *after* swap + "replacing X with Y".
CPU games = offense-only (D macros N/A). Validation badges unchanged.
"""

from __future__ import annotations

import html
import os
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from cfb_coach.browser_open import is_wsl as _is_wsl, try_cmd as _try_cmd
from cfb_coach.install_sheet import build_prep_plan

ET = ZoneInfo("America/New_York")


def default_prep_dir() -> Path:
    env = os.environ.get("CFB_COACH_DB")
    if env:
        return Path(env).expanduser().resolve().parent
    try:
        d = Path.home() / ".cfb-coach"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        d = Path("/workspace/cfb-coach/data")
        d.mkdir(parents=True, exist_ok=True)
        return d


def prep_html_path(opponent_id: str) -> Path:
    return default_prep_dir() / f"prep_{opponent_id.lower()}.html"


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _action_badge(action: str) -> str:
    a = (action or "").upper()
    cls = {
        "ADD": "badge add",
        "REMOVE": "badge remove",
        "BENCH": "badge bench",
        "UNBENCH": "badge unbench",
    }.get(a, "badge edit")
    return f'<span class="{cls}">{_esc(a)}</span>'


def _val_badge(status: str) -> str:
    from cfb_coach.macros import normalize_status

    s = normalize_status(status)
    if s == "proven":
        return '<span class="vbadge proven">proven</span>'
    if s == "meta_grounded":
        return '<span class="vbadge meta-grounded">meta_grounded</span>'
    if s == "failed":
        return '<span class="vbadge failed">failed</span>'
    return '<span class="vbadge unvalidated">unvalidated</span>'


def _render_settings(settings: dict[str, Any]) -> str:
    if not settings:
        return '<p class="muted">No full_settings on file.</p>'

    # Group by section when exact sheets present
    sections: dict[str, list[str]] = {}
    order: list[str] = []
    for key, val in settings.items():
        if isinstance(val, dict):
            value = val.get("value", "")
            tag = (val.get("status") or val.get("tag") or "confirmed").lower()
            if tag in ("confirmed", "exact", "proven"):
                tag = "confirmed"
            elif tag not in ("confirmed", "approx"):
                tag = "approx"
            section = val.get("section") or (
                key.split(" / ", 1)[0] if " / " in key else "Settings"
            )
            label = key.split(" / ", 1)[1] if " / " in key else key
            note = val.get("note") or ""
            note_html = (
                f"<div class='set-note'>{_esc(note)}</div>" if note else ""
            )
            row = (
                "<div class='set-row'>"
                f"<span class='set-k'>{_esc(label)}</span>"
                f"<span class='set-v'>{_esc(value)}</span>"
                f"<span class='set-tag {tag}'>{tag}</span>"
                f"{note_html}"
                "</div>"
            )
        else:
            section = "Settings"
            label = key
            row = (
                "<div class='set-row'>"
                f"<span class='set-k'>{_esc(label)}</span>"
                f"<span class='set-v'>{_esc(val)}</span>"
                "<span class='set-tag confirmed'>confirmed</span>"
                "</div>"
            )
        if section not in sections:
            sections[section] = []
            order.append(section)
        sections[section].append(row)

    preferred = [
        "General",
        "DL/LB",
        "Secondary",
        "Zone Drops",
        "Strategy",
        "Coverage Checks",
        "Individuals",
        "Notes",
        "Settings",
    ]
    ordered = [s for s in preferred if s in sections] + [
        s for s in order if s not in preferred
    ]
    blocks: list[str] = []
    for sec in ordered:
        blocks.append(
            f"<div class='set-section'><h5>{_esc(sec)}</h5>"
            f"{''.join(sections[sec])}</div>"
        )
    return f"<div class='settings'>{''.join(blocks)}</div>"


def _render_delta_cards(deltas: list[dict[str, Any]], empty_msg: str) -> str:
    if not deltas:
        return f'<div class="empty">{_esc(empty_msg)}</div>'
    parts: list[str] = []
    for d in deltas:
        field = d.get("field") or ""
        before = d.get("before") or ""
        after = d.get("after") or ""
        why = d.get("why") or ""
        change = ""
        if before or after:
            change = (
                f'<div class="change">'
                f'<span class="before">{_esc(before)}</span>'
                f'<span class="arrow">→</span>'
                f'<span class="after">{_esc(after)}</span>'
                f"</div>"
            )
        field_bit = f'<span class="field">{_esc(field)}</span>' if field else ""
        why_bit = f'<div class="why">{_esc(why)}</div>' if why else ""
        val_bit = _val_badge(str(d.get("validated_status") or "meta_grounded"))
        swap = d.get("swap_plan")
        swap_bit = ""
        if swap:
            steps = "".join(
                f"<li>{_esc(s)}</li>" for s in (swap.get("xbox_steps") or [])
            )
            swap_bit = f"""
            <div class="swap-box">
              <div class="swap-title">SWAP REQUIRED — deactivate
                <b>{_esc(swap.get("bench"))}</b>
                ({_esc(swap.get("bench_side"))}) before Activating
                <b>{_esc(swap.get("add"))}</b>
              </div>
              <div class="why">{_esc(swap.get("why", ""))}</div>
              <ol class="swap-steps">{steps}</ol>
            </div>
            """
        parts.append(
            f"""
            <article class="card">
              <div class="card-head">
                {_action_badge(d.get("action", "EDIT"))}
                <strong class="target">{_esc(d.get("target", ""))}</strong>
                {field_bit}
                {val_bit}
              </div>
              <p class="detail">{_esc(d.get("detail", ""))}</p>
              {change}
              {why_bit}
              {swap_bit}
            </article>
            """
        )
    return "\n".join(parts)


def _render_swap_banners(banners: list[dict[str, Any]]) -> str:
    if not banners:
        return ""
    parts: list[str] = []
    for sp in banners:
        steps = "".join(f"<li>{_esc(s)}</li>" for s in (sp.get("xbox_steps") or []))
        parts.append(
            f"""
            <div class="banner swap">
              <strong>SWAP PLAN:</strong> ADD {_esc(sp.get("add"))}
              → deactivate <b>{_esc(sp.get("bench"))}</b>
              ({_esc(sp.get("bench_side"))}) first
              <span class="muted"> — {_esc(sp.get("why", ""))}</span>
              <ol>{steps}</ol>
            </div>
            """
        )
    return "\n".join(parts)


def _render_macro_accordion(
    cards: list[dict[str, Any]],
    budget: dict[str, Any],
    *,
    offense_only: bool = False,
    replacing_lines: list[str] | None = None,
    loadout: dict[str, Any] | None = None,
    details: bool = False,
) -> str:
    """Render ONLY active loadout cards (≤8). Never benched (FLOOD/SCREEN) catalog.

    v1.15: offense cards carrying ``ingame`` expand to the exact CFB 27 Custom
    Adjustment rows (in-game order), the book plays they pair with, and when to fire.
    ``details=True`` (details page) adds research notes + citations."""
    # Safety: drop any non-active / benched cards if a caller passed full catalog
    cards = [c for c in cards if (c.get("slot") or "active") == "active"]
    # Never show FLOOD/SCREEN in main prep view
    cards = [
        c
        for c in cards
        if (c.get("name") or c.get("id") or "").upper() not in ("FLOOD", "SCREEN")
    ]

    if offense_only:
        d_note = '<div class="empty">D macros: <b>N/A — offense only</b> (CPU coaching)</div>'
        o_cards = [c for c in cards if (c.get("side") or "") == "offense"]
        cards = o_cards
    else:
        d_note = ""

    replace_bit = ""
    for line in replacing_lines or []:
        replace_bit += f'<div class="banner swap"><strong>Loadout:</strong> {_esc(line)}</div>'

    meter = _esc(
        (loadout or {}).get("meter")
        or budget.get("meter")
        or "Active ?/?"
    )
    meter_cls = "meter hot" if budget.get("at_cap") else "meter ok"
    meter_label = (
        "Active slot budget (offense only — CPU)"
        if offense_only
        else "Active loadout (USER O+D)"
    )

    if not cards:
        empty = (
            d_note
            or '<div class="empty">No active macros in this loadout</div>'
        )
        return f"""
    {replace_bit}
    <div class="{meter_cls}">
      <span class="meter-label">{meter_label}</span>
      <span class="meter-value">{meter}</span>
      <span class="meter-note">Aidan hard cap 8 — EA UI may show 10</span>
    </div>
    {empty}
    """

    items: list[str] = []
    for c in cards:
        mid = str(c.get("id") or c.get("name") or "?")
        name = c.get("name") or mid
        slot = c.get("slot") or "active"
        side = c.get("side") or "?"
        status = c.get("validated_status") or "unvalidated"
        purpose = c.get("purpose") or ""
        when = c.get("when_to_arm") or ""
        copy_block = c.get("copy_block") or ""
        settings = c.get("full_settings") or {}
        steps = c.get("xbox_steps") or []
        step_lis = "".join(f"<li>{_esc(s)}</li>" for s in steps)
        cid = "copy-" + "".join(ch if ch.isalnum() else "-" for ch in mid)
        open_attr = " open" if mid.upper() == "CROSS" else ""
        if c.get("ingame"):
            items.append(_render_ingame_card(c, cid, details=details))
            continue
        items.append(
            f"""
            <details class="macro-card slot-{_esc(slot)}"{open_attr}>
              <summary>
                <span class="mname">{_esc(name)}</span>
                <span class="mside">{_esc(side)}</span>
                <span class="mslot">{_esc(slot)}</span>
                {_val_badge(str(status))}
                {('<span class="mwhy">' + _esc(c.get("why")) + '</span>') if c.get("why") else ""}
              </summary>
              <div class="macro-body">
                <p class="purpose">{_esc(purpose)}</p>
                <p class="when"><b>When to arm:</b> {_esc(when)}</p>
                <h4>Xbox path</h4>
                <ol class="xbox">{step_lis}</ol>
                <h4>Full settings</h4>
                {_render_settings(settings)}
                <div class="copy-wrap">
                  <div class="copy-head">
                    <h4>Copy block (tick-by-tick checklist)</h4>
                    <button type="button" class="copy-btn" data-target="{cid}">Copy</button>
                  </div>
                  <pre id="{cid}" class="copy-block">{_esc(copy_block)}</pre>
                </div>
              </div>
            </details>
            """
        )

    return f"""
    {replace_bit}
    <div class="{meter_cls}">
      <span class="meter-label">{meter_label}</span>
      <span class="meter-value">{meter}</span>
      <span class="meter-note">Aidan hard cap 8 — show only Active loadout (never benched catalog)</span>
    </div>
    {d_note}
    <div class="macro-list">
      {"".join(items)}
    </div>
    """


def _render_ingame_card(c: dict[str, Any], cid: str, *, details: bool = False) -> str:
    """One offense macro row: collapsed summary; expands to Aidan's exact settings (verbatim
    from his notes), the book plays it pairs with, and when to fire it."""
    det = c.get("ingame") or {}
    mid = str(c.get("id") or "")
    name = det.get("xbox_name") or c.get("name") or mid
    ncol = 3 if details else 2
    rows_html: list[str] = []
    cur = None
    for r in det.get("settings") or []:
        single = r["setting"] == r["section"]
        if not single and r["section"] != cur:
            cur = r["section"]
            rows_html.append(f"<tr class='ca-sec'><td colspan='{ncol}'>{_esc(cur)}</td></tr>")
        if single:
            cur = None
        note = f"<td class='muted'>{_esc(r.get('research') or '')}</td>" if details else ""
        rows_html.append(f"<tr class='ca-row'><td class='ca-k'>{_esc(r['setting'])}</td>"
                         f"<td class='ca-v'>{_esc(r['value'])}</td>{note}</tr>")
    if not det.get("settings"):
        rows_html.append(f"<tr class='ca-row'><td colspan='{ncol}' class='muted'>No exact settings from Aidan on file for this macro.</td></tr>")
    rows_html.append(f"<tr class='ca-row dflt'><td class='ca-k'>Everything else</td><td class='ca-v'>Default</td>{'<td></td>' if details else ''}</tr>")
    pairs = det.get("pairs_with") or []
    pairs_html = "".join(f"<span class='pchip'>{_esc(p)}</span>" for p in pairs) or "<span class='muted'>none of its plays are in your book</span>"
    src_html = ""
    if details:
        srcs = "".join(
            f"<li><b>{_esc(s.get('id'))}</b>: {_esc(s.get('title'))}"
            + (f" — <a href='{_esc(s.get('url'))}'>{_esc(s.get('url'))}</a>" if s.get("url") else "")
            + f"<div class='muted'>{_esc(s.get('supports'))}</div></li>"
            for s in det.get("sources") or []
        )
        gaps = "".join(f"<li>{_esc(g)}</li>" for g in det.get("gaps") or [])
        src_html = (
            f"<div class='ca-assume'>Settings: {_esc(det.get('settings_source') or '')}, shown exactly as written. "
            f"Research notes (right column) are for reference only — nothing was remapped. "
            f"Hot-route menu reference: {_esc(det.get('hot_route_menu_source') or '')}. {_esc(det.get('limits') or '')}</div>"
            + (f"<h4>Not exact in his notes (not shown as settings)</h4><ul class='ca-src'>{gaps}</ul>" if gaps else "")
            + f"<h4>Research sources</h4><ul class='ca-src'>{srcs}</ul>"
        )
    status = c.get("validated_status") or "unvalidated"
    return f"""
            <details class="macro-card slot-active" data-macro="{_esc(mid)}">
              <summary>
                {f'<span class="mrank">{int(c["rank"])}.</span>' if c.get("rank") else ""}
                <span class="mname">{_esc(name)}</span>
                <span class="mside">{_esc(c.get("side") or "offense")}</span>
                {_val_badge(str(status)) if details else ""}
                {'<span class="mneeds">NEEDS SETTINGS</span>' if c.get("missing_settings") else ""}
                <span class="mkey">{_esc(det.get("key") or "")}</span>
              </summary>
              <div class="macro-body">
                <p class="purpose">{_esc(c.get("purpose") or "")}</p>
                <div class="mfire"><b>Fire it when:</b> {_esc(det.get("fire_when") or "")}</div>
                <div class="mpairs"><b>Pairs with (your book):</b> {pairs_html}</div>
                <h4>Settings — {_esc(" › ".join(det.get("editor_path") or []))}</h4>
                <table class="ca-set">{"".join(rows_html)}</table>
                <div class="muted ca-ingame">{_esc(det.get("in_game") or "")}</div>
                {src_html}
                <div class="copy-wrap">
                  <div class="copy-head">
                    <h4>Copy block (tick-by-tick checklist)</h4>
                    <button type="button" class="copy-btn" data-target="{cid}">Copy</button>
                  </div>
                  <pre id="{cid}" class="copy-block">{_esc(c.get("copy_block") or "")}</pre>
                </div>
              </div>
            </details>
            """


def _render_inventory(inv: dict[str, Any]) -> str:
    o_parts: list[str] = []
    for name, meta in (inv.get("offense") or {}).items():
        plays = meta.get("plays") or []
        aud = meta.get("audibles") or []
        play_lis = "".join(f"<li>{_esc(p)}</li>" for p in plays)
        aud_bit = (
            f'<div class="aud">Audibles: {", ".join(_esc(a) for a in aud)}</div>'
            if aud
            else ""
        )
        o_parts.append(
            f"""
            <div class="inv-form">
              <h4>{_esc(name)} <span class="role">{_esc(meta.get("role", ""))}</span></h4>
              {aud_bit}
              <ul>{play_lis}</ul>
            </div>
            """
        )

    d_parts: list[str] = []
    for name, meta in (inv.get("defense") or {}).items():
        calls = meta.get("calls") or []
        call_lis = "".join(f"<li>{_esc(c)}</li>" for c in calls)
        d_parts.append(
            f"""
            <div class="inv-form">
              <h4>{_esc(name)} <span class="role">{_esc(meta.get("role", ""))}</span></h4>
              <ul>{call_lis}</ul>
            </div>
            """
        )

    macros_a = ", ".join(_esc(m) for m in inv.get("macros_active") or [])
    o_macros = inv.get("offensive_macros") or []
    o_mac_bit = ", ".join(_esc(m) for m in o_macros) if o_macros else "(none)"
    offense_only = bool(inv.get("_offense_only"))
    d_book_bit = (
        '<div><b>D book / macros:</b> N/A — offense only (CPU)</div>'
        if offense_only
        else (
            f'<div><b>D book:</b> {_esc(inv.get("defense_book"))}</div>'
            f'<div><b>D macros active (loadout):</b> {macros_a}</div>'
            f'<div><b>O macros:</b> {o_mac_bit}</div>'
        )
    )
    d_section = (
        ""
        if offense_only
        else (
            f'<h3>Defense packages → calls</h3>'
            f'<div class="inv-grid">{"".join(d_parts)}</div>'
        )
    )

    return f"""
    <details class="inventory">
      <summary>Current inventory <span class="muted">(read-only — already stocked; Active loadout only above)</span></summary>
      <div class="inv-meta">
        <div><b>O book:</b> {_esc(inv.get("offense_book"))}</div>
        {d_book_bit}
      </div>
      <h3>Offense formations → plays</h3>
      <div class="inv-grid">{"".join(o_parts)}</div>
      {d_section}
    </details>
    """


_CSS = """
  :root {
    --bg: #0f1419; --panel: #1a2332; --panel2: #243044;
    --text: #e7ecf3; --muted: #8b9bb4; --accent: #5b9fd4;
    --add: #3d9a6a; --edit: #d4a017; --remove: #c45c5c;
    --bench: #8b7bb8; --border: #2e3a4f; --ok: #2a6b4a; --warn: #a67c2a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.45;
  }
  .wrap { max-width: 960px; margin: 0 auto; padding: 28px 20px 64px; }
  header {
    background: linear-gradient(135deg, #1a2332 0%, #152030 100%);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 22px 24px; margin-bottom: 20px;
  }
  header h1 { margin: 0 0 6px; font-size: 1.55rem; font-weight: 650; letter-spacing: -0.02em; }
  header .meta { color: var(--muted); font-size: 0.92rem; }
  header .meta span { margin-right: 14px; }
  .banner {
    border-radius: 10px; padding: 12px 16px; margin-bottom: 18px;
    font-weight: 560; border: 1px solid var(--border);
  }
  .banner.ok { background: rgba(42,107,74,0.25); border-color: var(--ok); color: #9fdfb8; }
  .banner.info { background: rgba(91,159,212,0.12); border-color: var(--accent); color: #b8d4ec; }
  .banner.dyn.ser { border-color: #3d9a6a; background: rgba(61,154,106,0.12); }
  .banner.dyn.exp { border-color: #d4a017; background: rgba(212,160,23,0.12); }
  .banner.swap {
    background: rgba(166,124,42,0.18); border-color: var(--warn); color: #f0d9a0;
    font-weight: 500;
  }
  .banner.swap ol { margin: 8px 0 0; padding-left: 1.3rem; font-size: 0.88rem; color: #e7ecf3; }
  section {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 20px 20px; margin-bottom: 16px;
  }
  section h2 {
    margin: 0 0 14px; font-size: 1.05rem; font-weight: 600;
    color: var(--accent); letter-spacing: 0.02em; text-transform: uppercase;
  }
  .empty { color: var(--muted); font-style: italic; padding: 8px 0; }
  .card {
    background: var(--panel2); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
  }
  .card-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .badge {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    padding: 3px 8px; border-radius: 6px; color: #0f1419;
  }
  .badge.add { background: var(--add); }
  .badge.edit { background: var(--edit); }
  .badge.remove { background: var(--remove); color: #fff; }
  .badge.bench { background: var(--bench); color: #fff; }
  .badge.unbench { background: var(--add); }
  .vbadge {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em;
    padding: 2px 7px; border-radius: 999px; border: 1px solid var(--border);
  }
  .vbadge.proven { background: rgba(61,154,106,0.25); color: #9fdfb8; border-color: var(--add); }
  .vbadge.meta-grounded { background: rgba(91,159,212,0.22); color: #b8d4ec; border-color: var(--accent); }
  .vbadge.failed { background: rgba(196,92,92,0.22); color: #f0b0b0; border-color: var(--remove); }
  .vbadge.unvalidated { background: rgba(139,155,180,0.15); color: var(--muted); }
  .target { font-size: 1rem; }
  .field {
    color: var(--muted); font-size: 0.82rem;
    background: rgba(0,0,0,0.25); padding: 2px 8px; border-radius: 5px;
  }
  .detail { margin: 8px 0 0; color: var(--text); }
  .change {
    margin-top: 8px; font-size: 0.88rem;
    display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline;
  }
  .before { color: var(--muted); text-decoration: line-through; opacity: 0.85; }
  .arrow { color: var(--accent); }
  .after { color: #c5e0a5; font-weight: 560; }
  .why { margin-top: 6px; font-size: 0.82rem; color: var(--muted); }
  .why::before { content: "why: "; color: var(--accent); }
  .swap-box {
    margin-top: 10px; padding: 10px 12px; border-radius: 8px;
    background: rgba(166,124,42,0.12); border: 1px solid var(--warn);
  }
  .swap-title { color: #f0d9a0; font-size: 0.9rem; }
  .swap-steps { margin: 6px 0 0; padding-left: 1.2rem; font-size: 0.84rem; }
  ul.tips { margin: 0; padding-left: 1.2rem; }
  ul.tips li { margin-bottom: 6px; }
  details.inventory {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 14px 18px; margin-bottom: 16px;
  }
  details.inventory summary {
    cursor: pointer; font-weight: 600; color: var(--muted); list-style: none;
  }
  details.inventory summary::-webkit-details-marker { display: none; }
  details.inventory summary::before { content: "▸ "; color: var(--accent); }
  details.inventory[open] summary::before { content: "▾ "; }
  .muted { font-weight: 400; color: var(--muted); font-size: 0.88rem; }
  .inv-meta {
    display: grid; gap: 4px; margin: 14px 0;
    font-size: 0.9rem; color: var(--muted);
  }
  .inv-meta b { color: var(--text); }
  .inv-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px; margin-bottom: 12px;
  }
  .inv-form {
    background: var(--panel2); border-radius: 8px; padding: 10px 12px;
    border: 1px solid var(--border);
  }
  .inv-form h4 { margin: 0 0 6px; font-size: 0.92rem; }
  .inv-form .role { color: var(--muted); font-weight: 400; font-size: 0.78rem; }
  .inv-form .aud { font-size: 0.8rem; color: #c5e0a5; margin-bottom: 6px; }
  .inv-form ul { margin: 0; padding-left: 1.1rem; font-size: 0.82rem; color: var(--muted); }
  details.inventory h3 { font-size: 0.95rem; color: var(--accent); margin: 16px 0 8px; }
  .meter {
    display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline;
    padding: 12px 14px; border-radius: 10px; margin-bottom: 14px;
    border: 1px solid var(--border); background: var(--panel2);
  }
  .meter.ok { border-color: var(--ok); }
  .meter.hot { border-color: var(--warn); background: rgba(166,124,42,0.12); }
  .meter-label { color: var(--muted); font-size: 0.85rem; }
  .meter-value { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.02em; }
  .meter-note { color: var(--muted); font-size: 0.78rem; }
  .macro-list { display: flex; flex-direction: column; gap: 8px; }
  details.macro-card {
    background: var(--panel2); border: 1px solid var(--border);
    border-radius: 10px; padding: 0;
  }
  details.macro-card.slot-active { border-color: rgba(61,154,106,0.55); }
  details.macro-card.slot-create { border-color: rgba(212,160,23,0.45); }
  details.macro-card.slot-benched { opacity: 0.85; }
  details.macro-card > summary {
    cursor: pointer; list-style: none;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 12px 14px; font-weight: 600;
  }
  details.macro-card > summary::-webkit-details-marker { display: none; }
  details.macro-card > summary::before { content: "▸ "; color: var(--accent); }
  details.macro-card[open] > summary::before { content: "▾ "; }
  .mname { font-size: 1rem; }
  .mside { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }
  .mslot {
    font-size: 0.72rem; padding: 2px 7px; border-radius: 5px;
    background: rgba(0,0,0,0.3); color: var(--muted);
  }
  .macro-body { padding: 0 14px 14px; border-top: 1px solid var(--border); }
  .mkey { color: var(--muted); font-size: 0.78rem; margin-left: 10px; font-weight: 400; }
  .mrank { color: var(--muted); font-weight: 600; margin-right: 4px; }
  .mneeds { color: var(--warn); border: 1px solid var(--warn); border-radius: 4px; padding: 0 6px;
            font-size: 0.72rem; font-weight: 700; margin-left: 8px; }
  .mfire, .mpairs { margin: 8px 0; font-size: 0.9rem; }
  .mfire b, .mpairs b { color: var(--accent); }
  .pchip { display: inline-block; margin: 2px 6px 2px 0; padding: 2px 8px; border-radius: 999px;
    background: rgba(91,159,212,0.14); border: 1px solid var(--border); font-size: 0.8rem; }
  table.ca-set { width: 100%; border-collapse: collapse; font-size: 0.86rem; margin-top: 4px; }
  table.ca-set td { padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); vertical-align: top; }
  tr.ca-sec td { color: var(--accent); font-weight: 700; padding-top: 10px; font-size: 0.8rem;
    letter-spacing: 0.04em; text-transform: uppercase; }
  td.ca-k { width: 34%; color: var(--muted); }
  td.ca-v { font-weight: 600; }
  tr.ca-row.dflt td.ca-v { color: var(--muted); font-weight: 400; }
  .ca-secnote { text-transform: none; letter-spacing: 0; font-weight: 400; color: var(--muted); font-size: 0.76rem; margin-top: 2px; }
  .ca-ingame { margin-top: 6px; }
  .ca-assume { margin-top: 10px; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--warn);
    background: rgba(166,124,42,0.12); font-size: 0.84rem; }
  ul.ca-src { font-size: 0.8rem; padding-left: 1.1rem; }
  .macro-body h4 { margin: 14px 0 6px; font-size: 0.88rem; color: var(--accent); }
  .purpose { margin: 10px 0 4px; }
  .when { font-size: 0.88rem; color: var(--muted); }
  ol.xbox { margin: 0; padding-left: 1.2rem; font-size: 0.86rem; color: var(--muted); }
  .settings { display: grid; gap: 4px; }
  .set-row {
    display: grid; grid-template-columns: 160px 1fr auto; gap: 8px;
    font-size: 0.82rem; padding: 4px 0; border-bottom: 1px solid rgba(46,58,79,0.6);
  }
  .set-k { color: var(--accent); }
  .set-v { color: var(--text); }
  .set-tag {
    font-size: 0.68rem; padding: 1px 6px; border-radius: 4px; align-self: start;
  }
  .set-tag.confirmed { background: rgba(61,154,106,0.25); color: #9fdfb8; }
  .set-section { margin: 10px 0 14px; padding: 8px 10px; border: 1px solid var(--border); border-radius: 8px; }
  .set-section h5 { margin: 0 0 8px; color: var(--accent); font-size: 0.85rem; letter-spacing: 0.04em; text-transform: uppercase; }
  .set-note { grid-column: 1 / -1; font-size: 0.78rem; color: #f0c674; margin-top: -4px; }
  .set-tag.approx { background: rgba(139,155,180,0.2); color: var(--muted); }
  .copy-wrap { margin-top: 8px; }
  .copy-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
  .copy-btn {
    background: var(--accent); color: #0f1419; border: none;
    border-radius: 6px; padding: 6px 12px; font-weight: 700; cursor: pointer;
  }
  .copy-btn:hover { filter: brightness(1.1); }
  .copy-btn.copied { background: var(--add); }
  pre.copy-block {
    margin: 8px 0 0; padding: 12px 14px;
    background: #0c1016; border: 1px solid var(--border); border-radius: 8px;
    font-size: 0.78rem; line-height: 1.4; white-space: pre-wrap; color: #c5d0e0;
    max-height: 420px; overflow: auto;
  }
  .scout-grid { display: grid; gap: 12px; grid-template-columns: 1fr; }
  @media (min-width: 720px) {
    .scout-grid { grid-template-columns: 1fr 1fr; }
  }
  .scout-card {
    background: var(--panel2); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
  }
  .scout-card h3 {
    margin: 0 0 8px; font-size: 0.88rem; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .scout-card ul { margin: 0; padding-left: 1.1rem; font-size: 0.86rem; }
  .scout-card li { margin-bottom: 4px; }
  .scout-src { font-size: 0.78rem; margin-top: 8px; }
  .scout-src a { color: var(--accent); }
  .scout-affect { grid-column: 1 / -1; }
  footer { margin-top: 24px; color: var(--muted); font-size: 0.8rem; text-align: center; }
  code { background: rgba(0,0,0,0.35); padding: 1px 5px; border-radius: 4px; }
  table.srcs { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  table.srcs td { padding: 3px 6px; border-bottom: 1px solid rgba(255,255,255,0.06); vertical-align: top; }
  table.srcs a { color: var(--accent); }
  li.conflict { margin-bottom: 6px; }
"""



def _fmt_local_ts(ts: str) -> str:
    from datetime import datetime

    if not ts:
        return "?"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return ts
        return dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except ValueError:
        return ts


def _render_meta_freshness(scout: dict[str, Any]) -> str:
    """v1.12: when the meta was fetched, from where, and what changed."""
    mode = scout.get("mode") or ""
    if not mode and not scout.get("sources"):
        return ""
    if not mode:  # older / Madden scout results
        mode = "cache" if scout.get("from_cache") else ("live" if scout.get("available") else "seed")
    cache_label = "CACHED FALLBACK (live research did not run)" if scout.get("research_status") else "cached today"
    label = {"live": "LIVE — fetched this prep", "cache": cache_label, "seed": "seed research (offline / no cache)"}.get(
        mode, mode or "?"
    )
    rows = []
    for s in scout.get("sources") or []:
        ok = s.get("fetched")
        state = "ok" if ok else ("seed" if s.get("kind") == "seed" else (s.get("error") or "failed"))
        url = s.get("url") or ""
        name = s.get("label") or s.get("title") or url
        rows.append(
            f'<tr><td>{"✅" if ok else ("📚" if s.get("kind") == "seed" else "⚠️")}</td>'
            f'<td><a href="{_esc(url)}" target="_blank" rel="noopener">{_esc(name)}</a></td>'
            f"<td class='muted'>{_esc(s.get('kind') or '')}</td><td class='muted'>{_esc(state)}</td></tr>"
        )
    changes = "".join(f"<li>{_esc(c)}</li>" for c in (scout.get("changes_since_last") or [])[:10])
    heads = "".join(
        f'<li><span class="muted">{_esc(h.get("date") or "")}</span> '
        f'<a href="{_esc(h.get("link") or "")}" target="_blank" rel="noopener">{_esc(h.get("title") or "")}</a> '
        f'<span class="muted">· {_esc(h.get("source") or "")}</span></li>'
        for h in (scout.get("headlines") or [])[:8]
    )
    sig = scout.get("rz_signals") or {}
    sig_txt = ", ".join(f"{k} ×{v}" for k, v in sorted(sig.items(), key=lambda kv: -kv[1])[:8])
    prev = scout.get("previous_fetched_at") or ""
    return f"""
    <section id="meta-freshness">
      <h2>Current meta — freshness &amp; sources</h2>
      <div class="scout-grid">
        <div class="scout-card">
          <h3>Fetched</h3>
          <div><b>{_esc(label)}</b> · {_esc(_fmt_local_ts(scout.get("fetched_at") or ""))}</div>
          <div class="muted">previous cache: {_esc(_fmt_local_ts(prev) if prev else "none")}</div>
          <div class="muted">{_esc(scout.get("ttl_policy") or "")}</div>
          <div class="muted">{_esc(scout.get("message") or "")}</div>
          <div class="muted">red-zone signals: {_esc(sig_txt or "none")}</div>
        </div>
        <div class="scout-card">
          <h3>What changed since the last cache</h3>
          <ul>{changes or "<li class='muted'>(no comparison available)</li>"}</ul>
        </div>
        <div class="scout-card scout-affect">
          <h3>Sources used</h3>
          <table class="srcs">{"".join(rows) or "<tr><td class='muted'>none</td></tr>"}</table>
          <h3 style="margin-top:10px">Recent headlines</h3>
          <ul>{heads or "<li class='muted'>(none this pass)</li>"}</ul>
        </div>
      </div>
    </section>
    """


def _render_zone_plan(za: dict[str, Any] | None) -> str:
    """v1.12: RZ / GL / open plan from learned zone weights + meta priors, and conflicts."""
    za = za or {}
    if not za:
        return ""
    if za.get("error"):
        return f"<section><h2>Zone plan</h2><div class='empty'>Unavailable: {_esc(za['error'])}</div></section>"
    labels = {"rz": "Red zone (inside the 20)", "gl": "Goal line / goal-to-go", "open": "Open field"}
    cards = []
    for zone in ("rz", "gl", "open"):
        rows = (za.get("zone_plan") or {}).get(zone) or []
        trs = "".join(
            f"<tr><td><b>{r['p']:.0%}</b></td><td>{_esc(r['formation'])} — <b>{_esc(r['play'])}</b></td>"
            f"<td class='muted'>yours {r['learned']:+.2f} (n={r['n_zone']})</td>"
            f"<td class='muted'>meta {r['meta']:+.2f}</td></tr>"
            for r in rows
        )
        cards.append(f"<div class='scout-card'><h3>{_esc(labels[zone])}</h3><table class='srcs'>{trs}</table></div>")
    conf = "".join(
        f"<li class='conflict'>⚠️ {_esc(c['message'])}"
        + (f"<br/><span class='muted'>meta sources: {_esc('; '.join(c.get('sources') or []))}</span>" if c.get("sources") else "")
        + "</li>"
        for c in za.get("conflicts") or []
    )
    sup = "".join(f"<li>✅ {_esc(c['message'])}</li>" for c in (za.get("support") or [])[:5])
    labs = "".join(
        f"<li>🧪 <b>{_esc(c['formation'])} — {_esc(c['play'])}</b> ({_esc('/'.join(c.get('zones') or []))}): "
        f"{_esc(c.get('note') or '')} <span class='muted'>[{_esc(c.get('verified_in') or '')}]</span></li>"
        for c in za.get("lab_candidates") or []
    )
    leaders = za.get("leaders") or {}
    lead_bits = []
    for z, lab in (("general", "General"), ("rz", "Red zone"), ("gl", "Goal line")):
        lb = leaders.get(z) or {}
        best = ", ".join(f"{_esc(n)} {v:+.2f}" for n, v, _ in (lb.get("best") or [])[:3]) or "—"
        worst = ", ".join(f"{_esc(n)} {v:+.2f}" for n, v, _ in (lb.get("worst") or [])[:3]) or "—"
        lead_bits.append(f"<li><b>{lab}</b> — best: {best}<br/><span class='muted'>worst: {worst}</span></li>")
    li = za.get("learning") or {}
    consts = "".join(f"<tr><td>{_esc(k)}</td><td class='muted'>{_esc(v)}</td></tr>" for k, v in za.get("constants") or [])
    return f"""
    <section id="zone-plan">
      <h2>Zone plan — your results + current meta</h2>
      <div class="muted" style="margin-bottom:8px">Shares = how often the live caller will suggest each call in that zone
        (learned zone weights blended with general, plus a meta prior that fades as your own sample grows).
        Retrain rules {_esc(li.get("rules_version") or "?")} · DB rebuilt {_esc(li.get("rebuilt_at") or "n/a")}.</div>
      <div class="scout-grid">{"".join(cards)}</div>
      <div class="scout-grid" style="margin-top:10px">
        <div class="scout-card scout-affect">
          <h3>Meta vs your data — conflicts</h3>
          <ul>{conf or "<li class='muted'>No conflicts: the meta and your results agree (or your sample is too small).</li>"}{sup}</ul>
        </div>
        {"<div class='scout-card scout-affect'><h3>Ohio State lab candidates (in the OSU book, untested by you)</h3><ul>" + labs + "</ul></div>" if labs else ""}
        <div class="scout-card scout-affect">
          <h3>Learned leaders</h3>
          <ul>{"".join(lead_bits)}</ul>
          <details><summary class="muted">Retrain constants</summary><table class="srcs">{consts}</table></details>
        </div>
      </div>
    </section>
    """


def _render_meta_scout(scout: dict[str, Any] | None) -> str:
    """Top-of-prep Live meta scout panel (rich)."""
    scout = scout or {}
    available = bool(scout.get("available"))
    msg = scout.get("message") or ""
    if not available and not (
        scout.get("patch_notes")
        or scout.get("meta_offense")
        or scout.get("meta_defense")
    ):
        fallback = scout.get("baseline_fallback") or "cfb27-2026-09"
        return _render_meta_freshness(scout) + f"""
    <section>
      <h2>Live meta scout</h2>
      <div class="empty">Scout unavailable — using cached/baseline {_esc(fallback)}
        <span class="muted">({_esc(msg)})</span>
      </div>
    </section>
    """
    conf = _esc(scout.get("confidence") or "low")
    tags = []
    if scout.get("offline"):
        tags.append("offline")
    if scout.get("from_cache"):
        tags.append("cached")
    tag_html = " · ".join(_esc(t) for t in tags) if tags else "live"
    # Patch radar
    patch_bits = []
    for pn in (scout.get("patch_notes") or [])[:4]:
        bullets = "".join(
            f"<li>{_esc(b)}</li>" for b in (pn.get("bullets") or [])[:4]
        )
        patch_bits.append(
            f"<div><strong>{_esc(pn.get('version') or '?')}</strong> "
            f"<span class='muted'>{_esc(pn.get('date') or '')}</span> — "
            f"{_esc(pn.get('title') or '')}"
            f"<ul>{bullets}</ul></div>"
        )
    patch_html = "".join(patch_bits) or "<div class='empty'>No patch bullets this pass</div>"
    o_lis = "".join(f"<li>{_esc(t)}</li>" for t in (scout.get("meta_offense") or [])[:6])
    d_lis = "".join(f"<li>{_esc(t)}</li>" for t in (scout.get("meta_defense") or [])[:6])
    affect_lis = "".join(
        f"<li>{_esc(t)}</li>" for t in (scout.get("affect_this_prep") or [])[:8]
    )
    src_bits = []
    for s in (scout.get("sources") or []):
        if not s.get("fetched"):
            continue
        url = s.get("url") or ""
        title = s.get("title") or url
        src_bits.append(
            f'<div class="scout-src"><a href="{_esc(url)}" target="_blank" rel="noopener">'
            f"{_esc(title)}</a></div>"
        )
        if len(src_bits) >= 6:
            break
    fresh_html = _render_meta_freshness(scout)
    return fresh_html + f"""
    <section>
      <h2>Live meta scout
        <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:400">
          ({tag_html} · confidence={conf})
        </span>
      </h2>
      <div class="scout-grid">
        <div class="scout-card">
          <h3>Patch radar</h3>
          {patch_html}
        </div>
        <div class="scout-card">
          <h3>What's meta right now</h3>
          <div><b>O</b><ul>{o_lis or "<li class='muted'>(thin)</li>"}</ul></div>
          <div style="margin-top:8px"><b>D</b><ul>{d_lis or "<li class='muted'>(thin)</li>"}</ul></div>
          {"".join(src_bits)}
        </div>
        <div class="scout-card scout-affect">
          <h3>How it affects THIS prep</h3>
          <ul>{affect_lis or "<li class='muted'>Soft priors only — baseline still primary</li>"}</ul>
        </div>
      </div>
    </section>
    """


def render_prep_details_html(plan: dict[str, Any]) -> str:
    """v1.14 DETAILS page (written next to the prep page, not auto-opened): everything
    the coach computed this prep — book reasons/flags/history/limits, research,
    sources, zone plan, conflicts, headlines, constants, per-play table."""
    from cfb_coach.prep_book_html import BOOK_CSS, MIN_CSS, render_book, render_research

    shown = plan.get("shown_deltas") or []
    pb = [d for d in shown if d.get("kind") == "playbook"]
    mac = [d for d in shown if d.get("kind") == "macro"]
    tips = plan.get("tips") or []
    budget = plan.get("slot_budget") or {}
    cards = plan.get("macro_cards") or []
    banners = plan.get("swap_banners") or []

    try:
        ts_raw = plan.get("ts") or ""
        if ts_raw.endswith("Z"):
            ts_raw = ts_raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts_raw).astimezone(ET)
        ts_label = dt.strftime("%a %b %d, %Y · %I:%M %p ET")
    except Exception:
        ts_label = plan.get("ts", "")

    no_changes = not shown
    status = (
        '<div class="banner ok">No playbook changes — run baseline as-is</div>'
        if no_changes
        else (
            f'<div class="banner info">{len(shown)} adjustment'
            f'{"s" if len(shown) != 1 else ""} for this opponent</div>'
        )
    )
    tip_lis = "".join(f"<li>{_esc(t)}</li>" for t in tips)
    oid = _esc(plan.get("opponent_id") or "")
    dcfg = plan.get("dynasty_config") or {}
    dynasty_id = _esc(plan.get("dynasty") or dcfg.get("id") or "alabama")
    dynasty_label = _esc(dcfg.get("label") or dynasty_id)
    dynasty_mode = _esc(dcfg.get("mode") or "")
    exp = bool(dcfg.get("experimental_badge"))
    dynasty_banner = (
        f'<div class="banner dyn {("exp" if exp else "ser")}">'
        f"<strong>Dynasty:</strong> {dynasty_label} "
        f"<span class=\"muted\">({dynasty_mode})</span>"
        + (" <span class=\"vbadge meta-grounded\">experimental</span>" if exp else "")
        + f"<div class=\"why\">{_esc(dcfg.get('description') or '')}</div>"
        f"<div class=\"why\">{_esc(plan.get('doctrine') or '')}</div>"
        "</div>"
    )


    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prep details — {_esc(plan.get("display_name"))} · CFB27</title>
<style>{_CSS}{BOOK_CSS}{MIN_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Prep details — vs {_esc(plan.get("display_name"))}
        <span style="color:var(--muted);font-weight:500">({_esc(plan.get("team"))})</span>
      </h1>
      <div class="meta">
        <span>{_esc(plan.get("game", "CFB 27"))}</span>
        <span>META {_esc(plan.get("version"))}</span>
        <span>macros {_esc(plan.get("macro_catalog_version", ""))}</span>
        <span>{_esc(ts_label)}</span>
        <span>overlay: {_esc(plan.get("overlay_depth"))}</span>
        <span>dynasty: {dynasty_id}</span>
      </div>
    </header>

    {status}
    {dynasty_banner}
    {render_book(plan.get("cfb_book"))}
    {render_research(plan.get("meta_scout") or {})}
    {_render_meta_scout(plan.get("meta_scout") or {})}
    {_render_zone_plan(plan.get("zone_alignment"))}
    {_render_swap_banners(banners)}

    <section>
      <h2>Playbook adjustments</h2>
      {_render_delta_cards(pb, "No playbook changes — run baseline as-is")}
    </section>

    <section>
      <h2>Macro adjustments</h2>
      {_render_delta_cards(mac, "No macro changes — keep current set")}
    </section>

    <section>
      <h2>{"Active loadout — offense only (CPU)" if plan.get("offense_only") else "Active loadout (≤8) — click to expand · Copy settings"}</h2>
      {_render_macro_accordion(
          cards,
          budget,
          offense_only=bool(plan.get("offense_only")),
          replacing_lines=list(plan.get("replacing_lines") or []),
          loadout=plan.get("loadout") or dict(),
          details=True,
      )}
    </section>

    {_render_inventory({**(plan.get("inventory") or dict()), "_offense_only": bool(plan.get("offense_only"))})}

    <section>
      <h2>Call emphasis</h2>
      <ul class="tips">{tip_lis}</ul>
    </section>

    <footer>
      Inventory is already stocked from seed — only opponent-specific deltas above
      (prep suggests only <b>proven</b> / <b>meta_grounded</b> tweaks — no ungrounded invention).
      Active loadout hard-capped at <b>8 O+D combined</b> (prep shows only those 8 — never benched FLOOD/SCREEN). CPU games = offense-only.
      <b>Workflow:</b> bring a suggested macro into game → if cooked, adjust via postgame → if it holds, mark <b>proven</b>.
      Live caller prefers proven macros; tags meta_grounded / failed / unvalidated when suggesting others.
      <b>Doctrine:</b> do NOT auto-use macros from one concept appearance — most snaps Cover 3 Sky / Quarters / Tampa 2 with no macro.
      UI note: Safety Midpoint <b>Strong</b> = toward pass strength.
      Mark applied with <code>prep --opponent {oid} --mark-applied</code>.
    </footer>
  </div>
{_COPY_JS}
</body>
</html>
"""


_COPY_JS = """  <script>
    document.querySelectorAll(".copy-btn").forEach(function(btn) {
      btn.addEventListener("click", function() {
        var id = btn.getAttribute("data-target");
        var el = document.getElementById(id);
        if (!el) return;
        var text = el.innerText || el.textContent || "";
        function done() {
          var prev = btn.textContent;
          btn.textContent = "Copied";
          btn.classList.add("copied");
          setTimeout(function() {
            btn.textContent = prev;
            btn.classList.remove("copied");
          }, 1400);
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done).catch(function() {
            var r = document.createRange(); r.selectNodeContents(el);
            var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
            try { document.execCommand("copy"); } catch (e) {}
            s.removeAllRanges(); done();
          });
        } else {
          var r = document.createRange(); r.selectNodeContents(el);
          var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
          try { document.execCommand("copy"); } catch (e) {}
          s.removeAllRanges(); done();
        }
      });
    });
  </script>
"""


def render_prep_html(plan: dict[str, Any]) -> str:
    """v1.14 prep page — ONLY what Aidan needs at the console: the formations to have in
    the custom playbook (new / in book / remove + the one apply command), the audibles,
    and the macros with exact settings. Everything else lives in the details page."""
    from cfb_coach.prep_book_html import (
        BOOK_CSS,
        MIN_CSS,
        render_audibles_min,
        render_formations_min,
        research_fail_banner,
        research_status_line,
    )

    try:
        ts_raw = plan.get("ts") or ""
        if ts_raw.endswith("Z"):
            ts_raw = ts_raw[:-1] + "+00:00"
        ts_label = datetime.fromisoformat(ts_raw).astimezone(ET).strftime("%a %b %d · %-I:%M %p ET")
    except Exception:
        ts_label = plan.get("ts", "")
    scout = plan.get("meta_scout") or {}
    dcfg = plan.get("dynasty_config") or {}
    book = plan.get("cfb_book") or {}
    cards = plan.get("macro_cards") or []
    offense_only = bool(plan.get("offense_only"))
    details = plan.get("details_path") or ""
    macros = _render_macro_accordion(
        cards,
        plan.get("slot_budget") or {},
        offense_only=offense_only,
        replacing_lines=list(plan.get("replacing_lines") or []),
        loadout=plan.get("loadout") or {},
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prep — {_esc(plan.get("display_name"))} · CFB27</title>
<style>{_CSS}{BOOK_CSS}{MIN_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>vs {_esc(plan.get("display_name"))}
        <span style="color:var(--muted);font-weight:500">({_esc(dcfg.get("label") or plan.get("dynasty") or "")})</span>
      </h1>
      <div class="meta"><span>{_esc(plan.get("game", "CFB 27"))}</span><span>{_esc(ts_label)}</span></div>
    </header>
    {research_fail_banner(scout)}
    <div class="rline">{_esc(research_status_line(scout))}</div>
    {render_formations_min(book)}
    {render_audibles_min(book)}
    <section id="macros">
      <h2>{"Macros — offense custom adjustments (CPU game)" if offense_only else "Macros — Active loadout (≤8)"}</h2>
      {macros}
    </section>
    <footer class="muted">Details (reasons, research, sources, history): {_esc(details or "prep --details")}</footer>
  </div>
{_COPY_JS}
</body>
</html>
"""


def prep_details_path(opponent_id: str) -> Path:
    return default_prep_dir() / f"prep_details_{opponent_id}.html"


def write_prep_details_html(opponent_id: str, plan: dict[str, Any], *, path: Path | None = None) -> Path:
    out = path or prep_details_path(opponent_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_prep_details_html(plan), encoding="utf-8")
    return out


def write_prep_html(
    opponent_id: str,
    plan: dict[str, Any] | None = None,
    *,
    path: Path | None = None,
) -> Path:
    if plan is None:
        plan = build_prep_plan(opponent_id, persist=False)
    out = path or prep_html_path(opponent_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    details = out.with_name(f"prep_details_{opponent_id}.html")
    plan.setdefault("details_path", str(details))
    out.write_text(render_prep_html(plan), encoding="utf-8")
    try:  # details page next to it — written every prep, never auto-opened
        write_prep_details_html(opponent_id, plan, path=details)
    except Exception:  # noqa: BLE001
        pass
    return out


def _windows_path(path: Path) -> str | None:
    try:
        r = subprocess.run(
            ["wslpath", "-w", str(path.resolve())],
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (FileNotFoundError, OSError):
        pass
    return None


def open_prep_html(path: Path, *, open_browser: bool = True) -> Path:
    """Open prep HTML. On Linux/WSL, fall back to wslview / explorer.exe / print path."""
    resolved = path.resolve()
    if not open_browser:
        return path

    uri = resolved.as_uri()
    opened = False
    try:
        opened = bool(webbrowser.open(uri))
    except Exception:
        opened = False

    if not opened:
        opened = _try_cmd(["xdg-open", str(resolved)])

    # On WSL, webbrowser/xdg-open often "succeed" without a real window.
    # Always try the WSL chain when detected; print a clear Windows path if all fail.
    if _is_wsl():
        if _try_cmd(["wslview", str(resolved)]):
            return path
        win = _windows_path(resolved)
        if win and _try_cmd(["explorer.exe", win]):
            return path
        if win:
            print(
                "Could not auto-open browser. Open this Windows path manually:\n"
                f"  {win}"
            )
        else:
            print(
                "Could not auto-open browser (WSL). Open this file manually:\n"
                f"  {resolved}\n"
                f"  (or: explorer.exe $(wslpath -w {resolved}))"
            )
        return path

    if not opened:
        print(f"Could not auto-open browser. Open this file manually:\n  {resolved}")
    return path


def generate_and_open(
    opponent_id: str,
    opp: dict[str, Any] | None = None,
    *,
    db: Any = None,
    persist: bool = True,
    open_browser: bool = True,
    mark_applied: bool = False,
    dynasty: str | None = None,
    offline: bool = False,
    refresh_meta: bool = False,
) -> tuple[Path, dict[str, Any]]:
    from cfb_coach.install_sheet import mark_prep_applied

    plan = build_prep_plan(
        opponent_id,
        opp,
        db=db,
        persist=persist,
        dynasty=dynasty,
        offline=offline,
        refresh_meta=refresh_meta,
    )
    if mark_applied and db is not None:
        mark_prep_applied(db, opponent_id, plan["proposed_deltas"])
        plan["applied_count"] = len(plan["proposed_deltas"])
        plan["shown_deltas"] = []
        plan["swap_banners"] = []
    path = write_prep_html(opponent_id, plan)
    try:  # plain-text twins next to the HTML (handy on a phone / for diffing preps)
        from cfb_coach.install_sheet import format_delta_text, format_prep_minimal_text

        path.with_suffix(".txt").write_text(format_prep_minimal_text(plan), encoding="utf-8")
        Path(plan["details_path"]).with_suffix(".txt").write_text(format_delta_text(plan), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    open_prep_html(path, open_browser=open_browser)
    return path, plan
