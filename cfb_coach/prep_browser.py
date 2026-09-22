"""Polished dark HTML prep sheet (deltas only) — open in browser.

v1.4: clickable macro accordion + Copy settings, Active 8/8 meter,
swap-plan banners when ADD would exceed Aidan's USER O+D cap,
validation badges (proven | meta_grounded | failed | unvalidated).
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
    rows: list[str] = []
    for key, val in settings.items():
        if isinstance(val, dict):
            value = val.get("value", "")
            tag = (val.get("status") or "approx").lower()
            if tag not in ("confirmed", "approx"):
                tag = "approx"
            rows.append(
                "<div class='set-row'>"
                f"<span class='set-k'>{_esc(key)}</span>"
                f"<span class='set-v'>{_esc(value)}</span>"
                f"<span class='set-tag {tag}'>{tag}</span>"
                "</div>"
            )
        else:
            rows.append(
                "<div class='set-row'>"
                f"<span class='set-k'>{_esc(key)}</span>"
                f"<span class='set-v'>{_esc(val)}</span>"
                "<span class='set-tag approx'>approx</span>"
                "</div>"
            )
    return f"<div class='settings'>{''.join(rows)}</div>"


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
    cards: list[dict[str, Any]], budget: dict[str, Any]
) -> str:
    if not cards:
        return '<div class="empty">No macro catalog loaded</div>'

    order = {"active": 0, "benched": 1, "create": 2, "reference": 3}
    cards = sorted(
        cards, key=lambda c: (order.get(c.get("slot", ""), 9), c.get("name") or "")
    )

    meter = _esc(budget.get("meter") or "Active ?/?")
    meter_cls = "meter hot" if budget.get("at_cap") else "meter ok"

    items: list[str] = []
    for c in cards:
        mid = str(c.get("id") or c.get("name") or "?")
        name = c.get("name") or mid
        slot = c.get("slot") or "reference"
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
        items.append(
            f"""
            <details class="macro-card slot-{_esc(slot)}"{open_attr}>
              <summary>
                <span class="mname">{_esc(name)}</span>
                <span class="mside">{_esc(side)}</span>
                <span class="mslot">{_esc(slot)}</span>
                {_val_badge(str(status))}
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
    <div class="{meter_cls}">
      <span class="meter-label">Active slot budget (USER O+D)</span>
      <span class="meter-value">{meter}</span>
      <span class="meter-note">Aidan hard cap 8 — EA UI may show 10</span>
    </div>
    <div class="macro-list">
      {"".join(items)}
    </div>
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
    macros_b = ", ".join(_esc(m) for m in inv.get("macros_benched") or [])
    o_macros = inv.get("offensive_macros") or []
    o_mac_bit = ", ".join(_esc(m) for m in o_macros) if o_macros else "(none)"

    return f"""
    <details class="inventory">
      <summary>Current inventory <span class="muted">(read-only — already stocked)</span></summary>
      <div class="inv-meta">
        <div><b>O book:</b> {_esc(inv.get("offense_book"))}</div>
        <div><b>D book:</b> {_esc(inv.get("defense_book"))}</div>
        <div><b>D macros active:</b> {macros_a}</div>
        <div><b>D macros benched:</b> {macros_b}</div>
        <div><b>O macros:</b> {o_mac_bit}</div>
      </div>
      <h3>Offense formations → plays</h3>
      <div class="inv-grid">{"".join(o_parts)}</div>
      <h3>Defense packages → calls</h3>
      <div class="inv-grid">{"".join(d_parts)}</div>
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
  footer { margin-top: 24px; color: var(--muted); font-size: 0.8rem; text-align: center; }
  code { background: rgba(0,0,0,0.35); padding: 1px 5px; border-radius: 4px; }
"""


def render_prep_html(plan: dict[str, Any]) -> str:
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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prep — {_esc(plan.get("display_name"))} · CFB27</title>
<style>{_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>vs {_esc(plan.get("display_name"))}
        <span style="color:var(--muted);font-weight:500">({_esc(plan.get("team"))})</span>
      </h1>
      <div class="meta">
        <span>{_esc(plan.get("game", "CFB 27"))}</span>
        <span>META {_esc(plan.get("version"))}</span>
        <span>macros {_esc(plan.get("macro_catalog_version", ""))}</span>
        <span>{_esc(ts_label)}</span>
        <span>overlay: {_esc(plan.get("overlay_depth"))}</span>
      </div>
    </header>

    {status}
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
      <h2>Macro catalog — click to expand · Copy settings</h2>
      {_render_macro_accordion(cards, budget)}
    </section>

    {_render_inventory(plan.get("inventory") or {})}

    <section>
      <h2>Call emphasis</h2>
      <ul class="tips">{tip_lis}</ul>
    </section>

    <footer>
      Inventory is already stocked from seed — only opponent-specific deltas above
      (prep suggests only <b>proven</b> / <b>meta_grounded</b> tweaks — no ungrounded invention).
      Active Custom Adjustments hard-capped at <b>8 O+D combined</b> for USER dynasty (Aidan rule).
      <b>Workflow:</b> bring a suggested macro into game → if cooked, adjust via postgame → if it holds, mark <b>proven</b>.
      Live caller prefers proven macros; tags meta_grounded / failed / unvalidated when suggesting others.
      Mark applied with <code>prep --opponent {oid} --mark-applied</code>.
    </footer>
  </div>
  <script>
    document.querySelectorAll(".copy-btn").forEach(function(btn) {{
      btn.addEventListener("click", function() {{
        var id = btn.getAttribute("data-target");
        var el = document.getElementById(id);
        if (!el) return;
        var text = el.innerText || el.textContent || "";
        function done() {{
          var prev = btn.textContent;
          btn.textContent = "Copied";
          btn.classList.add("copied");
          setTimeout(function() {{
            btn.textContent = prev;
            btn.classList.remove("copied");
          }}, 1400);
        }}
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          navigator.clipboard.writeText(text).then(done).catch(function() {{
            var r = document.createRange(); r.selectNodeContents(el);
            var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
            try {{ document.execCommand("copy"); }} catch (e) {{}}
            s.removeAllRanges(); done();
          }});
        }} else {{
          var r = document.createRange(); r.selectNodeContents(el);
          var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
          try {{ document.execCommand("copy"); }} catch (e) {{}}
          s.removeAllRanges(); done();
        }}
      }});
    }});
  </script>
</body>
</html>
"""


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
    out.write_text(render_prep_html(plan), encoding="utf-8")
    return out


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        ver = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "microsoft" in ver.lower()


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


def _try_cmd(argv: list[str]) -> bool:
    try:
        r = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
        )
        return r.returncode == 0
    except (FileNotFoundError, OSError):
        return False


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
) -> tuple[Path, dict[str, Any]]:
    from cfb_coach.install_sheet import mark_prep_applied

    plan = build_prep_plan(opponent_id, opp, db=db, persist=persist)
    if mark_applied and db is not None:
        mark_prep_applied(db, opponent_id, plan["proposed_deltas"])
        plan["applied_count"] = len(plan["proposed_deltas"])
        plan["shown_deltas"] = []
        plan["swap_banners"] = []
    path = write_prep_html(opponent_id, plan)
    open_prep_html(path, open_browser=open_browser)
    return path, plan
