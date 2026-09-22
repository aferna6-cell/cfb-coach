"""Generate polished dark HTML prep sheet (deltas only) and open in browser."""

from __future__ import annotations

import html
import os
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from cfb_coach.install_sheet import build_prep_plan


ET = ZoneInfo("America/New_York")


def default_prep_dir() -> Path:
    """Same data dir as the coach DB (~/.cfb-coach or fallback)."""
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


def _badge_class(action: str) -> str:
    a = (action or "").upper()
    if a == "ADD":
        return "badge add"
    if a == "REMOVE":
        return "badge remove"
    if a in ("BENCH",):
        return "badge bench"
    if a in ("UNBENCH",):
        return "badge unbench"
    return "badge edit"


def _render_delta_cards(deltas: list[dict[str, str]], empty_msg: str) -> str:
    if not deltas:
        return f'<div class="empty">{_esc(empty_msg)}</div>'
    parts: list[str] = []
    for d in deltas:
        action = d.get("action", "EDIT")
        field = d.get("field") or ""
        before = d.get("before") or ""
        after = d.get("after") or ""
        why = d.get("why") or ""
        change_row = ""
        if before or after:
            change_row = (
                f'<div class="change">'
                f'<span class="before">{_esc(before)}</span>'
                f'<span class="arrow">→</span>'
                f'<span class="after">{_esc(after)}</span>'
                f"</div>"
            )
        field_bit = f'<span class="field">{_esc(field)}</span>' if field else ""
        why_bit = f'<div class="why">{_esc(why)}</div>' if why else ""
        parts.append(
            f"""
            <article class="card">
              <div class="card-head">
                <span class="{_badge_class(action)}">{_esc(action)}</span>
                <strong class="target">{_esc(d.get("target", ""))}</strong>
                {field_bit}
              </div>
              <p class="detail">{_esc(d.get("detail", ""))}</p>
              {change_row}
              {why_bit}
            </article>
            """
        )
    return "\n".join(parts)


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
    o_mac_bit = (
        ", ".join(_esc(m) for m in o_macros) if o_macros else "(none)"
    )

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


def render_prep_html(plan: dict[str, Any]) -> str:
    shown = plan.get("shown_deltas") or []
    pb = [d for d in shown if d.get("kind") == "playbook"]
    mac = [d for d in shown if d.get("kind") == "macro"]
    tips = plan.get("tips") or []

    # Local ET timestamp for display
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
        else f'<div class="banner info">{len(shown)} adjustment{"s" if len(shown) != 1 else ""} for this opponent</div>'
    )

    tip_lis = "".join(f"<li>{_esc(t)}</li>" for t in tips)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prep — {_esc(plan.get("display_name"))} · CFB27</title>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #1a2332;
    --panel2: #243044;
    --text: #e7ecf3;
    --muted: #8b9bb4;
    --accent: #5b9fd4;
    --add: #3d9a6a;
    --edit: #d4a017;
    --remove: #c45c5c;
    --bench: #8b7bb8;
    --border: #2e3a4f;
    --ok: #2a6b4a;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.45;
  }}
  .wrap {{ max-width: 920px; margin: 0 auto; padding: 28px 20px 64px; }}
  header {{
    background: linear-gradient(135deg, #1a2332 0%, #152030 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 20px;
  }}
  header h1 {{ margin: 0 0 6px; font-size: 1.55rem; font-weight: 650; letter-spacing: -0.02em; }}
  header .meta {{ color: var(--muted); font-size: 0.92rem; }}
  header .meta span {{ margin-right: 14px; }}
  .banner {{
    border-radius: 10px; padding: 12px 16px; margin-bottom: 18px;
    font-weight: 560; border: 1px solid var(--border);
  }}
  .banner.ok {{ background: rgba(42,107,74,0.25); border-color: var(--ok); color: #9fdfb8; }}
  .banner.info {{ background: rgba(91,159,212,0.12); border-color: var(--accent); color: #b8d4ec; }}
  section {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px 20px;
    margin-bottom: 16px;
  }}
  section h2 {{
    margin: 0 0 14px; font-size: 1.05rem; font-weight: 600;
    color: var(--accent); letter-spacing: 0.02em; text-transform: uppercase;
  }}
  .empty {{ color: var(--muted); font-style: italic; padding: 8px 0; }}
  .card {{
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
  }}
  .card-head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .badge {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    padding: 3px 8px; border-radius: 6px; color: #0f1419;
  }}
  .badge.add {{ background: var(--add); }}
  .badge.edit {{ background: var(--edit); }}
  .badge.remove {{ background: var(--remove); color: #fff; }}
  .badge.bench {{ background: var(--bench); color: #fff; }}
  .badge.unbench {{ background: var(--add); }}
  .target {{ font-size: 1rem; }}
  .field {{
    color: var(--muted); font-size: 0.82rem;
    background: rgba(0,0,0,0.25); padding: 2px 8px; border-radius: 5px;
  }}
  .detail {{ margin: 8px 0 0; color: var(--text); }}
  .change {{
    margin-top: 8px; font-size: 0.88rem;
    display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline;
  }}
  .before {{ color: var(--muted); text-decoration: line-through; opacity: 0.85; }}
  .arrow {{ color: var(--accent); }}
  .after {{ color: #c5e0a5; font-weight: 560; }}
  .why {{ margin-top: 6px; font-size: 0.82rem; color: var(--muted); }}
  .why::before {{ content: "why: "; color: var(--accent); }}
  ul.tips {{ margin: 0; padding-left: 1.2rem; }}
  ul.tips li {{ margin-bottom: 6px; }}
  details.inventory {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 16px;
  }}
  details.inventory summary {{
    cursor: pointer; font-weight: 600; color: var(--muted);
    list-style: none;
  }}
  details.inventory summary::-webkit-details-marker {{ display: none; }}
  details.inventory summary::before {{ content: "▸ "; color: var(--accent); }}
  details.inventory[open] summary::before {{ content: "▾ "; }}
  .muted {{ font-weight: 400; color: var(--muted); font-size: 0.88rem; }}
  .inv-meta {{
    display: grid; gap: 4px; margin: 14px 0;
    font-size: 0.9rem; color: var(--muted);
  }}
  .inv-meta b {{ color: var(--text); }}
  .inv-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
  }}
  .inv-form {{
    background: var(--panel2); border-radius: 8px; padding: 10px 12px;
    border: 1px solid var(--border);
  }}
  .inv-form h4 {{ margin: 0 0 6px; font-size: 0.92rem; }}
  .inv-form .role {{ color: var(--muted); font-weight: 400; font-size: 0.78rem; }}
  .inv-form .aud {{ font-size: 0.8rem; color: #c5e0a5; margin-bottom: 6px; }}
  .inv-form ul {{ margin: 0; padding-left: 1.1rem; font-size: 0.82rem; color: var(--muted); }}
  details.inventory h3 {{ font-size: 0.95rem; color: var(--accent); margin: 16px 0 8px; }}
  footer {{
    margin-top: 24px; color: var(--muted); font-size: 0.8rem; text-align: center;
  }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>vs {_esc(plan.get("display_name"))} <span style="color:var(--muted);font-weight:500">({_esc(plan.get("team"))})</span></h1>
      <div class="meta">
        <span>{_esc(plan.get("game", "CFB 27"))}</span>
        <span>META {_esc(plan.get("version"))}</span>
        <span>{_esc(ts_label)}</span>
        <span>overlay: {_esc(plan.get("overlay_depth"))}</span>
      </div>
    </header>

    {status}

    <section>
      <h2>Playbook adjustments</h2>
      {_render_delta_cards(pb, "No playbook changes — run baseline as-is")}
    </section>

    <section>
      <h2>Macro adjustments</h2>
      {_render_delta_cards(mac, "No macro changes — keep current set")}
    </section>

    {_render_inventory(plan.get("inventory") or {})}

    <section>
      <h2>Call emphasis</h2>
      <ul class="tips">{tip_lis}</ul>
    </section>

    <footer>
      Inventory is already stocked from seed — only opponent-specific deltas above.
      Live caller unchanged. Mark applied with <code>prep --opponent {_esc(plan.get("opponent_id"))} --mark-applied</code>.
    </footer>
  </div>
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


def open_prep_html(path: Path, *, open_browser: bool = True) -> Path:
    if open_browser:
        uri = path.resolve().as_uri()
        webbrowser.open(uri)
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
    """Build plan, write HTML, optionally open browser / mark applied."""
    from cfb_coach.install_sheet import mark_prep_applied

    plan = build_prep_plan(opponent_id, opp, db=db, persist=persist)
    if mark_applied and db is not None:
        mark_prep_applied(db, opponent_id, plan["proposed_deltas"])
        plan["applied_count"] = len(plan["proposed_deltas"])
        # Re-filter shown for the HTML (all applied → empty)
        plan["shown_deltas"] = []
    path = write_prep_html(opponent_id, plan)
    open_prep_html(path, open_browser=open_browser)
    return path, plan
