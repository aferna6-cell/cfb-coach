"""Madden 27 Franchise prep HTML — same dark UX as CFB prep (renderers reused)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from cfb_coach.prep_browser import (
    _CSS,
    ET,
    _esc,
    _render_delta_cards,
    _render_macro_accordion,
    _render_meta_scout,
    _render_swap_banners,
    default_prep_dir,
    open_prep_html,
)

_COPY_JS = """
    document.querySelectorAll(".copy-btn").forEach(function(btn) {
      btn.addEventListener("click", function() {
        var el = document.getElementById(btn.getAttribute("data-target"));
        if (!el) return;
        var text = el.innerText || el.textContent || "";
        function done() {
          var prev = btn.textContent; btn.textContent = "Copied";
          setTimeout(function() { btn.textContent = prev; }, 1400);
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done).catch(done);
        } else { done(); }
      });
    });
"""


def prep_html_path(opponent_id: str) -> Path:
    from cfb_coach.games import GAMES, MADDEN27

    return default_prep_dir() / f"{GAMES[MADDEN27].prep_prefix}{opponent_id.lower()}.html"


def _ts_label(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).astimezone(ET).strftime("%a %b %d, %Y · %I:%M %p ET")
    except (TypeError, ValueError):
        return ts or ""


def _render_playbook(plan: dict[str, Any]) -> str:
    """Playbook of record per side: mode, build checklist / stock pick, full-book toggle."""
    oid = _esc(plan.get("opponent_id"))
    sides = ("offense",) if plan.get("offense_only") else ("offense", "defense")
    parts: list[str] = []
    for side in sides:
        bp = (plan.get("playbook") or {}).get(side) or {}
        rec = bp.get("record") or {}
        mode = rec.get("mode") or "stock"
        badge = (
            '<span class="vbadge proven">STOCK in-game book</span>'
            if mode == "stock"
            else '<span class="vbadge meta-grounded">CUSTOM book</span>'
        )
        change = bp.get("change")
        if bp.get("checklist"):
            items = "".join(
                f"<li><label><input type='checkbox'/> <b>{_esc(i['formation'])}</b>"
                f" <span class='muted'>({_esc(i['books'])})</span> — {_esc(', '.join(i['plays']))}</label></li>"
                for i in bp["checklist"]
            )
            head = (
                f"<div class='banner swap'><strong>BUILD CUSTOM {side.upper()} BOOK</strong> — "
                f"install exactly these {len(bp['checklist'])} formations "
                "(Create &amp; Share → custom playbook)<ol>" + items + "</ol></div>"
            )
        elif change == "stock_select":
            head = (
                f"<div class='banner ok'>Use the in-game stock book <b>{_esc(rec.get('name'))}</b>"
                " — nothing to build.</div>"
            )
        elif change == "diff":
            head = "<div class='banner info'>Custom book diff — formation ADD / REMOVE cards under Playbook adjustments.</div>"
        else:
            head = "<div class='banner ok'>Book unchanged since last prep.</div>"
        forms = "".join(
            f"<div class='inv-form'><h4>{_esc(f)}</h4><ul>"
            + "".join(f"<li>{_esc(p)}</li>" for p in plays)
            + "</ul></div>"
            for f, plays in (rec.get("formations") or {}).items()
        )
        flag = "--o-book" if side == "offense" else "--d-book"
        parts.append(f"""
        <div class="scout-card" style="margin-bottom:12px">
          <h3>{side.title()}: {_esc(rec.get('name'))} {badge}
            <span class="muted">rev {_esc(rec.get('rev'))}</span></h3>
          <div class="why">{_esc(bp.get('reason'))}</div>
          {head}
          <details class="inventory" open>
            <summary>Show full playbook ({len(rec.get('formations') or {})} formations) — live calls are locked to this list</summary>
            <div class="inv-grid">{forms}</div>
          </details>
          <div class="why">Switch any time: <code>prep --game madden27 -o {oid} {flag} stock:&lt;book&gt;</code>
            or <code>{flag} custom</code> · print it: <code>playbook --game madden27</code></div>
        </div>""")
    return "<section><h2>Playbook of record (locked by this prep)</h2>" + "".join(parts) + "</section>"


def render_prep_html(plan: dict[str, Any]) -> str:
    shown = plan.get("shown_deltas") or []
    pb = [d for d in shown if d.get("kind") == "playbook"]
    mac = [d for d in shown if d.get("kind") == "macro"]
    pcfg = plan.get("profile_config") or {}
    oid = _esc(plan.get("opponent_id"))
    offense_only = bool(plan.get("offense_only"))
    status = (
        '<div class="banner ok">No playbook changes — keep the locked book as-is</div>'
        if not shown
        else f'<div class="banner info">{len(shown)} adjustment{"s" if len(shown) != 1 else ""} for this opponent</div>'
    )
    exp = bool(pcfg.get("experimental_badge"))
    team_line = (
        f"Primary team: <b>{_esc(plan.get('primary_team'))}</b>"
        if plan.get("primary_team")
        else "Primary team: <b>TBD</b> — books picked from the verified meta catalog "
        "(Buccaneers / Shotgun Classic / custom O · 49ers Saleh D). Set later: "
        "<code>config --game madden27 --primary-team &lt;NFL team&gt;</code>"
    )
    franchise_banner = (
        f'<div class="banner dyn {"exp" if exp else "ser"}">'
        f"<strong>Franchise mode:</strong> {_esc(pcfg.get('label'))} "
        f'<span class="muted">({_esc(pcfg.get("mode"))} · team {_esc(pcfg.get("team_label"))})</span>'
        + (' <span class="vbadge meta-grounded">experimental</span>' if exp else "")
        + f'<div class="why">{team_line}</div>'
        f'<div class="why">Persona <b>{_esc(plan.get("display_name"))}</b> shared with CFB — '
        f'archetype {_esc(plan.get("archetype"))}, persona conf {_esc(plan.get("persona_confidence"))}, '
        f'Madden film conf {_esc(plan.get("film_confidence"))}.</div>'
        f'<div class="why">{_esc(plan.get("doctrine"))}</div>'
        "</div>"
    )
    patch_lis = "".join(f"<li>{_esc(p)}</li>" for p in plan.get("patch_notes") or [])
    tip_lis = "".join(f"<li>{_esc(t)}</li>" for t in plan.get("tips") or [])
    loadout_h = (
        "Active loadout — offense only (CPU)"
        if offense_only
        else "Active loadout (≤8 O+D) — click to expand · Copy recipe"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prep — {_esc(plan.get("display_name"))} · Madden 27 Franchise</title>
<style>{_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>vs {_esc(plan.get("display_name"))}
        <span style="color:var(--muted);font-weight:500">(persona · {_esc(plan.get("archetype"))})</span>
      </h1>
      <div class="meta">
        <span>{_esc(plan.get("game"))}</span>
        <span>META {_esc(plan.get("version"))}</span>
        <span>macros {_esc(plan.get("macro_catalog_version"))}</span>
        <span>{_esc(_ts_label(plan.get("ts", "")))}</span>
        <span>profile: {_esc(plan.get("profile"))}</span>
        <span>team: {_esc(pcfg.get("team_label"))}</span>
      </div>
    </header>

    {status}
    {franchise_banner}
    {_render_meta_scout(plan.get("meta_scout") or {})}

    <section>
      <h2>Baseline patch radar ({_esc(plan.get("version"))})</h2>
      <ul class="tips">{patch_lis}</ul>
    </section>

    {_render_playbook(plan)}

    {_render_swap_banners(plan.get("swap_banners") or [])}

    <section>
      <h2>Playbook adjustments</h2>
      {_render_delta_cards(pb, "No playbook changes — keep the locked book as-is")}
    </section>

    <section>
      <h2>Macro adjustments</h2>
      {_render_delta_cards(mac, "No macro changes — keep current Active-8")}
    </section>

    <section>
      <h2>{loadout_h}</h2>
      {_render_macro_accordion(
          plan.get("macro_cards") or [],
          plan.get("slot_budget") or {},
          offense_only=offense_only,
          replacing_lines=list(plan.get("replacing_lines") or []),
          loadout=plan.get("loadout") or {},
      )}
    </section>

    <section>
      <h2>Call emphasis</h2>
      <ul class="tips">{tip_lis}</ul>
    </section>

    <footer>
      Madden 27 <b>Franchise</b> (not MUT / MCS). Every prep picks a playbook of record per side:
      a <b>stock</b> in-game book by exact name, or a <b>custom</b> book (full formation checklist on first
      build or switch; later preps only ADD / REMOVE whole formations). Live <code>play</code> calls are hard-locked
      to that book. Macros are Custom Adjustments you create (Create &amp; Share → Custom Adjustments).
      Meta is community-derived (~Sep 2026, cross-checked 2026-09-23) and drifts after title updates;
      macro option labels tagged <b>approx</b> need an in-game confirm.
      Active loadout hard-capped at <b>8 O+D</b> for user games; CPU = offense-only.
      Personas are shared with CFB. Primary team stays TBD until you set it.
      Mark applied with <code>prep --game madden27 --opponent {oid} --mark-applied</code>.
    </footer>
  </div>
  <script>{_COPY_JS}</script>
</body>
</html>
"""


def write_prep_html(opponent_id: str, plan: dict[str, Any], *, path: Path | None = None) -> Path:
    out = path or prep_html_path(opponent_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_prep_html(plan), encoding="utf-8")
    return out


def generate_and_open(
    opponent_id: str,
    *,
    db: Any = None,
    persist: bool = True,
    open_browser: bool = True,
    mark: bool = False,
    profile: str | None = None,
    offline: bool = False,
    refresh_meta: bool = False,
    o_book: str | None = None,
    d_book: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    from cfb_coach.madden.prep import build_prep_plan, mark_applied

    plan = build_prep_plan(
        opponent_id, db=db, persist=persist, profile=profile,
        offline=offline, refresh_meta=refresh_meta, o_book=o_book, d_book=d_book,
    )
    if mark and db is not None:
        mark_applied(db, opponent_id, plan["proposed_deltas"])
        plan["applied_count"] = len(plan["proposed_deltas"])
        plan["shown_deltas"] = []
        plan["swap_banners"] = []
    path = write_prep_html(opponent_id, plan)
    open_prep_html(path, open_browser=open_browser)
    return path, plan
