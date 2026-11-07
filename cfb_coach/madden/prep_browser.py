"""Madden 27 Franchise prep HTML — same dark UX as CFB prep (renderers reused)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from cfb_coach.prep_book_html import (
    MIN_CSS,
    render_audibles_min,
    render_formations_min,
    research_fail_banner,
    research_status_line,
)
from cfb_coach.prep_browser import (
    _CSS,
    ET,
    _esc,
    _render_delta_cards,
    _render_ingame_card,
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


_GROUP_CSS = """
  h3.macro-group { margin: 14px 0 6px; font-size: 1rem; letter-spacing: 0.02em; }
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
        elif rec.get("source_book"):
            head = (
                f"<div class='banner ok'>Custom plan — trimmed from the in-game <b>{_esc(rec.get('source_book'))}</b> "
                f"{side} book. These formations are the book of record. Nothing to build in the custom editor.</div>"
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
        status = (
            "<div class='banner swap'><strong>PENDING</strong> — build it in-game, then run "
            f"<code>prep --game madden27 -o {oid} --mark-applied</code>. Live calls keep using the last locked book until then.</div>"
            if bp.get("status") == "pending"
            else "<div class='banner ok'><strong>LOCKED</strong> — live calls use only this book.</div>"
        )
        parts.append(f"""
        <div class="scout-card" style="margin-bottom:12px">
          <h3>{side.title()}: {_esc(rec.get('name'))} {badge}
            <span class="muted">rev {_esc(rec.get('rev'))}</span></h3>
          <div class="why">{_esc(bp.get('reason'))}</div>
          {head}
          {status}
          <details class="inventory" open>
            <summary>Show full playbook ({len(rec.get('formations') or {})} formations) — live calls are locked to this list</summary>
            <div class="inv-grid">{forms}</div>
          </details>
          <div class="why">Switch any time: <code>prep --game madden27 -o {oid} {flag} stock:&lt;book&gt;</code>
            or <code>{flag} custom</code> · print it: <code>playbook --game madden27</code></div>
        </div>""")
    return "<section><h2>Playbook of record (locked by this prep)</h2>" + "".join(parts) + "</section>"


def details_path(opponent_id: str) -> Path:
    from cfb_coach.games import GAMES, MADDEN27

    return default_prep_dir() / f"{GAMES[MADDEN27].prep_prefix}details_{opponent_id.lower()}.html"


def _min_book(plan: dict[str, Any], side: str) -> dict[str, Any]:
    """CFB `render_formations_min` / `render_audibles_min` input from a Madden side plan."""
    bp = (plan.get("playbook") or {}).get(side) or {}
    rec = bp.get("record") or {}
    label = "stock" if rec.get("mode") == "stock" else "custom"
    return {
        "name": f"{side.title()}: {rec.get('name')} ({label} book)",
        "formation_list": rec.get("formation_list") or [
            {"formation": f, "status": "applied", "source_book": rec.get("name"), "n_plays": len(ps), "note": ""}
            for f, ps in (rec.get("formations") or {}).items()],
        "pending": bp.get("status") == "pending",
        "apply_cmd": f"PYTHONPATH=. python3 -m cfb_coach prep --game madden27 -o {plan.get('opponent_id')} --mark-applied",
        "audibles": rec.get("audibles") or {},
        "game_label": "Madden 27",
    }


def _book_pick_line(plan: dict[str, Any], side: str) -> str:
    bp = (plan.get("playbook") or {}).get(side) or {}
    rec = bp.get("record") or {}
    team = ""
    rows = (bp.get("recommendation") or {}).get("rows") or []
    for r in rows:
        if r.get("book") == rec.get("name") and r.get("team"):
            team = f" ({r['team']} book)"
    if rec.get("source_book"):
        how = (f"Custom plan trimmed from the <b>{_esc(rec.get('source_book'))}</b> {side} book{_esc(team)}. "
               "Custom Adjustments pair only with these formations.")
    elif rec.get("mode") == "stock":
        how = f"In game: select the <b>{_esc(rec.get('name'))}</b> {side} playbook{_esc(team)} — nothing to build."
    else:
        how = "Build the custom book with the formations below."
    return (f"<div class='rline'><b>{side.title()} book: {_esc(rec.get('name'))}</b> — {how}"
            f"<div class='muted'>{_esc(bp.get('reason') or '')}</div></div>")


def _render_vod(plan: dict[str, Any]) -> str:
    from cfb_coach.vod_model.report import render_vod_html

    return render_vod_html(plan)


def _render_gameplan(plan: dict[str, Any]) -> str:
    """Call sheet (plays). Not Custom Adjustments — kept off the prep headline."""
    gp = plan.get("gameplan") or {}
    warns = "".join(
        f"<div class='banner fail'><strong>Playbook warning.</strong> {_esc(w)}</div>"
        for w in (plan.get("playbook_warnings") or [])
    )

    def side_html(title: str, macros: list[dict[str, Any]]) -> str:
        if not macros:
            empty = "N/A — offense only (CPU)" if title == "Defense" and gp.get("offense_only") else "none"
            return f"<h3>{_esc(title)}</h3><div class='empty'>{empty}</div>"
        rows = []
        for i, m in enumerate(macros, 1):
            rows.append(
                "<article class='gp'>"
                f"<h4>{i}. {_esc(m.get('label'))} <span class='muted'>{_esc(m.get('id'))}</span></h4>"
                f"<div class='call'><b>CALL:</b> {_esc(m.get('play'))} ({_esc(m.get('formation'))})</div>"
                f"<div><b>When:</b> {_esc(m.get('when'))} {_esc(m.get('score_situation') or '')}</div>"
                f"<div><b>Pre-snap:</b> {_esc(m.get('adjustments'))}</div>"
                f"<div><b>Read:</b> {_esc(m.get('read'))}</div>"
                f"<div><b>Counter:</b> {_esc(m.get('counter'))}</div>"
                + (f"<div class='muted'>{_esc(m.get('why'))}</div>" if m.get("why") else "")
                + "</article>"
            )
        return f"<h3>{_esc(title)} ({len(macros)})</h3>" + "".join(rows)

    note = _esc(gp.get("opponent_note") or "")
    return (
        f"<section id='gameplan'><h2>Call sheet — plays, not Custom Adjustments</h2>"
        f"<div class='why'>These are formation + play calls from the trimmed book. "
        f"They are not macros. Macros are the Custom Adjustments at the top of the prep page. {note}</div>"
        f"{warns}{side_html('Offense calls', list(gp.get('offense') or []))}"
        f"{side_html('Defense calls', list(gp.get('defense') or []))}</section>"
    )


def _render_offense_adjustments(plan: dict[str, Any]) -> str:
    """Cited one-off adjustments used only when no offense Custom Adjustment matches the look."""
    rows = "".join(
        f"<tr><td>{_esc(a['vs'])}</td><td><b>{_esc(a['label'])}</b><div class='muted'>{_esc(a['why'])}</div></td>"
        f"<td><code>{_esc(a['buttons'])}</code></td></tr>"
        for a in plan.get("offense_adjustments") or [])
    return ("<section id='offense-adjustments'><h2>Pre-snap adjustments (not macros)</h2>"
            "<div class='why'>Used live only when no Custom Adjustment matches the look — "
            "the call line says <code>ADJ: … — press …</code>.</div>"
            f"<table class='ca-set'><tr><th>vs look</th><th>Adjustment</th><th>Buttons (Xbox)</th></tr>{rows}</table></section>")


def _render_controls(plan: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{_esc(c['side'])}</td><td>{_esc(c['action'])}</td><td><code>{_esc(c['buttons'])}</code></td>"
        f"<td>{_esc(c['confidence'])}</td><td class='muted'>{_esc(c['sources'])}</td></tr>"
        for c in plan.get("controls") or [])
    return ("<section><h2>Xbox pre-snap controls (research DB, cited)</h2>"
            "<table class='ca-set'><tr><th>Side</th><th>Action</th><th>Buttons</th><th>Confidence</th><th>Sources</th></tr>"
            f"{rows}</table></section>")


def _render_macro_groups(plan: dict[str, Any], *, details: bool = False) -> str:
    """8 offense + 8 defense Custom Adjustments. Each card is the editor settings and LB."""
    offense_only = bool(plan.get("offense_only"))
    blocks = []
    for side, title in (("offense", "Offense"), ("defense", "Defense")):
        cards = [c for c in plan.get("macro_cards") or [] if c.get("side") == side]
        if not cards and offense_only and side == "defense":
            items = "<div class='empty'>N/A — defense off (CPU)</div>"
        else:
            items = "".join(
                _render_ingame_card(c, f"copy-{side[0]}-" + "".join(ch if ch.isalnum() else "-" for ch in str(c.get("id"))),
                                    details=details)
                for c in cards) or "<div class=empty>none</div>"
        blocks.append(f"<h3 class='macro-group' id='macros-{side}'>{title} ({len(cards)})</h3>"
                      f"<div class='macro-list'>{items}</div>")
    n_o = len([c for c in plan.get("macro_cards") or [] if c.get("side") == "offense"])
    n_d = len([c for c in plan.get("macro_cards") or [] if c.get("side") == "defense"])
    checklist = (
        f"<div class='copy-wrap'><div class='copy-head'><h4>Copy checklist — {n_o} offense + {n_d} defense</h4>"
        "<button type='button' class='copy-btn' data-target='copy-all-macros'>Copy</button></div>"
        f"<pre id='copy-all-macros' class='copy-block'>{_esc(plan.get('copy_checklist') or '')}</pre></div>")
    meter = _esc((plan.get("loadout") or {}).get("meter") or "")
    return (f"<div class='meter ok'><span class='meter-label'>Custom Adjustments</span>"
            f"<span class='meter-value'>{meter}</span>"
            "<span class='meter-note'>LB in game · unnamed fields = Default · cited value or Aidan's notes only</span></div>"
            + "".join(blocks) + checklist)


def render_prep_html(plan: dict[str, Any]) -> str:
    """Prep page headline is the 8+8 Custom Adjustments. Play calls stay on the details page."""
    oid = _esc(plan.get("opponent_id"))
    offense_only = bool(plan.get("offense_only"))
    scout = plan.get("meta_scout") or {}
    det = plan.get("details_path") or str(details_path(plan.get("opponent_id") or ""))
    o = _min_book(plan, "offense")
    from cfb_coach.scouting import opponent_study_html

    rstat = plan.get("research_db") or {}
    miss_html = (f"<div class='banner {'fail' if rstat.get('stale') else 'info'}'>{_esc(rstat.get('line') or '')}</div>"
                 if rstat else "")
    sel = plan.get("macro_selection") or {}
    if offense_only:
        head = (f"Custom Adjustments — {len(sel.get('offense') or [])} offense "
                "(CPU — defense off)")
    else:
        head = (f"Custom Adjustments — {len(sel.get('offense') or [])} offense + "
                f"{len(sel.get('defense') or [])} defense")
    macro_section = (f"<section id='macros'><h2>{head}</h2>{miss_html}{_render_macro_groups(plan)}</section>")
    parts = [research_fail_banner(scout), macro_section,
             opponent_study_html(plan.get("scouting_lines") or [], plan.get("opponent_research") or []),
             _book_pick_line(plan, "offense"), render_formations_min(o), render_audibles_min(o),
             _render_offense_adjustments(plan)]
    if not offense_only:
        d = _min_book(plan, "defense")
        parts += [_book_pick_line(plan, "defense"), render_formations_min(d)]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prep — {_esc(plan.get("display_name"))} · Madden 27 Franchise</title>
<style>{_CSS}{MIN_CSS}{_GROUP_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>vs {_esc(plan.get("display_name"))}
        <span style="color:var(--muted);font-weight:500">· Madden 27 Franchise · {_esc((plan.get("profile_config") or {}).get("team_label"))}</span>
      </h1>
      <div class="rline">{_esc(research_status_line(scout))} · <a href="file:///{_esc(str(det).lstrip('/'))}">details</a></div>
    </header>
    {"".join(parts)}
    <footer>Live: <code>play --game madden27 -o {oid}</code> — calls come only from these formations.
      A Custom Adjustment is <code>MACRO: name — press LB → name</code>. A one-off hot route or
      protection is <code>ADJ:</code>, and only when no macro matches. Details: {_esc(str(det))}</footer>
  </div>
  <script>{_COPY_JS}</script>
</body>
</html>
"""


def _render_research_madden(plan: dict[str, Any]) -> str:
    """Research used for the book pick: named books per side + book scores + top named formations."""
    scout = plan.get("meta_scout") or {}
    named = scout.get("named_signals") or {}
    yt = scout.get("youtube") or {}
    rows = []
    for side in ("offense",) if plan.get("offense_only") else ("offense", "defense"):
        rec = ((plan.get("playbook") or {}).get(side) or {}).get("recommendation") or {}
        score_rows = "".join(
            f"<tr><td>{_esc(r['book'])}</td><td>{r['score']:.2f}</td><td class='muted'>"
            + _esc(", ".join(f"{k} {v:+.2f}" for k, v in (r.get("parts") or {}).items() if v))
            + f"</td><td class='muted'>{_esc(r.get('research_docs', 0))} src</td></tr>"
            for r in (rec.get("rows") or [])[:8])
        forms = list(((named.get(side) or {}).get("formations") or {}).items())[:8]
        form_txt = ", ".join(f"{f} ({v.get('docs', 0)})" for f, v in forms) or "none named"
        books = list(((named.get("books") or {}).get(side) or {}).items())[:6]
        book_txt = ", ".join(f"{b} ({v.get('docs', 0)} src)" for b, v in books) or "none named"
        rows.append(f"<h3>{side.title()}</h3><div class='why'>Books named: {_esc(book_txt)}</div>"
                    f"<div class='why'>Formations named: {_esc(form_txt)}</div>"
                    + (f"<table class='ca-set'><tr><th>Book</th><th>Score</th><th>Why</th><th></th></tr>{score_rows}</table>" if score_rows else "")
                    + f"<div class='why'>{_esc(rec.get('reason') or '')}</div>")
    vids = "".join(
        f"<li><a href='{_esc(v.get('url'))}'>{_esc(v.get('title'))}</a> <span class='muted'>{_esc(v.get('channel') or '')} · "
        f"{_esc((v.get('published') or '')[:10])} · {_esc(v.get('transcript_status') or '')}</span></li>"
        for v in (yt.get("videos") or [])[:10])
    yt_line = (f"YouTube: {yt.get('found', 0)} Madden 27 videos, {yt.get('transcripts', 0)} transcripts"
               if yt else "YouTube research not run (offline / fallback).")
    return (f"<section><h2>Research → book + formation pick</h2><div class='rline'>{_esc(research_status_line(scout))}</div>"
            + "".join(rows) + f"<div class='why'>{_esc(yt_line)}</div><ul class='tips'>{vids}</ul></section>")


def render_prep_details_html(plan: dict[str, Any]) -> str:
    """Everything else (research, scores, book of record with every play, adjustments,
    macro research notes) — written next to the minimal page, never auto-opened."""
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
        else "Primary team: <b>not set</b> — books picked from research + the verified catalog. Set: "
        "<code>config --game madden27 --primary-team \"Detroit Lions\"</code>"
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
        "Custom Adjustments — 8 offense (CPU — defense off) · click to expand · Copy recipe"
        if offense_only
        else "Custom Adjustments — 8 offense + 8 defense · click to expand · Copy recipe"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prep details — {_esc(plan.get("display_name"))} · Madden 27 Franchise</title>
<style>{_CSS}{_GROUP_CSS}</style>
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
    {research_fail_banner(plan.get("meta_scout") or {})}
    {_render_research_madden(plan)}
    {_render_controls(plan)}
    {_render_meta_scout(plan.get("meta_scout") or {})}

    <section>
      <h2>Baseline patch radar ({_esc(plan.get("version"))})</h2>
      <ul class="tips">{patch_lis}</ul>
    </section>

    {_render_playbook(plan)}
    {_render_vod(plan)}
    {_render_gameplan(plan)}

    {_render_swap_banners(plan.get("swap_banners") or [])}

    <section>
      <h2>Playbook adjustments</h2>
      {_render_delta_cards(pb, "No playbook changes — keep the locked book as-is")}
    </section>

    <section>
      <h2>Macro adjustments</h2>
      {_render_delta_cards(mac, "No macro changes")}
    </section>

    <section>
      <h2>{loadout_h}</h2>
      {_render_macro_groups(plan, details=True)}
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
      Every prep researches live (web + YouTube transcripts) and may recommend a different stock book
      (start: stock Buccaneers O / 49ers D for the Lions; switches need live research, a clear margin and a cooldown).
      Defense macros and their settings come from the research DB (daily research routine, pulled every
      prep): every editor field shows the value a cited source names, else Default. Offense uses no macros —
      live calls add one researched adjustment (hot route / audible / protection) with its Xbox buttons.
      <b>10 defense macros</b> per opponent for user games; CPU = offense-only (adjustments).
      Personas are shared with CFB. Primary team: Detroit Lions (config --game madden27 --primary-team).
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
    det = out.with_name(details_path(opponent_id).name)
    plan.setdefault("details_path", str(det))
    out.write_text(render_prep_html(plan), encoding="utf-8")
    try:  # details page next to it — written every prep, never auto-opened
        det.write_text(render_prep_details_html(plan), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
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
    opp_team: str | None = None,
    n_gameplan: int = 8,
) -> tuple[Path, dict[str, Any]]:
    from cfb_coach.madden.prep import build_prep_plan, mark_applied

    plan = build_prep_plan(
        opponent_id, db=db, persist=persist, profile=profile,
        offline=offline, refresh_meta=refresh_meta, o_book=o_book, d_book=d_book,
        apply_books=mark, opp_team=opp_team, n_gameplan=n_gameplan,
    )
    if mark and db is not None:
        mark_applied(db, opponent_id, plan["proposed_deltas"])
        plan["applied_count"] = len(plan["proposed_deltas"])
        plan["shown_deltas"] = []
        plan["swap_banners"] = []
    path = write_prep_html(opponent_id, plan)
    open_prep_html(path, open_browser=open_browser)
    return path, plan
