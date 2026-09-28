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
    stock = rec.get("mode") == "stock"
    how = (f"In game: select the <b>{_esc(rec.get('name'))}</b> {side} playbook{_esc(team)} — nothing to build."
           if stock else "Build the custom book with the formations below.")
    return (f"<div class='rline'><b>{side.title()} book: {_esc(rec.get('name'))}</b> — {how}"
            f"<div class='muted'>{_esc(bp.get('reason') or '')}</div></div>")


def render_prep_html(plan: dict[str, Any]) -> str:
    """Minimal prep page (CFB parity): O formations + audibles, D formations, Active-8 macros
    with your exact settings. Everything else is on the details page."""
    oid = _esc(plan.get("opponent_id"))
    offense_only = bool(plan.get("offense_only"))
    scout = plan.get("meta_scout") or {}
    det = plan.get("details_path") or str(details_path(plan.get("opponent_id") or ""))
    o = _min_book(plan, "offense")
    parts = [research_fail_banner(scout), _book_pick_line(plan, "offense"), render_formations_min(o),
             render_audibles_min(o)]
    if not offense_only:
        d = _min_book(plan, "defense")
        parts += [_book_pick_line(plan, "defense"), render_formations_min(d)]
    missing = plan.get("missing_settings") or []
    miss_html = ""
    if missing:
        miss_html = ("<div class='banner fail'><strong>Missing macro settings — nothing is invented.</strong> "
                     "Enter yours with <code>macro-settings --game madden27 NAME --set \"Section: Setting = value\"</code>:"
                     "<ul>" + "".join(f"<li>{_esc(m)}</li>" for m in missing) + "</ul></div>")
    head = "Macros — Active 8 (offense only · CPU)" if offense_only else "Macros — Active 8 (O + D)"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prep — {_esc(plan.get("display_name"))} · Madden 27 Franchise</title>
<style>{_CSS}{MIN_CSS}</style>
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
    <section id="macros">
      <h2>{head}</h2>
      {miss_html}
      {_render_macro_accordion(
          plan.get("macro_cards") or [],
          plan.get("slot_budget") or {},
          offense_only=offense_only,
          replacing_lines=list(plan.get("replacing_lines") or []),
          loadout=plan.get("loadout") or {},
      )}
    </section>
    <footer>Live: <code>play --game madden27 -o {oid}</code> — calls come only from these formations
      (PLAY: X + MACRO: Y). Details: {_esc(str(det))}</footer>
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
        "Active loadout — offense only (CPU)"
        if offense_only
        else "Active loadout (≤8 O+D) — click to expand · Copy recipe"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Prep details — {_esc(plan.get("display_name"))} · Madden 27 Franchise</title>
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
    {research_fail_banner(plan.get("meta_scout") or {})}
    {_render_research_madden(plan)}
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
          details=True,
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
      Every prep researches live (web + YouTube transcripts) and may recommend a different stock book
      (start: stock Buccaneers O / 49ers D for the Lions; switches need live research, a clear margin and a cooldown).
      Macro settings shown as exact are ONLY the ones you entered (<code>macro-settings --game madden27</code>);
      research guesses are listed here for reference and never used as settings.
      Active loadout hard-capped at <b>8 O+D</b> for user games; CPU = offense-only.
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
) -> tuple[Path, dict[str, Any]]:
    from cfb_coach.madden.prep import build_prep_plan, mark_applied

    plan = build_prep_plan(
        opponent_id, db=db, persist=persist, profile=profile,
        offline=offline, refresh_meta=refresh_meta, o_book=o_book, d_book=d_book,
        apply_books=mark,
    )
    if mark and db is not None:
        mark_applied(db, opponent_id, plan["proposed_deltas"])
        plan["applied_count"] = len(plan["proposed_deltas"])
        plan["shown_deltas"] = []
        plan["swap_banners"] = []
    path = write_prep_html(opponent_id, plan)
    open_prep_html(path, open_browser=open_browser)
    return path, plan
