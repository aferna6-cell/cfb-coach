"""v1.13 prep-page panels: the autonomous custom playbook + live research detail
(research status, YouTube transcripts, named formation/play mentions)."""

from __future__ import annotations

import html
from typing import Any

CMD = "PYTHONPATH=. python3 -m cfb_coach"

BOOK_CSS = """
  .banner.fail { background: rgba(200,60,60,0.25); border: 2px solid #e05555; color: #ffd0d0; font-weight: 700; }
  .banner.pend { background: rgba(240,198,116,0.14); border: 2px solid #f0c674; }
  .book-grid { display: grid; gap: 12px; grid-template-columns: 1fr; }
  @media (min-width: 900px) { .book-grid { grid-template-columns: 1.1fr 0.9fr; } }
  .flag { font-size: 0.7rem; padding: 1px 6px; border-radius: 4px; background: rgba(139,155,180,0.2); color: var(--muted); }
  .flag.working { background: rgba(61,154,106,0.3); color: #9fdfb8; }
  .flag.demoted, .flag.on_notice { background: rgba(224,85,85,0.25); color: #ffb3b3; }
  .flag.trial { background: rgba(94,166,255,0.25); color: #b8d8ff; }
  .flag.fading { background: rgba(240,198,116,0.25); color: #f0c674; }
  table.book td { padding: 3px 6px; border-bottom: 1px solid rgba(255,255,255,0.06); vertical-align: top; font-size: 0.82rem; }
  table.book { width: 100%; border-collapse: collapse; }
  .edit-add { color: #9fdfb8; } .edit-rm { color: #ffb3b3; }
"""


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def _copy(cid: str, text: str, label: str = "Copy") -> str:
    return (
        f'<div class="copy-wrap"><div class="copy-head"><span></span>'
        f'<button type="button" class="copy-btn" data-target="{cid}">{_esc(label)}</button></div>'
        f'<pre class="copy-block" id="{cid}">{_esc(text)}</pre></div>'
    )


def _cite_html(cites: list[dict[str, Any]]) -> str:
    bits = []
    for c in cites or []:
        url = c.get("url") or ""
        name = c.get("source") or url
        d = c.get("date") or ""
        bits.append(f'<a href="{_esc(url)}" target="_blank" rel="noopener">{_esc(name[:60])}</a>' + (f" <span class='muted'>({_esc(d[:10])})</span>" if d else ""))
    return " · ".join(bits)


def render_book(book: dict[str, Any] | None) -> str:
    """DETAILS page: the formation-level custom playbook with every reason, flag,
    per-formation / per-play evidence, history, limits and decision constants."""
    book = book or {}
    if not book:
        return ""
    if book.get("error"):
        return f"<section id='cfb-book'><h2>Custom playbook</h2><div class='empty'>Unavailable: {_esc(book['error'])}</div></section>"
    dyn = book.get("dynasty") or ""
    cur = book.get("current") or {}
    pend = book.get("pending") or {}
    edits = book.get("edits") or []
    apply_cmd = f"{CMD} book apply --dynasty {dyn}" + (f" --rev {pend.get('rev')}" if pend else "")
    if pend:
        if cur:
            status = (f'<div class="banner pend"><b>{len(edits)} pending formation change(s) — rev {pend.get("rev")}.</b> '
                      f'Live calls stay locked to your applied book (rev {cur.get("rev")}) until you make these changes in-game and confirm '
                      f'(<code>book apply</code>, or the button in the live window).</div>')
        else:
            status = (f'<div class="banner pend"><b>First build — {len(edits)} formation(s).</b> Build this custom book in CFB 27, then confirm with '
                      f'<code>book apply</code>. Until then the live caller uses it but marks every call UNCONFIRMED.</div>')
    else:
        status = (f'<div class="banner ok">No playbook changes this prep — rev {_esc(cur.get("rev"))} stands '
                  f'(hysteresis: a formation needs a meaningful signal before it is swapped).</div>')
    seeded = (f"<div class='muted'>Seeded rev {_esc(cur.get('rev'))} this prep: {_esc(book.get('seed_summary'))}.</div>"
              if book.get("seeded_now") else "")
    edit_rows = "".join(
        f"<tr><td class='{'edit-add' if e['op'] != 'remove_formation' else 'edit-rm'}'>{_esc(e['op'].replace('_', ' '))}</td>"
        f"<td><b>{_esc(e['formation'])}</b> <span class='muted'>({_esc(e.get('source_book') or 'any book')}, {len(e.get('plays') or [])} plays)</span></td>"
        f"<td class='muted'>{_esc(e.get('reason') or '')}{('<br/>' + _cite_html(e.get('cites') or [])) if e.get('cites') else ''}</td></tr>"
        for e in edits
    )
    edits_html = ""
    if edits:
        edits_html = (
            "<h3>Formation edits — CFB 27 › Create &amp; Share › Custom Playbooks</h3>"
            + _copy("book-edits", book.get("edit_text") or "", "Copy edit list")
            + f"<table class='book'>{edit_rows}</table>"
            + "<h3 style='margin-top:10px'>After you make the edits</h3>"
            + _copy("book-apply-cmd", apply_cmd, "Copy apply command")
        )
    frows = "".join(
        f"<tr><td><b>{_esc(r['formation'])}</b><br/><span class='muted'>{_esc(r.get('source_book') or 'any')} · {_esc(r['n_plays'])} plays</span></td>"
        f"<td><span class='flag {_esc(r['flag'])}'>{_esc(r['flag'] or '-')}</span></td>"
        f"<td class='muted'>meta {r['meta']:+.2f}<br/>(specific {r['specific']:+.2f})</td><td class='muted'>{_esc(r['stats'])}</td>"
        f"<td class='muted'>{_esc((r.get('why') or '')[:220])}"
        + ("<br/>" + _esc(' · '.join(r.get('reasons') or [])[:260]) if r.get('reasons') else "")
        + (('<br/>' + _cite_html(r.get('cites') or [])) if r.get('cites') else '') + "</td></tr>"
        for r in book.get("formation_table") or []
    )
    rows = "".join(
        f"<tr><td>{_esc(r['formation'])} — <b>{_esc(r['play'])}</b>{' <span class=flag>audible</span>' if r.get('audible') else ''}</td>"
        f"<td><span class='flag {_esc(r['flag'])}'>{_esc(r['flag'] or '-')}</span></td>"
        f"<td class='muted'>meta {r['meta']:+.2f}</td><td class='muted'>{_esc(r['stats'])}</td>"
        f"<td class='muted'>{_esc((r.get('why') or '')[:180])}{('<br/>' + _cite_html(r.get('cites') or [])) if r.get('cites') else ''}</td></tr>"
        for r in book.get("meta_table") or []
    )
    fmeta_rows = "".join(
        f"<tr><td>{_esc(f)}</td><td class='muted'>{m.get('meta', 0):+.2f}</td><td class='muted'>{_esc(m.get('book') or '')}</td>"
        f"<td class='muted'>{_esc('; '.join(m.get('reasons') or [])[:200])}</td></tr>"
        for f, m in (book.get("formation_meta") or {}).items()
    )
    hist = "".join(
        f"<tr><td>rev {_esc(h['rev'])}</td><td><span class='flag'>{_esc(h['status'])}</span></td><td class='muted'>{_esc(h['kind'])}</td>"
        f"<td class='muted'>{_esc((h.get('created_ts') or '')[:16].replace('T', ' '))} UTC"
        + (f" · applied {_esc((h.get('applied_ts') or '')[:16].replace('T', ' '))}" if h.get("applied_ts") else "")
        + f"</td><td class='muted'>{_esc(h.get('n_formations'))} formations / {_esc(h.get('n_plays'))} plays — {_esc(h.get('summary') or '')}</td></tr>"
        for h in book.get("history") or []
    )
    lim = book.get("limits") or {}
    pr = book.get("practical") or {}
    lsrc = "".join(
        f"<li>{_esc(s['claim'])} — " + (f"<a href='{_esc(s['url'])}' target='_blank' rel='noopener'>{_esc(s['source'])}</a> " if s.get('url') else _esc(s['source']) + " ")
        + f"<span class='muted'>({_esc(s['date'])}, {_esc(s['title_era'])})</span></li>"
        for s in book.get("limit_sources") or []
    )
    consts = "".join(f"<tr><td>{_esc(k)}</td><td class='muted'>{_esc(v)}</td></tr>" for k, v in book.get("constants") or [])
    cands = "".join(
        f"<li>{c['score']:+.2f} {_esc(c['formation'])} <span class='muted'>(from {_esc(c.get('book') or 'any')}) {_esc(c['why'][:160])}</span></li>"
        for c in book.get("candidates") or []
    )
    n_forms = len(book.get("target_plays") or {})
    n_plays = sum(len(v) for v in (book.get("target_plays") or {}).values())
    return f"""
    <section id="cfb-book">
      <h2>Custom playbook — {_esc(book.get('name'))} <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:400">(formation-level · managed by the coach · {_esc(dyn)})</span></h2>
      {status}
      {seeded}
      {edits_html}
      <div class="scout-card scout-affect" style="margin-top:10px">
        <h3>Formations — why each is in (or out of) the book</h3>
        <table class="book">{frows}</table>
      </div>
      <div class="scout-card" style="margin-top:10px">
        <h3>Full resulting book ({n_forms} formations, {n_plays} callable plays)</h3>
        {_copy("book-full", book.get("book_text") or "", "Copy full book")}
      </div>
      <div class="scout-card scout-affect" style="margin-top:10px">
        <h3>Every callable play — flags + evidence (failing plays are called less, never removed)</h3>
        <table class="book">{rows}</table>
      </div>
      <details style="margin-top:10px"><summary class="muted">Formation meta ranking this prep (top 12)</summary><table class="book">{fmeta_rows}</table></details>
      <details><summary class="muted">Revision history (rollback: <code>{CMD} book rollback --dynasty {_esc(dyn)} --to N</code>)</summary>
        <table class="book">{hist or "<tr><td class='muted'>none</td></tr>"}</table></details>
      <details><summary class="muted">Add candidates this prep</summary><ul>{cands or "<li class='muted'>none cleared the bar</li>"}</ul></details>
      <details><summary class="muted">Custom playbook limits (CFB 27) + decision rules</summary>
        <div class="muted">Game limits: ≤{_esc(lim.get('max_plays'))} plays, ≤{_esc(lim.get('max_formation_sets'))} formation sets,
          ≤{_esc(lim.get('max_plays_per_set'))} plays per set, {_esc(lim.get('audibles_per_formation'))} audibles per formation; formations from any book; no reordering.
          Coach's self-imposed size: ≤{_esc(pr.get('max_formations'))} formations.</div>
        <div class="muted">{_esc(book.get('limit_assumption'))}</div>
        <ul>{lsrc}</ul>
        <table class="srcs">{consts}</table>
      </details>
    </section>
    """


# ---------------------------------------------------------------------------
# v1.14 minimal prep page pieces
# ---------------------------------------------------------------------------

def _local_hm(ts: str) -> str:
    from datetime import datetime

    try:
        dt = datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return ts
        from zoneinfo import ZoneInfo

        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%-I:%M %p ET")
    except (ValueError, ImportError):
        return ts or "?"


def research_failed(scout: dict[str, Any] | None) -> bool:
    scout = scout or {}
    if not scout:
        return False
    return (scout.get("mode") or "") in ("cache", "seed") or (scout.get("research_status") or "") in ("failed", "offline")


def research_status_line(scout: dict[str, Any] | None) -> str:
    """One line: 'Meta researched 2:31 PM ET · 11 sources · 2 YouTube transcripts'."""
    scout = scout or {}
    if not scout:
        return "Meta research: not run"
    n_ok = sum(1 for s in (scout.get("sources") or []) if s.get("fetched"))
    yt = (scout.get("youtube") or {}).get("transcripts", 0)
    when = _local_hm(scout.get("fetched_at") or "")
    if research_failed(scout):
        age = scout.get("fallback_age_hours")
        age_txt = f", cache {age:.0f}h old" if isinstance(age, (int, float)) else ""
        return f"Meta research FAILED this prep — using {'cached' if scout.get('mode') == 'cache' else 'seed'} research{age_txt}"
    partial = " (partial)" if scout.get("research_status") == "partial" else ""
    return f"Meta researched {when}{partial} · {n_ok} sources · {yt} YouTube transcript{'s' if yt != 1 else ''}"


def research_fail_banner(scout: dict[str, Any] | None) -> str:
    scout = scout or {}
    if not research_failed(scout):
        return ""
    age = scout.get("fallback_age_hours")
    age_txt = f"{age:.0f} hours old" if isinstance(age, (int, float)) else "age unknown"
    what = f"cached research ({age_txt})" if scout.get("mode") == "cache" else "the built-in seed research"
    return (f'<div class="banner fail">⚠️ LIVE META RESEARCH DID NOT RUN THIS PREP ({_esc(scout.get("research_status") or scout.get("mode"))}). '
            f'Using {what} — the formations below are based on older information. {_esc(scout.get("message") or "")}</div>')


_STATUS_LABEL = {"new": "NEW", "applied": "in book", "remove": "REMOVE", "changed": "RE-ADD"}


def render_formations_min(book: dict[str, Any] | None) -> str:
    """Formations to have in the custom playbook (+ the single apply command if pending)."""
    book = book or {}
    if not book:
        return ""
    if book.get("error"):
        return f"<section id='formations'><h2>Formations</h2><div class='empty'>Unavailable: {_esc(book['error'])}</div></section>"
    items = []
    for r in book.get("formation_list") or []:
        st = r.get("status") or "applied"
        note = f"<div class='fnote'>{_esc(r['note'])}</div>" if r.get("note") else ""
        items.append(
            f"<li class='frow {st}'><span class='fbadge {st}'>{_esc(_STATUS_LABEL.get(st, st))}</span>"
            f"<span class='fname'>{_esc(r['formation'])}</span>"
            f"<span class='fsrc'>{_esc(r.get('source_book') or 'any')} playbook · {_esc(r['n_plays'])} plays</span>{note}</li>"
        )
    pending = bool(book.get("pending")) and bool(book.get("apply_cmd"))
    apply_html = ""
    if pending:
        apply_html = ("<div class='apply'><div class='muted'>After you add/remove the marked formations in CFB 27 "
                      "(Create &amp; Share › Custom Playbooks), confirm:</div>" + _copy("book-apply-cmd", book["apply_cmd"], "Copy apply command") + "</div>")
    return f"""
    <section id="formations">
      <h2>Formations — {_esc(book.get('name'))}</h2>
      <ul class="flist">{"".join(items)}</ul>
      {apply_html}
    </section>
    """


def render_audibles_min(book: dict[str, Any] | None) -> str:
    book = book or {}
    auds = book.get("audibles") or {}
    if not auds:
        return ""
    rows = "".join(
        f"<div class='aud'><div class='aform'>{_esc(f)}</div><ol>{''.join(f'<li>{_esc(p)}</li>' for p in ps)}</ol></div>"
        for f, ps in auds.items()
    )
    return f"""
    <section id="audibles">
      <h2>Audibles (4 per formation)</h2>
      <div class="auds">{rows}</div>
    </section>
    """


MIN_CSS = """
  .rline { color: var(--muted); font-size: 0.85rem; margin: 6px 0 14px; }
  ul.flist { list-style: none; padding: 0; margin: 0; }
  .frow { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.07); }
  .fbadge { display: inline-block; min-width: 64px; text-align: center; font-size: 0.7rem; font-weight: 700; padding: 2px 6px;
            border-radius: 4px; margin-right: 10px; background: rgba(139,155,180,0.2); color: var(--muted); }
  .fbadge.new { background: rgba(94,166,255,0.3); color: #cfe3ff; }
  .fbadge.remove { background: rgba(224,85,85,0.3); color: #ffc4c4; }
  .fbadge.changed { background: rgba(240,198,116,0.3); color: #f6dca8; }
  .frow.remove .fname { text-decoration: line-through; opacity: 0.8; }
  .fname { font-weight: 700; font-size: 1.02rem; }
  .fsrc { color: var(--muted); font-size: 0.8rem; margin-left: 10px; }
  .fnote { color: var(--muted); font-size: 0.8rem; margin: 3px 0 0 78px; }
  .apply { margin-top: 12px; }
  .auds { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); }
  .aud { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 12px; }
  .aform { font-weight: 700; margin-bottom: 4px; }
  .aud ol { margin: 0; padding-left: 20px; }
  .mwhy { color: var(--muted); font-size: 0.78rem; margin-left: 8px; }
"""


def render_research(scout: dict[str, Any] | None) -> str:
    """Research status (loud on fallback) + YouTube + named mentions."""
    scout = scout or {}
    if not scout:
        return ""
    status = scout.get("research_status") or ""
    mode = scout.get("mode") or ""
    age = scout.get("fallback_age_hours")
    banner = ""
    if mode in ("cache", "seed") or status in ("failed", "offline"):
        age_txt = f"{age:.0f} hours old" if isinstance(age, (int, float)) else ("seed research " if mode == "seed" else "age unknown")
        banner = (f'<div class="banner fail">⚠️ LIVE META RESEARCH DID NOT RUN THIS PREP ({_esc(status or mode)}). '
                  f'Using {("cached research " + age_txt) if mode == "cache" else "the built-in seed research"} — '
                  f'playbook decisions below are based on older information. {_esc(scout.get("message") or "")}</div>')
    elif status == "partial":
        banner = f'<div class="banner pend">Live research partially succeeded — {_esc(scout.get("message") or "")}</div>'
    yt = scout.get("youtube") or {}
    vids = "".join(
        f"<tr><td class='muted'>{_esc((v.get('published') or '')[:10])}{'' if v.get('published_exact') else '~'}</td>"
        f"<td><a href='{_esc(v.get('url'))}' target='_blank' rel='noopener'>{_esc(v.get('title'))}</a></td>"
        f"<td class='muted'>{_esc(v.get('channel') or '')}</td>"
        f"<td class='muted'>{('📝 transcript (' + _esc(v.get('transcript_method') or 'cache') + ')') if v.get('transcript_status') in ('ok', 'cached') else _esc(v.get('transcript_status') or 'not tried')}</td></tr>"
        for v in sorted(yt.get("videos") or [], key=lambda v: (v.get("transcript_status") not in ("ok", "cached"), -(v.get("relevance") or 0)))[:14]
    )
    yt_line = (
        f"YouTube: <b>{_esc(yt.get('found', 0))}</b> CFB 27 videos found "
        f"(search {_esc(yt.get('search_ok', 0))}/{_esc(yt.get('search_total', 0))}, channel RSS {_esc(yt.get('rss_ok', 0))}/{_esc(yt.get('rss_total', 0))}; "
        f"{_esc(yt.get('rejected_old_title', 0))} older-title/Madden videos filtered) · "
        f"<b>{_esc(yt.get('transcripts', 0))}</b> transcripts used ({_esc(yt.get('from_cache', 0))} cached, {_esc(yt.get('fetched_now', 0))} fetched now"
        + (f"; methods {_esc(', '.join(f'{k}×{v}' for k, v in (yt.get('methods') or {}).items()))}" if yt.get("methods") else "")
        + (f"; {len(yt.get('blocked') or [])} attempt(s) blocked by YouTube (common on cloud IPs — works better from a home connection)" if yt.get("blocked") else "")
        + ")"
    ) if yt else "YouTube research not run."
    notes = "".join(f"<li class='muted'>{_esc(n)}</li>" for n in (yt.get("notes") or [])[:5])
    named = scout.get("named_signals") or {}

    def _rows(d: dict[str, Any], n: int) -> str:
        out = []
        for k, v in sorted(d.items(), key=lambda kv: -float(kv[1].get("score", 0)))[:n]:
            srcs = v.get("sources") or []
            src_html = " · ".join(
                f"<a href='{_esc(s.get('url'))}' target='_blank' rel='noopener'>{_esc((s.get('label') or '')[:40])}</a>"
                + (f" <span class='muted'>({_esc((s.get('date') or '')[:10])}, ×{_esc(s.get('mentions', 1))})</span>")
                for s in srcs[:4]
            )
            out.append(f"<tr><td><b>{_esc(k.replace('::', ' — '))}</b></td><td class='muted'>{float(v.get('score', 0)):.2f}</td>"
                       f"<td class='muted'>{_esc(v.get('docs', 0))} src · {_esc(v.get('mentions', 0))} mentions</td><td>{src_html}</td></tr>")
        return "".join(out)

    pairs = _rows(named.get("pairs") or {}, 14)
    forms = _rows(named.get("formations") or {}, 10)
    return f"""
    <section id="research">
      <h2>Meta research this prep</h2>
      {banner}
      <div class="muted">Status: <b>{_esc(status or mode)}</b> · {_esc(scout.get('elapsed_s') or '?')}s · {_esc(named.get('docs_scanned', 0))} documents scanned for CFB 27 formation/play names (recency-weighted, half-life 21 days).</div>
      <div class="scout-grid" style="margin-top:8px">
        <div class="scout-card scout-affect">
          <h3>YouTube</h3>
          <div>{yt_line}</div>
          <ul>{notes}</ul>
          <table class="srcs">{vids or "<tr><td class='muted'>no CFB 27 videos this pass</td></tr>"}</table>
        </div>
        <div class="scout-card">
          <h3>Plays named in current sources (formation — play)</h3>
          <table class="srcs">{pairs or "<tr><td class='muted'>none matched the CFB 27 play vocabulary</td></tr>"}</table>
        </div>
        <div class="scout-card">
          <h3>Formations named</h3>
          <table class="srcs">{forms or "<tr><td class='muted'>none</td></tr>"}</table>
        </div>
      </div>
    </section>
    """
