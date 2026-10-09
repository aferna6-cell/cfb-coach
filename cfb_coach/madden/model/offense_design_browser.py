"""Standalone Madden-style preview of ML-designed formations and sourced macros."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Mapping

from cfb_coach.games import data_dir
from cfb_coach.madden import catalog, playbook, research_db
from cfb_coach.madden.model import offense_designer as designer


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def output_path(opponent_id: str) -> Path:
    slug = re.sub(r"[^a-z0-9_-]", "-", opponent_id.lower())[:50] or "cpu"
    return Path(data_dir()) / f"madden27_offense_design_{slug}.html"


def _link(item: Mapping[str, Any]) -> str:
    title = esc(item.get("title") or item.get("id") or "Reference")
    url = str(item.get("url") or "")
    if not url.startswith(("https://", "http://")):
        return f"{title} <small>(local reference)</small>"
    return f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{title}</a>'


def installed_design(db: Any, opponent_id: str) -> dict[str, Any] | None:
    """Reconstruct status from a user-confirmed active book and saved blueprints."""
    book = playbook.load_books(db).get("offense") or {}
    if book.get("name") != "ML Designed Offense (custom)":
        return None
    try:
        obj = json.loads(db.get_meta(designer.META_BLUEPRINTS) or "{}")
    except (ValueError, TypeError):
        obj = {}
    return {
        "proposal_id": obj.get("proposal_id") or "installed",
        "opponent_id": opponent_id, "book": book,
        "macro_blueprints": obj.get("macro_blueprints") or [],
        "changes": [], "model_version": "See stored model artifact",
        "supervised_examples": "See ml inspect",
    }


def current_macro_html(db: Any, opponent_id: str, applied: Mapping[str, Any]) -> str:
    """Existing known Custom Adjustments, with user-confirmed rows and gaps.

    Read meta directly: load_selection() can migrate records and must not run
    while this HTML preview has the database open in read-only mode.
    """
    from cfb_coach.madden.offense_macros import offense_detail

    try:
        rec = json.loads(db.get_meta("active_macros:" + opponent_id) or "{}")
    except (TypeError, ValueError):
        rec = {}
    ids = list(rec.get("offense") or []) if rec.get("schema") == 2 else []
    cards = []
    for mid in ids:
        detail = offense_detail(mid, applied.get("formations") or {})
        settings = detail.get("settings") or []
        compatible = bool(detail.get("pairs_with"))
        rows = [
            f'<tr><td>{esc(row.get("section"))}</td><td>{esc(row.get("setting"))}</td>'
            f'<td>{esc(row.get("value"))}</td><td>{esc(row.get("source") or "user notes")}</td></tr>'
            for row in settings
        ]
        status = (
            "Selected in loadout; editor verification needed"
            if settings and compatible and not detail.get("gaps")
            else "Not callable until settings and in-book play are verified"
        )
        cards.append(
            f'<details><summary><b>{esc(detail.get("xbox_name") or mid)}</b>'
            f'<span class="pill">{esc(status)}</span></summary>'
            f'<p><b>When to call:</b> {esc(detail.get("fire_when") or "See matchup conditions")}</p>'
            f'<p><b>Compatible play pairs:</b> {esc(", ".join(detail.get("pairs_with") or []))}</p>'
            f'<p><b>At the line:</b> {esc(detail.get("in_game"))}</p>'
            '<div class="scroll"><table><thead><tr><th>Section</th><th>Field</th><th>Value</th>'
            f'<th>Evidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="muted">{esc(detail.get("settings_source"))}</p>'
            f'<p class="warning">Unverified fields: {esc(", ".join(detail.get("gaps") or []))}</p>'
            '</details>'
        )
    return "".join(cards) or '<section><p class="muted">No existing offensive macros in the saved loadout.</p></section>'


def render_html(
    db: Any, proposal: Mapping[str, Any], *,
    mode: str = "preview", opponent_id: str = "cpu",
) -> str:
    if mode not in ("preview", "staged", "installed"):
        raise ValueError("Unknown designer browser mode")
    book = (proposal.get("book") or {})
    forms = book.get("formations") or {}
    sources = book.get("formation_sources") or {}
    applied = (playbook.load_books(db).get("offense") or {})
    is_installed = mode == "installed" and applied.get("formations") == forms
    approved = {m["name"] for m in designer.verified_created_macros(db, opponent_id)}
    configured = {m["name"] for m in designer.verified_macro_configurations(db)}
    formation_details = {
        str(item.get("formation")): item
        for item in proposal.get("chosen_formation_details") or []
    }
    from cfb_coach.madden.model.offense_inventory import inventory_report

    # Counts are logged recommendations, not confirmed executions. The
    # inventory includes never-called plays so the browser can reveal gaps.
    used_inventory = inventory_report(db, opponent_id=opponent_id)
    used_pairs = {
        (p["formation"], p["play"]): p["recommended_calls"]
        for p in used_inventory["play_usage"]
    }
    source_docs = research_db.sources()
    proposal_id = str(proposal.get("proposal_id") or "")
    status = "INSTALLED — CALLABLE" if is_installed else (
        "STAGED — NOT CALLABLE" if mode == "staged" else "PREVIEW — NO CHANGES"
    )
    groups = []
    required = 0
    for form, plays in forms.items():
        detail = formation_details.get(str(form), {})
        source = sources.get(form) or ""
        stock = (catalog.books("offense").get(source) or {})
        stock_url = str(stock.get("url") or "")
        source_link = (
            f'<a href="{esc(stock_url)}" target="_blank" rel="noopener noreferrer">{esc(source)}</a>'
            if stock_url.startswith(("https://", "http://")) else esc(source)
        )
        # Review/install one complete formation, not dozens of individual
        # plays. All sourced plays are listed for reference, never pruned.
        built = is_installed and list((applied.get("formations") or {}).get(form) or []) == list(plays)
        if built:
            mark = '<span class="ok">✓</span>'
            label = "Installed & callable"
        else:
            required += 1
            mark = (
                f'<input class="build-check" type="checkbox" '
                f'aria-label="I installed all plays in formation {esc(form)}">'
            )
            label = "Check full formation in Madden"
        entries = [
            f'<li class="play"><b>{esc(p)}</b>'
            + (
                f'<small>{used_pairs.get((form, p), 0)} calls recommended</small>'
                if is_installed else '<small>Available after installation</small>'
            )
            + '</li>'
            for p in plays
        ]
        verify_header = f'<div class="play">{mark}<strong>{esc(label)}</strong></div>'
        situations = ", ".join(str(item).replace("_", " ")
                               for item in detail.get("addresses") or [])
        audibles = ", ".join(
            f"{item.get('role')}: {item.get('play')}"
            for item in detail.get("suggested_audibles") or []
        )
        groups.append(
            f'<section><div class="head"><h3>{esc(form)}</h3><span class="pill">{len(plays)} plays</span></div>'
            f'<p class="muted">Stock source: {source_link} · complete formation, all {len(plays)} catalogued plays</p>'
            f'<p><b>Why selected:</b> {esc(detail.get("selection_reason") or "installed formation")}</p>'
            f'<p><b>Strongest situation coverage:</b> {esc(situations or "not recorded")}</p>'
            f'<p><b>Suggested audibles (not installed automatically):</b> {esc(audibles or "none")}</p>'
            f'{verify_header}<ul>{"".join(entries)}</ul></section>'
        )
    deltas = [
        f'<li class="delta"><span class="pill">{esc(d.get("action"))}</span>'
        f'<span>{esc(d.get("formation"))} {esc(d.get("play") or "")}</span>'
        f'<small>{esc(d.get("source_book") or "")}</small></li>'
        for d in (proposal.get("changes") or [])
    ]
    current_macros = current_macro_html(db, opponent_id, applied)
    from cfb_coach.madden.model.offense_action_inventory import offense_actions_report

    action_readiness = offense_actions_report(db, opponent_id)
    macros = []
    for index, m in enumerate(proposal.get("macro_blueprints") or []):
        name = str(m.get("name") or "UNNAMED")
        verified = is_installed and name in approved
        config_verified = is_installed and name in configured
        macro_status = (
            "Verified and armed — callable when eligible" if verified
            else "Verified settings — not armed" if config_verified
            else "Draft — not callable"
        )
        setting_rows = []
        for row in m.get("settings") or []:
            cited = [
                _link(source_docs.get(source_id, {"id": source_id}))
                for source_id in row.get("sources") or []
            ]
            setting_rows.append(
                f'<tr><td>{esc(row.get("section"))}</td><td>{esc(row.get("setting"))}</td>'
                f'<td><strong>{esc(row.get("value"))}</strong></td>'
                f'<td>{" · ".join(cited) or "Source unavailable"}</td></tr>'
            )
        srcs = [
            _link(source_docs.get(source_id, {"id": source_id}))
            for source_id in m.get("source_ids") or []
        ]
        pairs = ", ".join(
            esc(str(p.get("formation")) + " — " + str(p.get("play")))
            for p in m.get("base_pairs") or []
        )
        verify_config_cmd = (
            f'python -m cfb_coach ml offense-design -o {opponent_id} '
            f'--verify-macro-config {name} '
            '--attest "I checked every listed setting and confirmed this configuration is supported in Madden."'
        )
        arm_cmd = (
            f'python -m cfb_coach ml offense-design -o {opponent_id} '
            f'--verify-macro {name} '
            '--attest "I built this Custom Adjustment with the sourced settings and armed it in Madden."'
        )
        disable_cmd = (
            f"python -m cfb_coach ml offense-design -o {opponent_id} --unverify-macro {name}"
        )
        cmd_html = (
            (
                f'<pre id="macro-config-{index}">{esc(verify_config_cmd)}</pre>'
                f'<button data-copy="macro-config-{index}">Copy settings-verification command</button>'
                if not config_verified else ""
            )
            + (
                f'<pre id="macro-cmd-{index}">{esc(arm_cmd)}</pre>'
                f'<button data-copy="macro-cmd-{index}">Copy armed-slot command</button>'
                if config_verified and not verified else ""
            )
            if is_installed and not verified else (
                f'<p class="muted">If you remove this macro from Madden, disable it in the coach:</p>'
                f'<pre id="macro-disable-{index}">{esc(disable_cmd)}</pre>'
                f'<button data-copy="macro-disable-{index}">Copy disable-macro command</button>'
                if verified else ""
            )
        )
        macros.append(
            f'<details><summary><b>{esc(name)}</b><span class="pill">{esc(macro_status)}</span></summary>'
            f'<p><b>Trigger:</b> {esc(m.get("fire_when"))}</p>'
            f'<p><b>Exact eligible plays:</b> {pairs}</p>'
            '<div class="scroll"><table><thead><tr><th>Section</th><th>Field</th><th>Setting</th>'
            f'<th>Source</th></tr></thead><tbody>{"".join(setting_rows)}</tbody></table></div>'
            f'<p class="warning">{esc(m.get("other_editor_settings") or "Check unspecified editor fields in Madden.")}</p>'
            f'<p><b>Research:</b> {" · ".join(srcs)}</p>'
            f'<p class="muted">Draft, verified configuration, and armed slot are separate states. '
            f'Only armed macros are eligible for live calls.</p>'
            f'{cmd_html}</details>'
        )
    stage_cmd = f"python -m cfb_coach ml offense-design -o {opponent_id} --stage"
    confirm_cmd = (
        f'python -m cfb_coach ml offense-design --confirm-installed {proposal_id} '
        '--attest "I installed and checked every proposed formation and play inside Madden."'
    )
    workflow = ""
    if mode == "preview":
        workflow = (
            '<p>Review the formations, then stage the design:</p>'
            f'<pre id="stage-cmd">{esc(stage_cmd)}</pre><button data-copy="stage-cmd">Copy stage command</button>'
        )
    elif mode == "staged":
        workflow = (
            f'<p>Install each complete formation inside Madden. Check all {required} formations after verifying their complete play lists. '
            'These checkboxes are local to this browser, not a database confirmation.</p>'
            f'<pre id="confirm-cmd">{esc(confirm_cmd)}</pre>'
            '<button id="confirm-copy" data-copy="confirm-cmd" disabled>Check installed plays first</button>'
        )
    elif is_installed:
        workflow = '<p class="ok">This playbook is confirmed in SQLite and is available to the live ML coach.</p>'
    status_color = "ok" if is_installed else "warning"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Madden 27 — Offensive Designer</title>
<style>
:root{{--bg:#0e1117;--fg:#e6edf3;--muted:#8b949e;--accent:#3fb950;--call:#58a6ff;--panel:#161b22;--border:#30363d;--warn:#d29922}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,sans-serif}}
main{{max-width:920px;margin:auto;padding:22px 16px 55px}}
h1{{color:var(--call);font-size:clamp(1.5rem,3vw,2.3rem)}}h2{{font-size:.85rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
h3{{margin:0;font-size:1.15rem}}section,details,.stat{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:15px;margin:12px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:10px}}.grid .stat{{margin:0}}
.stat strong{{display:block;margin-top:8px;color:var(--call);overflow-wrap:anywhere}}
.muted,small{{color:var(--muted)}}.warning{{color:var(--warn)}}.ok{{color:var(--accent)}}.head{{display:flex;align-items:center;justify-content:space-between;gap:8px}}
.pill{{background:#21262d;border:1px solid var(--border);border-radius:5px;font-size:.75rem;padding:3px 8px;color:var(--accent);margin-left:6px}}
ul{{list-style:none;margin:8px 0;padding:0}}.play,.delta{{display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border);padding:9px 0}}
.play b,.delta span:nth-child(2){{flex:1}}input[type=checkbox]{{width:18px;height:18px;accent-color:var(--accent)}}a{{color:var(--call);overflow-wrap:anywhere}}
summary{{display:flex;align-items:center;gap:10px;cursor:pointer;flex-wrap:wrap}}summary b{{font-size:1rem}}
table{{border-collapse:collapse;width:100%;font-size:.83rem}}th,td{{border-bottom:1px solid var(--border);text-align:left;vertical-align:top;padding:8px}}
.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem;background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:12px}}
button{{border:1px solid var(--border);border-radius:6px;background:#21262d;color:var(--fg);font-weight:600;padding:9px 13px;cursor:pointer}}
button:disabled{{opacity:.45;cursor:not-allowed}}button:hover:not(:disabled){{border-color:var(--call)}}
.note{{border-left:3px solid var(--warn);padding-left:11px;line-height:1.5}}
</style></head><body><main>
<p class="{status_color}">{esc(status)} · {esc(opponent_id.upper())}</p>
<h1>AI Offensive Playbook Designer</h1>
<p class="muted">Matches the live Madden coach. Proposals never change playable formations or
Custom Adjustments until you confirm their actual installation in Madden.</p>
<div class="grid">
<div class="stat"><small>Active offense</small><strong>{esc(applied.get("name") or "None locked")}</strong></div>
<div class="stat"><small>Model</small><strong>{esc(proposal.get("model_version") or "Prior-only")}</strong></div>
<div class="stat"><small>Designed formations / plays</small><strong>{len(forms)} / {sum(len(ps) for ps in forms.values())}</strong></div>
<div class="stat"><small>Installed plays recommended</small><strong>{used_inventory["used_plays"]} / {used_inventory["play_count"]}</strong></div>
<div class="stat"><small>Macro designs: drafted / verified</small><strong>{len(macros)} / {len(approved)}</strong></div>
<div class="stat"><small>Saved loadout macros with eligible book pairs</small><strong>{action_readiness["ready_in_saved_loadout"]} / {action_readiness["saved_loadout_macros"]}</strong></div>
</div>
<section><h2>Build and confirm</h2>
<p><b>Design ID:</b> {esc(proposal_id)}</p>
<p class="muted">Applied inventory ID: <code>{esc(used_inventory["inventory_id"])}</code>
 · {used_inventory["unused_plays"]} installed plays have not yet been recommended
 for {esc(opponent_id)}. These usage counts are recommendations, not verified executions.</p>
<p class="note">Source-book links identify catalogued plays; they do not guarantee a particular
play is present in your Madden custom editor. Confirm it in game before marking installed.</p>
{workflow}<p id="progress" class="muted"></p></section>
<section><h2>Playbook edits</h2><ul>{"".join(deltas) or "<li>No pending edits.</li>"}</ul></section>
<h2>Formations and individual plays</h2>{"".join(groups) or "<p>No formations available.</p>"}
<h2>New Custom Adjustments and researched settings</h2>
<p class="muted">Expand each macro for its trigger, compatible plays, settings and linked research.
Unspecified Madden editor fields remain unverified; generated blueprints are never auto-armed.</p>
{"".join(macros) or '<section><p>No new macros were proposed.</p></section>'}
<h2>Existing offensive Custom Adjustments in your saved loadout</h2>
<p class="muted">A compatible saved macro is not assumed to be physically armed.
Check the full readiness breakdown with
<code>python -m cfb_coach ml offense-actions -o {esc(opponent_id)}</code>.
On each snap, the model chooses the best researched eligible action—or
<b>no adjustment</b> when the trigger is insufficient.</p>
<p class="muted">Listed macros have research or user-noted settings. Being selected during prep
does not alone prove they were built and armed in Madden. Review missing settings and compatibility.</p>
{current_macros}
<section><h2>How the live model knows what is installed</h2>
<p>Live ML scores <b>every situationally eligible play</b> from the full confirmed
formation inventory recorded in SQLite—not just five favorite plays or two plays
per concept. Normal decisions favor model-rated plays; controlled experimentation
can sample the entire situationally eligible installed inventory.
New macros become eligible only after a separate verified-and-armed confirmation for this
opponent and when the exact formation, play, and observed pre-snap coverage match.</p>
<p class="muted">This is a read-only HTML file. Checkmarks do not change the database.
After running a copied verification command, rerun <code>ml offense-design --show</code>
to refresh what the model knows.</p></section>
<script>
(function(){{
 const c=Array.from(document.querySelectorAll('.build-check'));
 const b=document.getElementById('confirm-copy');
 const p=document.getElementById('progress');
 function update(){{
  const n=c.filter(x=>x.checked).length;
  if(b){{b.disabled=c.length>0&&n!==c.length;b.textContent=b.disabled?'Check installed formations first':'Copy install-confirmation command';}}
  if(p) p.textContent=c.length?(n+' / '+c.length+' formations checked on this page (not saved)'):'';
 }}
 c.forEach(x=>x.addEventListener('change',update));update();
 document.querySelectorAll('button[data-copy]').forEach(x=>x.addEventListener('click',async()=>{{
  const src=document.getElementById(x.getAttribute('data-copy'));if(!src)return;
  try{{await navigator.clipboard.writeText(src.textContent);x.textContent='Copied';}}
  catch(e){{const t=document.createElement('textarea');t.value=src.textContent;document.body.appendChild(t);t.select();document.execCommand('copy');t.remove();x.textContent='Copied';}}
 }}));
}})();
</script>
</main></body></html>"""


def write_and_open(
    db: Any, proposal: Mapping[str, Any], *,
    mode: str = "preview", opponent_id: str = "cpu",
    path: Path | None = None, open_browser: bool = True,
) -> Path:
    out = path or output_path(opponent_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(db, proposal, mode=mode, opponent_id=opponent_id), encoding="utf-8")
    if open_browser:
        # Reuse the same WSL-aware opening flow as Madden's prep browser:
        # webbrowser, xdg-open, wslview, explorer.exe, then Windows path hints.
        from cfb_coach.prep_browser import open_prep_html

        open_prep_html(out, open_browser=True)
    return out
