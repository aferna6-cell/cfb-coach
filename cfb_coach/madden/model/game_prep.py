"""`ml game-prep`: design both books, write the setup page and checklist.

Read-only against the database. The only files written are the proposal JSON,
the browser page and the Markdown checklist under ``<data dir>/game_prep``.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden import catalog_full, playbook
from cfb_coach.madden.model import game_evidence, playbook_architect, playbook_install

MADDEN_STEPS_SOURCE = (
    "ClutchPoints 'Madden 27 How to Create Custom Playbooks' and EA Help 'Madden NFL 27 playbooks guide'; "
    "menu wording may differ slightly in your build."
)
CONCENTRATION_ALERT = 0.6


def _opponent_name(db: Any, opponent_id: str) -> str:
    try:
        row = db.conn.execute("SELECT display_name FROM opponents WHERE id=?", (opponent_id,)).fetchone()
        return str(row[0]) if row and row[0] else opponent_id
    except Exception:  # noqa: BLE001
        return opponent_id


def _load_offense_artifact(db: Any) -> tuple[Any, str]:
    from cfb_coach.madden.model import experimental_model
    from cfb_coach.madden.model.experimental_live import resolve_artifact_path

    path = resolve_artifact_path(db)
    if path is None:
        return None, "none"
    try:
        art = experimental_model.load_artifact(path)
    except Exception:  # noqa: BLE001
        return None, "none"
    return art, str(getattr(art, "evidence_quality", "prior_driven") or "prior_driven")


def build_steps(side: str, design: Mapping[str, Any], applied: Mapping[str, Any] | None) -> dict[str, Any]:
    """Exact editor work: base book, formations to delete from it, formations to add, play fixes."""
    forms = design["formations"]
    counts: dict[str, int] = {}
    for f in forms:
        counts[f["source_book"]] = counts.get(f["source_book"], 0) + 1
    base = max(counts, key=lambda k: (counts[k], k)) if counts else None
    base_forms = (catalog_full.architect_books(side).get(base or "") or {}).get("formations") or {}
    wanted = {f["formation"] for f in forms}
    delete = sorted(f for f in base_forms if f not in wanted)
    per_formation = []
    for f in forms:
        in_base = base_forms.get(f["formation"])
        base_plays = list((in_base or {}).get("plays") or [])
        if in_base and f["source_book"] == base:
            action = f"Already in the {base} copy - keep it and check the plays."
            add, remove = [], []
        elif in_base:
            add = [p for p in f["plays"] if p not in base_plays]
            remove = [p for p in base_plays if p not in f["plays"]]
            action = (f"In the {base} copy, but its play list differs from the {f['source_book']} book - "
                      f"add {len(add)} and remove {len(remove)} plays.")
        else:
            add, remove = list(f["plays"]), []
            action = f"Add it from All Plays: {f['editor_path']['formation']} > {f['editor_path']['set']}, {f['source_book']} plays."
        per_formation.append({"formation": f["formation"], "action": action, "add_plays": add, "remove_plays": remove})
    label = "OFF" if side == "offense" else "DEF"
    name = f"ML {label} {design['fingerprint'].split('-', 1)[1].replace('-', '')}"
    editing_existing = bool(applied and applied.get("architect_proposal_id"))
    steps = [
        "Main Menu > Creation Center > Playbooks.",
        (f"Open your existing custom book from the last confirmed proposal and edit it (My Playbooks / All Plays), "
         f"or create a fresh one as below." if editing_existing else
         f"Select Create {side.title()} and choose the {base} playbook as the base."),
        f"Name it {name} so it matches this proposal.",
        f"In My Playbooks, delete the {len(delete)} {base} formations listed under 'Delete from the base copy'.",
        "In All Plays, add each formation marked 'Add it from All Plays' and activate every play in its checklist.",
        "For every formation, make the plays match the checklist exactly - the complete source-book list, nothing extra.",
        "Set audibles for new formations (Audibles page); otherwise you cannot audible out of them.",
        "Press Start to save.",
        "Franchise: on Gameday open Advanced Settings and select this playbook before Ready.",
    ]
    return {"base_book": base, "custom_name": name, "delete_from_base": delete, "formations": per_formation,
            "steps": steps, "source": MADDEN_STEPS_SOURCE}


def _defense_model_coverage(formations: Sequence[str]) -> dict[str, Any]:
    from cfb_coach.madden.model import defense_v1

    path = defense_v1.default_artifact_path()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"artifact": str(path), "available": False, "known": [], "unknown": list(formations)}
    known_forms = {str(c.get("formation")) for c in raw.get("candidates") or [] if isinstance(c, dict)}
    return {"artifact": str(path), "available": True, "known": [f for f in formations if f in known_forms],
            "unknown": [f for f in formations if f not in known_forms]}


def run_game_prep(db: Any, *, opponent_id: str, franchise: str = "primary", nfl_team: str | None = None,
                  opponent_type: str | None = None, roster: Mapping[str, Any] | None = None,
                  injuries: Sequence[str] = (), scouting: str | None = None,
                  max_offense: int = playbook_architect.MAX_FORMATIONS,
                  max_defense: int = playbook_architect.MAX_FORMATIONS,
                  root: Path | str | None = None, now: str | None = None) -> dict[str, Any]:
    from cfb_coach.opponents import is_cpu_opponent

    generated = now or datetime.now(timezone.utc).isoformat()
    kind = (opponent_type or ("cpu" if is_cpu_opponent(opponent_id) else "human")).lower()
    applied = playbook.load_books(db)
    art, quality = _load_offense_artifact(db)
    off_ev = game_evidence.offense_evidence(db, opponent_id, before_ts=now)
    def_ev = game_evidence.defense_evidence(db, opponent_id, before_ts=now)
    designs = {
        "offense": playbook_architect.design_side(
            "offense", opponent_id=opponent_id, db=db, applied=applied.get("offense"), evidence=off_ev, art=art,
            max_formations=max_offense, evidence_quality=quality, opponent_type=kind, roster_input=roster),
        "defense": playbook_architect.design_side(
            "defense", opponent_id=opponent_id, db=db, applied=applied.get("defense"), evidence=def_ev,
            max_formations=max_defense, evidence_quality="empirical" if def_ev["verified_executions"] >= 60
            else "prior_driven", opponent_type=kind),
    }
    proposals = {side: playbook_install.build_proposal(d, opponent_id=opponent_id, applied=applied.get(side),
                                                       generated_ts=generated) for side, d in designs.items()}
    paths = playbook_install.save_proposals(root, opponent_id, proposals)
    games = game_evidence.recent_games(db, opponent_id=opponent_id, limit=5, before_ts=now)
    latest_any = game_evidence.recent_games(db, limit=3, before_ts=now)
    audit = catalog_full.completeness_audit()
    sides = {}
    for side, d in designs.items():
        sides[side] = {
            "design": d,
            "proposal_id": proposals[side]["proposal_id"],
            "fingerprint": d["fingerprint"],
            "proposal_path": paths[side],
            "steps": build_steps(side, d, applied.get(side)),
            "status": playbook_install.side_status(db, root, opponent_id, side),
            "applied": {"name": (applied.get(side) or {}).get("name"), "rev": (applied.get(side) or {}).get("rev"),
                        "formations": list(((applied.get(side) or {}).get("formations") or {}))},
        }
    sides["defense"]["model_coverage"] = _defense_model_coverage(
        [f["formation"] for f in designs["defense"]["formations"]])
    concentration = [c for c in def_ev["call_concentration"] if c["top_share"] >= CONCENTRATION_ALERT]
    pkg = {
        "schema": "madden27.game_prep.v1",
        "generated_ts": generated,
        "opponent": {"id": opponent_id, "name": _opponent_name(db, opponent_id), "type": kind, "nfl_team": nfl_team},
        "franchise": franchise,
        "inputs": {"roster": "provided for this run" if roster else "stored verified snapshot, if any",
                   "injuries": list(injuries), "scouting": scouting},
        "sides": sides,
        "evidence": {"offense": {k: off_ev[k] for k in ("verified_rows", "labeled_rows", "prior_success", "situation",
                                                         "repeated_failures", "label")},
                     "defense": {k: def_ev[k] for k in ("snaps", "verified_executions", "opponent_success_allowed",
                                                         "situation_allowed", "call_concentration",
                                                         "opponent_profile", "labels")}},
        "games_vs_opponent": games,
        "latest_games": latest_any,
        "concentration_alerts": concentration,
        "catalog_audit": {k: audit[k] for k in ("complete", "claim", "parse_failures", "not_verified", "accessed")}
        | {"sides": {s: {k: v for k, v in a.items() if k != "book_names"} for s, a in audit["sides"].items()}},
        "editor_limits": catalog_full.EDITOR_LIMITS,
        "offense_model": {"evidence_quality": quality, "model_version": getattr(art, "model_version", None),
                          "rows": getattr(art, "n_rows", None)},
        "no_schedule": playbook_architect.no_schedule_statement(),
        "cpu_defense_note": ("CPU games: the coach calls offense only unless you start play with --cpu-defense. "
                             "The defensive book is still designed so it is ready if you opt in.")
        if kind == "cpu" else None,
    }
    pkg["summary"] = executive_summary(pkg)
    pkg["limitations"] = limitations(pkg)
    pkg["game_ready"] = game_ready(pkg)
    return pkg


def game_ready(pkg: Mapping[str, Any]) -> dict[str, Any]:
    needed = ["offense"] if pkg["opponent"]["type"] == "cpu" else ["offense", "defense"]
    states = {s: pkg["sides"][s]["status"]["state"] for s in ("offense", "defense")}
    ready = all(states[s] == "ready" for s in needed)
    return {"ready": ready, "needed": needed, "states": states,
            "label": "GAME READY" if ready else "NOT READY - install and confirm: " +
            ", ".join(s for s in needed if states[s] != "ready")}


def _change_line(side: str, d: Mapping[str, Any]) -> str:
    c = d["changes"]
    return (f"{side.title()}: {d['limits']['formations']} formations, {d['limits']['plays']} plays "
            f"(keep {len(c['keep'])}, add {len(c['add'])}, remove {len(c['remove'])}"
            + (f", switch source book {len(c['new_source'])}" if c.get("new_source") else "") + ").")


def executive_summary(pkg: Mapping[str, Any]) -> list[str]:
    opp = pkg["opponent"]
    lines = [f"Opponent: {opp['name']} ({'CPU' if opp['type'] == 'cpu' else 'human'})"
             + (f", NFL team {opp['nfl_team']}" if opp.get("nfl_team") else "") + "."]
    if pkg["games_vs_opponent"]:
        g = pkg["games_vs_opponent"][0]
        fix = f" (stored as {g['stored_score']}, corrected)" if g["corrected"] else ""
        lines.append(f"Last game: {g['result_wl']} {g['score']}{fix}. Offense called by {g['callers']['offense']}; "
                     f"defense called by {g['callers']['defense']}.")
    for side in ("offense", "defense"):
        lines.append(_change_line(side, pkg["sides"][side]["design"]))
    for alert in pkg["concentration_alerts"]:
        lines.append(f"Last game the defense logged {alert['top_formation']} on {alert['top_count']} of "
                     f"{alert['calls']} calls ({round(100 * alert['top_share'])}%). The new book gives the caller "
                     f"{len({f['package'] for f in pkg['sides']['defense']['design']['formations']})} packages. "
                     "Once it is confirmed, the defense model calls only installed plays and penalizes a "
                     "formation above 40% of its last ten calls when a close installed alternative exists.")
    lines.append(pkg["no_schedule"])
    return lines


def limitations(pkg: Mapping[str, Any]) -> list[str]:
    out = []
    ev = pkg["evidence"]
    out.append(f"Offense evidence: {ev['offense']['labeled_rows']} verified, labeled executions in scope; the success "
               f"model's evidence quality is {pkg['offense_model']['evidence_quality']}.")
    out.append(f"Defense evidence: {ev['defense']['verified_executions']} verified defensive executions. Defensive "
               "formation choice rests on football priors, opponent tendencies and situation-level results, not "
               "verified call outcomes.")
    if ev["defense"]["opponent_profile"].get("n"):
        out.append(f"Opponent tendencies: {ev['defense']['opponent_profile']['summary']}.")
    out.append("Formation play lists come from Huddle.gg (third party). Nothing here was verified in your copy of Madden.")
    out.append("Editor limits (15 formations, 56 sets, 500 plays) come from madden.tools and are not verified in game.")
    out.append("Offensive personnel is inferred from formation names; defensive personnel from the package convention.")
    if pkg["inputs"]["injuries"]:
        out.append("Injuries were recorded but are not modeled numerically: " + "; ".join(pkg["inputs"]["injuries"]))
    if pkg["inputs"]["scouting"]:
        out.append("Scouting notes were recorded for you; they do not change the scores.")
    cov = pkg["sides"]["defense"].get("model_coverage") or {}
    if cov.get("unknown"):
        out.append("The installed defensive ML model has never seen " + ", ".join(cov["unknown"]) +
                   "; it calls only formations it knows and the established coordinator covers the rest.")
    out.append("Scores are planning estimates, not win probabilities. No simulated alternative is claimed to have won a past game.")
    return out


# --------------------------------------------------------------------------- rendering


def confirm_command(pkg: Mapping[str, Any], side: str, db_path: str = "~/.cfb-coach/madden27.db") -> str:
    s = pkg["sides"][side]
    return (f"python3 -m cfb_coach ml game-prep --opponent {pkg['opponent']['id']} --franchise {pkg['franchise']} "
            f"--db {db_path} --confirm {side} --proposal-id {s['proposal_id']} --fingerprint {s['fingerprint']} "
            f"--attest \"I built {s['fingerprint']} in Madden and checked every play\"")


def rollback_command(pkg: Mapping[str, Any], side: str, db_path: str = "~/.cfb-coach/madden27.db") -> str:
    return (f"python3 -m cfb_coach ml game-prep --opponent {pkg['opponent']['id']} --db {db_path} "
            f"--rollback {side} --attest \"I reinstalled the previous {side} playbook in Madden\"")


def render_markdown(pkg: Mapping[str, Any]) -> str:
    opp = pkg["opponent"]
    out = [f"# Game prep vs {opp['name']} - Madden 27", "", f"Generated {pkg['generated_ts']}", "",
           f"**Status: {pkg['game_ready']['label']}**", "", "## Summary", ""]
    out += [f"- {line}" for line in pkg["summary"]]
    for side in ("offense", "defense"):
        s = pkg["sides"][side]
        d = s["design"]
        out += ["", f"## {side.title()} - proposal {s['proposal_id']} - fingerprint {s['fingerprint']}", "",
                f"Status: {s['status']['label']}", ""]
        if side == "defense" and pkg.get("cpu_defense_note"):
            out += [pkg["cpu_defense_note"], ""]
        out += ["### Changes", ""]
        for f in d["formations"]:
            out.append(f"- **{f['status']}** {f['formation']} ({f['source_book']}) - {f['why']}")
        for r in d["removed"]:
            out.append(f"- **REMOVE** {r['formation']} ({r['source_book']}) - {r['why']}")
        st = s["steps"]
        out += ["", "### Build it in Madden", ""]
        out += [f"{i}. {step}" for i, step in enumerate(st["steps"], 1)]
        out += ["", f"Delete from the base copy ({st['base_book']}):", ""]
        out += [f"- [ ] {f}" for f in st["delete_from_base"]] or ["- (nothing)"]
        out += ["", "### Formation checklists", ""]
        by_form = {x["formation"]: x for x in st["formations"]}
        for f in d["formations"]:
            fx = by_form[f["formation"]]
            out += [f"#### [ ] {f['formation']} - {f['source_book']} ({f['n_plays']} plays)", "",
                    f"Find: {f['editor_path']['formation']} > {f['editor_path']['set']} > {f['source_book']} plays. "
                    f"{fx['action']}", ""]
            if fx["remove_plays"]:
                out.append("Remove from the base version: " + ", ".join(fx["remove_plays"]))
                out.append("")
            out += [f"- [ ] {p}" for p in f["plays"]]
            out.append("")
        out += ["### Confirm after building", "", "```", confirm_command(pkg, side), "```", ""]
    out += ["## Limitations", ""] + [f"- {x}" for x in pkg["limitations"]]
    out += ["", "## Restore the previous books", "", "```", rollback_command(pkg, "offense"),
            rollback_command(pkg, "defense"), "```", ""]
    return "\n".join(out)


_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f5f6f8;color:#1b1f24}
main{max-width:1080px;margin:0 auto;padding:24px}
h1{margin:0 0 4px}h2{margin-top:32px;border-bottom:2px solid #d0d5dc;padding-bottom:4px}
.badge{display:inline-block;padding:4px 10px;border-radius:12px;font-weight:600;font-size:14px}
.ready{background:#d4f5dc;color:#11632a}.wait{background:#fff1c9;color:#7a5500}.stale{background:#fde0de;color:#8f1d14}
.flow{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0}.flow span{background:#fff;border:1px solid #cdd3da;border-radius:6px;padding:6px 10px}
.card{background:#fff;border:1px solid #dde2e8;border-radius:8px;padding:14px 16px;margin:10px 0}
.tag{font-weight:700;padding:2px 8px;border-radius:4px;margin-right:6px;font-size:13px}
.KEEP{background:#e3ecff;color:#1d3f9a}.SOURCE{background:#fff1cc;color:#7a5200}.ADD{background:#d9f7e3;color:#13652f}.REMOVE{background:#fde4e1;color:#9a2216}
.plays{columns:3;column-gap:24px;margin:8px 0 0;padding:0;list-style:none}.plays li{break-inside:avoid;margin:2px 0}
pre{background:#11161c;color:#e7edf3;padding:10px;border-radius:6px;white-space:pre-wrap;word-break:break-all}
.muted{color:#5b6672;font-size:14px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #e3e7ec;padding:6px;text-align:left;font-size:14px}
details summary{cursor:pointer;font-weight:600}
@media print{.noprint{display:none}body{background:#fff}.card{break-inside:avoid}}
"""

_JS = """
document.querySelectorAll('input[type=checkbox][data-k]').forEach(function(b){
  var k='gp:'+b.dataset.k; b.checked=localStorage.getItem(k)==='1';
  b.addEventListener('change',function(){localStorage.setItem(k,b.checked?'1':'0')});
});
"""


def _badge(state: str, label: str) -> str:
    cls = {"ready": "ready", "stale": "stale"}.get(state, "wait")
    return f'<span class="badge {cls}">{html.escape(label)}</span>'


def render_html(pkg: Mapping[str, Any]) -> str:
    e = html.escape
    opp = pkg["opponent"]
    gr = pkg["game_ready"]
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Game prep vs {e(opp['name'])}</title>",
             f"<style>{_CSS}</style></head><body><main>",
             f"<h1>Game prep vs {e(opp['name'])}</h1>",
             f"<div class='muted'>Madden 27 · {'CPU' if opp['type'] == 'cpu' else 'human'} opponent · "
             f"generated {e(pkg['generated_ts'])}</div>",
             f"<p>{_badge('ready' if gr['ready'] else 'wait', gr['label'])} "
             + " ".join(_badge(pkg['sides'][s]['status']['state'], f"{s.title()}: {pkg['sides'][s]['status']['label']}")
                        for s in ("offense", "defense")) + "</p>",
             "<div class='flow'><span>1 Generate ✓</span><span>2 Review this page</span><span>3 Build in Madden</span>"
             "<span>4 Confirm each side</span><span>5 Ready to play</span></div>",
             "<h2>Summary</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in pkg["summary"]) + "</ul>"]
    for side in ("offense", "defense"):
        s = pkg["sides"][side]
        d = s["design"]
        st = s["steps"]
        parts.append(f"<h2>{side.title()} playbook</h2>")
        parts.append(f"<p>{_badge(s['status']['state'], s['status']['label'])} "
                     f"<span class='muted'>Proposal {e(s['proposal_id'])} · fingerprint <b>{e(s['fingerprint'])}</b> · "
                     f"currently callable: {e(str(s['applied']['name']))} rev {e(str(s['applied']['rev']))}</span></p>")
        if side == "defense" and pkg.get("cpu_defense_note"):
            parts.append(f"<p class='muted'>{e(pkg['cpu_defense_note'])}</p>")
        rows = "".join(
            f"<tr><td><span class='tag {f['status']}'>{f['status']}</span></td><td>{e(f['formation'])}</td>"
            f"<td>{e(f['source_book'])}</td><td>{f['n_plays']}</td><td>{e(f['why'])}</td></tr>" for f in d["formations"])
        rows += "".join(
            f"<tr><td><span class='tag REMOVE'>REMOVE</span></td><td>{e(r['formation'])}</td>"
            f"<td>{e(r['source_book'])}</td><td>{r['n_plays']}</td><td>{e(r['why'])}</td></tr>" for r in d["removed"])
        parts.append("<div class='card'><table><tr><th></th><th>Formation</th><th>Source book</th><th>Plays</th>"
                     f"<th>Why</th></tr>{rows}</table></div>")
        parts.append("<div class='card'><b>Build it in Madden</b><ol>" +
                     "".join(f"<li>{e(x)}</li>" for x in st["steps"]) + "</ol>"
                     f"<details><summary>Delete from the base copy ({e(str(st['base_book']))}): "
                     f"{len(st['delete_from_base'])} formations</summary><ul class='plays'>" +
                     "".join(f"<li><label><input type='checkbox' data-k='{side}-del-{e(f)}'> {e(f)}</label></li>"
                             for f in st["delete_from_base"]) +
                     f"</ul></details><div class='muted'>{e(st['source'])}</div></div>")
        by_form = {x["formation"]: x for x in st["formations"]}
        for f in d["formations"]:
            fx = by_form[f["formation"]]
            prof = f.get("profile") or {}
            trait = (f"{prof.get('runs', 0)} runs / {prof.get('passes', 0)} passes" if side == "offense" else
                     ", ".join(f"{k} {v}" for k, v in sorted((prof.get('coverage_families') or {}).items(),
                                                             key=lambda kv: -kv[1])))
            parts.append(
                f"<div class='card'><label><input type='checkbox' data-k='{side}-form-{e(f['formation'])}'> "
                f"<span class='tag {f['status']}'>{f['status']}</span><b>{e(f['formation'])}</b></label> "
                f"<span class='muted'>from {e(f['source_book'])} · {f['n_plays']} plays · {e(str(f.get('personnel')))} · "
                f"{e(trait)}</span>"
                f"<div><b>Where:</b> {e(f['editor_path']['formation'])} › {e(f['editor_path']['set'])} › "
                f"{e(f['source_book'])} plays. {e(fx['action'])}</div>"
                + (f"<div><b>Remove from the base version:</b> {e(', '.join(fx['remove_plays']))}</div>"
                   if fx["remove_plays"] else "")
                + "<ul class='plays'>" + "".join(
                    f"<li><label><input type='checkbox' data-k='{side}-{e(f['formation'])}-{e(p)}'> {e(p)}</label></li>"
                    for p in f["plays"]) + "</ul></div>")
        parts.append(f"<div class='card noprint'><b>After building, confirm {side}:</b><pre>{e(confirm_command(pkg, side))}</pre>"
                     "<div class='muted'>This page cannot confirm anything. Only the command above, typed by you, "
                     "changes what the coach may call.</div></div>")
    parts.append("<h2>Football reasoning</h2>")
    for side in ("offense", "defense"):
        d = pkg["sides"][side]["design"]
        obj = d["objective"]
        parts.append(f"<div class='card'><b>{side.title()}</b><p class='muted'>Portfolio score: current {obj['current']}, "
                     f"proposed {obj['proposed']}, best from scratch {obj['ideal']}. A change must gain more than "
                     f"{obj['switch_threshold']} (evidence quality {e(str(obj['evidence_quality']))}, "
                     f"{obj['evidence_rows']} evidence rows). {len(d['ideal'])} formations in the from-scratch plan.</p>")
        if d["moves"]:
            parts.append("<ul>" + "".join(
                f"<li>{e(m['type'])}: {e(m.get('removed') or '')}{' → ' if m.get('removed') and m.get('added') else ''}"
                f"{e(m.get('added') or '')} (gain {m.get('gain')}) {e('; '.join(m.get('situations') or []))}</li>"
                for m in d["moves"]) + "</ul>")
        else:
            parts.append("<p>No change clears the threshold; the installed book stays as it is.</p>")
        if d["deferred"]:
            parts.append("<p class='muted'>Considered but not worth the change yet: " + e("; ".join(
                f"{x['formation']} ({x['why_not']})" for x in d["deferred"])) + "</p>")
        w = obj["situation_weights"]
        parts.append("<p class='muted'>Situation weights: " + e(", ".join(f"{k} {v:.2f}" for k, v in w.items())) + "</p></div>")
    ev = pkg["evidence"]["defense"]
    if pkg["concentration_alerts"] or ev["opponent_profile"].get("n"):
        parts.append("<div class='card'><b>Defense vs this opponent</b><ul>")
        if ev["opponent_profile"].get("n"):
            parts.append(f"<li>Tendencies: {e(str(ev['opponent_profile']['summary']))}</li>")
        for k, v in (ev.get("situation_allowed") or {}).items():
            parts.append(f"<li>{e(k)}: opponent converted {round(100 * v['rate'])}% of {v['n']} logged snaps</li>")
        for a in pkg["concentration_alerts"]:
            parts.append(f"<li>Logged calls last game: {e(json.dumps(a['formations']))} - "
                         f"{round(100 * a['top_share'])}% one formation (recommendations; execution unverified)</li>")
        parts.append("</ul></div>")
    games = pkg["games_vs_opponent"] or pkg["latest_games"]
    if games:
        parts.append("<div class='card'><b>Recent results and who called them</b><table><tr><th>Date</th><th>Opponent</th>"
                     "<th>Result</th><th>Offense caller</th><th>Defense caller</th></tr>" + "".join(
                         f"<tr><td>{e(str(g['started_ts'])[:10])}</td><td>{e(str(g['opponent_id']))}</td>"
                         f"<td>{e(str(g['result_wl']))} {e(str(g['score']))}"
                         f"{' (corrected from ' + e(str(g['stored_score'])) + ')' if g['corrected'] else ''}</td>"
                         f"<td>{e(g['callers']['offense'])}</td><td>{e(g['callers']['defense'])}</td></tr>"
                         for g in games) + "</table></div>")
    audit = pkg["catalog_audit"]
    parts.append("<h2>Confidence and limitations</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in pkg["limitations"]) + "</ul>")
    parts.append("<div class='card'><b>Catalog coverage</b><p>" + e(audit["claim"]) + "</p><ul>" + "".join(
        f"<li>{e(s)}: {a['books_catalogued']} books, {a['book_formation_pairs']} book-formation pairs, "
        f"{a['plays_listed']} play listings, {a['unique_formations']} distinct formations; index coverage "
        f"{a['formation_index_covered']}/{a['formation_index_size']}</li>" for s, a in audit["sides"].items()) +
        f"</ul><div class='muted'>Huddle.gg accessed {e(str(audit['accessed']))}.</div></div>")
    parts.append("<h2 class='noprint'>Restore the previous books</h2><div class='card noprint'><pre>" +
                 e(rollback_command(pkg, "offense")) + "\n" + e(rollback_command(pkg, "defense")) + "</pre>"
                 "<div class='muted'>Rebuild or reselect the previous book in Madden first, then run the matching command.</div></div>")
    parts.append(f"</main><script>{_JS}</script></body></html>")
    return "".join(parts)


def write_outputs(pkg: Mapping[str, Any], root: Path | str | None = None) -> dict[str, str]:
    folder = playbook_install.proposal_dir(root, pkg["opponent"]["id"])
    folder.mkdir(parents=True, exist_ok=True)
    page = folder / "game_prep.html"
    md = folder / "game_prep_checklist.md"
    data = folder / "game_prep.json"
    page.write_text(render_html(pkg), encoding="utf-8")
    md.write_text(render_markdown(pkg), encoding="utf-8")
    data.write_text(json.dumps(pkg, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return {"html": str(page), "markdown": str(md), "json": str(data)}


__all__ = ["build_steps", "confirm_command", "game_ready", "render_html", "render_markdown", "rollback_command",
           "run_game_prep", "write_outputs"]
