"""Prep text and HTML for VOD beaters and playbook changes.

Absent when no model is loaded or the prior is off, so existing prep snapshots
stay put. The strings here do not include a ``CALL:`` marker.
"""

from __future__ import annotations

from typing import Any


def format_vod_section(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return []
    weight = report.get("quality_weight")
    nudge = report.get("log_nudge")
    lines = [
        f"## VOD beaters — {report.get('version')} {report.get('opponent_type')} "
        f"(VOD weight {weight:g}, log nudge {nudge:g})",
        "  Call quality uses VOD success as the primary signal. Your logged games nudge a near-tie.",
        "  Logged tendencies pick which defensive look to expect. Thin samples are display-only.",
    ]
    lines.extend(list(report.get("beater_lines") or []) or ["  (no beater rows for this opponent type)"])
    lines.append("## Playbook changes")
    changes = list(report.get("changes") or [])
    if report.get("frozen"):
        lines.append("  Book frozen. VOD may still choose among plays already in the book. Nothing was added or swapped.")
    elif not changes:
        lines.append("  No playbook changes. Samples below the n and tier bar do not add or swap plays.")
    else:
        for change in changes:
            replaced = f" — replaced {change['replaced']}" if change.get("replaced") else ""
            lines.append(f"  {change.get('action')}: {change.get('changed')}{replaced}")
            lines.append(
                f"    why: {change.get('why')} | n={change.get('n')} success={float(change.get('success') or 0):.3f} "
                f"lb={float(change.get('lower_bound') or 0):.3f} tier={change.get('tier')} "
                f"model={change.get('model_version')}"
            )
        lines.append("  Live play uses this book on the next snap. No confirmation step.")
    return lines


def render_vod_html(plan: dict[str, Any] | None) -> str:
    report = (plan or {}).get("vod_report")
    if not report:
        return ""
    from cfb_coach.prep_browser import _esc

    items = "".join(f"<li>{_esc(str(line).strip())}</li>" for line in (report.get("beater_lines") or []))
    changes = list(report.get("changes") or [])
    if report.get("frozen"):
        body = "<p>Book frozen. Nothing was added or swapped.</p>"
    elif not changes:
        body = "<p>No playbook changes. Thin samples stay display-only.</p>"
    else:
        body = "<ul>" + "".join(
            f"<li>{_esc(str(change.get('action')))}: {_esc(str(change.get('changed')))}"
            + (f" — replaced {_esc(str(change.get('replaced')))}" if change.get("replaced") else "")
            + f" (n={_esc(str(change.get('n')))}, success={_esc(str(change.get('success')))}, "
            f"lb={_esc(str(change.get('lower_bound')))}, tier={_esc(str(change.get('tier')))}, "
            f"model={_esc(str(change.get('model_version')))})</li>"
            for change in changes
        ) + "</ul><p>Live play uses this book on the next snap.</p>"
    return (
        "<section id='vod-beaters'><h2>VOD beaters</h2>"
        f"<p>VOD weight {_esc(str(report.get('quality_weight')))}, "
        f"log nudge {_esc(str(report.get('log_nudge')))}. "
        f"{_esc(str(report.get('version')))} / {_esc(str(report.get('opponent_type')))}.</p>"
        f"<ul>{items}</ul><h2>Playbook changes</h2>{body}</section>"
    )
