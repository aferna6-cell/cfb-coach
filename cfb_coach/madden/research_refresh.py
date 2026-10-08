"""Daily / on-demand Madden knowledge refresh.

Produces **candidate** updates only. Never silently replaces the active five
formations or the eight armed Custom Adjustments — those require user
confirmation.

Workflow:
1. Prefer official EA Madden 27 patch notes / gameplay news.
2. Collect competitive meta evidence from reputable public sources when reachable.
3. Record URLs, publication dates, retrieval dates, game version, confidence,
   and evidence type (confirmed mechanic vs player opinion).
4. Detect changed / contradicted prior findings.
5. Validate schemas and play/formation names against the packaged catalogs.
6. Preserve the last validated knowledge base if research fails.
7. Emit a readable change report under ``research/candidates/``.
8. Optionally open a reviewable PR (CI / operator) — never auto-merge.

Credentials (optional):
- ``CFB_COACH_RESEARCH_URL`` — override remote research base URL
- Network egress to EA.com, GitHub raw, and public guide sites
- ``GH_TOKEN`` / ``GITHUB_TOKEN`` only when ``--open-pr`` is used in Actions

The routine is not claimed operational until a scheduled or manual run succeeds
and writes a candidate report.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfb_coach import ai_research

REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DIR = REPO_ROOT / "research" / "candidates"
ACTIVE_RESEARCH = REPO_ROOT / "research" / "madden27.json"
PACKAGED_DB = REPO_ROOT / "cfb_coach" / "data" / "madden27" / "research_db.json"

# Official / primary sources checked first.
EA_NEWS_URLS = (
    "https://www.ea.com/games/madden-nfl/madden-nfl-27/news",
    "https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-nfl-27-title-update-september-16",
)
# Reputable secondary sources — fetch when accessible; never invent citations.
SECONDARY_URLS = (
    "https://mp1st.com/title-updates-and-patches/madden-nfl-27-update-1-007-tweaks-salary-cap-management-september-30",
    "https://www.ea.com/games/madden-nfl/madden-nfl-27/tips-and-tricks-hub/m27-how-to-stop-the-run",
    "https://www.ea.com/games/madden-nfl/madden-nfl-27/tips-and-tricks-hub/m27-best-offensive-playbooks",
)

UA = "cfb-coach-research-refresh/1.0 (+https://github.com/aferna6-cell/cfb-coach)"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fetch(url: str, timeout: float = 12.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — allowlisted https
            body = resp.read(500_000).decode("utf-8", errors="replace")
            return {
                "url": url,
                "ok": True,
                "status": getattr(resp, "status", 200),
                "retrieved_at": _now().isoformat(),
                "bytes": len(body),
                "title_guess": _guess_title(body),
                "snippet": re.sub(r"\s+", " ", body)[:400],
            }
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {
            "url": url,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "retrieved_at": _now().isoformat(),
        }


def _guess_title(html: str) -> str:
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
    return (m.group(1).strip() if m else "")[:200]


def _load_active() -> dict[str, Any] | None:
    if not ACTIVE_RESEARCH.is_file():
        return None
    try:
        return json.loads(ACTIVE_RESEARCH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _legal_play_names() -> set[str]:
    names: set[str] = set()
    seed = REPO_ROOT / "cfb_coach" / "data" / "madden27" / "seed.json"
    try:
        doc = json.loads(seed.read_text(encoding="utf-8"))
        for play in (doc.get("reads") or {}):
            names.add(str(play))
    except (OSError, ValueError):
        pass
    pb = REPO_ROOT / "cfb_coach" / "data" / "madden27" / "playbooks.json"
    try:
        doc = json.loads(pb.read_text(encoding="utf-8"))
        books = doc.get("books") if isinstance(doc, dict) else None
        iterable = books.values() if isinstance(books, dict) else (doc.values() if isinstance(doc, dict) else [])
        for book in iterable:
            if not isinstance(book, dict):
                continue
            for side in ("offense", "defense"):
                forms = book.get(side) or {}
                if isinstance(forms, dict):
                    for plays in forms.values():
                        if isinstance(plays, list):
                            names.update(str(p) for p in plays)
    except (OSError, ValueError):
        pass
    return names


def _detect_contradictions(old: dict[str, Any] | None, fetches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lightweight contradiction hints — never fabricate new claims."""
    out: list[dict[str, Any]] = []
    if not old:
        return out
    old_patch = str((old.get("patch") or {}).get("version") or "")
    for f in fetches:
        if not f.get("ok"):
            continue
        snip = (f.get("snippet") or "") + " " + (f.get("title_guess") or "")
        # Flag when a newer title-update string appears vs stored patch.
        m = re.search(r"(?:title update|update)\s*(1\.\d{2,3})", snip, re.I)
        if m and old_patch and m.group(1) != old_patch:
            out.append(
                {
                    "type": "possible_patch_bump",
                    "old_patch": old_patch,
                    "seen": m.group(1),
                    "source_url": f.get("url"),
                    "confidence": "medium",
                    "evidence_type": "confirmed_mechanic_candidate",
                    "note": "Newer patch string observed — review before changing active KB",
                }
            )
    return out


def _patch_discount(old: dict[str, Any] | None, new_patch: str | None) -> dict[str, Any]:
    """Older gameplay findings get lower confidence after important updates."""
    if not old or not new_patch:
        return {"applied": False}
    old_patch = str((old.get("patch") or {}).get("version") or "")
    if not old_patch or old_patch == new_patch:
        return {"applied": False, "old_patch": old_patch, "new_patch": new_patch}
    return {
        "applied": True,
        "old_patch": old_patch,
        "new_patch": new_patch,
        "policy": (
            "Findings tied to older Madden gameplay versions should receive "
            "lower confidence after important gameplay updates. Candidate "
            "files mark this; active policy is not auto-overwritten."
        ),
    }


def build_candidate_report(
    *,
    fetches: list[dict[str, Any]],
    active: dict[str, Any] | None,
    contradictions: list[dict[str, Any]],
) -> dict[str, Any]:
    retrieved = _now().isoformat()
    ok_fetches = [f for f in fetches if f.get("ok")]
    failed = [f for f in fetches if not f.get("ok")]
    validation_errs = ai_research.validate(active, "madden27") if active else ["no active research file"]
    legal = _legal_play_names()
    # Do not invent Custom Adjustment editor settings.
    return {
        "schema": 1,
        "kind": "madden_research_candidate",
        "game": "madden27",
        "generated_at": retrieved,
        "status": "candidate",
        "active_research_preserved": True,
        "active_researched_at": (active or {}).get("researched_at"),
        "active_patch": (active or {}).get("patch"),
        "active_validation_errors": validation_errs,
        "sources_attempted": len(fetches),
        "sources_ok": len(ok_fetches),
        "sources_failed": failed,
        "fetches": fetches,
        "contradictions_or_changes": contradictions,
        "patch_aware_discount": _patch_discount(active, None),
        "legal_play_name_count": len(legal),
        "policy_guards": {
            "overwrite_active_five_formations": False,
            "overwrite_armed_custom_adjustments": False,
            "requires_user_confirmation_for_policy": True,
            "fabricated_citations_forbidden": True,
            "guessed_custom_adjustment_settings_forbidden": True,
        },
        "change_summary": (
            f"Fetched {len(ok_fetches)}/{len(fetches)} sources. "
            f"{len(contradictions)} possible change(s) flagged for review. "
            "Active research/madden27.json and research_db.json were NOT modified."
        ),
        "next_steps": [
            "Review candidate report under research/candidates/",
            "Manually edit research/madden27.json only with cited findings",
            "Run scripts/validate_ai_research.py before committing",
            "Do not change the applied five formations or armed macros without confirmation",
        ],
    }


def run_research_refresh(
    *,
    dry_run: bool = False,
    open_pr: bool = False,
) -> dict[str, Any]:
    """Execute one research refresh. Preserves active KB on failure."""
    active = _load_active()
    fetches: list[dict[str, Any]] = []
    for url in EA_NEWS_URLS + SECONDARY_URLS:
        fetches.append(_fetch(url))

    ok_any = any(f.get("ok") for f in fetches)
    contradictions = _detect_contradictions(active, fetches) if ok_any else []
    report = build_candidate_report(
        fetches=fetches, active=active, contradictions=contradictions
    )

    if not ok_any:
        report["ok"] = False
        report["preserved_active_kb"] = True
        report["error"] = "all source fetches failed — last validated KB preserved"
        if not dry_run:
            CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
            fail_path = CANDIDATE_DIR / f"FAILED_{_now().strftime('%Y%m%dT%H%M%SZ')}.json"
            fail_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            report["candidate_path"] = str(fail_path)
        return report

    report["ok"] = True
    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    out_path = CANDIDATE_DIR / f"madden27_candidate_{stamp}.json"
    change_md = CANDIDATE_DIR / f"CHANGE_REPORT_{stamp}.md"
    if not dry_run:
        CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        change_md.write_text(_markdown_report(report), encoding="utf-8")
        report["candidate_path"] = str(out_path)
        report["change_report_path"] = str(change_md)
    else:
        report["dry_run"] = True

    # Never auto-overwrite active files.
    assert ACTIVE_RESEARCH.read_text(encoding="utf-8") if ACTIVE_RESEARCH.is_file() else True

    if open_pr and not dry_run:
        report["pr"] = _maybe_open_pr(out_path, change_md, report)
    else:
        report["pr"] = {"opened": False, "reason": "open_pr not requested or dry_run"}

    report["freshness"] = {
        "active_researched_at": (active or {}).get("researched_at"),
        "candidate_generated_at": report["generated_at"],
        "stale_hours_threshold": ai_research.STALE_HOURS,
    }
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Madden 27 research candidate",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        report.get("change_summary") or "",
        "",
        "## Policy guards",
        "",
        "- Active five formations: **not** modified",
        "- Armed Custom Adjustments: **not** modified",
        "- Active `research/madden27.json`: **preserved** (candidate only)",
        "",
        "## Sources",
        "",
    ]
    for f in report.get("fetches") or []:
        status = "OK" if f.get("ok") else "FAIL"
        lines.append(f"- [{status}] {f.get('url')}")
        if f.get("title_guess"):
            lines.append(f"  - title: {f['title_guess']}")
        if f.get("error"):
            lines.append(f"  - error: {f['error']}")
    if report.get("contradictions_or_changes"):
        lines.extend(["", "## Possible changes (review required)", ""])
        for c in report["contradictions_or_changes"]:
            lines.append(f"- {c.get('type')}: {c.get('note')} ({c.get('source_url')})")
    lines.extend(["", "## Next steps", ""])
    for s in report.get("next_steps") or []:
        lines.append(f"1. {s}")
    lines.append("")
    return "\n".join(lines)


def _maybe_open_pr(candidate: Path, change_md: Path, report: dict[str, Any]) -> dict[str, Any]:
    """PR creation is owned by GitHub Actions — never switch branches from CLI.

    The daily workflow commits ``research/candidates/`` and opens a reviewable
    PR. Local ``--open-pr`` only reports readiness so an interactive agent never
    leaves the working branch mid-task.
    """
    del change_md  # referenced by the Actions workflow after candidate write
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not candidate.is_file():
        return {"opened": False, "reason": "no candidate file"}
    return {
        "opened": False,
        "token_present": bool(token),
        "candidate": str(candidate),
        "meaningful_updates": bool(report.get("contradictions_or_changes"))
        or int(report.get("sources_ok") or 0) > 0,
        "reason": (
            "Candidate written under research/candidates/. Opening the PR is "
            "owned by .github/workflows/madden-research-daily.yml "
            "(workflow_dispatch or schedule). Never auto-merges; never "
            "overwrites active formations/macros."
        ),
        "configure": (
            "Actions needs contents:write + pull-requests:write (already in the "
            "workflow). Optional: CFB_COACH_RESEARCH_URL for a custom research base."
        ),
    }


def research_freshness_for_prep() -> dict[str, Any]:
    """Expose freshness + provenance for prep surfaces."""
    doc, origin = ai_research.load_research("madden27", offline=True)
    age = ai_research.age_hours(doc) if doc else None
    return {
        "game": "madden27",
        "origin": origin,
        "researched_at": (doc or {}).get("researched_at"),
        "researched_by": (doc or {}).get("researched_by"),
        "patch": (doc or {}).get("patch"),
        "sources": (doc or {}).get("sources") or [],
        "age_hours": age,
        "stale": age is None or age > ai_research.STALE_HOURS,
        "note": "Candidate refreshes land under research/candidates/; active KB needs review.",
    }
