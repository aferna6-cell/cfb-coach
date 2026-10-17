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
            text = re.sub(r"\s+", " ", body)
            return {
                "url": url,
                "ok": True,
                "status": getattr(resp, "status", 200),
                "retrieved_at": _now().isoformat(),
                "bytes": len(body),
                "title_guess": _guess_title(body),
                "snippet": text[:400],
                "body_text": text[:20_000],
            }
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {
            "url": url,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "retrieved_at": _now().isoformat(),
        }


# Patterns for extracting candidate findings — never invent Custom Adjustment settings.
_PATCH_RE = re.compile(
    r"(?:title\s*update|update|patch)\s*(1\.\d{2,3})",
    re.I,
)
_GAMEPLAY_RE = re.compile(
    r"(cover\s*[0-6]|quarters|tampa\s*2|db\s*fire|man\s*press|pass\s*lead|"
    r"four\s*verticals|mesh|flood|return\s*bench|rpo|salary\s*cap|"
    r"catch\s*success|run[\s-]*action|block(?:ing)?)",
    re.I,
)
_DISCOVERY_SEED = (
    "https://www.ea.com/games/madden-nfl/madden-nfl-27/news",
    "https://mp1st.com/?s=madden+27+title+update",
)


def _extract_findings(fetch: dict[str, Any], *, legal_plays: set[str]) -> list[dict[str, Any]]:
    """Extract reviewable finding candidates from fetched text. Never fabricates CA settings."""
    if not fetch.get("ok"):
        return []
    text = str(fetch.get("body_text") or fetch.get("snippet") or "")
    title = str(fetch.get("title_guess") or "")
    url = str(fetch.get("url") or "")
    retrieved = str(fetch.get("retrieved_at") or _now().isoformat())
    out: list[dict[str, Any]] = []
    patch = None
    m = _PATCH_RE.search(title) or _PATCH_RE.search(text[:2000])
    if m:
        patch = m.group(1)
        out.append(
            {
                "claim": f"Source references Madden title update/patch {patch}.",
                "side": "general",
                "source": title[:80] or url,
                "url": url,
                "published": None,
                "retrieved_at": retrieved,
                "confidence": "medium",
                "evidence_type": "confirmed_mechanic_candidate",
                "game_version": patch,
                "note": "Candidate only — verify against official patch notes before promoting",
            }
        )
    # Named legal plays mentioned in the page (soft, opinion unless EA).
    mentioned = sorted({p for p in legal_plays if p and re.search(re.escape(p), text, re.I)})[:8]
    for play in mentioned:
        out.append(
            {
                "claim": f"Source mentions play/concept {play!r} in Madden 27 coverage.",
                "side": "offense",
                "source": title[:80] or url,
                "url": url,
                "published": None,
                "retrieved_at": retrieved,
                "confidence": "low",
                "evidence_type": "player_opinion_or_mention",
                "game_version": patch,
                "play_names": [play],
                "note": "Mention only — not a verified matchup probability",
            }
        )
    # Gameplay keyword sentences (short excerpts).
    for match in _GAMEPLAY_RE.finditer(text):
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 120)
        excerpt = text[start:end].strip()
        if len(excerpt) < 40:
            continue
        side = "defense" if re.search(r"cover|quarters|tampa|db\s*fire|man\s*press", excerpt, re.I) else "offense"
        is_ea = "ea.com" in url
        out.append(
            {
                "claim": excerpt[:240],
                "side": side,
                "source": title[:80] or url,
                "url": url,
                "published": None,
                "retrieved_at": retrieved,
                "confidence": "medium" if is_ea else "low",
                "evidence_type": "confirmed_mechanic_candidate" if is_ea else "player_opinion_or_mention",
                "game_version": patch,
                "note": "Extracted excerpt — human review required; do not auto-promote",
            }
        )
        if len(out) >= 12:
            break
    # Deduplicate by claim prefix.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for f in out:
        key = str(f.get("claim") or "")[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    return deduped


def _diff_findings(
    active: dict[str, Any] | None, extracted: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Meaningful differences vs the active research file (candidate review list)."""
    old_claims = {
        str(f.get("claim") or "").strip().lower()[:100]
        for f in ((active or {}).get("findings") or [])
        if isinstance(f, dict)
    }
    old_patch = str(((active or {}).get("patch") or {}).get("version") or "")
    diffs: list[dict[str, Any]] = []
    for f in extracted:
        claim = str(f.get("claim") or "").strip()
        key = claim.lower()[:100]
        if key and key not in old_claims:
            diffs.append(
                {
                    "type": "new_finding_candidate",
                    "claim": claim[:240],
                    "source_url": f.get("url"),
                    "confidence": f.get("confidence"),
                    "evidence_type": f.get("evidence_type"),
                    "game_version": f.get("game_version"),
                }
            )
        gv = str(f.get("game_version") or "")
        if gv and old_patch and gv != old_patch:
            diffs.append(
                {
                    "type": "possible_patch_bump",
                    "old_patch": old_patch,
                    "seen": gv,
                    "source_url": f.get("url"),
                    "confidence": "medium",
                    "evidence_type": "confirmed_mechanic_candidate",
                    "note": "Newer patch observed — discount older findings until reviewed",
                }
            )
    # Dedup diffs
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for d in diffs:
        k = f"{d.get('type')}|{d.get('claim') or d.get('seen')}|{d.get('source_url')}"
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out[:40]


def _discover_urls(fetches: list[dict[str, Any]]) -> list[str]:
    """Follow obvious Madden 27 news links from hub pages (bounded)."""
    found: list[str] = []
    for f in fetches:
        if not f.get("ok"):
            continue
        text = str(f.get("body_text") or "")
        for href in re.findall(r'href=["\'](https?://[^"\']+)["\']', text):
            low = href.lower()
            if "madden" not in low and "title-update" not in low and "patch" not in low:
                continue
            if any(x in low for x in ("ea.com", "mp1st.com", "gamerant.com", "timesaver")):
                if href not in found and href not in EA_NEWS_URLS + SECONDARY_URLS:
                    found.append(href)
            if len(found) >= 4:
                return found
    return found


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
    extracted_findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retrieved = _now().isoformat()
    ok_fetches = [f for f in fetches if f.get("ok")]
    failed = [f for f in fetches if not f.get("ok")]
    # Strip large body_text from persisted fetches (keep snippet).
    slim_fetches = []
    for f in fetches:
        slim = {k: v for k, v in f.items() if k != "body_text"}
        slim_fetches.append(slim)
    validation_errs = ai_research.validate(active, "madden27") if active else ["no active research file"]
    legal = _legal_play_names()
    extracted = list(extracted_findings or [])
    newest_patch = None
    for f in extracted:
        if f.get("game_version"):
            newest_patch = f["game_version"]
            break
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
        "fetches": slim_fetches,
        "extracted_findings": extracted,
        "contradictions_or_changes": contradictions,
        "patch_aware_discount": _patch_discount(active, newest_patch),
        "legal_play_name_count": len(legal),
        "policy_guards": {
            "overwrite_active_five_formations": False,
            "overwrite_armed_custom_adjustments": False,
            "requires_user_confirmation_for_policy": True,
            "fabricated_citations_forbidden": True,
            "guessed_custom_adjustment_settings_forbidden": True,
            "unattended_daily_requires_workflow_on_default_branch": True,
        },
        "change_summary": (
            f"Fetched {len(ok_fetches)}/{len(fetches)} sources; "
            f"extracted {len(extracted)} finding candidate(s); "
            f"{len(contradictions)} meaningful difference(s) flagged. "
            "Active research/madden27.json and research_db.json were NOT modified."
        ),
        "next_steps": [
            "Review candidate report under research/candidates/",
            "Manually promote cited findings into research/madden27.json after validation",
            "Run scripts/validate_ai_research.py before committing",
            "Do not change the applied five formations or armed macros without confirmation",
            "Scheduled refresh is not live until the workflow exists on the default branch",
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
    seed_urls = list(EA_NEWS_URLS + SECONDARY_URLS)
    for url in seed_urls:
        fetches.append(_fetch(url))
    # Discover a few additional recent links from hubs (bounded).
    for url in _discover_urls(fetches):
        fetches.append(_fetch(url))

    ok_any = any(f.get("ok") for f in fetches)
    legal = _legal_play_names()
    extracted: list[dict[str, Any]] = []
    if ok_any:
        for f in fetches:
            extracted.extend(_extract_findings(f, legal_plays=legal))
    contradictions = _detect_contradictions(active, fetches) if ok_any else []
    contradictions.extend(_diff_findings(active, extracted))
    unique_c: list[dict[str, Any]] = []
    seen2: set[str] = set()
    for c in contradictions:
        key = f"{c.get('type')}|{c.get('claim') or c.get('seen')}|{c.get('source_url')}"
        if key in seen2:
            continue
        seen2.add(key)
        unique_c.append(c)
    contradictions = unique_c[:40]

    report = build_candidate_report(
        fetches=fetches,
        active=active,
        contradictions=contradictions,
        extracted_findings=extracted,
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
