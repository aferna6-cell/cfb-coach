"""AceMadden-compatible catalogue ingestion: permitted public metadata discovery + authorized export import, with layout-drift guards.

Safety design (so a website layout change cannot corrupt the catalogue):
  * every run parses into a STAGING list; it is validated as a whole and only then written atomically (temp file + rename);
  * rows that fail structural checks go to ``quarantine.jsonl`` with the reason, never into ``catalog.jsonl``;
  * a run aborts (writes nothing) if a page yields no tiles, or more than ``MAX_QUARANTINE_FRACTION`` of its tiles are bad;
  * sync never deletes: records that stop appearing are marked ``not_seen`` and the change is appended to ``history.jsonl``.

Access rules enforced here: robots.txt is honoured, requests are rate-limited and identify the tool, no login/cookie/auth is ever sent, 401/403
stops the run, 429/5xx back off. Anything behind a login (membership) is out of reach by design. Raw third-party images are written only to the
private store directory, which must not be inside a git work tree unless explicitly overridden.
"""
from __future__ import annotations

import csv
import hashlib
import html as htmllib
import json
import os
import re
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

SCHEMA = "madden.catalog_record.v1"
BASE = "https://acemadden.com"
USER_AGENT = "cfb-coach-catalog/1.0 (owner-run research tool; contact: aidanfernandes31@gmail.com)"
SIDES = ("offense", "defense")
MAX_QUARANTINE_FRACTION = 0.2
PERMISSION_CONTACT = "support@acemadden.com"
EVIDENCE_TIERS = ("unknown", "football_inference", "source_documented", "diagram_extracted_unreviewed", "diagram_verified", "in_game_verified")
VERIFIED = ("diagram_verified", "in_game_verified")


SCOPES = ("metadata_bulk", "images_download", "images_local_storage", "ml_training", "redistribution")


def load_authorization(store: "Store") -> dict[str, Any] | None:
    p = store.root / "authorization.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def record_authorization(store: "Store", *, sender: str, date: str, scopes: Iterable[str], restrictions: str, private_reference: str, summary: str) -> dict[str, Any]:
    """Store a MINIMAL authorization record: who/when, which scopes the grant actually covers, restrictions, and a private pointer to the email.
    ``scopes`` must come from what the email says; redistribution is never implied by any other scope."""
    scopes = sorted(set(scopes))
    bad = [x for x in scopes if x not in SCOPES]
    if bad:
        raise ValueError(f"unknown scope(s) {bad}; choose from {SCOPES}")
    if not (sender and date and summary and private_reference):
        raise ValueError("sender, date, summary and private_reference are all required (no invented or empty authorizations)")
    rec = {"sender": sender, "date": date, "scopes": scopes, "restrictions": restrictions, "private_reference": private_reference,
           "summary": summary, "recorded_at": now(), "note": "summary written by the owner from the actual email; the email itself is not stored here"}
    (store.root / "authorization.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def require_scope(store: "Store", scope: str) -> dict[str, Any]:
    a = load_authorization(store)
    if not a or scope not in a["scopes"]:
        raise PermissionError(f"no recorded authorization covers {scope!r}; run `authorize` with what the email actually grants (or stop)")
    return a


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def record_id(game: str, side: str, formation_slug: str, play_slug: str) -> str:
    return f"{game}|{side}|{formation_slug}|{play_slug}"


def content_hash(rec: Mapping[str, Any]) -> str:
    keep = {k: rec.get(k) for k in ("game", "side", "formation_slug", "play_slug", "play_name", "personnel_group", "playbooks", "art_url")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------------------ store
class Store:
    def __init__(self, root: str | Path, *, allow_in_repo: bool = False):
        self.root = Path(root).expanduser().resolve()
        if not allow_in_repo and _inside_git_tree(self.root):
            raise ValueError(f"{self.root} is inside a git work tree: third-party material must stay outside the repository (pass allow_in_repo to override)")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "raw").mkdir(exist_ok=True)

    @property
    def catalog(self) -> Path:
        return self.root / "catalog.jsonl"

    def read(self) -> dict[str, dict[str, Any]]:
        if not self.catalog.is_file():
            return {}
        return {r["id"]: r for r in map(json.loads, self.catalog.read_text(encoding="utf-8").splitlines()) if r}

    def write_atomic(self, recs: Mapping[str, Mapping[str, Any]]) -> None:
        tmp = self.catalog.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(recs[k], sort_keys=True) for k in sorted(recs)) + ("\n" if recs else ""), encoding="utf-8")
        os.replace(tmp, self.catalog)

    def append(self, name: str, row: Mapping[str, Any]) -> None:
        with (self.root / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _inside_git_tree(p: Path) -> bool:
    return any((a / ".git").exists() for a in [p, *p.parents])


# ------------------------------------------------------------------------------------------------------------------------ canonical record
def new_record(*, game: str, version: str, side: str, formation_slug: str, play_slug: str, play_name: str, formation_label: str | None = None,
               personnel_group: str | None = None, playbooks: Iterable[str] = (), detail_url: str | None = None, art_url: str | None = None,
               source: str = "acemadden", source_version: str = "", permission: Mapping[str, Any] | None = None, retrieved_at: str | None = None) -> dict[str, Any]:
    rec = {
        "schema": SCHEMA, "id": record_id(game, side, formation_slug, play_slug), "game": game, "version": version, "side": side,
        "playbooks": sorted(set(playbooks)), "formation_slug": formation_slug, "formation_label": formation_label, "personnel_group": personnel_group,
        "play_slug": play_slug, "play_name": play_name,
        "source": {"name": source, "detail_url": detail_url, "version": source_version, "retrieved_at": retrieved_at or now()},
        "art_url": art_url, "art_sha256": None,
        "permitted_use": dict(permission or {"status": "unrecorded"}),
        # football knowledge: nothing is asserted until evidence exists
        "offense": {"formation_personnel": None, "assignments": None, "read_progression": None, "motion": None, "run_blocking": None, "hot_routes": None}
        if side == "offense" else None,
        "defense": {"front_personnel": None, "coverage_family": None, "variant": None, "responsibilities": None, "pressure": None, "run_fits": None}
        if side == "defense" else None,
        "evidence": {"tier": "unknown", "basis": "catalogue listing only; names and diagrams are not assignments"},
        "status": "active",
    }
    rec["content_hash"] = content_hash(rec)
    return rec


# ------------------------------------------------------------------------------------------------------------------------ parsing (layout-guarded)
TILE_RE = re.compile(r'<a [^>]*href="(/playbooks/plays/(?P<game>madden\d+)/(?P<side>offense|defense)/(?P<f>[a-z0-9-]+)/(?P<p>[a-z0-9-]+))"[^>]*>(?P<body>.*?)</a>', re.S)
IMG_RE = re.compile(r'<img [^>]*src="(?P<src>/play-art/(?P<game>madden\d+)/(?P<side>offense|defense)/(?P<f>[a-z0-9-]+)__(?P<p>[a-z0-9-]+)\.png)"[^>]*alt="(?P<alt>[^"]*)"')
TEXT_RE = re.compile(r">([^<>]{1,80})</div>")


@dataclass
class ParsedPage:
    records: list[dict[str, Any]]
    quarantined: list[dict[str, Any]]
    tiles_seen: int


def parse_listing(page_html: str, *, version: str = "", side_hint: str | None = None, permission: Mapping[str, Any] | None = None, retrieved_at: str | None = None) -> ParsedPage:
    recs, bad = [], []
    tiles = list(TILE_RE.finditer(page_html))
    for m in tiles:
        body = m.group("body")
        img = IMG_RE.search(body)
        texts = [htmllib.unescape(t).strip() for t in TEXT_RE.findall(body)]
        why = None
        if not img:
            why = "tile has no play-art image"
        elif (img.group("f"), img.group("side")) != (m.group("f"), m.group("side")):
            why = "image formation/side does not match the link (layout change or mis-linked tile)"
        elif side_hint and m.group("side") != side_hint:
            why = f"tile side {m.group('side')} does not match the requested {side_hint}"
        elif len(texts) < 1:
            why = "tile has no visible play name"
        if why:
            bad.append({"reason": why, "href": m.group(1), "snippet": body[:200]})
            continue
        name = htmllib.unescape(img.group("alt")).strip() or texts[-1]
        if texts and slug(texts[-1]) != slug(name):
            bad.append({"reason": "visible name and image alt disagree", "href": m.group(1), "snippet": f"{texts[-1]!r} vs {name!r}"})
            continue
        art_slug_differs = img.group("p") != m.group("p")     # observed on the live site (e.g. a typo in the art file name): the LINK slug is the identity
        recs.append(new_record(game=m.group("game"), version=version, side=m.group("side"), formation_slug=m.group("f"), play_slug=m.group("p"),
                               play_name=name, personnel_group=texts[0] if len(texts) > 1 else None, detail_url=BASE + m.group(1),
                               art_url=BASE + img.group("src"), source_version=version, permission=permission, retrieved_at=retrieved_at))
        if art_slug_differs:
            recs[-1]["source"]["art_slug_differs_from_link"] = True
    return ParsedPage(recs, bad, len(tiles))


def parse_detail(page_html: str) -> dict[str, Any]:
    """Title, formation label and the 'Playbooks With' memberships. Returns only what is present; unknown stays absent."""
    out: dict[str, Any] = {}
    t = re.search(r"<title>([^<]+)</title>", page_html)
    if t:
        title = htmllib.unescape(t.group(1))
        m = re.match(r"(.+?) — (.+?) \|", title)
        if m:
            out["play_name"], out["formation_label"] = m.group(1).strip(), m.group(2).strip()
    i = page_html.find("Playbooks With")
    if i >= 0:
        chunk = re.sub(r"<script.*?</script>", "", page_html[i:i + 4000], flags=re.S)
        names = [htmllib.unescape(x).strip() for x in re.findall(r">([^<>]{2,40})</(?:a|span|div|li)>", chunk)]
        out["playbooks_text_unverified"] = [n for n in names[1:12] if n and not n.lower().startswith(("playbooks", "let us", "copy", "download"))]   # structure not understood: kept as a hint, never as membership
    return out


# ------------------------------------------------------------------------------------------------------------------------ fetching (permitted, polite)
class Fetcher:
    def __init__(self, store: Store, *, delay: float = 1.5, base: str = BASE, opener=None, permission_note: str | None = None, sleep=time.sleep):
        self.store, self.delay, self.base, self.sleep = store, delay, base, sleep
        self.opener = opener or urllib.request.build_opener()
        self.permission_note = permission_note
        self._last = 0.0
        self.robots = urllib.robotparser.RobotFileParser()
        self.requests = 0
        try:
            self.robots.parse(self._get("/robots.txt", check_robots=False).decode("utf-8", "replace").splitlines())
        except Exception:                       # no robots file readable: be conservative and stay slow, not permissive about paths
            self.robots.parse(["User-agent: *", "Allow: /"])

    def _get(self, path: str, *, check_robots: bool = True, binary: bool = False) -> bytes:
        if check_robots and not self.robots.can_fetch(USER_AGENT, self.base + path):
            raise PermissionError(f"robots.txt disallows {path}")
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            self.sleep(wait)
        for attempt in range(4):
            req = urllib.request.Request(self.base + path, headers={"User-Agent": USER_AGENT})     # no cookies, no auth
            try:
                self._last = time.monotonic()
                self.requests += 1
                with self.opener.open(req, timeout=30) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise PermissionError(f"HTTP {e.code} for {path}: access restricted; not retrying or bypassing") from e
                if e.code == 404:
                    raise FileNotFoundError(path) from e
                if e.code == 429 or e.code >= 500:
                    self.sleep(self.delay * (2 ** (attempt + 1)))
                    continue
                raise
        raise RuntimeError(f"giving up on {path} after retries")

    def listing(self, game: str, side: str, page: int) -> str:
        q = f"?page={page}" + ("&side=defense" if side == "defense" else "")
        return self._get(f"/playbooks/plays/{game}{q}").decode("utf-8", "replace")

    def detail(self, rec: Mapping[str, Any]) -> str:
        return self._get(rec["source"]["detail_url"].removeprefix(self.base)).decode("utf-8", "replace")

    def art(self, rec: Mapping[str, Any]) -> bytes:
        if not self.permission_note:
            raise PermissionError("downloading play art requires --permission-note (a record of the written permission you hold)")
        return self._get(rec["art_url"].removeprefix(self.base))


def discover(fetcher: Fetcher, *, game: str = "madden27", sides: Iterable[str] = SIDES, max_pages: int | None = None, version: str = "",
             permission: Mapping[str, Any] | None = None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Walk listing pages until one repeats or is empty. Returns (records, quarantined, stats). Writes nothing."""
    out: dict[str, dict[str, Any]] = {}
    quarantine: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"pages": {}, "tiles": 0}
    for side in sides:
        seen_first, page = None, 1
        while max_pages is None or page <= max_pages:
            parsed = parse_listing(fetcher.listing(game, side, page), version=version, side_hint=side, permission=permission)
            if parsed.tiles_seen == 0:
                if page == 1:
                    raise RuntimeError(f"{side}: first listing page had no tiles; the layout probably changed. Nothing written.")
                break
            if parsed.tiles_seen and len(parsed.quarantined) / parsed.tiles_seen > MAX_QUARANTINE_FRACTION:
                raise RuntimeError(f"{side} page {page}: {len(parsed.quarantined)}/{parsed.tiles_seen} tiles failed checks; the layout probably changed. Nothing written.")
            ids = tuple(r["id"] for r in parsed.records)
            if ids and ids == seen_first:
                break
            seen_first = seen_first or ids
            fresh = [r for r in parsed.records if r["id"] not in out]
            if not fresh:
                break                       # the site serves the LAST page again for any page number past the end (observed): no new plays = done
            for r in fresh:
                out.setdefault(r["id"], r)
            quarantine += [{**q, "side": side, "page": page} for q in parsed.quarantined]
            stats["pages"][f"{side}:{page}"] = parsed.tiles_seen
            stats["tiles"] += parsed.tiles_seen
            page += 1
    stats["records"], stats["quarantined"] = len(out), len(quarantine)
    return out, quarantine, stats


# ------------------------------------------------------------------------------------------------------------------------ authorized export import
EXPORT_REQUIRED = ("side", "formation", "play")


def read_export(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".csv":
        with p.open(encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
    else:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def import_export(path: str | Path, *, game: str = "madden27", version: str = "", source: str = "authorized_export",
                  permission: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Map an authorized export (jsonl/csv: side, formation, play, playbook(s), personnel_group, art_url, optional assignments) to canonical records.

    Optional ``assignments`` (JSON text or object) is accepted ONLY with ``assignment_source`` and ``assignment_tier``; otherwise it is rejected."""
    recs, bad = {}, []
    for i, row in enumerate(read_export(path), 1):
        miss = [k for k in EXPORT_REQUIRED if not str(row.get(k) or "").strip()]
        if miss or str(row["side"]).lower() not in SIDES:
            bad.append({"row": i, "reason": f"missing/invalid {miss or 'side'}", "data": row})
            continue
        side = str(row["side"]).lower()
        books = row.get("playbooks") or row.get("playbook") or ""
        books = books if isinstance(books, list) else [b.strip() for b in str(books).split("|") if b.strip()]
        rec = new_record(game=game, version=version, side=side, formation_slug=slug(row["formation"]), play_slug=slug(row["play"]), play_name=str(row["play"]),
                         formation_label=str(row["formation"]), personnel_group=row.get("personnel_group") or None, playbooks=books,
                         art_url=row.get("art_url") or None, source=source, source_version=version, permission=permission)
        asg = row.get("assignments")
        if asg:
            tier = str(row.get("assignment_tier") or "")
            if tier not in EVIDENCE_TIERS or tier == "unknown" or not row.get("assignment_source"):
                bad.append({"row": i, "reason": "assignments need assignment_tier and assignment_source", "data": {k: row.get(k) for k in EXPORT_REQUIRED}})
                continue
            key = "offense" if side == "offense" else "defense"
            rec[key]["assignments" if side == "offense" else "responsibilities"] = json.loads(asg) if isinstance(asg, str) else asg
            rec["evidence"] = {"tier": tier, "basis": str(row["assignment_source"])}
        if rec["id"] in recs and recs[rec["id"]]["content_hash"] != rec["content_hash"]:
            recs[rec["id"]]["playbooks"] = sorted(set(recs[rec["id"]]["playbooks"]) | set(rec["playbooks"]))   # same play in another book: membership, not a new play
            recs[rec["id"]]["content_hash"] = content_hash(recs[rec["id"]])
            continue
        recs[rec["id"]] = rec
    return recs, bad


# ------------------------------------------------------------------------------------------------------------------------ sync / validate / audit
def sync(store: Store, incoming: Mapping[str, Mapping[str, Any]], *, scope_sides: Iterable[str] = SIDES) -> dict[str, Any]:
    """Reconcile incoming records into the store. Never deletes; preserves knowledge fields already earned (evidence/assignments/art hash)."""
    cur = store.read()
    scope = set(scope_sides)
    added, changed, unchanged = [], [], []
    merged = dict(cur)
    for k, rec in incoming.items():
        old = cur.get(k)
        if old is None:
            merged[k] = dict(rec)
            added.append(k)
            continue
        if old.get("content_hash") == rec["content_hash"]:
            unchanged.append(k)
            old["status"] = "active"
            continue
        new = dict(rec)
        if old.get("evidence", {}).get("tier") != "unknown" or old.get("art_sha256"):
            for keep in ("offense", "defense", "evidence", "art_sha256"):      # never overwrite earned knowledge with a bare listing
                new[keep] = old[keep]
        new["playbooks"] = sorted(set(old.get("playbooks", [])) | set(rec["playbooks"]))
        new["content_hash"] = content_hash(new)
        merged[k] = new
        changed.append(k)
    missing = [k for k, r in cur.items() if k not in incoming and r["side"] in scope and r.get("status") != "not_seen"]
    for k in missing:
        merged[k] = {**cur[k], "status": "not_seen"}
    store.write_atomic(merged)
    run = {"ts": now(), "added": len(added), "changed": len(changed), "unchanged": len(unchanged), "not_seen": len(missing)}
    store.append("history.jsonl", {**run, "changed_ids": changed[:200], "added_ids": added[:200], "not_seen_ids": missing[:200]})
    return run


def validate(recs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    errs: list[str] = []
    for k, r in recs.items():
        if r.get("schema") != SCHEMA:
            errs.append(f"{k}: schema {r.get('schema')!r}")
        if k != record_id(r["game"], r["side"], r["formation_slug"], r["play_slug"]):
            errs.append(f"{k}: id does not match its identity fields")
        if r["side"] not in SIDES or (r["side"] == "offense") == (r.get("offense") is None) or (r["side"] == "defense") == (r.get("defense") is None):
            errs.append(f"{k}: side/knowledge block mismatch")
        ev = r.get("evidence") or {}
        if ev.get("tier") not in EVIDENCE_TIERS:
            errs.append(f"{k}: unknown evidence tier {ev.get('tier')!r}")
        block = r.get("offense") or r.get("defense") or {}
        has_detail = any(v not in (None, {}, []) for v in block.values())
        if has_detail and ev.get("tier") == "unknown":
            errs.append(f"{k}: football detail present with no evidence tier")
        if ev.get("tier") in VERIFIED and not r.get("art_sha256") and "first-party" not in str(ev.get("basis", "")) and "export" not in str(ev.get("basis", "")):
            errs.append(f"{k}: verified tier without a recorded art hash or first-party/export basis")
        if r.get("permitted_use", {}).get("status") == "unrecorded" and r.get("art_sha256"):
            errs.append(f"{k}: art stored without a recorded permission")
        if r["content_hash"] != content_hash(r):
            errs.append(f"{k}: content hash is stale")
    by_art: dict[str, set[str]] = {}
    for k, r in recs.items():
        if r.get("art_sha256"):
            by_art.setdefault(r["art_sha256"], set()).add(f"{r['side']}|{r['formation_slug']}|{r['play_slug']}")
    for h, ids in by_art.items():
        if len(ids) > 1:
            errs.append(f"info: identical art {h[:10]} used by {len(ids)} different plays (equivalence needs review, not auto-merge)")
    return errs


def audit(recs: Mapping[str, Mapping[str, Any]], *, reference: Mapping[str, Iterable[tuple[str, str]]] | None = None, permission: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"generated": now(), "records": len(recs), "permission": dict(permission or {"status": "unrecorded"}), "sides": {}}
    for side in SIDES:
        rs = [r for r in recs.values() if r["side"] == side and r.get("status") != "not_seen"]
        blk = "offense" if side == "offense" else "defense"
        tiers = {t: sum(r["evidence"]["tier"] == t for r in rs) for t in EVIDENCE_TIERS}
        out["sides"][side] = {
            "catalog_entries": len(rs), "distinct_formations": len({r["formation_slug"] for r in rs}),
            "distinct_playbooks": len({b for r in rs for b in r["playbooks"]}),
            "fully_indexed": sum(bool(r["formation_slug"] and r["play_slug"] and r["play_name"] and r["source"]["detail_url"]) for r in rs),
            "with_usable_diagram": sum(bool(r.get("art_sha256")) for r in rs),
            "with_diagram_url_only": sum(bool(r.get("art_url")) and not r.get("art_sha256") for r in rs),
            "verified_individual_assignments": sum(r["evidence"]["tier"] in VERIFIED for r in rs),
            "partial_assignments": sum(any(v not in (None, {}, []) for v in (r[blk] or {}).values()) and r["evidence"]["tier"] not in VERIFIED for r in rs),
            "lacking_sufficient_knowledge": sum(r["evidence"]["tier"] in ("unknown", "football_inference") for r in rs),
            "evidence_tiers": tiers, "not_seen_in_latest_sync": sum(1 for r in recs.values() if r["side"] == side and r.get("status") == "not_seen"),
        }
    if reference is not None:
        out["discrepancies"] = {}
        for side in SIDES:
            mine = {(r["formation_slug"], r["play_slug"]) for r in recs.values() if r["side"] == side}
            ref = {(slug(f), slug(p)) for f, p in reference.get(side, ())}
            out["discrepancies"][side] = {"in_catalog_not_in_reference": len(mine - ref), "in_reference_not_in_catalog": len(ref - mine),
                                          "in_both": len(mine & ref), "note": "slug match only; different sources count different units"}
    return out
