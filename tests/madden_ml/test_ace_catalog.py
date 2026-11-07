"""Ingestion tests on SYNTHETIC fixtures (invented names). Nothing here touches the network."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from cfb_coach.madden.model import ace_catalog as ac
from cfb_coach.madden.model import play_art_extract as pe

FX = Path(__file__).resolve().parents[1] / "fixtures" / "ace"
ROOT = Path(__file__).resolve().parents[2]
PERM = {"status": "owner_attested", "note": "test"}


def html(name):
    return (FX / name).read_text()


class FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeOpener:
    """Serves fixtures; records requests; can inject errors."""
    def __init__(self, pages, errors=None):
        self.pages, self.errors, self.log = pages, errors or {}, []

    def open(self, req, timeout=0):
        path = req.full_url.removeprefix(ac.BASE)
        self.log.append((path, req.headers.get("User-agent"), req.headers.get("Cookie"), req.headers.get("Authorization")))
        if path in self.errors:
            raise urllib.error.HTTPError(req.full_url, self.errors[path], "x", {}, None)
        if path not in self.pages:
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
        return FakeResp(self.pages[path].encode())


def fetcher(tmp_path, pages, errors=None, **kw):
    st = ac.Store(tmp_path / "store")
    pages = {"/robots.txt": "User-agent: *\nAllow: /\n", **pages}
    return st, ac.Fetcher(st, delay=0, opener=FakeOpener(pages, errors), sleep=lambda s: None, **kw)


def test_listing_parser_builds_canonical_records_and_nothing_is_asserted_beyond_the_listing():
    p = ac.parse_listing(html("listing_offense_p1.html"), version="v1", permission=PERM)
    assert [r["id"] for r in p.records] == ["madden27|offense|fixture-formation-a|test-play-one", "madden27|offense|fixture-formation-a|test-play-two"]
    r = p.records[0]
    assert r["evidence"]["tier"] == "unknown" and r["offense"]["assignments"] is None and r["defense"] is None and r["art_sha256"] is None
    assert r["personnel_group"] == "3WR" and ac.validate({x["id"]: x for x in p.records}) == []


def test_layout_drift_is_quarantined_not_ingested():
    p = ac.parse_listing(html("listing_mismatch.html"))
    assert p.records == [] and p.quarantined and "does not match" in p.quarantined[0]["reason"]


def test_a_play_slug_that_differs_from_the_art_filename_is_kept_and_flagged_not_dropped():
    page = html("listing_offense_p1.html").replace("fixture-formation-a__test-play-two", "fixture-formation-a__test-play-tw0")     # real-site style typo in the art name
    p = ac.parse_listing(page)
    two = next(r for r in p.records if r["play_slug"] == "test-play-two")
    assert two["source"]["art_slug_differs_from_link"] is True and not p.quarantined


def test_discovery_aborts_without_writing_when_the_layout_breaks(tmp_path):
    st, f = fetcher(tmp_path, {"/playbooks/plays/madden27?page=1": "<html>totally new layout</html>"})
    with pytest.raises(RuntimeError, match="layout probably changed"):
        ac.discover(f, sides=("offense",))
    assert not st.catalog.exists()
    st, f = fetcher(tmp_path, {"/playbooks/plays/madden27?page=1": html("listing_mismatch.html")})
    with pytest.raises(RuntimeError, match="tiles failed checks"):
        ac.discover(f, sides=("offense",))


def test_discovery_walks_pages_until_empty_and_never_sends_credentials(tmp_path):
    pages = {"/playbooks/plays/madden27?page=1": html("listing_offense_p1.html"), "/playbooks/plays/madden27?page=2": "<html></html>",
             "/playbooks/plays/madden27?page=1&side=defense": html("listing_defense_p1.html"), "/playbooks/plays/madden27?page=2&side=defense": "<html></html>"}
    st, f = fetcher(tmp_path, pages)
    recs, q, stats = ac.discover(f)
    assert len(recs) == 3 and {r["side"] for r in recs.values()} == {"offense", "defense"} and q == []
    assert all(ua == ac.USER_AGENT and not cookie and not auth for _, ua, cookie, auth in f.opener.log)


def test_robots_disallow_and_auth_walls_stop_the_run(tmp_path):
    st = ac.Store(tmp_path / "s")
    f = ac.Fetcher(st, delay=0, opener=FakeOpener({"/robots.txt": "User-agent: *\nDisallow: /playbooks/\n"}), sleep=lambda s: None)
    with pytest.raises(PermissionError, match="robots"):
        f.listing("madden27", "offense", 1)
    st, f = fetcher(tmp_path, {}, errors={"/playbooks/plays/madden27?page=1": 403})
    with pytest.raises(PermissionError, match="not retrying or bypassing"):
        f.listing("madden27", "offense", 1)
    assert sum(1 for p, *_ in f.opener.log if p.startswith("/playbooks")) == 1               # no retry on 403


def test_art_download_requires_a_recorded_permission(tmp_path):
    st, f = fetcher(tmp_path, {})
    with pytest.raises(PermissionError, match="permission-note"):
        f.art({"art_url": ac.BASE + "/play-art/x.png"})


def test_export_import_dedupes_memberships_rejects_bad_rows_and_demands_evidence(tmp_path):
    recs, bad = ac.import_export(FX / "export_sample.jsonl", version="v1", permission=PERM)
    one = recs["madden27|offense|fixture-formation-a|test-play-one"]
    assert one["playbooks"] == ["Book X", "Book Y", "Book Z"]                              # same play in more books = memberships, one record
    assert len(recs) == 2 and {b["row"] for b in bad} == {4, 5}
    assert any("assignment_tier" in b["reason"] for b in bad)


def test_sync_adds_changes_never_deletes_and_keeps_earned_knowledge(tmp_path):
    st = ac.Store(tmp_path / "s")
    recs, _ = ac.import_export(FX / "export_sample.jsonl", permission=PERM)
    assert ac.sync(st, recs)["added"] == 2
    cur = st.read()
    k = "madden27|offense|fixture-formation-a|test-play-one"
    cur[k]["evidence"] = {"tier": "diagram_verified", "basis": "first-party screenshot"}
    cur[k]["art_sha256"] = "ab" * 32
    cur[k]["offense"]["assignments"] = {"routes": ["earned"]}
    cur[k]["content_hash"] = ac.content_hash(cur[k])
    st.write_atomic(cur)
    newer = {i: dict(r) for i, r in recs.items()}
    newer[k] = dict(newer[k], art_url="https://x/y.png")
    newer[k]["content_hash"] = ac.content_hash(newer[k])
    del newer["madden27|defense|fixture-front|test-cover-3"]
    run = ac.sync(st, newer, scope_sides=("offense", "defense"))
    after = st.read()
    assert run["changed"] == 1 and run["not_seen"] == 1 and len(after) == 2                  # nothing deleted
    assert after[k]["evidence"]["tier"] == "diagram_verified" and after[k]["offense"]["assignments"] == {"routes": ["earned"]}
    assert after["madden27|defense|fixture-front|test-cover-3"]["status"] == "not_seen"
    assert (st.root / "history.jsonl").read_text().count("\n") == 2


def test_validator_catches_fabricated_knowledge_and_stale_hashes(tmp_path):
    recs, _ = ac.import_export(FX / "export_sample.jsonl", permission=PERM)
    k = "madden27|offense|fixture-formation-a|test-play-one"
    bad = json.loads(json.dumps(recs))
    bad[k]["offense"]["assignments"] = {"WR1": "go"}                                         # detail but evidence tier unknown
    assert any("no evidence tier" in e for e in ac.validate(bad))
    bad = json.loads(json.dumps(recs))
    bad[k]["evidence"]["tier"] = "diagram_verified"
    assert any("verified tier without" in e for e in ac.validate(bad))
    bad = json.loads(json.dumps(recs))
    bad[k]["play_name"] = "Changed"
    assert any("stale" in e for e in ac.validate(bad))


def test_audit_reports_every_requested_unit_and_flags_an_empty_store(tmp_path):
    recs, _ = ac.import_export(FX / "export_sample.jsonl", permission=PERM)
    rep = ac.audit(recs, reference={"offense": [("Fixture Formation A", "Test Play One"), ("Other", "Missing")], "defense": []}, permission=PERM)
    o = rep["sides"]["offense"]
    assert o["catalog_entries"] == 1 and o["distinct_formations"] == 1 and o["distinct_playbooks"] == 3 and o["lacking_sufficient_knowledge"] == 1
    assert o["verified_individual_assignments"] == 0 and o["with_usable_diagram"] == 0
    assert rep["discrepancies"]["offense"] == {"in_catalog_not_in_reference": 0, "in_reference_not_in_catalog": 1, "in_both": 1, "note": rep["discrepancies"]["offense"]["note"]}
    out = subprocess.run([sys.executable, str(ROOT / "scripts/madden_ml/ace_catalog.py"), "--store", str(tmp_path / "empty"), "audit"], capture_output=True, text=True)
    assert out.returncode == 0 and "has NOT occurred" in out.stdout


def test_store_refuses_to_live_inside_the_repository():
    with pytest.raises(ValueError, match="git work tree"):
        ac.Store(ROOT / "third_party_cache")


def test_cli_import_needs_permission_and_validates(tmp_path):
    env = [sys.executable, str(ROOT / "scripts/madden_ml/ace_catalog.py"), "--store", str(tmp_path / "s")]
    r = subprocess.run(env + ["import", "--file", str(FX / "export_sample.jsonl")], capture_output=True, text=True)
    assert r.returncode == 2 and "permission-note" in r.stderr
    r = subprocess.run(env + ["--permission-note", "test licence", "import", "--file", str(FX / "export_sample.jsonl")], capture_output=True, text=True)
    assert r.returncode == 0 and '"added": 2' in r.stdout
    assert subprocess.run(env + ["validate"], capture_output=True, text=True).returncode == 0


# ---------------------------------------------------------------------------------------------- extractor on synthetic art
def _panel_offense():
    img = Image.new("RGB", pe.PANEL, (0, 0, 0))
    d = ImageDraw.Draw(img)
    for x, y in ((100, 180), (300, 180), (450, 180)):
        d.line([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255), width=2)
        d.line([x - 4, y + 4, x + 4, y - 4], fill=(255, 255, 255), width=2)
    d.line([100, 176, 100, 20], fill=(210, 210, 60), width=4)                                # straight vertical
    d.line([300, 176, 300, 100, 380, 100], fill=(210, 210, 60), width=4)                    # up then out
    d.line([450, 176, 400, 120], fill=(200, 30, 60), width=4)                                # red diagonal
    buf = io.BytesIO(); img.save(buf, "PNG")
    return buf.getvalue()


def test_extractor_finds_routes_owners_and_never_names_slots_or_verifies():
    ex = pe.analyze(_panel_offense(), "offense")
    assert ex["tier"] == "diagram_extracted_unreviewed" and ex["needs_review"] and "UNVERIFIED" in ex["colour_meaning"]
    assert ex["counts"]["routes"] == 3 and ex["counts"]["markers_with_route"] == 3 and ex["flags"] == []
    shapes = {r["owner_marker_index"]: r["suggested_shape"] for r in ex["routes"]}
    assert shapes[0] == "vertical" and shapes[2] in ("diagonal_upfield", "stem_then_diagonal")
    assert all("slot" not in r for r in ex["routes"]) and "read_order" not in ex


def test_extractor_flags_a_route_with_no_start_marker():
    img = Image.new("RGB", pe.PANEL, (0, 0, 0))
    ImageDraw.Draw(img).line([200, 150, 200, 40], fill=(210, 210, 60), width=4)
    buf = io.BytesIO(); img.save(buf, "PNG")
    ex = pe.analyze(buf.getvalue(), "offense")
    assert "route_without_start_marker" in ex["flags"]


def test_ingest_cli_extract_review_and_approval_rules(tmp_path):
    st = ac.Store(tmp_path / "s")
    recs, _ = ac.import_export(FX / "export_sample.jsonl", permission=PERM)
    ac.sync(st, recs)
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "fixture-formation-a__test-play-one.png").write_bytes(_panel_offense())
    cli = [sys.executable, str(ROOT / "scripts/madden_ml/ingest_first_party_art.py"), "--store", str(tmp_path / "s")]
    assert subprocess.run(cli + ["extract", "--images", str(imgs), "--side", "offense"], capture_output=True, text=True).returncode == 0
    assert subprocess.run(cli + ["review"], capture_output=True, text=True).returncode == 0
    q = (tmp_path / "s" / "review_queue.csv").read_text()
    assert "fixture-formation-a|test-play-one" in q
    rid = "madden27|offense|fixture-formation-a|test-play-one"
    csvp = tmp_path / "rev.csv"
    csvp.write_text(f"id,approved,source\n{rid},yes,whatever\n")
    r = subprocess.run(cli + ["approve", "--file", str(csvp)], capture_output=True, text=True)
    assert "promoted 0" in r.stdout                                                           # bad source -> nothing promoted
    assert ac.Store(tmp_path / "s").read()[rid]["evidence"]["tier"] == "unknown"
    csvp.write_text(f"id,approved,source\n{rid},yes,user_screenshot:play1.png\n")
    assert "promoted 1" in subprocess.run(cli + ["approve", "--file", str(csvp)], capture_output=True, text=True).stdout
    done = ac.Store(tmp_path / "s").read()[rid]
    assert done["evidence"]["tier"] == "diagram_verified" and done["offense"]["assignments"]["routes"] and ac.validate({rid: done}) == []


def test_authorization_is_structured_scoped_and_never_invented(tmp_path):
    st = ac.Store(tmp_path / "s")
    with pytest.raises(PermissionError, match="no recorded authorization"):
        ac.require_scope(st, "images_download")
    with pytest.raises(ValueError, match="required"):
        ac.record_authorization(st, sender="", date="2026-11-07", scopes=["metadata_bulk"], restrictions="", private_reference="r", summary="s")
    with pytest.raises(ValueError, match="unknown scope"):
        ac.record_authorization(st, sender="x@y", date="2026-11-07", scopes=["everything"], restrictions="", private_reference="r", summary="s")
    ac.record_authorization(st, sender="support@example.test", date="2026-11-07", scopes=["metadata_bulk"], restrictions="private use only", private_reference="mailbox ref", summary="metadata only")
    assert ac.require_scope(st, "metadata_bulk")["scopes"] == ["metadata_bulk"]
    for scope in ("images_download", "ml_training", "redistribution"):                     # a grant for one scope never implies another
        with pytest.raises(PermissionError):
            ac.require_scope(st, scope)


def test_cli_art_refuses_without_the_images_scope(tmp_path):
    base = [sys.executable, str(ROOT / "scripts/madden_ml/ace_catalog.py"), "--store", str(tmp_path / "s")]
    subprocess.run(base + ["authorize", "--sender", "a@b.test", "--date", "2026-11-07", "--scopes", "metadata_bulk", "--reference", "r", "--summary", "metadata only"], check=True, capture_output=True)
    r = subprocess.run(base + ["art"], capture_output=True, text=True)
    assert r.returncode != 0 and "images_download" in (r.stderr + r.stdout)
