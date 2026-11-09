"""The owner-machine procedure, exercised end-to-end on a fixture 'primary' database (never the real one)."""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from cfb_coach.madden.unified import guard, localrun as LR
from cfb_coach.madden.unified.cli import main as cli_main
from cfb_coach.madden.unified.coordinator import inventory_fp
from cfb_coach.madden.unified.types import DEF, OFF
from tests.unified.helpers import inventory
from tests.unified.test_history import make_db


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_local_prep_never_touches_the_source_and_stops_before_installation(tmp_path, monkeypatch):
    home = tmp_path / "home" / ".cfb-coach"
    home.mkdir(parents=True)
    monkeypatch.setattr(guard, "DEFAULT_HOME", home)
    src = make_db(home / "madden27.db", n_games=12, per_game=12, extra=False)  # stands in for the PRIMARY franchise db
    before = sha(src)
    rep = LR.local_prep(src, tmp_path / "work")
    assert sha(src) == before and rep["steps"]["1_protected_backup"]["db_backup"]["source_unchanged"]
    t = rep["steps"]["2_isolated_test_db"]
    assert t["row_counts_match"] and t["source_unchanged"] and "work" in t["copy"] and str(home) not in t["copy"]
    assert rep["steps"]["4_history"]["usable_rows"] == 144
    assert rep["steps"]["5_game_prep"][OFF]["formations"] <= 15 and rep["steps"]["5_game_prep"][DEF]["formations"] <= 15
    assert "STOPPED" in rep["stop"] and not (tmp_path / "work" / "books.json").exists()  # nothing was marked installed
    guide = (tmp_path / "work" / "prep" / "INSTALL_GUIDE.md").read_text()
    assert "NOT CONFIRMED" in guide and "UNVERIFIED_IN_EDITOR" in guide
    with pytest.raises(guard.ProtectedPathError):
        LR.local_prep(src, home / "work2")  # workdir inside the primary data dir is refused
    with pytest.raises(FileExistsError):
        LR.safe_copy_db(__import__("pathlib").Path(src), tmp_path / "work" / "testing_franchise" / "madden27_test.db")


def test_discover_is_read_only_and_reports_primary_location(tmp_path, monkeypatch):
    home = tmp_path / "h" / ".cfb-coach"
    home.mkdir(parents=True)
    monkeypatch.setattr(guard, "DEFAULT_HOME", home)
    monkeypatch.setenv("CFB_COACH_MADDEN_DB", make_db(home / "madden27.db", n_games=2, per_game=3, extra=False))
    p = home / "madden27.db"
    before = sha(p)
    d = LR.discover()
    assert sha(p) == before and d["databases"][0]["primary_data_dir"] is True and d["databases"][0]["ok"]


def test_local_finish_reaches_go_only_with_attested_books(tmp_path, monkeypatch):
    home = tmp_path / "home" / ".cfb-coach"
    home.mkdir(parents=True)
    monkeypatch.setattr(guard, "DEFAULT_HOME", home)
    src = make_db(home / "madden27.db", n_games=6, per_game=10, extra=False)
    LR.local_prep(src, tmp_path / "work")
    # the owner installs, checks and attests the PROPOSED books (here: done by the test, standing in for the owner)
    books = {}
    for s in (OFF, DEF):
        prop = json.loads((tmp_path / "work" / "prep" / f"proposal_{s}.json").read_text())
        books[s] = {"side": s, "fingerprint": prop["proposed_fingerprint"], "inventory": prop["inventory"], "attested_by": "owner"}
    bf = tmp_path / "books.json"
    bf.write_text(json.dumps(books))
    rep = LR.local_finish(tmp_path / "work", bf, calls=20)
    assert rep["verdict"] == "GO", rep["blocking"]
    assert rep["launch"].startswith("python -m cfb_coach.madden.unified game ") and "--store" in rep["launch"]
    # without the attestation the same procedure cannot reach GO
    only_off = tmp_path / "off_only.json"
    only_off.write_text(json.dumps({OFF: books[OFF]}))
    assert LR.local_finish(tmp_path / "work", only_off, calls=20)["verdict"] == "SHADOW ONLY"
