"""Priority 1 (#116): the exact defensive call typed into the existing post-play form is kept, beside the unchanged normalized label.

HTML form -> LivePlayController.result_and_call -> snaps.coverage_typed (additive column) -> next offensive decision.
Training data here is SYNTHETIC; this proves capture, backward compatibility and wiring, not model quality.
"""
from __future__ import annotations

import sqlite3

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import defense_semantics as ds
from cfb_coach.madden.model import offense_live_coverage as lc
from cfb_coach.madden.model import offense_time_correct as tc
from tests.madden_ml.test_defense_knowledge_live import _ctrl, _prime, _world, artifacts, rows, kdb, ddb  # noqa: F401

CASES = [  # typed -> expected exact catalog key; the second column documents what today's form stores
    ("Cover 3 Cloud", "Cover 3 Sky"), ("Fire Zone 3", "pressure"), ("Cover 3 Buzz Match", "Cover 3 Match"),
    ("Cover 4 Quarters", "Cover 4 Quarters"), ("Cover 4 Palms", None), ("Zero Blitz", "Cover 0"),
    ("Cover 2 Man", "Cover 2 Man"), ("Cover 1 Robber", "Cover 1"),
]


def _stored(db, c):
    return db.conn.execute("SELECT coverage_seen, coverage_typed FROM snaps WHERE side='offense' AND session_id=? "
                           "AND result IS NOT NULL ORDER BY id", (c.session_id,)).fetchall()


def test_exact_typed_names_are_stored_beside_the_unchanged_normalized_label(kdb):
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    sits = ["2&4 my 31", "1&10 my 41", "2&8 my 43", "3&2 my 45", "1&10 my 50", "2&7 my 53", "3&3 my 56", "1&10 my 60"]
    for (typed, _), sit in zip(CASES, sits):
        c.result_and_call(outcome="+4", sit_raw=sit, their=typed)
    got = _stored(kdb, c)
    assert [r["coverage_typed"] for r in got] == [t for t, _ in CASES]            # exactly what was entered
    for r, (typed, via_form) in zip(got, CASES):
        if via_form is not None:
            assert r["coverage_seen"] == via_form                               # normalized interpretation unchanged
        assert ds.resolve(r["coverage_typed"]).key                              # and recognised by the catalog


@pytest.mark.parametrize("typed,key", [("Cover 3 Cloud", None), ("Fire Zone 3", None), ("Cover 3 Buzz Match", None),
                                       ("Cover 4 Palms", None), ("Zero Blitz", None), ("Cover 4 Quarters", None)])
def test_next_offensive_decision_sees_the_exact_defense_not_the_collapsed_label(kdb, ddb, typed, key):
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+6", sit_raw="2&4 my 31", their=typed)
    exact = ds.resolve(typed).key
    from cfb_coach.madden.model import offense_route_ranker as rr
    cov = rr.info_of(c.last_call)["coverage"]
    top = dict(cov["top"])
    named = {k: v for k, v in top.items() if k != ds.GENERIC_PRESSURE_KEY}      # the synthetic prior carries some generic pressure mass
    if exact == ds.GENERIC_PRESSURE_KEY:
        named = top
    assert exact and max(named, key=named.get) == exact, (typed, exact, top)       # the exact name is the leading observed defense
    assert cov["basis"].endswith("+defense_knowledge")
    # and the same entry read the OLD way (stored label only) is a blurry mixture, never the exact name alone
    stored_only = ds.stored_label_weights(_stored(kdb, c)[0]["coverage_seen"])
    assert stored_only.get(exact, 0.0) < 1.0 or typed == "Cover 4 Quarters"


def test_default_artifact_path_is_unchanged_by_the_new_column(ddb):
    _prime(ddb)
    c = _ctrl(ddb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+6", sit_raw="2&4 my 31", their="Cover 3 Cloud")
    from cfb_coach.madden.model import offense_route_ranker as rr
    cov = rr.info_of(c.last_call)["coverage"]
    assert cov["basis"] == "closed_manual_postplay_coverage_with_historical_prior" and not any(k.startswith("dk:") for k, _ in cov["top"])


def test_old_rows_without_typed_text_stay_ambiguous():
    w = ds.stored_label_weights("Cover 3 Sky")                                   # legacy row: typed is NULL
    assert len(w) > 3 and abs(sum(w.values()) - 1) < 1e-9
    assert ds.stored_label_weights("Cover 3 Sky", None) == w == ds.stored_label_weights("Cover 3 Sky", "")
    assert ds.stored_label_weights("Cover 3 Sky", "some made up call") == w     # unrecognized typed text invents nothing
    assert ds.stored_label_weights("Cover 3 Sky", "Cover 3 Cloud") == {ds.resolve("Cover 3 Cloud").key: 1.0}
    assert ds.stored_label_weights("pressure", "Fire Zone 3") == {ds.resolve("Fire Zone 3").key: 1.0}
    assert ds.stored_label_weights("pressure") == {ds.GENERIC_PRESSURE_KEY: 1.0}


def test_additive_migration_leaves_existing_rows_untouched(tmp_path):
    p = tmp_path / "old.db"
    db = CoachDB(p, seed={"opponents": {"cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu", "confidence": "low", "profile_json": "{}"}}})
    db.log_snap(opponent_id="cpu", side="offense", situation_raw="1&10", our_call="x", result="+3", coverage_seen="Cover 3 Sky", session_id="s1")
    db.conn.execute("ALTER TABLE snaps DROP COLUMN coverage_typed") if "coverage_typed" in {r[1] for r in db.conn.execute("PRAGMA table_info(snaps)")} else None
    db.conn.commit()
    db.close()
    con = sqlite3.connect(p)
    assert "coverage_typed" not in {r[1] for r in con.execute("PRAGMA table_info(snaps)")}
    con.close()
    db = CoachDB(p)                                                              # reopen: old DB, no column yet
    assert db.conn.execute("SELECT coverage_seen FROM snaps").fetchone()[0] == "Cover 3 Sky"
    ctx = lc.coverage_for_next_snap(db.conn, session_id="s1", opponent_id="cpu", prior_mix=None, next_snap_seq=None, knowledge=True)  # reads a DB lacking the column
    assert ctx.history_n == 1
    db.log_snap(opponent_id="cpu", side="offense", situation_raw="2&5", our_call="y", result="+4", coverage_seen="Cover 3 Sky",
                coverage_typed="Cover 3 Cloud", session_id="s1")
    rows_ = db.conn.execute("SELECT coverage_seen, coverage_typed FROM snaps ORDER BY id").fetchall()
    assert [tuple(r) for r in rows_] == [("Cover 3 Sky", None), ("Cover 3 Sky", "Cover 3 Cloud")]
    db.close()


def test_time_correct_annotation_uses_the_typed_name_only_when_present():
    base = {"side": "offense", "result": "+3", "opponent_id": "o", "session_id": "g1", "game_id": "g1"}
    rows_ = [dict(base, coverage_seen="Cover 3 Sky", coverage_typed="Cover 3 Cloud", snap_seq=1),
             dict(base, coverage_seen="Cover 3 Sky", snap_seq=2), dict(base, coverage_seen="Cover 3 Sky", snap_seq=3)]
    out = tc.annotate(rows_, knowledge=True)
    cloud = ds.resolve("Cover 3 Cloud").key
    key = tc.KEY
    assert out[1][key]["dk_counts"] == {cloud: 1.0}
    assert out[2][key]["dk_counts"].get(cloud, 0.0) > 1.0 and len(out[2][key]["dk_counts"]) > 3   # + the ambiguous older-style row


def test_undo_removes_the_row_and_its_typed_call_together(kdb):
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+6", sit_raw="2&4 my 31", their="Cover 3 Cloud")
    c.result_and_call(outcome="+2", sit_raw="3&2 my 33", their="Palms")
    assert [r["coverage_typed"] for r in _stored(kdb, c)] == ["Cover 3 Cloud", "Palms"]
    c.undo_last()
    assert [r["coverage_typed"] for r in _stored(kdb, c)] == ["Cover 3 Cloud"]


# ---- names the old parser cannot recognise at all (form_store is None): the exact text must still be kept -----------------------
def _dropped_installed():
    import json
    from cfb_coach.madden.model import offense_assignments as oa  # noqa: F401
    snap = json.loads((__import__("pathlib").Path(ds.__file__).parents[2] / "data" / "madden27" / "installed_playbook_snapshot.json").read_text())
    installed = {p for side in snap["sides"].values() for plays in side["formations"].values() for p in plays} | \
                {f for side in snap["sides"].values() for f in side["formations"]}
    return [n for n in ds.channel_audit()["dropped_names"] if n in installed]


def test_fixture_names_genuinely_have_no_stored_form():
    names = _dropped_installed()
    assert names, "expected installed names the form drops"
    for n in names:
        assert ds.form_store(n) is None and ds.resolve(n).key


def test_dropped_names_are_kept_exactly_and_reach_the_next_decision(kdb):
    name = _dropped_installed()[0]
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+5", sit_raw="2&5 my 30", their=name)
    row = _stored(kdb, c)[0]
    assert row["coverage_typed"] == name and not row["coverage_seen"]               # nothing recognised, nothing invented
    from cfb_coach.madden.model import offense_route_ranker as rr
    cov = rr.info_of(c.last_call)["coverage"]
    exact = ds.resolve(name).key
    assert cov["basis"].endswith("+defense_knowledge") and cov["named_n"] == 1
    assert max({k: v for k, v in cov["top"] if k != ds.GENERIC_PRESSURE_KEY}.items(), key=lambda kv: kv[1])[0] == exact


def test_unknown_typed_text_is_preserved_but_never_becomes_a_defense(kdb):
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+5", sit_raw="2&5 my 30", their="  some   made up call ")
    row = _stored(kdb, c)[0]
    assert row["coverage_typed"] == "some made up call"                             # whitespace normalised only
    from cfb_coach.madden.model import offense_route_ranker as rr
    cov = rr.info_of(c.last_call)["coverage"]
    assert cov["named_n"] == 0 and cov["pressure_n"] == 0


def test_result_only_retry_does_not_wipe_the_typed_call_and_a_resubmit_is_idempotent(kdb):
    name = _dropped_installed()[0]
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+5", sit_raw="2&5 my 30", their=name)
    n_rows = kdb.conn.execute("SELECT COUNT(*) FROM snaps WHERE session_id=?", (c.session_id,)).fetchone()[0]
    book = c._book()
    # correction of the result for the same logged snap, nothing typed in "they ran"
    if book.logged_id is not None:
        kdb.update_snap(book.logged_id, result="+6")
    assert _stored(kdb, c)[0]["coverage_typed"] == name
    assert kdb.conn.execute("SELECT COUNT(*) FROM snaps WHERE session_id=?", (c.session_id,)).fetchone()[0] == n_rows


def test_next_situation_text_is_never_attributed_to_the_snap(kdb):
    from cfb_coach.last_snap import LastSnapBook  # noqa: F401
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+5", sit_raw="2&5 my 30 cover 3 cloud", their="")   # text on the NEXT situation line
    assert _stored(kdb, c)[0]["coverage_typed"] is None


def test_request_key_retry_is_idempotent_for_the_typed_call(kdb):
    name = _dropped_installed()[0]
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    first = c.result_and_call(outcome="+5", sit_raw="2&5 my 30", their=name, request_key="k-1")
    again = c.result_and_call(outcome="+5", sit_raw="2&5 my 30", their=name, request_key="k-1")
    assert first == again or again.get("ok") is not False
    got = _stored(kdb, c)
    assert [r["coverage_typed"] for r in got] == [name]


def test_undo_after_a_dropped_name(kdb):
    names = _dropped_installed()
    _prime(kdb)
    c = _ctrl(kdb)
    c.call_only("1&10 my 25")
    c.result_and_call(outcome="+6", sit_raw="2&4 my 31", their="Cover 3 Cloud")
    c.result_and_call(outcome="+2", sit_raw="3&2 my 33", their=names[0])
    assert [r["coverage_typed"] for r in _stored(kdb, c)] == ["Cover 3 Cloud", names[0]]
    c.undo_last()
    assert [r["coverage_typed"] for r in _stored(kdb, c)] == ["Cover 3 Cloud"]
