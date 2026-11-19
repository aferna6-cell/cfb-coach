"""t477u: every live Patriots 6-1 / Nickel defensive call shows sourced pre-snap adjustments."""
from __future__ import annotations

from cfb_coach.madden.ravens_defense import catalog as rc
from cfb_coach.madden.ravens_defense import coordinator as co
from cfb_coach.madden.ravens_defense import legality as lg
from cfb_coach.madden.ravens_defense import presnap as ps
from cfb_coach.madden.scheme import play_cooldown as pcd
from tests.test_chiefs_play_cooldown import ravens_env  # noqa: F401
from tests.test_civil_scheme_only import _db, _sit, civil_env  # noqa: F401
from tests.test_lions_cpu_book_t445u import _forms
from tests.test_titans_play_cooldown import _giants, _lions


def _payload_for(cand, *, shape=None, cat=None):
    cat = cat or rc.load_61_defense()
    decision = {
        "status": "CALL",
        "candidate": cand.as_dict(),
        "candidate_raw": dict(cand.raw),
        "macro_display": cand.macro_display(cat),
        "zone_it_selected": cand.optional_macro == "ZONE_IT",
        "user": {"status": "UNVERIFIED", "user": None, "offense_shape": shape},
    }
    return ps.collect(decision, cat), decision


def test_every_live_call_has_shown_adjustments_and_none_are_invented():
    cat = rc.load_61_defense()
    assert set(c.call_id for c in cat.candidates).isdisjoint(rc.NSM_CALL_IDS)
    assert "dime_dbfire2" not in {c.call_id for c in cat.candidates}

    kinds = {}
    flagged = []
    even61_ok = []
    for cand in cat.candidates:
        payload, _ = _payload_for(cand, cat=cat)
        assert payload["shown"] and payload["lines"], cand.candidate_id
        assert payload["lines"] != [""]
        kinds.setdefault(payload["kind"], []).append(cand.candidate_id)
        if payload["flagged"]:
            flagged.append(cand.candidate_id)
        if cand.package == "even_61":
            steps = [s for s in (cand.raw.get("setup_steps") or []) if not ps._is_video(s)]
            text = "\n".join(payload["lines"])
            for step in steps:
                assert step in text, (cand.candidate_id, step)
            assert payload["kind"] == "even61_pdf"
            assert "PDF" in (payload["source"] or "")
            even61_ok.append(cand.call_id)
        if cand.package == "match_coverage" and cand.optional_macro != "ZONE_IT":
            assert payload["kind"] == "match_guide"
            assert "not verified" in (payload["source"] or "").lower() or "Match" in (payload["source"] or "")
        if cand.package == "spot_drop_fallback":
            assert payload["kind"] == "stock_default"
            assert payload["lines"] == [ps.STOCK_DEFAULT_LINE]

    assert len(even61_ok) == 14
    assert not flagged, flagged
    # Cover 4 Zone is the one expected stock-default live call.
    assert kinds["stock_default"] == ["stock_cover4_zone"] or set(kinds["stock_default"]) == {"stock_cover4_zone"}


def test_even61_card_and_compact_ui_print_pdf_adjustments():
    cat = rc.load_61_defense()
    cand = cat.by_id()["even61_gun5wrtight_d2_tampa2_61_base"]
    payload, decision = _payload_for(cand, cat=cat)
    ps.attach(decision, cat)
    lines = co.card_lines(decision)
    assert lines[0] == "PLAY: 4-3 Even 6-1 — Tampa 2"
    assert "ADJUSTMENTS:" in lines
    joined = "\n".join(lines)
    for needed in (
        "Default Shell",
        "Put Safety On Tight Side On Outside 3rd",
        "Put Safety On Bunch Side On Inside Quarter",
        "Conservative Zone Strategy Entire Defense",
        "Aggressive Zone Strategy Both Hooks",
    ):
        assert needed in joined, needed
    block = ps.variation_block(decision, payload)
    assert "ADJUSTMENTS:" in block and "Default Shell" in block
    assert "MACRO: 61 Base" in block
    assert "SITUATION: Gun 5 WR Tight Defense 2" in block


def test_nickel_guide_and_match_checks_are_labeled_not_invented():
    cat = rc.load_61_defense()
    cloud, _ = _payload_for(cat.by_id()["nickel_cloud"], shape="base", cat=cat)
    assert cloud["kind"] == "nickel_guide_reference"
    assert any("narrow safeties" in ln or "cloud-side safety" in ln for ln in cloud["lines"])
    assert "reference only" in (cloud["source"] or "").lower() or "not verified" in (cloud["source"] or "").lower()

    match, _ = _payload_for(cat.by_id()["match_cover3"], shape="trips", cat=cat)
    assert match["kind"] == "match_guide"
    assert any("Skinny MEG" in ln for ln in match["lines"])
    assert all("not verified" in ln.lower() or "alignment not entered" in ln.lower() or "if " in ln
               or "Skinny MEG" in ln or "Hook Curl" in ln or "Match" in ln
               for ln in match["lines"])


def test_live_html_shows_defense_adjustments(tmp_path, civil_env, ravens_env):
    from cfb_coach.live_server import LivePlayController, _call_main, render_live_html
    from cfb_coach.madden import playcaller
    from cfb_coach.madden.situation import parse_madden_situation
    import html as _html

    db = _db(tmp_path, formations=_forms())
    _lions(db, "d-adj")
    ctrl = LivePlayController(db=db, opponent_id="jaxon", make_call=playcaller.make_call,
                              parse_situation=parse_madden_situation, learn_summary=lambda: "ok",
                              brand="Madden 27 Franchise", cpu_only=False, enable_execution_verify=True)
    sit = _sit(1, 10, 40, session="d-adj")
    sit.side = "defense"
    sit.extras = {"session_id": "d-adj", "snap_seq": 1}
    call = playcaller.make_call(sit, "jaxon", db, cpu_defense=False)
    rd = call.ravens_defense
    assert rd["status"] == "CALL" and rd.get("adjustment_lines")
    ctrl.last_call, ctrl.call_text = call, call.format()
    rec = ctrl.state()["pending_recommendation"]
    label = rec["variation"]
    assert label
    assert rec["variation_block"] and "ADJUSTMENTS:" in rec["variation_block"]
    for line in rd["adjustment_lines"]:
        assert line in rec["variation_block"], line
    main = _call_main(ctrl)
    assert main == f"{call.formation} — {call.play} · {label}"
    page = render_live_html(ctrl)
    assert "ADJUSTMENTS:" in page
    for line in rd["adjustment_lines"]:
        assert _html.escape(line) in page, line


def test_giants_quick_defense_sim_shows_adjustments(tmp_path, civil_env, ravens_env):
    from cfb_coach.madden import playcaller

    db = _db(tmp_path, formations=_forms())
    _giants(db, "giants-d")
    nsm_ids = set(rc.NSM_CALL_IDS)
    seen, forms, statuses = set(), set(), []
    n = 120
    for i in range(n):
        s = _sit(1 + i % 3, 5 + i % 8, 30 + i % 40, session="giants-d")
        s.side = "defense"
        s.extras = {"session_id": "giants-d", "snap_seq": i}
        call = playcaller.make_call(s, "giants_dallas_cowboys", db, cpu_defense=False)
        rd = call.ravens_defense
        statuses.append(rd["status"])
        assert rd["status"] == "CALL", rd
        assert call.formation in rc.SIXTY_ONE_FORMATIONS
        cid = rd["candidate"]["call_id"]
        assert cid != "dime_dbfire2" and cid not in nsm_ids
        assert rd.get("adjustments_shown") and rd.get("adjustment_lines")
        assert any(ln == "ADJUSTMENTS:" or ln.startswith("ADJUSTMENTS:") for ln in (rd.get("card") or []))
        assert not (rd.get("play_cooldown") or {}).get("active")
        assert pcd.cooldown_spec(db, "giants-d", side="defense") is None
        seen.add(cid)
        forms.add(call.formation)
    assert "NO_LEGAL_RAVENS_CALL" not in statuses and statuses.count("CALL") == n
    assert "dime_dbfire2" not in seen and seen.isdisjoint(nsm_ids)
    assert rc.DIME not in forms and rc.NSM not in forms
    print("GIANTS_DEFENSE_QUICK_SIM", {
        "snaps": n, "unique_calls": len(seen), "forms": sorted(forms),
        "nsm": 0, "dime": 0, "adjustments_shown": n,
        "even61": sum(1 for x in seen if x.startswith("even61_")),
    })
