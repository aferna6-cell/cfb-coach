#!/workspace/vod_data/tools/.venv/bin/python
"""VOD snap extractor + replay-style labeler (standalone box tool, outside the cfb-coach repo).

Subcommands (all idempotent, safe to re-run):
  run      OCR + extract every Scout manifest VOD with status "downloaded" (or --video-ids / --spec), set manifest
           status to "extracted" / "extract_failed", then build a new label version if anything new was extracted.
  (run --spec FILE / --video-ids X: extract specific VODs, e.g. the pilot spec; OCR is cached + resumable)
  label    Build a versioned label set /workspace/vod_data/labels/vX.Y/ from snap batches and point LATEST.json at it.
  score    Accuracy / coverage / confidence report for a label version.

Snap extraction logic is copied from cfb-coach draft PR #16 (vod_pilot @ 9f112f4) into ./vodx_engine; success follows
cfb_coach.madden.postgame.snap_success / learning.evaluate_snap from cfb-coach main @ ef8294f (PR #15), re-implemented here.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, fcntl, gzip, hashlib, json, math, os, re, statistics as st, sys, time, traceback
from pathlib import Path
# (v0.31) always run inside the tools venv (cv2/rapidocr live there); the 19:00 run died on system python
_VENV = Path(__file__).resolve().parent / ".venv"
if Path(sys.prefix).resolve() != _VENV.resolve() and os.environ.get("VODX_NO_VENV_REEXEC") != "1":
    _py = _VENV / "bin" / "python"
    if not _py.exists():
        sys.exit(f"vodx: tools venv missing at {_py}")
    os.environ["VODX_NO_VENV_REEXEC"] = "1"  # loop guard
    os.execv(str(_py), [str(_py), str(Path(__file__).resolve()), *sys.argv[1:]])
from collections import Counter

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
ROOT = Path("/workspace/vod_data")
MANIFEST = ROOT / "manifest.jsonl"
SNAPS = ROOT / "snaps"
LABELS = ROOT / "labels"
SPEC_PILOT = TOOLS / "pilot_vods.jsonl"
SCHEMA_SNAPS = "vod_snaps.v1"
SCHEMA_LABELS = "vod_labels.v0"
ENGINE_REF = "cfb-coach PR #16 vod_pilot@9f112f4 (copied into tools/vodx_engine)"
SUCCESS_REF = "cfb-coach main@ef8294f learning.evaluate_snap / madden.postgame.snap_success (PR #15), offense perspective"
CURRENT_FIXES = ""  # set by `vodx.py run --fixes` for this process, including cache-only redos


def script_version() -> str:
    h = hashlib.sha256()
    for p in sorted([TOOLS / "vodx.py", TOOLS / "vodx_ocr.py", TOOLS / "vodx_refine.py", TOOLS / "vodx_field.py", TOOLS / "vodx_speech.py",
                    TOOLS / "vodx_names.py", TOOLS / "vodx_chatjunk.py", TOOLS / "vodx_formlist.py", TOOLS / "names_lexicon.json", TOOLS / "vodx_cards.py", TOOLS / "vodx_windows.py",
                    TOOLS / "vodx_possession.py",
                    TOOLS / "vodx_expert.py", TOOLS / "vodx_quality.py", TOOLS / "vodx_labelfix.py", TOOLS / "opponent_overrides.json",
                    *sorted((TOOLS / "vodx_engine").glob("*.py"))]):
        if not p.exists():
            continue
        h.update(p.read_bytes())
    return "vodx-" + h.hexdigest()[:10]


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

# ----------------------------------------------------------------------------------------------- manifest

def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def norm_vid(v: str) -> str:
    v = str(v)
    return v[1:] if re.fullmatch(r"v\d+", v) else v


def scout_index() -> dict[str, dict]:
    # clip-window rows (kind=window) share the parent's video_id; they must never shadow the VOD's own record
    return {norm_vid(r["video_id"]): r for r in read_jsonl(MANIFEST) if r.get("kind") != "window"}


def update_manifest_status(video_id: str, fields: dict) -> bool:
    """Edit Scout's manifest line in place under Scout's own lock (flock on manifest.jsonl, same protocol as
    /workspace/vod_data/set_status.py): no rename, so Scout's lock holders and appenders keep the same inode."""
    if not MANIFEST.exists():
        return False
    with open(MANIFEST, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        out, hit = [], False
        for l in f.read().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if norm_vid(r.get("video_id", "")) == norm_vid(video_id) and r.get("kind") != "window":
                r.update(fields); hit = True
                if "status" in fields:
                    r["status_updated_at"] = now_iso()
                l = json.dumps(r)
            out.append(l)
        if hit:
            f.seek(0); f.write("\n".join(out) + "\n"); f.truncate(); f.flush(); os.fsync(f.fileno())
        return hit

# ----------------------------------------------------------------------------------------------- lexicons

def _n(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())

# Pre-snap adjustment markers (Madden 27 / CFB 27 HUD wording). Matched on whole normalized OCR tokens
# from frames inside the snap's pre-snap window. Abstain when nothing matches; never inferred.
ADJ = {
    "audible": ["AUDIBLE", "AUDIBLES", "AUDIBLED", "PLAYMAKERAUDIBLE", "QUICKAUDIBLE", "FLIPPLAY", "FLIP"],
    "hot_routes": ["HOTROUTE", "HOTROUTES", "SMARTROUTE", "SMARTROUTES", "SMARTROUTEON", "SMARTROUTEOFF"],
    "off_macros": ["MOTION", "SLIDEPROTECT", "SLIDEPROTECTION", "SLIDELEFT", "SLIDERIGHT", "MAXPROTECT", "PASSBLOCK",
                   "RBBLOCK", "TEBLOCK", "CHIPBLOCK", "COMMIT", "RUNCOMMIT", "PASSCOMMIT", "CUSTOMADJUSTMENT",
                   "CUSTOMADJUSTMENTS", "MACRO", "MACROS", "QUICKADJUST", "QUICKADJUSTMENTS", "BACKFIELDMOTION",
                   "JETMOTION", "SHIFTMOTION", "OLINEADJUST", "BLOCKINGADJUSTMENTS", "RPOREAD", "SNAPCOUNT"],
    "def_adjustments": ["SHADEINSIDE", "SHADEOUTSIDE", "SHADEUNDER", "SHADEOVER", "SHADEUNDERNEATH", "SHADEDEEP",
                        "SHADE", "PRESS", "BACKOFF", "PRESSCOVERAGE", "SHOWBLITZ", "SHOWCOVERAGE", "PINCH", "SPREAD",
                        "CRASH", "CRASHLEFT", "CRASHRIGHT", "LBSHIFT", "DLSHIFT", "DLINEADJUST", "LINEBACKERADJUST",
                        "SHIFTLEFT", "SHIFTRIGHT", "STEM", "MANALIGN", "BASEALIGN", "COVERAGEAUDIBLE", "COVERAGEADJUST",
                        "COVERAGEADJUSTMENTS", "COVERAGESHELL", "COVERAGEHOTROUTE", "DEFENSIVEHOTROUTE", "PLAYMAKER",
                        "QBSPY", "QBCONTAIN", "BLITZ", "OVERLOAD", "CHEAT", "SINGLEHIGH", "TWOHIGH", "SPLITSAFETIES",
                        "SHIFTDLINE", "SHIFTLB", "PASSRUSH", "CONTAIN"],
}
# Zone assignment overlays (CFB/Madden coach-cam): defensive look evidence, not an adjustment.
DEF_OVERLAY = ["DEEPZONE", "DEEPHALF", "DEEPTHIRD", "DEEPQUARTER", "FLATZONE", "HOOKZONE", "CURLFLAT", "BUZZZONE",
               "3RECHOOK", "VERTHOOK", "SEAMFLAT", "QUARTERFLAT", "CLOUDFLAT", "MATCHZONE", "MANCOVERAGE"]
ROUTES = ["STREAK", "FADE", "SLANT", "CURL", "COMEBACK", "DRAG", "INSIDE", "OUTSIDE", "QUICKOUT", "OUT", "IN", "POST",
          "CORNER", "WHEEL", "SPOT", "DIG", "SMARTROUTE", "BLOCK", "FLAT", "ANGLE", "TEXAS", "SEAM", "SHALLOW"]
ADJ_N = {k: {_n(x) for x in v} for k, v in ADJ.items()}
OVERLAY_N = {_n(x) for x in DEF_OVERLAY}
ROUTES_N = {_n(x) for x in ROUTES}
HR_RX = re.compile(r"^(WR|TE|HB|RB|FB|SLOT|QB|X|Y|Z)?HOTROUTES?$")
MENU_BLOCK = (0.55, 0.92)  # y-band of the hot-route option wheel (scorebug sits below 0.93)
SKIP_FRAME = re.compile(r"PENALTY|SETTINGS|GAMEPLAYOPTIONS|PAUSEMENU|MAINMENU|CREATESHARE|CUSTOMADJUSTMENTSEDIT")
# Play-call screen chrome: its button legend prints AUDIBLES / FLIP PLAY / SMART ROUTE on every call screen, so those
# frames never count as an adjustment (verified on z5ne-uBM0YI t=34-45s).
CALL_SCREEN = re.compile(r"SELECTAPLAY|PICKAPLAY|COACHSUGGEST|PERSONNELGROUP|PLAYTYPE|KEYPLAYERS|PREVIOUSPLAY|AVGYDS|PLAYCALL|FAVORITES|PLAYBOOKS")
LEGEND_Y = 0.93  # tokens below this line are controller-button legends

# ----------------------------------------------------------------------------------------------- extract

def vod_specs(video_ids: list[str] | None, spec: Path | None) -> list[dict]:
    out = {}
    for r in read_jsonl(MANIFEST):
        if r.get("kind") == "window":  # Scout clip windows: handled by vodx_windows, never a VOD spec
            continue
        out[norm_vid(r["video_id"])] = dict(r, _from="scout_manifest")
    for p in ([spec] if spec else []):
        for r in read_jsonl(p):
            k = norm_vid(r["video_id"])
            if k not in out:
                out[k] = dict(r, _from=str(p))
            else:  # same video in Scout's manifest: keep Scout's record, but remember the pilot key
                out[k].setdefault("pilot_key", r.get("pilot_key"))
    rows = list(out.values())
    if video_ids:
        want = {norm_vid(v) for v in video_ids}
        rows = [r for r in rows if norm_vid(r["video_id"]) in want]
    return rows


def platform_of(r: dict) -> str:
    s = (r.get("source") or "").lower()
    return "twitch" if "twitch" in s or "twitch" in (r.get("url") or "") else ("youtube" if "youtu" in (r.get("url") or "") or s == "youtube" else s or "local")


LAYOUTS = json.loads((TOOLS / "layouts.json").read_text())
EXTRACT_LOCK = ROOT / ".extract.lock"


OPP_OVERRIDES_PATH = TOOLS / "opponent_overrides.json"


def opp_override(vid: str, opp: str, src: str, basis: str, conf: float):
    """tools/opponent_overrides.json: box-side opponent_type for VODs Scout never tagged (a Scout tag always wins)."""
    if src == "scout_manifest" or not OPP_OVERRIDES_PATH.exists():
        return opp, src, basis, conf
    ov = {norm_vid(k): v for k, v in json.loads(OPP_OVERRIDES_PATH.read_text()).get("videos", {}).items()}
    o = ov.get(norm_vid(vid))
    if not o or o.get("opponent_type") not in ("cpu", "human"):
        return opp, src, basis, conf
    return o["opponent_type"], o.get("basis", "override"), o.get("note", ""), float(o.get("confidence", 0.7))


def current_opponent(vid: str, stored: tuple, scout: dict | None = None):
    """Label-time opponent_type: Scout's current manifest tag (cpu/human) wins; else tools/opponent_overrides.json;
    else what the snap batch stored. Returns (type, source, basis, conf)."""
    if scout is None:
        scout = scout_index().get(norm_vid(vid))
    t = (scout or {}).get("opponent_type")
    if t in ("cpu", "human"):
        return t, "scout_manifest", (scout or {}).get("opponent_type_basis", ""), 1.0
    typ, src, basis, conf = stored
    src = src or "missing_scout_tag"
    if src == "scout_manifest":  # Scout tag withdrawn since extraction -> fall back
        typ, src, basis, conf = "unknown", "missing_scout_tag", "", 0.0
    o = opp_override(vid, typ, src, basis, conf)
    return o


def _relabel_log() -> Path:
    return SNAPS / "opponent_relabels.json"


def assign_match_index(labs: list[dict], starts) -> None:
    """(v0.32) layouts video `match_starts` [abs VOD s]: several matches of the SAME game in one VOD (e.g. two H2H games).
    match_index = number of starts <= the row's VOD time; drives match_id, possession/drive and label-fix sequences."""
    if not starts:
        return
    st = sorted(float(x) for x in starts)
    for l in labs:
        t = (l.get("section_offset_s") or 0) + (l.get("t_start") or 0) if l.get("t_start_vod") is None else l["t_start_vod"]
        l["match_index"] = sum(1 for s in st if t >= s)


def layout_for(r: dict) -> dict:
    lay = dict(LAYOUTS["creators"].get(r.get("creator") or "", {}))
    lay.update(LAYOUTS["videos"].get(norm_vid(r["video_id"]), {}))
    panel = r.get("previous_play_panel")
    lay["calls"] = "panel" if panel is True else ("no_panel" if panel is False else "unknown")
    return lay


def section_offset(r: dict) -> int:
    m = re.match(r"\s*(\d+)\s*-\s*\d+", str(r.get("section") or ""))
    return int(m.group(1)) if m else 0


def fps_for(r: dict, fps_arg: str) -> float:
    if fps_arg != "auto":
        return float(fps_arg)
    # v0.7: 0.5 fps regular grid for every VOD + dense call-screen frames around each snap (cmd_run) ~= 1 fps where it
    # matters (pre-snap call screen / Previous Play panel) at ~70% of the cost of a full 1 fps pass
    return 0.5


def ts_url(url: str, platform: str, t: float) -> str:
    if not url:
        return ""
    if platform == "twitch":
        s = int(t); return f"{url}?t={s//3600}h{(s%3600)//60}m{s%60}s"
    return f"{url}{'&' if '?' in url else '?'}t={int(t)}s"


def presnap_adjustments(frames: list[dict], t0: float, t1: float, lexicon) -> dict:
    """Scan OCR tokens between t0 and t1 for adjustment markers. Returns per-field values, conf, evidence."""
    hits = {k: {} for k in ADJ}      # field -> {term: [scores]}
    overlay: dict[str, list[float]] = {}
    routes: dict[str, list[float]] = {}
    audible_plays: list[str] = []
    window = [f for f in frames if t0 <= f["t"] <= t1]
    for f in window:
        toks = [(x[0], x[1]) for x in f["k"] if x[1] >= 0.6]
        blob = _n(" ".join(t for t, _ in toks))
        if SKIP_FRAME.search(blob) or CALL_SCREEN.search(blob) or f.get("g", 0) < 0.12:
            continue
        toks = [(x[0], x[1]) for x in f["k"] if x[1] >= 0.6 and x[3] < LEGEND_Y]
        # Hot-route menu ("WR HOT ROUTES" + the option wheel): record that it was opened and for which position.
        # The listed options (Streak, Post, Pass Block...) are the menu, not the chosen route, so they are not recorded
        # and they don't feed any other field.
        hr = [(m.group(1) or "?", sc) for t, sc in toks for m in [HR_RX.search(_n(t))] if m]
        if hr:
            for pos, sc in hr:
                hits["hot_routes"].setdefault(f"menu:{pos}", []).append(sc)
            continue
        audible_menu = any(_n(t) in ADJ_N["audible"] for t, _ in toks)
        for text, sc in toks:
            nt = _n(text)
            if not nt:
                continue
            for field, terms in ADJ_N.items():
                if field == "hot_routes":
                    continue  # handled above
                if audible_menu and field != "audible":
                    continue  # audible card lists play names / tags, nothing else counts on that frame
                if nt in terms:
                    hits[field].setdefault(nt, []).append(sc)
            if nt in OVERLAY_N:
                overlay.setdefault(nt, []).append(sc)
        if any(_n(t) in ADJ_N["audible"] for t, _ in toks):
            p = lexicon.match_play([t for t, _ in toks])
            if p:
                audible_plays.append(p)
    out = {}
    nwin = max(1, len(window))
    for field in ADJ:
        terms = hits[field]
        if not terms:
            out[field] = ("", 0.0, "")
            continue
        vals = sorted(terms, key=lambda k: -len(terms[k]))
        if field == "audible" and audible_plays:
            top = max(set(audible_plays), key=audible_plays.count)
            vals = vals + [f"play:{top}"]
        scores = [s for v in terms.values() for s in v]
        frames_seen = sum(len(v) for v in terms.values())
        # evidence-based: mean OCR score x persistence (frames seen, saturating at 3) x 0.8 cap for marker-only reads
        # evidence-based but capped at 0.5: a marker word proves the menu was up, not which option was chosen
        conf = round(min(0.5, st.mean(scores) * min(1.0, frames_seen / 3) * 0.5 + (0.1 if frames_seen >= 2 else 0)), 2)
        out[field] = (";".join(vals), conf, "ocr_presnap_overlay")
    out["def_overlay"] = (";".join(sorted(overlay, key=lambda k: -len(overlay[k]))[:4]),
                          round(min(0.8, st.mean([s for v in overlay.values() for s in v]) * 0.7), 2) if overlay else 0.0,
                          "ocr_presnap_overlay" if overlay else "")
    return out


def offense_success(down, distance, yardline, kind, yards):
    """learning.evaluate_snap (cfb-coach main @ ef8294f) success, offense perspective. None when result unknown."""
    if kind in ("", None, "unknown"):
        return None
    if kind == "td":
        return True
    if kind in ("int", "fumble"):
        return False
    if kind == "sack":
        return False
    if kind in ("incomplete", "stop", "incomplete_or_no_gain"):
        return False
    if kind == "convert":
        return True
    y = yards or 0
    if down is None or distance is None:
        return y > 0
    need = {1: 0.4, 2: 0.6, 3: 1.0, 4: 1.0}.get(int(down), 1.0) * float(distance)
    if yardline not in (None, ""):
        need = min(need, float(100 - int(yardline)))
    need = max(need, 0.5)
    return y >= need


LAYOUT_KEYS = ("mask", "transform", "gameplay_window")  # the layout fields extraction actually uses


def _layout_sig(lay: dict | None) -> str:
    return json.dumps({k: (lay or {}).get(k) for k in LAYOUT_KEYS}, sort_keys=True)


def layout_stale_batches() -> list[dict]:
    """Snap batches extracted with different masks/transform than tools/layouts.json now gives (layout edited after
    extraction). Redone from the OCR cache only; needs the cache, never the raw. Pilot imports (no cache) are skipped."""
    out = []
    idx = scout_index()
    for mp in sorted(SNAPS.glob("*.meta.json")):
        m = json.loads(mp.read_text())
        r = idx.get(norm_vid(m["video_id"]))
        if not r or m.get("from") == "vod_pilot2/snaps.csv":
            continue
        if _layout_sig(m.get("layout")) == _layout_sig(layout_for(r)):
            continue
        if not list(Path("/workspace/vod_data/cache/ocr").glob(f"{r['video_id']}.fps*.jsonl")):
            log(f"[layout] {r['video_id']}: layout changed but no OCR cache; left as is")
            continue
        out.append(r)
    return out


def _panel_rev() -> int:
    import vodx_field
    return getattr(vodx_field, "PANEL_ATTR_REV", 1)


PANEL_REDO_PER_RUN = 25  # cache-only redos per hourly run for the panel-attribution change (spread so new VODs are not held up)


def panel_attr_stale_batches(limit: int | None = PANEL_REDO_PER_RUN) -> list[dict]:
    """OCR-extracted batches written before the current panel->row attribution rule (meta panel_attr_rev older);
    redone from the OCR/dense cache only, at most `limit` per run (oldest first)."""
    out, idx, cur = [], scout_index(), _panel_rev()
    for p in sorted(SNAPS.glob("*.meta.json"), key=lambda q: q.stat().st_mtime):
        m = json.loads(p.read_text())
        if m.get("from") == "vod_pilot2/snaps.csv" or m.get("panel_attr_rev", 1) >= cur:
            continue
        r = idx.get(norm_vid(m.get("video_id", "")))
        if r and list(Path("/workspace/vod_data/cache/ocr").glob(f"{m['video_id']}.fps*.jsonl")):
            out.append(r)
        if limit and len(out) >= limit:
            break
    return out


def def_guard_stale_batches() -> list[dict]:
    """Snap batches whose play field still holds a defensive call name (extracted before vodx_names.guard_def_in_play);
    labels already fix them at label time, this rewrites the batch itself from the OCR cache."""
    import vodx_names
    out, idx = [], scout_index()
    for p in sorted(SNAPS.glob("*.jsonl")):
        m = p.with_suffix(".meta.json")
        if not m.exists() or json.loads(m.read_text()).get("from") == "vod_pilot2/snaps.csv":
            continue
        rows = read_jsonl(p)
        if not any(x.get("play") and not x.get("is_kickoff") and vodx_names._is_def(vodx_names._n(x["play"])) for x in rows):
            continue
        r = idx.get(norm_vid(rows[0]["video_id"])) if rows else None
        if r and list(Path("/workspace/vod_data/cache/ocr").glob(f"{r['video_id']}.fps*.jsonl")):
            out.append(r)
    return out


def mixed_needing_resplit() -> list[dict]:
    """Non-window manifest rows with game=mixed + game_segments whose snaps meta is not a matching per-segment batch
    and whose OCR cache exists."""
    out = []
    if not MANIFEST.exists():
        return out
    for l in MANIFEST.read_text().splitlines():
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("kind") == "window" or r.get("game") != "mixed" or not r.get("game_segments"):
            continue
        mp = SNAPS / f"{norm_vid(r['video_id'])}.meta.json"
        if not mp.exists():
            continue  # never extracted: the normal queue handles it per segment
        m = json.loads(mp.read_text())
        want = [(g["game"], float(g.get("start_s") or 0)) for g in r["game_segments"]]
        have = [(g["game"], float(g.get("start_s") or 0)) for g in (m.get("game_segments") or [])]
        if m.get("game") == "mixed" and have == want:
            continue
        if not list((Path("/workspace/vod_data/cache/ocr")).glob(f"{r['video_id']}.fps*.jsonl")):
            log(f"[mixed] {r['video_id']}: needs per-segment re-split but has no OCR cache (raw needed)")
            continue
        out.append(r)
    return out


def game_segments(r: dict) -> list[tuple[str, float, float]]:
    """[(game, start_s, end_s)] in absolute VOD seconds. game=="mixed" needs Scout's game_segments; otherwise one segment."""
    segs = r.get("game_segments") or []
    if r.get("game") == "mixed" and segs:
        out = []
        for g in sorted(segs, key=lambda g: float(g.get("start_s") or 0)):
            out.append((g["game"], float(g.get("start_s") or 0), float(g["end_s"]) if g.get("end_s") is not None else float("inf")))
        return out
    if r.get("game") == "mixed":
        raise ValueError(f"{r.get('video_id')}: game=mixed without game_segments")
    return [(r.get("game") or "madden27", 0.0, float("inf"))]


def extract_one(r: dict, fps: float, max_seconds: float | None, threads: int, cache_only: bool = False) -> dict:
    segs = game_segments(r)
    if len(segs) > 1:
        # one VOD, several games: run the full per-game pipeline (scorebug, refine rules) on each segment's frames,
        # then write one snap batch with per-row `game` and a per-row `game_segment` index
        rows, stats, nfr = [], {}, 0
        for i, (g, lo, hi) in enumerate(segs):
            sub = dict(r, game=g, game_segments=None, _seg=(lo, hi), _seg_index=i)
            res = extract_one(sub, fps, max_seconds, threads, cache_only)
            for x in res["rows"]:
                x["game_segment"] = i
            rows += res["rows"]; stats[f"seg{i}:{g}"] = res["refine_stats"]; nfr += res["frames_ocr"]
            last = res
        rows.sort(key=lambda x: x["t_start"])
        m = last["meta"]
        m.update({"game": "mixed", "game_segments": [{"game": g, "start_s": lo, "end_s": (None if hi == float("inf") else hi),
                  "snaps": sum(1 for x in rows if x["game_segment"] == i)} for i, (g, lo, hi) in enumerate(segs)],
                  "snaps": len(rows), "frames_ocr": nfr, "refine_stats": stats})
        return _write_batch(r, rows, m)
    from vodx_ocr import run_ocr, load_cache, load_frames
    from vodx_engine.lexicon import load_lexicon
    from vodx_engine.segment import classify_frame, build_snaps
    vid = r["video_id"]
    path = r.get("local_path")
    if (not path or not Path(path).exists()) and not cache_only:
        raise FileNotFoundError(f"local_path missing: {path}")
    if not cache_only:
        run_ocr(path, vid, fps, max_seconds, threads, log=log)
    meta, frames = load_cache(vid, fps)
    # densified frames (vodx.py densify) feed the panel / scorebug / adjustment readers, not snap segmentation,
    # so snap ids stay stable across label versions
    dmeta, frames_all = load_frames(vid, fps)
    import vodx_windows  # re-pulled clip windows (raw gone): reader frames only, shifted to section time
    wframes = vodx_windows.load_window_frames(vid)
    if wframes:
        frames_all = sorted(frames_all + wframes, key=lambda f: f["t"])
    lay = layout_for(r)
    off = section_offset(r)
    if max_seconds is not None:
        frames = [f for f in frames if f["t"] <= max_seconds]
        frames_all = [f for f in frames_all if f["t"] <= max_seconds]
    if r.get("_seg"):  # one game segment of a mixed VOD (segment bounds are absolute VOD seconds)
        lo, hi = r["_seg"]
        frames = [f for f in frames if lo <= f["t"] + off < hi]
        frames_all = [f for f in frames_all if lo <= f["t"] + off < hi]
    # Masks and the gameplay window run on cached token x/y before classification,
    # including --cache-only. Pre-game and ticker chat never become snaps.
    for extra in (str(TOOLS), str(TOOLS.parent)):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    import vodx_pr88
    frames = vodx_pr88.apply_layout(frames, lay, section_offset_s=off)
    frames_all = vodx_pr88.apply_layout(frames_all, lay, section_offset_s=off)
    lex = load_lexicon()
    game = r.get("game") or "madden27"
    extract = vodx_pr88.flags_from_fixes(CURRENT_FIXES)
    snaps, obs = vodx_pr88.observe_frames(
        frames, video_id=vid, channel=r.get("creator") or "", game=game, extract=extract, lexicon=lex,
    )
    plat = platform_of(r)
    scout = scout_index().get(norm_vid(vid))
    opp = (scout or {}).get("opponent_type") or "unknown"
    if opp not in ("cpu", "human", "unknown"):
        opp = "unknown"
    opp_src = "scout_manifest" if scout and opp != "unknown" else "missing_scout_tag"
    opp_basis = (scout or {}).get("opponent_type_basis", "")
    opp_conf = 1.0 if opp_src == "scout_manifest" else 0.0
    if opp == "unknown" and r.get("opponent_type_fallback") in ("cpu", "human"):
        opp, opp_src = r["opponent_type_fallback"], "pilot_metadata_no_scout_tag"
        opp_basis, opp_conf = r.get("opponent_type_fallback_basis", ""), 0.7
    opp, opp_src, opp_basis, opp_conf = opp_override(vid, opp, opp_src, opp_basis, opp_conf)
    rows = []
    prev_end = 0.0
    for i, s in enumerate(snaps):
        d = s.to_full_dict()
        t0, t1 = float(d["t_start"]), float(d["t_end"])
        m = re.search(r"q(\d)\s+(\d{1,2}:\d{2})", d.get("situation_raw") or "")
        clock = m.group(2) if m else ""
        adj = presnap_adjustments(frames_all, max(prev_end, t0 - 16), t1, lex) if d.get("play") != "Kickoff" else \
            {k: ("", 0.0, "") for k in list(ADJ) + ["def_overlay"]}
        prev_end = t1
        c = d.get("confidence") or {}
        kind = d.get("result_kind") or ""
        if kind == "no_gain":
            kind = "gain"
        yards = d.get("yards")
        row = {
            "snap_id": f"{norm_vid(vid)}:{t0:.1f}", "video_id": vid, "game": game, "creator": r.get("creator") or "",
            "source_platform": plat, "source_url": r.get("url") or "", "ts_url": ts_url(r.get("url") or "", plat, t0 + off),
            "t_start": t0, "t_end": t1, "section_offset_s": off, "t_start_vod": t0 + off, "opponent_type": opp, "opponent_type_source": opp_src,
            "opponent_type_basis": opp_basis,
            "side": d.get("side") or "", "quarter": d.get("quarter"), "clock": clock, "score": None,
            "down": vodx_pr88.as_int(d.get("down")), "distance": vodx_pr88.as_int(d.get("distance")),
            "yardline": vodx_pr88.as_int(d.get("yardline")),
            "yard_from_glyph": bool(d.get("yard_from_glyph")),
            "base_yardline": vodx_pr88.as_int(d.get("base_yardline")),
            "glyph_yardline": d.get("glyph_yardline"),
            "glyph_phrase": d.get("glyph_phrase") or "",
            "glyph_conf": float(d.get("glyph_conf") or 0.0),
            "field_spot": d.get("field_spot"),
            "baseline_distance": vodx_pr88.as_int(d.get("baseline_distance")),
            "situation_raw": d.get("situation_raw") or "",
            "formation": d.get("formation") or "", "play": d.get("play") or "",
            "coverage_seen": d.get("coverage_seen") or "", "concept_seen": d.get("concept_seen") or "",
            "def_overlay": adj["def_overlay"][0],
            "audible": adj["audible"][0], "hot_routes": adj["hot_routes"][0],
            "off_macros": adj["off_macros"][0], "def_adjustments": adj["def_adjustments"][0],
            "macro": adj["off_macros"][0],
            "result": d.get("result") or "", "result_kind": kind, "yards": yards,
            "is_kickoff": d.get("play") == "Kickoff",
            "def_formation": d.get("def_formation") or "",
            "team_left": d.get("team_left") or "", "team_right": d.get("team_right") or "",
            "score_left": d.get("score_left"), "score_right": d.get("score_right"),
            "record_left": d.get("record_left") or "", "record_right": d.get("record_right") or "",
            "adjustments": d.get("adjustments") or [],
            "goal_to_go": bool(d.get("goal_to_go")),
            "is_menu_or_lobby": bool(d.get("is_menu_or_lobby")),
            "menu_confidence": float(d.get("menu_confidence") or 0.0),
            "result_flag": d.get("result_flag") or "",
            "extractor_version": script_version(), "panel_attr_rev": _panel_rev(), "engine": ENGINE_REF,
        }
        if row["score_left"] is not None and row["score_right"] is not None:
            row["score"] = f"{row['score_left']}-{row['score_right']}"
        if extract.read_adjustments:
            for key, text in vodx_pr88._adjustment_fields(d.get("adjustments") or []).items():
                if float(c.get("adjustments") or 0.0) > float(adj[key][1] or 0.0):
                    row[key] = text
        dd_conf = c.get("down", 0.0) if row["down"] is not None else 0.0
        conf = {
            "down": dd_conf, "distance": dd_conf,
            "quarter": round(sum(1 for f in obs if t0 <= f.t <= t1 and f.quarter == row["quarter"]) /
                             max(1, sum(1 for f in obs if t0 <= f.t <= t1)), 2) if row["quarter"] else 0.0,
            "clock": 0.8 if clock else 0.0,
            "yardline": c.get("yardline", 0.0) if row["yardline"] is not None else 0.0,
            "formation": c.get("formation", 0.0) if row["formation"] else 0.0,
            "play": (0.9 if row["is_kickoff"] else c.get("play", 0.0)) if row["play"] else 0.0,
            "coverage_seen": c.get("coverage", 0.0) if row["coverage_seen"] else 0.0,
            "concept_seen": 0.3 if row["concept_seen"] else 0.0,
            "result": c.get("result", 0.0) if row["result"] else 0.0,
            "audible": adj["audible"][1], "hot_routes": adj["hot_routes"][1],
            "off_macros": adj["off_macros"][1], "def_adjustments": adj["def_adjustments"][1],
            "def_overlay": adj["def_overlay"][1],
            "opponent_type": opp_conf,
            "score": 0.0,
        }
        src = {
            "down": "ocr_scorebug", "distance": "ocr_scorebug", "quarter": "ocr_scorebug", "clock": "ocr_scorebug",
            "yardline": "engine", "formation": "ocr_call_screen_lexicon",
            "play": "ocr_tooltip_or_previous_play_panel", "coverage_seen": "ocr_previous_play_panel",
            "concept_seen": "ocr_call_screen_route_words", "result": "dd_delta_or_banner",
            "audible": adj["audible"][2], "hot_routes": adj["hot_routes"][2], "off_macros": adj["off_macros"][2],
            "def_adjustments": adj["def_adjustments"][2], "def_overlay": adj["def_overlay"][2],
            "opponent_type": opp_src,
        }
        if lay["calls"] == "no_panel":
            for k in ("formation", "play", "coverage_seen"):
                if row[k] and not row["is_kickoff"]:
                    conf[k] = min(conf[k], 0.2); src[k] = src[k] + ":no_panel_low_conf"
        row["conf"] = {k: float(v) for k, v in conf.items()}
        row["field_sources"] = {k: v for k, v in src.items() if row.get(k) not in ("", None) or k == "opponent_type"}
        # PR #88 readers win over refine / field / score / cards unless that stage is stricter.
        row["_pr88_conf"] = {}
        if extract.read_defense:
            row["_pr88_conf"]["def_formation"] = float(c.get("def_formation") or 0.0)
            row["conf"]["def_formation"] = row["_pr88_conf"]["def_formation"]
        if extract.read_score:
            row["_pr88_conf"]["score"] = float(c.get("score") or 0.0)
            row["conf"]["score"] = row["_pr88_conf"]["score"]
        if extract.read_adjustments:
            row["_pr88_conf"]["adjustments"] = float(c.get("adjustments") or 0.0)
            row["conf"]["adjustments"] = row["_pr88_conf"]["adjustments"]
        if extract.read_offense_formation:
            row["_pr88_conf"]["formation"] = float(c.get("formation") or 0.0)
        if extract.dd_norm or extract.goal_to_go:
            row["_pr88_conf"]["down"] = float(c.get("down") or 0.0)
            row["_pr88_conf"]["distance"] = float(c.get("down") or 0.0)
        if extract.goal_to_go:
            row["_pr88_conf"]["goal_to_go"] = 0.8 if row["goal_to_go"] else 0.0
        if extract.read_result:
            row["_pr88_conf"]["result"] = float(c.get("result") or 0.0)
        if extract.presnap_clock and row["clock"]:
            row["_pr88_conf"]["clock"] = 0.85
            row["conf"]["clock"] = 0.85
        if extract.flag_menu:
            row["_pr88_conf"]["menu"] = float(row["menu_confidence"] or 0.0)
        rows.append(row)
    for row in rows:
        vodx_pr88.coerce_row_numbers(row)
        # Glyph and penalty-card labels must not be trusted before vodx_field.
        vodx_pr88.prepare_yardline_for_field(row)
    pr88_snapshot = vodx_pr88.snapshot_rows(rows)
    vodx_pr88.install_quality_guard()
    from vodx_refine import refine_rows
    refine_stats = refine_rows(rows, frames_all, game)
    import vodx_speech  # (v0.8) spoken adjustments from cache/speech/<id>.json; separate speech_* fields
    speech_stats = vodx_speech.apply_speech(rows, vid)
    # vodx_field has reconciled the engine line with the scorebug. Lock that
    # published value. The raw engine number must not be restored over it.
    from vod_pilot.segment import capture_base_yardlines, preserve_base_yardlines

    capture_base_yardlines(rows)
    # vodx_refine, vodx_field, vodx_score, and vodx_cards have already written
    # through refine_rows / apply_speech. Restore a PR #88 field when that
    # stage was not strictly more confident.
    vodx_pr88.restore_rows(rows, pr88_snapshot)
    preserve_base_yardlines(rows)
    if extract.yardline_glyphs:
        vodx_pr88.gate_glyph_yardlines(rows)
    preserve_base_yardlines(rows)
    # A later field pass may have flagged spot_contradicts_plus0 and still
    # left a +0. That +0 is not a result. Filled gains and losses stay.
    vodx_pr88.withhold_contradicted_plus0(rows)
    batch_meta = {"schema": SCHEMA_SNAPS, "video_id": vid, "game": game, "creator": r.get("creator"),
                  "source_platform": plat, "source_url": r.get("url"), "local_path": path, "opponent_type": opp,
                  "opponent_type_source": opp_src, "sample_fps": fps, "max_seconds": max_seconds,
                  "frames_ocr": len(frames), "dense_frames": dmeta.get("dense_frames", 0), "window_frames": len(wframes), "seconds_covered": frames[-1]["t"] if frames else 0,
                  "snaps": len(rows), "extractor_version": script_version(), "panel_attr_rev": _panel_rev(), "engine": ENGINE_REF,
                  "layout": lay, "section_offset_s": off, "cache_only": cache_only,
                  "extracted_at": now_iso(), "pilot_key": r.get("pilot_key"), "from": r.get("_from"),
                  "refine": "vodx_refine.py (analyst note_20261007_2004 fixes c/d/e)", "refine_stats": refine_stats, "speech_stats": speech_stats,
                  "fixes": extract.enabled(), "source": {"fixes": extract.enabled(), "cache_only": cache_only}}
    if r.get("_seg"):
        return {"rows": rows, "meta": batch_meta, "refine_stats": refine_stats, "frames_ocr": len(frames)}
    return _write_batch(r, rows, batch_meta)


def _write_batch(r: dict, rows: list[dict], batch_meta: dict) -> dict:
    batch_meta["snaps_unique"] = len({x["snap_id"] for x in rows}); batch_meta["rows"] = len(rows)
    if batch_meta["snaps_unique"] != len(rows):  # never publish duplicate snap ids: keep the first (they are sorted)
        seen, keep = set(), []
        for x in rows:
            if x["snap_id"] not in seen:
                seen.add(x["snap_id"]); keep.append(x)
        log(f"[extract] {r['video_id']}: dropped {len(rows) - len(keep)} duplicate snap_id rows")
        batch_meta["duplicates_dropped"] = len(rows) - len(keep)
        rows[:] = keep; batch_meta["snaps"] = batch_meta["rows"] = batch_meta["snaps_unique"] = len(keep)
    SNAPS.mkdir(parents=True, exist_ok=True)
    stem = SNAPS / norm_vid(r["video_id"])
    tmp = stem.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(x) + "\n" for x in rows)); os.replace(tmp, stem.with_suffix(".jsonl"))
    write_csv(stem.with_suffix(".csv"), rows)
    stem.with_suffix(".meta.json").write_text(json.dumps(batch_meta, indent=2))
    return batch_meta


def flat(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k not in ("conf", "field_sources")}
    for k, v in (row.get("conf") or {}).items():
        out[f"conf_{k}"] = v
    for k, v in (row.get("field_sources") or {}).items():
        out[f"src_{k}"] = v
    return out


def write_csv(p: Path, rows: list[dict]) -> None:
    fl = [flat(r) for r in rows]
    cols: list[str] = []
    for r in fl:
        for k in r:
            if k not in cols:
                cols.append(k)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols or ["snap_id"]); w.writeheader(); w.writerows(fl)
    os.replace(tmp, p)


def write_batch_notes() -> None:
    metas = [json.loads(p.read_text()) for p in sorted(SNAPS.glob("*.meta.json"))]
    lines = ["# Snap batches: notes", "", f"Regenerated {now_iso()} by `vodx.py` ({script_version()}).", "",
             "opponent_type comes from VOD Scout's per-VOD tag in /workspace/vod_data/manifest.jsonl and is repeated on every row.",
             "Scout tags one value per VOD, so every game inside a VOD gets that VOD's value.", "",
             "| video_id | creator | game | snaps | seconds covered | opponent_type | source |", "|---|---|---|---|---|---|---|"]
    unknown = []
    for m in metas:
        lines.append(f"| {m['video_id']} | {m.get('creator')} | {m['game']} | {m['snaps']} | {m['seconds_covered']:.0f}"
                     f"{' (window '+str(int(m['max_seconds']))+'s)' if m.get('max_seconds') else ''} | {m['opponent_type']} | {m['opponent_type_source']} |")
        if m["opponent_type"] == "unknown" or m["opponent_type_source"] != "scout_manifest":
            unknown.append(m)
    lines += ["", "## FLAG: no Scout opponent_type tag", ""]
    lines += [(f"- {m['video_id']} ({m.get('creator')}, {m['game']}): not in Scout's manifest -> `unknown` on all {m['snaps']} rows"
               if m["opponent_type"] == "unknown" else
               f"- {m['video_id']} ({m.get('creator')}, {m['game']}): not in Scout's manifest; `{m['opponent_type']}` from pilot metadata "
               f"(conf 0.7, source {m['opponent_type_source']}) - needs a Scout tag to confirm")
              for m in unknown] or ["- none"]
    if _relabel_log().exists():
        rl = json.loads(_relabel_log().read_text())
        lines += ["", f"## opponent_type changed at label time ({rl.get('version')}; current Scout manifest tag > override > stored batch value)", ""]
        lines += [f"- {x['video_id']}: {x['from']} ({x['from_source']}) -> {x['to']} ({x['to_source']}); {x['rows_type_changed']} of {x['rows']} rows changed type"
                  for x in rl.get("relabels", [])] or ["- none"]
    (SNAPS / "NOTES.md").write_text("\n".join(lines) + "\n")

# ----------------------------------------------------------------------------------------------- labels

LABEL_FIELDS = ["label_id", "video_id", "snap_id", "game", "creator", "source_platform", "source_url", "ts_url",
                "t_start", "t_end", "section_offset_s", "t_start_vod", "opponent_type", "opponent_type_source",
                "down", "distance", "yardline", "yards_to_goal", "territory", "score", "team_left", "team_right", "score_left", "score_right", "clock", "quarter", "situation_raw", "side",
                "formation", "call_screen_formation", "play", "coverage_seen", "def_call", "def_formation", "def_overlay", "concept_seen", "field_spot",
                "audible", "hot_routes", "off_macros", "def_adjustments", "macro",
                "speech_audible", "speech_hot_routes", "speech_off_macros", "speech_def_adjustments", "speech_adjustments", "has_speech_adjustment",
                "result", "result_kind", "yards", "turnover", "touchdown", "sack", "success", "success_known",
                "clean_pair", "has_adjustment", "result_check", "play_class", "possession_seg", "side_evidence",
                "quality_flags", "pilot2_seg", "t_approx", "name_ambiguous", "name_fixes", "name_candidates",
                "possession", "coverage_family", "result_fix", "result_unverified", "game_segment",
                "goal_to_go", "is_menu_or_lobby", "menu_confidence", "adjustments", "result_flag",
                "record_left", "record_right"]


def call_pair(row: dict) -> bool:
    return bool(row.get("play")) and not row.get("is_kickoff") and row["conf"].get("play", 0) >= 0.5 \
        and any(bool(row.get(k)) and row["conf"].get(k, 0) >= 0.5 for k in ("coverage_seen", "def_call", "def_overlay"))


FIELD_DEFINITIONS = {
    "call_pair": "offensive call (play, conf >= 0.5) AND a defensive look (coverage_seen / def_call / def_overlay, "
                 "conf >= 0.5) on the same scrimmage snap, regardless of the result (= clean_pair before v0.14)",
    "clean_pair": "call_pair AND success_known (v0.14+); the training pair with a known outcome",
    "success": "learning.evaluate_snap rule, offense perspective; null when the result is unknown or unverified",
    "result_fix": "(v0.14) result corrected/nulled by the next-snap consistency pass: {kind, before, after}",
    "result_unverified": "(v0.14) reason the result could not be verified (yards/success/sack nulled)",
    "possession": "streamer_offense | streamer_defense | special_teams | unknown: the streamer's unit on the snap",
    "score_margin_streamer": "streamer score - opponent score (streamer perspective)",
    "score_margin_offense": "score margin from the offense's perspective",
    "opponent_type_source": "scout_manifest (Scout tag) | pilot_metadata (tools/opponent_overrides.json, conf 0.7) | "
                            "pilot_metadata_no_scout_tag | missing_scout_tag",
}


def to_label(row: dict) -> dict:
    kind = row.get("result_kind") or ""
    succ = offense_success(row.get("down"), row.get("distance"), row.get("yardline"), kind, row.get("yards"))
    if row.get("result_unverified"):  # (v0.14) e.g. a "sack" whose positive yards the next snap does not support
        succ = None
    lab = {k: row.get(k) for k in LABEL_FIELDS if k in row}
    yl = lab.get("yardline")
    if isinstance(yl, (int, float)) and 0 <= yl <= 100 and lab.get("yards_to_goal") is None:  # e.g. pilot2-imported rows
        lab["yards_to_goal"] = 100 - yl
        lab["territory"] = "own" if yl < 50 else "opp" if yl > 50 else "mid"
        row["conf"]["yards_to_goal"] = row["conf"]["territory"] = row["conf"].get("yardline", 0.0)
        row["field_sources"]["yards_to_goal"] = row["field_sources"]["territory"] = "derived:100-yardline"
    lab.update({
        "label_id": "L:" + row["snap_id"],
        "turnover": kind in ("int", "fumble") if kind else None,
        "touchdown": kind == "td" if kind else None,
        "sack": (kind == "sack" if kind else None) if not row.get("result_unverified") else None,
        "success": succ, "success_known": succ is not None,
        "offense_call": {"formation": row.get("formation") or None, "play": row.get("play") or None},
        "defense_look": {"coverage_seen": row.get("coverage_seen") or None, "def_call": row.get("def_call") or None,
                         "def_formation": row.get("def_formation") or None, "def_overlay": row.get("def_overlay") or None},
        "adjustments": {k: {"value": row.get(k) or None, "conf": row["conf"].get(k, 0.0),
                            "src": row["field_sources"].get(k)} for k in ("audible", "hot_routes", "off_macros", "def_adjustments")},
        # (v0.14) call_pair = offensive call + defensive look both read (conf >= 0.5), regardless of the result
        # (the pre-v0.14 meaning of clean_pair); clean_pair = call_pair AND success_known
        "call_pair": call_pair(row),
        "clean_pair": call_pair(row) and succ is not None,
        "has_adjustment": any(row.get(k) for k in ("audible", "hot_routes", "off_macros", "def_adjustments")),
        "has_speech_adjustment": bool(row.get("speech_adjustments")),
        "has_adjustment_any": any(row.get(k) for k in ("audible", "hot_routes", "off_macros", "def_adjustments"))
                              or bool(row.get("speech_adjustments")),
        "conf": dict(row["conf"], success=row["conf"].get("result", 0.0) if succ is not None else 0.0),
        "field_sources": dict(row["field_sources"], **({"success": "rule:" + SUCCESS_REF} if succ is not None else {})),
    })
    return lab


def next_version(explicit: str | None) -> str:
    LABELS.mkdir(parents=True, exist_ok=True)
    existing = sorted([p.name for p in LABELS.iterdir() if p.is_dir() and re.fullmatch(r"v\d+\.\d+", p.name)],
                      key=lambda s: tuple(int(x) for x in s[1:].split(".")))
    if explicit:
        if (LABELS / explicit).exists():
            raise SystemExit(f"{explicit} already exists; label dirs are never overwritten")
        return explicit
    if not existing:
        return "v0.1"
    maj, mn = (int(x) for x in existing[-1][1:].split("."))
    return f"v{maj}.{mn + 1}"


def prev_capture_rates() -> dict | None:
    """capture_rates() of the current LATEST label version (before this build), for before/after in summary.json."""
    latest = LABELS / "LATEST.json"
    try:
        d = Path(json.loads(latest.read_text())["path"])
        labs = [json.loads(l) for p in sorted(d.glob("*.jsonl.gz")) for l in gzip.open(p, "rt")]
        import vodx_possession
        return dict(vodx_possession.capture_rates(labs), version=d.name)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def prev_quality() -> dict | None:
    latest = LABELS / "LATEST.json"
    try:
        d = Path(json.loads(latest.read_text())["path"])
        labs = [json.loads(l) for p in sorted(d.glob("*.jsonl.gz")) for l in gzip.open(p, "rt")]
        import vodx_quality
        return dict(vodx_quality.quality_metrics(labs), version=d.name)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def formlist_frames(video_id: str):
    """OCR (+dense) cache frames of a VOD for the label-time formation-list guard; None when there is no cache."""
    mp = SNAPS / f"{norm_vid(video_id)}.meta.json"
    if not mp.exists():
        return None
    m = json.loads(mp.read_text())
    if m.get("from") == "vod_pilot2/snaps.csv":
        return None
    import vodx_ocr
    try:
        return vodx_ocr.load_frames(m["video_id"], float(m.get("sample_fps") or 1.0))[1]
    except (FileNotFoundError, ValueError, KeyError):
        return None


def build_labels(version: str | None, video_ids: list[str] | None, note: str) -> Path:
    metas = {norm_vid(json.loads(p.read_text())["video_id"]): json.loads(p.read_text()) for p in SNAPS.glob("*.meta.json")}
    if video_ids:
        metas = {k: v for k, v in metas.items() if k in {norm_vid(x) for x in video_ids}}
    # skip if identical to LATEST (same batches + same script) -> idempotent hourly calls
    hsig = hashlib.sha256(script_version().encode())
    for k in sorted(metas):
        hsig.update(k.encode()); hsig.update((SNAPS / f"{k}.jsonl").read_bytes())
    sig = hsig.hexdigest()[:12]
    latest = LABELS / "LATEST.json"
    if latest.exists() and json.loads(latest.read_text()).get("input_signature") == sig and not version:
        log(f"[label] inputs unchanged since {json.loads(latest.read_text())['latest']}; nothing to do")
        return Path(json.loads(latest.read_text())["path"])
    ver = next_version(version)
    out = LABELS / ver
    tmpdir = LABELS / f".{ver}.tmp"
    if tmpdir.exists():
        for p in tmpdir.iterdir():
            p.unlink()
    tmpdir.mkdir(parents=True, exist_ok=True)
    all_labels = []
    per_vod = []
    staged = []
    sidx, relabels = scout_index(), []
    for k, m in sorted(metas.items()):
        labs = [to_label(r) for r in read_jsonl(SNAPS / f"{k}.jsonl")]
        assign_match_index(labs, LAYOUTS["videos"].get(norm_vid(m["video_id"]), {}).get("match_starts"))
        old = (m.get("opponent_type") or "unknown", m.get("opponent_type_source") or "", "", 0.0)
        typ, src, basis, conf = current_opponent(m["video_id"], old, sidx.get(norm_vid(m["video_id"])))
        changed = 0
        for l in labs:
            if l.get("opponent_type") != typ or l.get("opponent_type_source") != src:
                changed += l.get("opponent_type") != typ
                l["opponent_type"], l["opponent_type_source"], l["opponent_type_basis"] = typ, src, basis
                (l.setdefault("conf", {}))["opponent_type"] = conf if typ != "unknown" else 0.0
                (l.setdefault("field_sources", {}))["opponent_type"] = src
        if typ != m.get("opponent_type") or src != m.get("opponent_type_source"):
            relabels.append({"video_id": m["video_id"], "from": m.get("opponent_type"), "from_source": m.get("opponent_type_source"),
                             "to": typ, "to_source": src, "basis": basis, "rows": len(labs), "rows_type_changed": changed, "version": ver})
        m = dict(m, opponent_type=typ, opponent_type_source=src)
        staged.append((k, m, labs)); all_labels += labs
    import vodx_possession  # streamer team per creator + streamer-perspective score margin (needs every VOD)
    pos_stats = vodx_possession.label_pass(all_labels)
    import vodx_names  # (v0.20) label-time guard too, so cached batches get it without re-extraction
    pos_stats.update({f"def_guard:{k}": v for k, v in vodx_names.guard_def_in_play(all_labels).items()})
    import vodx_labelfix  # (v0.29) analyst v0.28 fixes: sack/turnover yards, no-gain split, fragments, down jumps, penalty 1st
    _lf, _kept, _dedupe_log = vodx_labelfix.apply(all_labels)
    pos_stats.update({f"labelfix:{k}": v for k, v in _lf.items()})
    import vodx_chatjunk  # (v0.34) chat-overlay junk names (@user, yyyy, non-lexicon) -> nulled, raw kept in *_raw
    pos_stats.update({f"chatjunk:{k}": v for k, v in vodx_chatjunk.apply(all_labels).items()})
    if len(_kept) != len(all_labels):
        _keep_ids = {id(l) for l in _kept}
        staged = [(k, m, [l for l in labs if id(l) in _keep_ids]) for k, m, labs in staged]
        all_labels = _kept
        log(f"[label] dropped {len(_dedupe_log)} duplicate fragment rows: {[d['dropped'] for d in _dedupe_log][:20]}")
    import vodx_formlist  # (v0.39) formation-list-only Hail Mary / Goal Line / Field Goal play + dropped-play carry-over
    pos_stats.update({f"formlist:{k}": v for k, v in vodx_formlist.apply_labels(all_labels, formlist_frames).items()})
    import vodx_expert  # Role 4: provenance/permitted_use, review_status, match split, offense/defense expert sets
    _mrows = {}
    if MANIFEST.exists():
        for _l in MANIFEST.read_text().splitlines():
            try:
                _r = json.loads(_l)
            except ValueError:
                continue
            if _r.get("kind") != "window" and _r.get("video_id"):
                _mrows[_r["video_id"]] = _r
    role4 = vodx_expert.publish(tmpdir, ver, all_labels, _mrows)
    _relabel_log().write_text(json.dumps({"version": ver, "relabels": relabels}, indent=1))
    write_batch_notes()
    for k, m, labs in staged:
        with gzip.open(tmpdir / f"{k}.jsonl.gz", "wt") as fh:
            for l in labs:
                fh.write(json.dumps(l) + "\n")
        per_vod.append({"video_id": m["video_id"], "game": m["game"], "creator": m.get("creator"), "source_url": m.get("source_url"),
                        "opponent_type": m["opponent_type"], "opponent_type_source": m["opponent_type_source"],
                        "snaps": len(labs), "snaps_unique": len({l["snap_id"] for l in labs}), "label_rows": len(labs),
                        "offense_expert_rows": sum(1 for l in labs if l.get("possession") == "streamer_offense" and not l.get("excluded_from_training")),
                        "defense_expert_rows": sum(1 for l in labs if l.get("possession") == "streamer_defense" and not l.get("excluded_from_training")),
                        "seconds_covered": m["seconds_covered"], "max_seconds": m.get("max_seconds"),
                        "extracted_at": m["extracted_at"], "extractor_version": m["extractor_version"]})
    write_csv(tmpdir / "labels.csv", [dict(l, conf=l["conf"], field_sources=l["field_sources"]) |
                                      {"offense_call": "", "defense_look": "", "adjustments": ""} for l in all_labels])
    summary = summarize(all_labels)
    summary["possession_pass"] = dict(pos_stats)
    summary["labelfix_v029"] = {"stats": {k[9:]: v for k, v in pos_stats.items() if k.startswith("labelfix:")},
                                "fragments_dropped": _dedupe_log,
                                "result_kind_split": dict(Counter(l.get("result_kind") for l in all_labels if l.get("result_kind_raw")))}
    summary["capture_rates"] = vodx_possession.capture_rates(all_labels)
    summary["capture_rates_prev"] = prev_capture_rates()
    import vodx_quality
    summary["quality"] = vodx_quality.quality_metrics(all_labels)
    summary["quality_prev"] = prev_quality()
    summary["role4"] = role4
    if isinstance(role4, dict) and role4.get("trainable_with_unreviewed_waiver"):
        summary["trainable_with_unreviewed_waiver"] = role4["trainable_with_unreviewed_waiver"]
    summary["opponent_type_relabels"] = relabels
    _fixes = []
    for _m in metas.values():
        for _name in (_m.get("fixes") or (_m.get("source") or {}).get("fixes") or []):
            if _name not in _fixes:
                _fixes.append(_name)
    try:
        for _p in (str(TOOLS), str(TOOLS.parent)):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        import vodx_pr88 as _pr88
        _coverage = _pr88.coverage_report(all_labels)
    except Exception:  # noqa: BLE001
        _coverage = {}
    summary["source"] = {"fixes": _fixes, "cache_only": any(bool(m.get("cache_only")) for m in metas.values())}
    summary["field_coverage"] = _coverage
    manifest = {"version": ver, "schema_version": SCHEMA_LABELS, "snap_schema": SCHEMA_SNAPS, "created_at": now_iso(),
                "source": {"fixes": _fixes, "cache_only": any(bool(m.get("cache_only")) for m in metas.values())},
                "field_coverage": _coverage,
                "script_version": script_version(), "engine": ENGINE_REF, "success_rule": SUCCESS_REF,
                "input_signature": sig, "source_vods": per_vod, "counts": summary["counts"], "note": note,
                "score_margin_perspective": "score_margin_streamer = streamer score - opponent score (streamer perspective); "
                                            "score_margin_offense = same from the offense's perspective; possession = "
                                            "streamer_offense|streamer_defense|special_teams|unknown (the streamer's unit)",
                "capture_rates": summary["capture_rates"]["by_group"], "field_definitions": FIELD_DEFINITIONS,
                "row_model": "One label row per snap in <video_id>.jsonl.gz (snap_id unique; snaps == snaps_unique == label_rows). "
                             "offense_expert.jsonl.gz and defense_expert.jsonl.gz are SUBSETS of those same rows (possession == "
                             "streamer_offense / streamer_defense, F8Hm324SMtg excluded), not extra snaps: do not glob *.jsonl.gz "
                             "together with them, or snaps are counted up to ~1.8x and repeat."}
    (tmpdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (tmpdir / "summary.json").write_text(json.dumps(summary, indent=2))
    if MANIFEST.exists():
        (LABELS / f"manifest_snapshot_for_{ver}.jsonl").write_text(MANIFEST.read_text())
    os.replace(tmpdir, out)
    tmp = latest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"latest": ver, "path": str(out), "schema_version": SCHEMA_LABELS,
                               "vods": len(per_vod), "rows": summary["counts"]["snaps_labeled"],
                               "clean_pairs": summary["counts"]["clean_pairs"], "script_version": script_version(),
                               "input_signature": sig, "created_at": manifest["created_at"],
                               "note": "Each label build is a full re-run into a new v0.<N>; old dirs are never overwritten."}, indent=1))
    os.replace(tmp, latest)
    log(f"[label] wrote {out} ({summary['counts']['snaps_labeled']} rows); LATEST -> {ver}")
    return out


COV_FIELDS = ["side", "possession", "coverage_family", "down", "distance", "yardline", "yards_to_goal", "territory", "field_spot", "quarter", "clock", "score", "formation", "call_screen_formation", "play",
              "coverage_seen", "def_call", "def_overlay", "concept_seen", "audible", "hot_routes", "off_macros", "def_adjustments", "result", "success",
              "opponent_type"]


def summarize(labels: list[dict]) -> dict:
    scrim = [l for l in labels if l.get("play") != "Kickoff"]
    n = len(labels)
    cov = {}
    for f in COV_FIELDS:
        if f == "opponent_type":
            pop = [l for l in labels if l.get(f) in ("cpu", "human")]
        elif f == "side":
            pop = [l for l in labels if l.get(f) in ("offense", "defense")]
        elif f == "success":
            pop = [l for l in labels if l.get("success") is not None]
        else:
            pop = [l for l in labels if l.get(f) not in ("", None)]
        cs = [l["conf"].get(f, 0.0) for l in pop]
        cov[f] = {"populated": len(pop), "coverage": round(len(pop) / n, 3) if n else None,
                  "mean_conf_when_populated": round(st.mean(cs), 3) if cs else None}
    split = {k: sum(1 for l in labels if l.get("opponent_type") == k) for k in ("cpu", "human", "unknown")}
    adj_any = sum(1 for l in scrim if l["has_adjustment"])
    succ = [l for l in labels if l.get("success") is not None]
    return {"counts": {"vods": len({l["video_id"] for l in labels}), "snaps_labeled": n, "scrimmage_snaps": len(scrim),
                       "clean_pairs": sum(1 for l in labels if l["clean_pair"]),
                       "call_pairs": sum(1 for l in labels if l.get("call_pair", l["clean_pair"])),
                       "offense_call_captured": sum(1 for l in scrim if l.get("play")),
                       "defense_look_captured": sum(1 for l in scrim if l.get("coverage_seen") or l.get("def_call") or l.get("def_overlay")),
                       "side_split": dict(Counter(l.get("side") or "" for l in labels)),
                       "result_kind_split": dict(Counter(l.get("result_kind") or "null" for l in scrim)),
                       "opponent_type_split": split,
                       "adjustment_capture_rate": round(adj_any / len(scrim), 3) if scrim else None,
                       "snaps_with_any_adjustment": adj_any,
                       "adjustment_capture_rate_screen_plus_speech": round(sum(1 for l in scrim if l.get("has_adjustment_any")) / len(scrim), 3) if scrim else None,
                       "snaps_with_speech_adjustment": sum(1 for l in scrim if l.get("has_speech_adjustment")),
                       "speech_only_adjustment_snaps": sum(1 for l in scrim if l.get("has_speech_adjustment") and not l["has_adjustment"]),
                       "speech_adjustment_by_type": {k: sum(1 for l in scrim if l.get("speech_" + k)) for k in ("audible", "hot_routes", "off_macros", "def_adjustments")},
                       "adjustment_by_type": {k: sum(1 for l in scrim if l.get(k)) for k in ("audible", "hot_routes", "off_macros", "def_adjustments")},
                       "name_fixed_rows": sum(1 for l in labels if l.get("name_fixes")),
                       "name_flagged_rows": sum(1 for l in labels if l.get("name_ambiguous")),
                       "name_fix_kinds": dict(Counter(f["kind"] for l in labels for f in l.get("name_fixes") or [])),
                       "formation_sources": dict(Counter((l["field_sources"].get("formation") or "").split(":")[0]
                                                         for l in scrim if l.get("formation"))),
                       "success_known": len(succ), "success_rate": round(sum(1 for l in succ if l["success"]) / len(succ), 3) if succ else None},
            "coverage": cov}

# ----------------------------------------------------------------------------------------------- CLI

PILOT2 = Path("/workspace/vod_pilot2/snaps.csv")
CONF_WORD = {"high": 0.9, "med": 0.6, "low": 0.3, "unknown": 0.0, "": 0.0}


PILOT2_COMP = {"z5ne-uBM0YI": Path("/workspace/vod_pilot2/m1/comp/meta.json")}


def pilot2_times(vid: str, ref: list[dict]) -> dict:
    """seg -> (t, approx). cNNN = pilot comp segment k (snap time, else pre-snap frame); pNNN = panel-only row for the
    snap before segment k: placed between segment k-1 post and segment k play-call start (approximate)."""
    p = PILOT2_COMP.get(norm_vid(vid))
    if not p or not p.exists():
        return {}
    meta = {e["k"]: e for e in json.loads(p.read_text())}
    out = {}
    for x in ref:
        m = re.fullmatch(r"([cp])(\d+)", x["seg"])
        if not m or int(m.group(2)) not in meta:
            continue
        e = meta[int(m.group(2))]
        if m.group(1) == "c":
            out[x["seg"]] = ((e["snap"] if e.get("snap") is not None else e["pre"]), e.get("snap") is None)
        else:
            prev = meta.get(int(m.group(2)) - 1)
            lo = prev["post"] if prev else max(0.0, e["pc_start"] - 30)
            out[x["seg"]] = (round((lo + e["pc_start"]) / 2, 1), True)
    return out


def v01_rows(vid: str) -> list[dict]:
    p = LABELS / "v0.1" / f"{norm_vid(vid)}.jsonl.gz"
    return [json.loads(l) for l in gzip.open(p, "rt")] if p.exists() else []


def match_v01(v01: list[dict], t: float, down, distance, taken: set):
    """v0.1 OCR snap whose run contains t (+-3 s); d&d must not contradict. Gives z5ne rows v0.1's snap_id/t_start."""
    best, bd = None, 1e9
    for l in v01:
        if l.get("t_start") is None or l.get("play") == "Kickoff":
            continue
        t0, t1 = l["t_start"], l.get("t_end") or l["t_start"]
        if not (t0 - 3 <= t <= t1 + 3):
            continue
        if l["snap_id"] in taken or (down is None) != (l.get("down") is None):
            continue
        if down is not None and (l["down"] != down or (distance is not None and l.get("distance") not in (None, distance))):
            continue
        d = 0 if t0 <= t <= t1 else min(abs(t - t0), abs(t - t1))
        if d < bd:
            best, bd = l, d
    return best


def import_pilot2(r: dict) -> dict:
    """Reuse vod_pilot2 vision-tagged rows for a VOD (e.g. z5ne-uBM0YI = pilot m1) instead of re-extracting."""
    vid = r["video_id"]
    ref = [x for x in csv.DictReader(open(PILOT2)) if norm_vid(x["video"]) == norm_vid(vid)]
    if not ref:
        raise RuntimeError(f"no vod_pilot2 rows for {vid}")
    scout = scout_index().get(norm_vid(vid)) or {}
    opp = scout.get("opponent_type") if scout.get("opponent_type") in ("cpu", "human") else "unknown"
    opp_src = "scout_manifest" if opp != "unknown" else "missing_scout_tag"
    if opp == "unknown" and r.get("opponent_type_fallback") in ("cpu", "human"):
        opp, opp_src = r["opponent_type_fallback"], "pilot_metadata_no_scout_tag"
    opp, opp_src, _, _ = opp_override(vid, opp, opp_src, "", 0.0)
    plat = platform_of(r)
    sys.path.insert(0, str(TOOLS))
    from vodx_score import silver_result
    times = pilot2_times(vid, ref)
    v01 = v01_rows(vid)
    off = section_offset(r)
    rows = []
    taken: set = set()
    links = {}
    num = lambda v: int(v) if str(v).lstrip("-").isdigit() else None
    for x in sorted(ref, key=lambda x: (times.get(x["seg"], (0, True))[1], x["seg"])):  # exact (cNNN) times claim first
        t = times.get(x["seg"], (None, True))[0]
        lk = match_v01(v01, t, num(x["down"]), num(x["distance"]), taken) if t is not None else None
        if lk:
            taken.add(lk["snap_id"]); links[x["seg"]] = lk
    for x in ref:
        kind, yards = silver_result(x["result"])
        cov = x["coverage_seen"] if "call unknown" not in x["coverage_seen"] else ""
        num = lambda v: int(v) if str(v).lstrip("-").isdigit() else None
        t, t_approx = times.get(x["seg"], (None, True))
        link = links.get(x["seg"])
        if link:
            sid, t0, t1 = link["snap_id"], link["t_start"], link["t_end"]
        else:
            sid, t0, t1 = (f"{norm_vid(vid)}:{t:.1f}" if t is not None else f"{norm_vid(vid)}:p2:{x['seg']}"), t, None
        if sid in {y["snap_id"] for y in rows} or (not link and sid in {l["snap_id"] for l in v01}):
            sid = f"{norm_vid(vid)}:{t:.1f}:p2{x['seg']}" if t is not None else f"{norm_vid(vid)}:p2:{x['seg']}"
        row = {"snap_id": sid, "video_id": vid, "game": x["game"], "creator": r.get("creator") or "",
               "source_platform": plat, "source_url": r.get("url") or "",
               "ts_url": ts_url(r.get("url") or "", plat, t0 + off) if t0 is not None else r.get("url") or "",
               "t_start": t0, "t_end": t1, "section_offset_s": off, "t_start_vod": (t0 + off) if t0 is not None else None,
               "pilot2_seg": x["seg"], "t_approx": bool(t_approx), "linked_v0_1": bool(link),
               "opponent_type": opp, "opponent_type_source": opp_src, "opponent_type_basis": scout.get("opponent_type_basis", ""),
               "side": x["side"], "quarter": num(x["quarter"]), "clock": x["clock"], "score": x["score"] or None,
               "down": num(x["down"]), "distance": num(x["distance"]), "yardline": num(x["yardline"]),
               "situation_raw": x["situation_raw"], "formation": x["formation"], "play": x["play"],
               "coverage_seen": cov, "concept_seen": x["concept_seen"], "def_overlay": "",
               "audible": "", "hot_routes": "", "off_macros": "", "def_adjustments": "", "macro": x.get("macro") or "",
               "result": x["result"], "result_kind": kind or "", "yards": yards, "is_kickoff": False,
               "extractor_version": script_version(), "engine": "vod_pilot2 round-2 vision tags (reused, not re-extracted)",
               "pilot2_notes": x["notes"], "field_spot": None, "def_call": "", "def_formation": "",
               "result_check": "pilot2_vision", "quality_flags": [] if link else ["no_v0.1_snap_match"]}
        dd = CONF_WORD.get(x["conf_dd"], 0.0)
        conf = {"down": dd if row["down"] is not None else 0.0, "distance": dd if row["distance"] is not None else 0.0,
                "quarter": dd if row["quarter"] else 0.0, "clock": dd if row["clock"] else 0.0,
                "yardline": dd if row["yardline"] is not None else 0.0, "score": dd if row["score"] else 0.0,
                "formation": CONF_WORD.get(x["conf_formation"], 0.0) if row["formation"] else 0.0,
                "play": CONF_WORD.get(x["conf_play"], 0.0) if row["play"] else 0.0,
                "coverage_seen": CONF_WORD.get(x["conf_coverage"], 0.0) if cov else 0.0,
                "concept_seen": 0.3 if row["concept_seen"] else 0.0,
                "result": CONF_WORD.get(x["conf_result"], 0.0) if row["result"] else 0.0,
                "audible": 0.0, "hot_routes": 0.0, "off_macros": 0.0, "def_adjustments": 0.0, "def_overlay": 0.0,
                "opponent_type": (1.0 if opp_src == "scout_manifest" else 0.7) if opp != "unknown" else 0.0,
                "side": dd if x["side"] in ("offense", "defense") else 0.0}
        row["conf"] = conf
        row["field_sources"] = {k: "vod_pilot2_vision_tags" for k in conf if row.get(k) not in ("", None) and k != "opponent_type"}
        row["field_sources"]["opponent_type"] = opp_src
        rows.append(row)
    stem = SNAPS / norm_vid(vid)
    SNAPS.mkdir(parents=True, exist_ok=True)
    tmp = stem.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(x) + "\n" for x in rows)); os.replace(tmp, stem.with_suffix(".jsonl"))
    write_csv(stem.with_suffix(".csv"), rows)
    meta = {"schema": SCHEMA_SNAPS, "video_id": vid, "game": rows[0]["game"], "creator": r.get("creator"), "source_platform": plat,
            "source_url": r.get("url"), "local_path": r.get("local_path"), "opponent_type": opp, "opponent_type_source": opp_src,
            "sample_fps": None, "max_seconds": None, "frames_ocr": 0, "seconds_covered": float(r.get("duration_s") or 0),
            "snaps": len(rows), "extractor_version": script_version(), "engine": "vod_pilot2 vision tags (reused)",
            "layout": layout_for(r), "section_offset_s": section_offset(r), "cache_only": False,
            "extracted_at": now_iso(), "pilot_key": r.get("pilot_key"), "from": "vod_pilot2/snaps.csv"}
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _ocr_job(args):
    path, vid, fps, max_seconds, threads = args
    import vodx_ocr
    vodx_ocr.run_ocr(path, vid, fps, max_seconds, threads, log=log)
    return vid


def acquire_extract_lock():
    if os.environ.get("VODX_LOCK_HELD") == "1":
        return None
    fh = open(EXTRACT_LOCK, "a+")
    log(f"[lock] waiting for {EXTRACT_LOCK}")
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.seek(0); fh.truncate(); fh.write(json.dumps(lock_note())); fh.flush()
    return fh


def lock_note(pid: int | None = None) -> dict:
    """Written the moment the extract lock is acquired. `since`/`acquired_at` = wall clock at acquisition (ISO+offset).
    `pid_start_ps` = the holder's start time the way `ps -o lstart/etime` computes it (boot time + monotonic ticks);
    on this box the two can differ by hours because the box is suspended between turns and ps/etime don't count
    suspended time. A staleness check should compare `pid` + `pid_start_ps` with ps, or `since` with wall clock."""
    pid = pid or os.getpid()
    st = {"holder": "vodx.py run", "pid": pid, "since": now_iso(), "acquired_at": now_iso()}
    try:
        ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        btime = next(int(l.split()[1]) for l in Path("/proc/stat").read_text().splitlines() if l.startswith("btime"))
        start = btime + ticks / os.sysconf("SC_CLK_TCK")
        st["pid_start_ps"] = dt.datetime.fromtimestamp(start).astimezone().isoformat(timespec="seconds")
        st["pid_start_wall"] = dt.datetime.fromtimestamp(Path(f"/proc/{pid}").stat().st_ctime).astimezone().isoformat(timespec="seconds")
    except Exception:  # noqa: BLE001
        pass
    return st


RUN_STATUSES = ("downloaded", "redownloaded")  # Scout statuses `vodx run` picks up by default


def dense_targets(vid: str, pre: float = 20.0, post: float = 2.0, max_frames: int | None = None,
                  sample_fps: float = 0.5) -> list[float]:
    """Timestamps between the regular sampling grid around each scrimmage snap's call screen ([t-pre, t+post]).
    At 0.5 fps that is the odd seconds; at >= 1 fps the grid is already that dense -> nothing to add.
    If the VOD would need more than max_frames, the pre-snap window shrinks (20 s -> 14 s -> 8 s) to stay in budget."""
    if sample_fps >= 1.0:
        return []
    step = 1.0 / sample_fps
    p = SNAPS / f"{norm_vid(vid)}.jsonl"
    rows = [r for r in (read_jsonl(p) if p.exists() else []) if not r.get("is_kickoff") and r.get("t_start") is not None]
    for pre_s in (pre, 14.0, 8.0):
        out = set()
        for r in rows:
            t = math.floor((r["t_start"] - pre_s) / step) * step + step / 2
            while t <= r["t_start"] + post:
                if t >= r["t_start"] - pre_s and t >= 0:
                    out.add(round(t, 3))
                t += step
        if not max_frames or len(out) <= max_frames:
            break
    out = sorted(out)
    return out[:max_frames] if max_frames else out


def _dense_job(args):
    path, vid, times, threads = args
    import vodx_ocr
    return vid, vodx_ocr.run_dense(path, vid, times, threads, log=log)


def _speech_job(args):
    """Twitch (or YouTube with an audio track): raw audio -> faster-whisper transcript; temp audio always deleted."""
    path, vid, window, threads = args
    import vodx_speech
    audio = vodx_speech.extract_audio(path, vid, duration=(window[1] if window else None))
    try:
        doc = vodx_speech.transcribe(audio, vid, threads=threads, window=window, log=log)
    finally:
        audio.unlink(missing_ok=True)
    return vid, {"segments": len(doc["segments"]), "wall_s": doc["wall_s"], "cpu_s": doc["cpu_s"]}


def _yt_speech_job(args):
    vid, window, threads = args
    import vodx_speech
    doc = vodx_speech.youtube_speech(vid, window=window, threads=threads, log=log)
    return vid, {"segments": len(doc["segments"]), "source": doc["source"], "wall_s": doc.get("wall_s"), "cpu_s": doc.get("cpu_s")}


def speech_plan(r: dict, max_seconds: float | None) -> tuple[str, str]:
    """('done'|'captions'|'whisper'|'skip', note) for a VOD whose raw is (still) on disk."""
    import vodx_speech
    vid = r["video_id"]
    if vodx_speech.load_transcript(vid):
        return "done", "transcript cached"
    plat = r.get("source") or r.get("source_platform") or ""
    if plat == "youtube":
        return "captions", "youtube auto-captions (fallback: audio-only stream + faster-whisper)"
    if r.get("local_path") and Path(r["local_path"]).exists():
        return "whisper", "no captions on twitch -> faster-whisper on raw audio"
    return "skip", "raw missing"


WINDOW_DONE = "extracted"  # Scout's window rows are flipped (by window_id) once folded into the parent's snaps


def window_readout(vid: str, wins: dict, tol: float = 8.0) -> dict:
    """Per window: the parent snap at target_snap_s (section time) and what the readers now have for it."""
    sid = norm_vid(vid)
    meta = json.loads((SNAPS / f"{sid}.meta.json").read_text())
    off = meta.get("section_offset_s") or 0
    rows = read_jsonl(SNAPS / f"{sid}.jsonl")
    tgt = {r["window_id"]: r for r in read_jsonl(MANIFEST) if r.get("kind") == "window" and r.get("window_id") in wins}
    out = {}
    for wid, upd in wins.items():
        w = tgt.get(wid) or {}
        T = w.get("target_snap_s")
        best = None
        if T is not None:
            cand = [r for r in rows if not r.get("is_kickoff") and abs(r["t_start"] - (T - off)) <= tol]
            best = min(cand, key=lambda r: abs(r["t_start"] - (T - off)), default=None)
        rd = {}
        if best:
            src = best.get("field_sources") or {}
            rd = {k: best.get(k) for k in ("formation", "play", "def_call", "def_formation", "coverage_seen") if best.get(k)}
            rd["sources"] = {k: src.get(k) for k in rd if src.get(k)}
        out[wid] = dict(upd, window_snap_id=best["snap_id"] if best else None, window_read=rd)
    return out


def run_windows(a) -> int:
    """Scout clip windows (kind=window, status window_downloaded): OCR + merge (vodx_windows), re-extract the parent
    VOD from cache so the window frames reach the readers (panel / cards / audible header / tooltip), match each window
    to the parent snap at target_snap_s, then flip the window rows (by window_id) to `extracted`."""
    try:
        import vodx_windows
        vodx_windows.MANIFEST, vodx_windows.SNAPS = MANIFEST, SNAPS  # same files as this run (tests point these at tmp)
        per_vod = vodx_windows.process_pending(video_ids=getattr(a, "video_ids", None) or None,
                                               workers=getattr(a, "workers", 2) or 2, set_status=False, log=log)
    except Exception as e:  # noqa: BLE001  (an enhancement: never blocks the hourly run)
        log(f"[windows] FAILED {e!r}")
        return 0
    n = 0
    if not per_vod:  # (vod_specs([]) would mean "all VODs")
        return 0
    for r in vod_specs(list(per_vod), SPEC_PILOT if SPEC_PILOT.exists() else None):
        m = SNAPS / f"{norm_vid(r['video_id'])}.meta.json"
        if not m.exists():
            continue
        om = json.loads(m.read_text())
        if not om.get("frames_ocr"):  # e.g. z5ne-uBM0YI: pilot2 vision-tag import, no OCR cache to fold windows into
            log(f"[windows] {r['video_id']}: parent has no OCR cache (pilot2 import); windows left at window_downloaded")
            continue
        try:
            meta = extract_one(r, om.get("sample_fps") or 0.5, om.get("max_seconds"), a.threads, cache_only=True)
            log(f"[windows] {r['video_id']}: re-extracted from cache with {meta.get('window_frames', 0)} window frames")
            wins = per_vod.get(r["video_id"]) or per_vod.get(norm_vid(r["video_id"])) or {}
            try:
                wins = window_readout(r["video_id"], wins)
            except Exception as e:  # noqa: BLE001
                log(f"[windows] {r['video_id']}: readout failed {e!r}")
            import vodx_windows
            vodx_windows.set_window_status(wins, WINDOW_DONE)
            n += 1
        except Exception as e:  # noqa: BLE001
            log(f"[windows] {r['video_id']}: re-extract FAILED {e!r}")
    return n


def cmd_windows(a) -> int:
    """`vodx windows [--video-ids ...] [--no-label]`: process Scout clip windows now (under the extract lock)."""
    lockfh = acquire_extract_lock()
    try:
        n = run_windows(a)
        log(f"[windows] {n} parent VODs updated")
        if n and not a.no_label:
            build_labels(None, None, a.note or "windows")
    finally:
        if lockfh:
            lockfh.close()
    return 0


def cmd_run(a) -> int:
    """Hourly: pick up Scout's `downloaded` + `redownloaded` VODs.
    downloaded   -> OCR (regular grid) -> extract -> dense call-screen frames around each snap -> re-extract -> `extracted`
    redownloaded -> (cached OCR) extract -> dense frames -> re-extract -> `extracted`
    The manifest only flips to `extracted` after the dense pass, because Scout deletes the raw at `extracted`."""
    global CURRENT_FIXES
    CURRENT_FIXES = getattr(a, "fixes", "") or ""
    lockfh = acquire_extract_lock()
    win_ok = 0
    if not a.video_ids and not a.spec and not a.cache_only and getattr(a, "windows", "auto") == "auto":
        win_ok = run_windows(a)
    specs = vod_specs(a.video_ids, Path(a.spec) if a.spec else None)
    if not a.video_ids and not a.spec:
        specs = [r for r in specs if r.get("status") in RUN_STATUSES or (a.retry_failed and r.get("status") == "extract_failed")]
    if a.video_ids:  # keep the caller's order (priority)
        order = {norm_vid(v): i for i, v in enumerate(a.video_ids)}
        specs.sort(key=lambda r: order.get(norm_vid(r["video_id"]), 1e9))
    else:  # Previous Play panel VODs first
        specs.sort(key=lambda r: (r.get("previous_play_panel") is not True,))
    reuse = {norm_vid(v) for v in (a.reuse_pilot2 or [])}
    in_manifest = {norm_vid(r["video_id"]) for r in read_jsonl(MANIFEST)}
    ok = fail = 0
    failures = {}

    def plan_fps(r):
        """(fps, max_seconds) for this VOD; a re-download reuses the cached OCR's grid."""
        fps, mx = fps_for(r, a.fps), a.max_seconds
        old = SNAPS / f"{norm_vid(r['video_id'])}.meta.json"
        if (a.cache_only or r.get("status") == "redownloaded") and a.fps == "auto" and old.exists():
            om = json.loads(old.read_text())
            fps = om.get("sample_fps") or fps
            mx = om.get("max_seconds") if a.max_seconds is None else a.max_seconds
        return fps, mx

    def record_fail(r, e):
        nonlocal fail
        vid = r["video_id"]
        fail += 1; failures[vid] = repr(e)[:500]
        log(f"[extract] {vid}: FAILED {e!r}")
        if sys.exc_info()[0] is not None:
            traceback.print_exc()
        if norm_vid(vid) in in_manifest and not a.cache_only:
            if r.get("status") == "redownloaded":  # keep it queued; Scout must not delete it yet
                update_manifest_status(vid, {"extract_error": repr(e)[:500], "extract_attempted_at": now_iso()})
            else:
                update_manifest_status(vid, {"status": "extract_failed", "extract_error": repr(e)[:500], "extracted_at": now_iso()})
        write_batch_notes()

    def record_ok(r, meta, dense_info):
        nonlocal ok
        vid = r["video_id"]
        ok += 1
        log(f"[extract] {vid}: {meta['snaps']} snaps, opponent_type={meta['opponent_type']}, dense_frames={meta.get('dense_frames', 0)}")
        if norm_vid(vid) in in_manifest and a.max_seconds is None and not a.cache_only:
            upd = {"status": "extracted", "extracted_at": meta["extracted_at"], "extract_snaps": meta["snaps"],
                   "extract_path": str(SNAPS / f"{norm_vid(vid)}.jsonl"), "extractor_version": meta["extractor_version"],
                   "extract_error": None, "dense_frames": meta.get("dense_frames", 0), "dense_note": dense_info}
            if r.get("status") == "redownloaded":
                upd["redownload_processed_at"] = now_iso()
            update_manifest_status(vid, upd)
        write_batch_notes()

    # mixed-game VODs whose snap batch predates per-segment handling (e.g. extracted as all-Madden before Scout set
    # game=mixed): redo from the OCR/dense caches only - no raw needed, manifest status untouched
    redo = {r["video_id"]: ("mixed", r) for r in mixed_needing_resplit()}
    for r in layout_stale_batches():  # general: any layouts.json edit re-labels the affected VODs from cache
        redo.setdefault(r["video_id"], ("layout", r))
    for r in def_guard_stale_batches():  # batches extracted before the def-name guard still carry the raw name
        redo.setdefault(r["video_id"], ("def_guard", r))
    for r in panel_attr_stale_batches():  # batches written before the panel attribution fix (call-screen fragments)
        redo.setdefault(r["video_id"], ("panel_attr", r))
    queued = {norm_vid(x["video_id"]) for x in specs}
    for vid, (why, r) in redo.items():
        if norm_vid(vid) in queued:
            continue  # being (re)extracted in this run anyway, with the current layout
        try:
            fps, mx = plan_fps(r)
            meta = extract_one(r, fps, mx, a.threads, cache_only=True)
            segs = [(g["game"], g["snaps"]) for g in meta.get("game_segments") or []]
            log(f"[{why}] {vid}: re-labelled from cache with current layout: {meta['snaps']} snaps {segs or ''}")
        except Exception as e:  # noqa: BLE001
            log(f"[{why}] {vid}: cache redo FAILED {e!r}")
    # cache-only / pilot2-reuse rebuilds: no raw video involved
    queue = []
    for r in specs:
        if norm_vid(r["video_id"]) in reuse or a.cache_only:
            try:
                if norm_vid(r["video_id"]) in reuse:
                    meta = import_pilot2(r)
                else:
                    fps, mx = plan_fps(r)
                    meta = extract_one(r, fps, mx, a.threads, cache_only=True)
                record_ok(r, meta, "cache-only rebuild")
            except Exception as e:  # noqa: BLE001
                record_fail(r, e)
        else:
            queue.append(r)
    if queue:
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        import vodx_ocr
        pending, notes = {}, {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:

            def start_dense(r):
                """extract on the regular grid (gives snap times), then queue the dense pass (or finish if none)."""
                fps, mx = plan_fps(r)
                meta = extract_one(r, fps, mx, a.threads, cache_only=True)
                if a.dense == "off":
                    return record_ok(r, meta, "dense off")
                times = dense_targets(r["video_id"], max_frames=a.dense_max_frames, sample_fps=fps)
                have = vodx_ocr.dense_times(r["video_id"])
                times = [t for t in times if round(t, 3) not in have]
                if not times:
                    return after_dense(r, fps, mx, "no dense frames needed" if fps < 1 else f"base grid already {fps} fps")
                pending[ex.submit(_dense_job, (r["local_path"], r["video_id"], times, a.threads))] = ("dense", r, fps, mx)

            def after_dense(r, fps, mx, note):
                """speech before 'extracted': captions (youtube) or local speech-to-text on the raw audio (twitch)."""
                notes[r["video_id"]] = note
                if a.speech == "off":
                    return finish_final(r, fps, mx)
                plan, why = speech_plan(r, mx)
                if plan == "captions":
                    win = (0.0, float(mx)) if mx else None
                    pending[ex.submit(_yt_speech_job, (r["video_id"], win, max(2, a.threads)))] = ("speech", r, fps, mx)
                    notes[r["video_id"]] += f"; speech: {why}"
                    return None
                if plan == "whisper":
                    win = (0.0, float(mx)) if mx else None
                    pending[ex.submit(_speech_job, (r["local_path"], r["video_id"], win, max(2, a.threads)))] = ("speech", r, fps, mx)
                    notes[r["video_id"]] += f"; speech: {why}"
                    return None
                notes[r["video_id"]] += f"; speech: {why}"
                return finish_final(r, fps, mx)

            def finish_final(r, fps, mx):
                meta = extract_one(r, fps, mx, a.threads, cache_only=True)
                return record_ok(r, meta, notes.get(r["video_id"], ""))

            ocr_backlog = []
            for r in queue:
                if not r.get("local_path") or not Path(r["local_path"]).exists():
                    record_fail(r, FileNotFoundError(f"local_path missing: {r.get('local_path')}"))
                    continue
                fps, mx = plan_fps(r)
                if r.get("status") == "redownloaded" and vodx_ocr.load_cache(r["video_id"], fps)[1]:
                    try:
                        start_dense(r)
                    except Exception as e:  # noqa: BLE001
                        record_fail(r, e)
                    continue
                ocr_backlog.append((r, fps, mx))

            def feed_ocr():
                """Start OCR only into free worker slots. Dense/speech jobs are submitted the moment a VOD's OCR ends, so
                they run before the next VOD's OCR and each VOD reaches `extracted` (raw freeable) as soon as its own
                work is done, not after every other VOD's multi-hour OCR (old: all OCR submitted up front, FIFO)."""
                while ocr_backlog and len(pending) < a.workers:
                    r, fps, mx = ocr_backlog.pop(0)
                    pending[ex.submit(_ocr_job, (r["local_path"], r["video_id"], fps, mx, a.threads))] = ("ocr", r, fps, mx)

            feed_ocr()
            while pending:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fut in done:
                    stage, r, fps, mx = pending.pop(fut)
                    try:
                        fut.result()
                        if stage == "ocr":
                            start_dense(r)
                        elif stage == "dense":
                            after_dense(r, fps, mx, f"dense call-screen frames around snaps (odd seconds, <= {a.dense_max_frames})")
                        else:
                            finish_final(r, fps, mx)
                    except Exception as e:  # noqa: BLE001
                        if stage == "speech":  # speech is an enhancement too: never blocks 'extracted'
                            log(f"[speech] {r['video_id']}: FAILED {e!r}")
                            notes[r["video_id"]] = notes.get(r["video_id"], "") + f"; speech failed: {e!r}"[:200]
                            finish_final(r, fps, mx)
                        elif stage == "dense":  # dense is an enhancement: keep the regular-grid extraction
                            log(f"[dense] {r['video_id']}: FAILED {e!r}; keeping regular-grid extraction")
                            if r.get("status") == "redownloaded":
                                record_fail(r, e)
                            else:
                                after_dense(r, fps, mx, f"dense failed: {e!r}"[:300])
                        else:
                            record_fail(r, e)
                feed_ocr()
    if not a.no_label and (ok or win_ok or a.force_label):
        build_labels(a.label_version, None, a.note or "auto label build")
    log(f"[run] extracted={ok} failed={fail} {json.dumps(failures)}")
    if lockfh:
        lockfh.close()
    return 0 if fail == 0 else 3


def cmd_densify(a) -> int:
    """Manual dense pass (raw video required) for VODs already extracted at < 1 fps. No manifest change.
    Normal processing does this inside `vodx run`; rebuild with `vodx.py run --cache-only` (under the lock) afterwards."""
    specs = vod_specs(a.video_ids, Path(a.spec) if a.spec else None)
    todo, skipped = [], {}
    for r in specs:
        meta = SNAPS / f"{norm_vid(r['video_id'])}.meta.json"
        m = json.loads(meta.read_text()) if meta.exists() else {}
        if not a.video_ids and r.get("previous_play_panel") is not True:
            continue
        fps = m.get("sample_fps") or 1.0
        if fps >= 1.0:
            if a.video_ids:
                skipped[r["video_id"]] = f"base grid already {fps} fps"
            continue
        if not r.get("local_path") or not Path(r["local_path"]).exists():
            skipped[r["video_id"]] = "raw video missing"; continue
        todo.append((r["local_path"], r["video_id"], dense_targets(r["video_id"], max_frames=a.dense_max_frames, sample_fps=fps), a.threads))
    for v, why in skipped.items():
        log(f"[dense] {v}: skipped ({why})")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(_dense_job, t) for t in todo]):
            try:
                vid, n = fut.result(); log(f"[dense] {vid}: +{n} frames")
            except Exception as e:  # noqa: BLE001
                log(f"[dense] FAILED {e!r}")
    return 0


def cmd_speech(a) -> int:
    """Backfill transcripts for already-extracted VODs (no manifest change): youtube -> auto-captions;
    twitch -> faster-whisper only if the raw is still on disk, otherwise skipped (raw deleted)."""
    specs = vod_specs(a.video_ids, Path(a.spec) if a.spec else None)
    res = {}
    for r in specs:
        vid = r["video_id"]
        meta = SNAPS / f"{norm_vid(vid)}.meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        plan, why = speech_plan(r, m.get("max_seconds"))
        try:
            win = (0.0, float(m["seconds_covered"]) + 30) if m.get("seconds_covered") and m["seconds_covered"] < 3000 else None
            if plan == "captions":
                _, info = _yt_speech_job((vid, win, a.threads)); why += f": ok {info}"
            elif plan == "whisper":
                _, info = _speech_job((r["local_path"], vid, win, a.threads)); why += f": ok {info}"
        except Exception as e:  # noqa: BLE001
            why += f": FAILED {e}"[:300]
        res[vid] = f"{plan}: {why}"
        log(f"[speech] {vid}: {res[vid]}")
    return 0


def cmd_label(a) -> int:
    build_labels(a.version, a.video_ids, a.note or "")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--video-ids", nargs="*"); r.add_argument("--spec")
    r.add_argument("--fps", default="auto", help="'auto' = 0.5 fps regular grid + dense call-screen frames around each snap")
    r.add_argument("--max-seconds", type=float); r.add_argument("--cache-only", action="store_true")
    r.add_argument("--fixes", default="", help="comma list of PR #88 label fixes, or all. Applied after layouts.json mask and gameplay_window.")
    r.add_argument("--workers", type=int, default=4); r.add_argument("--reuse-pilot2", nargs="*")
    r.add_argument("--label-version")
    r.add_argument("--threads", type=int, default=2); r.add_argument("--no-label", action="store_true")
    r.add_argument("--force-label", action="store_true"); r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--note", default="")
    r.add_argument("--dense", choices=["auto", "off"], default="auto", help="dense call-screen frames around snaps before 'extracted'")
    r.add_argument("--dense-max-frames", type=int, default=900, help="per-VOD cap for the dense pass")
    r.add_argument("--speech", choices=["auto", "off"], default="auto", help="captions / local speech-to-text before 'extracted'")
    r.add_argument("--windows", choices=["auto", "off"], default="auto", help="Scout clip windows (kind=window) -> OCR + re-extract parent")
    wd = sub.add_parser("windows", help="Scout clip windows -> OCR, fold into the parent snaps, mark extracted, label")
    wd.add_argument("--video-ids", nargs="*"); wd.add_argument("--workers", type=int, default=2)
    wd.add_argument("--threads", type=int, default=4); wd.add_argument("--no-label", action="store_true")
    wd.add_argument("--note", default=""); wd.set_defaults(func=cmd_windows)
    r.set_defaults(func=cmd_run)
    l = sub.add_parser("label"); l.add_argument("--version"); l.add_argument("--video-ids", nargs="*")
    l.add_argument("--note", default=""); l.set_defaults(func=cmd_label)
    dn = sub.add_parser("densify"); dn.add_argument("--video-ids", nargs="*"); dn.add_argument("--spec")
    dn.add_argument("--workers", type=int, default=4); dn.add_argument("--threads", type=int, default=2)
    dn.add_argument("--dense-max-frames", type=int, default=900)
    dn.set_defaults(func=cmd_densify)
    sp = sub.add_parser("speech"); sp.add_argument("--video-ids", nargs="*"); sp.add_argument("--spec")
    sp.add_argument("--threads", type=int, default=4); sp.set_defaults(func=cmd_speech)
    sc = sub.add_parser("score"); sc.add_argument("--version")
    sc.set_defaults(func=lambda a: __import__("vodx_score").main((["--version", a.version] if a.version else [])) or 0)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
