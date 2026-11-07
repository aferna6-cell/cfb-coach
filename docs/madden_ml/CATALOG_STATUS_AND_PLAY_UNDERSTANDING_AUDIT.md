# Catalogue status and play-understanding audit (answer to the review comment on #126, 2026-11-07)

## 1. Process status (this sandbox; nothing is running now)
| Job | Result | Notes |
|---|---|---|
| first crawl (PID 1798) | **killed by me after ~35 min** | it never terminated: past the last page the site serves the LAST page again for any page number (not empty, not page 1). Fixed + regression test; the crawl also hammered the site for ~15 minutes past the end at 1.5 s/request, which I should have caught with a progress/health check |
| metadata sync 2 | completed 22:06Z, exit 0 | 10,132 records, 40 tiles quarantined (20 art-less offense plays) |
| metadata sync 3 | completed 22:17Z, exit 0 | +20 art-less plays kept (no art URL) |
| art download, active books | completed, 203/203 | 93 offense + 123 defense installed pairs: 203 matched by slug, 13 unmatched (aliases such as `L Spot`/`Zl Spot Shake`, `Cover 3 Sky`, `Fire Zone 3`, `1 Double WR1/2`) |
| extraction | completed, 203/203 | geometry only |
The store lives in the sandbox container (`~/cfb_acemadden`, private, not committed, ephemeral). To keep it, run the same commands on your computer. New: `ace_catalog.py status` (waterfall + job health: healthy / STALE heartbeat / DEAD process) and a `progress.json` heartbeat written by `sync` and `art`; `art` checkpoints every 25 images and resumes.

## 2. Catalogue metadata status (private store)
* 10,152 records, 10,152 unique ids: **offense 8,574** (matches the site's own 8,574 listing count; 420 formations; 20 listed plays have no art tile) and **defense 1,578** (90 formations).
* Distinct art URLs: 10,119 (the same diagram is shared by some listings). Link slug differs from the art-file slug for 975 offense and 1,135 defense listings (handled; the link slug is identity).
* Same formation + same play name under two different slugs: 0. Distinct play names: 3,154 offense, 381 defense.
* **Playbook (source-book) membership: NOT parsed (0 of 10,152).** The listing has no membership; the detail page's "Playbooks With" block has not been decoded. Indexed listings != downloaded diagrams != knowledge.
* Reconciled against this repo's own catalogue by slug: offense 3,266 in both, 5,288 only on the site, 155 only in the repo; defense 932 / 646 / 139 (units differ; slug mismatches such as spelling variants inflate the "only" sets).

## 3. Football-knowledge waterfall
| | Offense | Defense |
|---|---:|---:|
| listed | 8,574 | 1,578 |
| art URL present | 8,554 | 1,578 |
| **art downloaded** | **90** (active book; 3 alias-unmatched) | **113** (installed; 10 unmatched) |
| **geometry extracted** (unreviewed) | 90 (68 with every route owned by a start marker, 22 flagged) | 113 (zones/rush arrows/links found on all; rusher counts plausible: 68 plays show 4 rushers) |
| geometry **reviewed** by a human | 0 | 0 |
| **player-specific assignments verified** | 0 new (repo: 8 plays partially documented, 40 of 1,023 slots, 0 complete) | 0 |
| concepts mapped (name hints only; NOT assignments) | 6,675 of 8,574 listings carry a concept hint; 7,002 a pass/run type hint | 1,017 of 1,578 listings resolve to a coverage/pressure structure (120 of 381 names) |
| QB read / progression known | 7 documented read orders (third-party, 8 plays) | n/a |

## 4. Play-understanding audit (looked at the art, not just counted)
I compared the overlay (what the extractor thinks it saw) against the original diagram for 10 offensive and 6 defensive installed plays.

**Offense (10: HB Slip Screen, Inside Zone, Hi Lo Cross, Mtn Stick Wheel, Switch HB Wheel, Pistol Trips Slants, Pistol Trips Inside Cross, Mtn Shuffle Verts Smash, Flood Seam, QB Draw)**
* The art also uses **cyan lines** (backs/motion), **white bars** (run-blocking assignments) and red vs yellow route colours. The first extractor ignored cyan and white, so it missed the HB routes in 3 of the 10 and all blocking bars; cyan is now captured and white bars are listed as `blocker_stubs` (only 2 of 90 offense plays hit my bar thresholds, so blocking is NOT really captured yet).
* Where it found a route it was geometrically right (verticals, the red route). It labels most routes `unknown_complex` (162 of 273 routes), and on crossing patterns (Hi Lo Cross, Flood Seam) it still merges or drops lines (e.g. 2 routes found where about 5 are drawn).
* Run plays: the RB path is a short line; no run scheme (inside zone vs draw) is recovered. QB Draw/Inside Zone read as one "diagonal" stub.
* **Verdict: geometry is partial and unverified; no receiver assignment, route depth, or read order can be certified from it.** A line is not a player: slot names (WR1, SLWR, HB1...) are not in the art and are not assigned.

**Defense (6: Nickel Over Cover 2 Man, 4-3 Over Cover 4 Quarters, Nickel Over Cover 4 Palms, Nickel Over Cover 3 Match, Nickel Over Cover 6, 4-3 Over DBL Safety Blitz)**
* Deep zones are right: two for Cover 2 Man, four for Quarters/Palms, three for Cover 3 Match, mixed for Cover 6, none for the zero-deep blitz. Rush arrows right (4 rushers; the blitz shows its three diagonal rushers).
* The first extractor split ovals crossed by link lines (overcounting zones); fixed by merging, but some under zones still merge with a neighbouring one.
* New and genuinely useful: the art carries its own **category chip** (MAN / MATCH / ZONE / BLITZ). Read from colour (text not OCR'd), it agrees with the name-derived coverage type for **103 of 106** resolvable installed calls. The 3 disagreements are `Nickel 2 Trap` (art says BLITZ; the name decomposition misses the pressure) - a real catalogue correction.
* **Verdict: coverage family and rusher count are usable hints; which defender owns which zone/man/match rule is not established** (a link line from a defender marker to an oval is geometry, not a rule).

**Propagation of the extracted representation (a)-(d): none of it reaches the models yet.** The extractions sit in the private store; the scorer, the adjustment search, QB read ranking and the defensive shadow read only `offense_assignments` (8 plays), name hints, and the football priors. The bridge (reviewed extraction -> slot-named annotation overlay with evidence tier) is not built; the slot-naming step needs formation alignment knowledge and your review. Until then every statement of play understanding rests on name hints at 0.35 trust or 8 source-documented plays.

## 5. Real-game validation: not run (blocker)
The sandbox cannot reach your computer or `~/.cfb-coach/madden27.db` (no shared folders). On your machine:
```bash
python - <<'PY'
import sqlite3, pathlib, shutil
src = pathlib.Path.home()/".cfb-coach"/"madden27.db"; dst = pathlib.Path.home()/"madden27_copy.db"
s = sqlite3.connect(f"file:{src}?mode=ro", uri=True); d = sqlite3.connect(dst); s.backup(d); d.close(); s.close(); print("backup copy:", dst)
PY
python scripts/madden_ml/football_priors_eval.py --db ~/madden27_copy.db --out ~/priors_eval
python scripts/madden_ml/cpu_experiment.py prepare --db ~/.cfb-coach/madden27.db --work ~/cfb_cpu_experiment_2   # backs up again + readiness + priors eval
```
Report to expect: verified folds, name-only vs structured-prior vs structured+learned log loss, game-clustered intervals, KEEP_CHAMPION/PROMOTE verdict. **CPU GO/NO-GO: priors stay OFF; the existing experimental coordinator remains the only recommended caller until that evaluation exists.**
