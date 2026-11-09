# Lions baseline rollback: 2026-11-04 offense + original heuristic defense

**Status:** rollback branch. No merge, training, model replacement, or main-database mutation.

This branch starts at **commit `0c4d1036372ad67be702ce0698f07de7857b7811`**, the version logged during the Detroit Lions 35–7 primary CPU win. It intentionally does **not** inherit Unified V1's decision solver or new HTML widgets.

## Exact preserved football setup

- **Offense:** `madden-ml.experimental.offense.428.empirical` (hierarchical shrinkage v1), **428 supervised rows**, model is selected in experimental mode, no VOD prior.
- **Offense playbook:** **5 formations / 93 plays**, captured as applied in the historical Madden snapshot. Gun 5WR Tight (15), Gun Doubles Clamp Stack (27), Gun Trips X Nasty (21), Pistol Deuce Close (12), Pistol Trips (18).
- **Defense:** the original `_pick_defense` heuristic, **no defensive ML**. Historical custom defense: **6 formations / 123 plays** (4-3 Over, 46 Normal, Dime 3-2 Odd, Nickel Double Mug, Nickel Over, Nickel Single Mug). No defensive macros during CPU rollback.
- **HTML:** established pre-Unified interface, with its established down, yardline and post-result controls. The owner confirms completed snaps as recommended by standing attestation.
- **CPU both sides:** this branch adds **`--coach-cpu-defense`** to opt into calling the original defensive heuristic against CPU. The original default CPU offense-only behavior remains unchanged without this flag.

These are *recorded historical installed* book names. **The owner must select the corresponding custom books in Madden and confirm the stock plays are still present.** Repository metadata cannot install plays in the game.

## Worktrees — no branch-switching in the running Unified folder

Stop the old/unified HTML server with Ctrl+C; do not click Game over.

The owner already has a `~/cfb-coach-lions` worktree; use it:

```bash
cd ~/cfb-coach-lions
git fetch origin
git switch rollback/lions-428-5form-heuristic-defense
export PYTHONPATH=.
```

**Do not** run `ml experimental --retrain`, `prep --mark-applied`, `offense-design --confirm-installed`, or `ml heuristic` on the primary database. Those commands alter the setup that we're trying to preserve.

## Read-only preview and protected clone

Preview which original 428-row model can actually be loaded:

```bash
python3 scripts/restore_lions_baseline.py
```

The preview names the source database, the 5/93 and 6/123 installed historical books, the artifact candidates and their model versions. It writes nothing.

If it finds the **exact** `madden-ml.experimental.offense.428.empirical` artifact:

```bash
python3 scripts/restore_lions_baseline.py --apply
```

This backs up the current primary DB **using SQLite's backup API into a separate working copy**, preserves all existing game rows in that copy, and changes ONLY the copy's current applied books and mode metadata:
- `$HOME/lions-legacy-rollback/madden27-lions.db`
- `$HOME/lions-legacy-rollback/models/offense_428_empirical.json`
- `$HOME/lions-legacy-rollback/RESTORE_REPORT.json`

If `~/Downloads/madden27.backup.20261105T013541Z.db` is available, its *historical* applied book records (including audibles) are reused when they match the recorded 5/93 and 6/123 snapshot. Otherwise the embedded manifest is used. Both choices are verified against the same exact play inventory.

**No overwrite**: `--apply` refuses to replace a previously restored game DB. Choose a distinct `--workdir` if you need a second copy.

**Missing artifact:** the script returns **NO GO**. Your current laptop's offense artifact contains **655 supervised rows**, not the original **428**. Do not substitute the 655-row model, train a new one, or overwrite it.

Search your model and ZIP backups read-only:

```bash
python3 scripts/restore_lions_baseline.py --scan-models
```

The scanner checks common local backup, Downloads, repository, and WSL folders for a JSON artifact with exactly the historical 428-row model identity. It does **not** touch any database or artifact.

If it finds a JSON file, preview it with `python3 scripts/restore_lions_baseline.py --model /path/to/old-428-offense.json`. If it reports a matching `ZIP :: member.json`, extract only that JSON to a **new** path first, then pass that path using `--model`. Never overwrite the current 655 model.

If the correct artifact is absent from all backups, exact historical restoration remains blocked. A 428-row model reconstructed from the old dataset would be a **new candidate**, not the originally deployed file.

## Tests and loaded-model smoke check

Run:

```bash
python3 -m pytest tests/test_lions_baseline_rollback.py -q --tb=short

python3 scripts/restore_lions_baseline.py --check
```

The smoke check loads the restored database, verifies the old book fingerprints/inventory, checks experimental O / heuristic D, and samples ten offensive plus four defensive situations **through the actual old playcaller**. If an offense call fell back, or any call took over 150ms, it reports **NO GO**. Source and Unified data are untouched.

## Game launcher — old HTML, old model, both sides against CPU

Only once the smoke check prints `LIONS RESTORE CHECK: PASS` and the physical books are selected in Madden:

```bash
cd ~/cfb-coach-lions
W="$HOME/lions-legacy-rollback"
PYTHONPATH=. python3 -m cfb_coach play \
  --game madden27 \
  --opponent cpu \
  --franchise primary \
  --db "$W/madden27-lions.db" \
  --no-vod-prior \
  --no-macros \
  --coach-cpu-defense
```

This opens **the familiar original HTML interface**. Offensive plays use the loaded `428 empirical` ML scorer. Defensive plays use the legacy heuristic; the CPU defensive ML opt-in is not enabled. Calls stay within the five-formation offense and six-formation defense books.

## Rollback and limits

- Stop the Lions server with Ctrl+C. Main (`~/.cfb-coach/madden27.db`) and Unified V1 (`~/unified_v1_cpu1/game/unified.db`) were never modified by this script.
- Do not merge draft PRs from Unified V1 into this baseline until those models prove better in prospective games.
- Last recorded primary experimental ML decisions had *timeouts on a newer 15-formation book*. The original five-formation setup had no observed timeouts in the November 4 game; that does **not** guarantee no timeouts today. The separate `--check` gate tests the actual laptop.
- CPU defensive recommendations were disabled in the 2026-11-04 release. The opt-in flag enables the same old heuristic selection for CPU defense, but its two-sided CPU HTML workflow must be verified locally before claiming it performs identically to the previous user-game defensive flow.
