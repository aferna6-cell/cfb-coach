# Lions 2026-11-04 coordinator — fresh training on all newer verified snaps

This is a **new training run using the exact same Lions algorithm**, not a claim of recovering the original `madden-ml.experimental.offense.428.empirical` weights.

## What stays unchanged

- Source coordinator code is the exact 2026-11-04 successful Lions revision `0c4d1036372ad67be702ce0698f07de7857b7811` (no Unified V1 scoring).
- Training calls the existing `cfb_coach.madden.model.experimental_model.train_experimental(rows, side="offense", seed=7)` function on the historical branch. It uses the same `hierarchical_shrinkage.v1` buckets, concept/cov/down priors, same row eligibility rules and the same capped research prior contributions.
- Offense: **5 formations / 93 plays** from the original installed Buccaneers-based custom Lions book.
- Defense: the **old heuristic caller**, no defensive ML, **6 formations / 123 plays**.
- Old HTML; the new `--coach-cpu-defense` flag adds legacy defensive calls in CPU sessions.
- No automatic book changes or live retraining.

## Data honesty

The **655** the owner saw was the count inside an existing trained file. This new run counts independently the currently eligible **verified offensive** rows in the live franchise SQLite database. The final new model version becomes `madden-ml.experimental.offense.N.empirical` with the actual observed `N`; it is not artificially renamed to 428 or 655.

The original algorithm also uses *discounted* uncertain-execution priors, but only genuinely eligible executions count toward `N`. The script requires at least **400** verified eligible offensive training rows by default and will refuse to produce a misleading prior-driven model.

The original 428 model's parameters cannot be exactly recreated without its original file; a model trained on the new data will have different weights and likely different calls. We should test its gameplay rather than assume it is already better.

## Quick start: separate worktree and untouched user databases

Make sure the HTML coach is stopped before launching. Do **not** delete old worktrees or databases.

```bash
cd ~/cfb-coach-lions
git fetch origin
git switch rollback/lions-retrained-legacy-offense
export PYTHONPATH=.

python3 -m pytest tests/test_lions_baseline_rollback.py tests/test_lions_retrained_offense.py -q --tb=short
python3 scripts/train_lions_legacy_offense.py           # read-only preview: actual eligible row count
```

Once the test and preview pass:

```bash
python3 scripts/train_lions_legacy_offense.py --apply    # training in an isolated DB copy
python3 scripts/train_lions_legacy_offense.py --check    # actual model-selected calls vs legacy defense
```

The new folder contains:
- `~/lions-retrained-legacy/madden27-lions.db`
- `~/lions-retrained-legacy/models/lions-hierarchical-shrinkage-new.json`
- `~/lions-retrained-legacy/TRAIN_REPORT.json`

The main `~/.cfb-coach/madden27.db`, existing `~/.cfb-coach/madden_ml_experimental/offense.json` (655-row artifact), and Unified V1 test DB are never written to. The clone retains all historical snaps/decisions so postgame feedback remains usable.

**If `--apply` fails**, do not use the new workdir. The script cleans incomplete staging. If a completed new workdir already exists, `--apply` refuses to overwrite it.

## Play

Only if `LIONS TRAINED VERDICT: GO` and the owner has physically selected the historical book:

```bash
cd ~/cfb-coach-lions
W="$HOME/lions-retrained-legacy"
PYTHONPATH=. python3 -m cfb_coach play \
  --game madden27 \
  --opponent cpu \
  --franchise primary \
  --db "$W/madden27-lions.db" \
  --no-vod-prior \
  --no-macros \
  --coach-cpu-defense
```

**Offensive formations:** Gun 5WR Tight, Gun Doubles Clamp Stack, Gun Trips X Nasty, Pistol Deuce Close, Pistol Trips.

**Defensive formations:** 4-3 Over, 46 Normal, Dime 3-2 Odd, Nickel Double Mug, Nickel Over, Nickel Single Mug.

If there is an unexpectedly low number of verified rows or the old scorer falls back during the live check, investigate that rather than force a GO. There is no need to change the already-working old defensive heuristics.

The older exact-artifact restoration script `scripts/restore_lions_baseline.py` is still available if a real 428-row file ever turns up; it is not used in this new training path.
