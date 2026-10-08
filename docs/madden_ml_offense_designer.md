# Sprint 6C — ML offensive playbook and Custom Adjustment designer

**Status:** opt-in/offline designer; no unapproved live playbook or macro edits.
Stacked on Sprint 6B's model-primary offensive coach.

## Model controls the next game's offensive inventory

This is a *new design stage*, not merely a new name for heuristic prep.
It asks the installed experimental success model to score Madden 27
catalogued formation/play pairs over multiple generic pre-snap situations.
The designer may:

- Replace entire formations (within the five-formation pilot cap).
- Remove and add individual plays within those formations (up to ten each).
- Select a formation's exact sourced plays from a *different* catalogued
  stock playbook; each formation has `formation_sources` provenance.
- Propose up to eight newly named offense Custom Adjustment **blueprints**
  derived from source-backed hot-route and protection primitives.
- Show exact ADD/REMOVE play and formation deltas against your locked book.

The ML model's **play-concept** prediction and evidence drive the
formation's aggregate score; the current experimental artifact does NOT yet
learn an independent formation-specific effect. This distinction matters
until we collect verified results for multiple formations. The heuristic
does not choose the winning play. A slight incumbency preference
avoids pointless churn; diverse concepts and at least one run are retained.
These are offline roster-construction constraints, not a live heuristic selector.

**Evidence limitation:** The experimental model remains low-data. High model
scores do not prove an unplayed formation or hot-route combination is better.
Playbook pairs originate in the Huddle.gg stock-book catalog, not verified
inside your particular custom editor. All are **pending** until you check
them in Madden yourself. Macro routes/control availability must also be
confirmed in the current game version.

## Browser workflow (Sprint 6D)

The default `python -m cfb_coach ml offense-design -o cpu` now writes a local
Madden-dark HTML dashboard and opens it in your default browser, like live
Madden and the pregame prep page. The browser uses the same dark palette,
prominent call-style headings, cards, and single-page layout as the live coach.

It includes:

- Current applied offense (what live ML is actually allowed to call).
- New proposed formations and every individual play, source-book pages,
  named change deltas and local installation checklists.
- Expanding macro cards with exactly researched editor settings,
  direct research/source links, trigger conditions, supported formation/play
  pairs, and a clear Draft vs Verified & Armed status.
- Existing selected offensive macros and their user-confirmed setting rows,
  unspecified fields and activation controls. A saved prep loadout is not
  itself proof of installation.
- Copyable `--stage`, `--confirm-installed`, and `--verify-macro`
  terminal commands, without unsafe automatic activation.

```bash
python -m cfb_coach ml offense-design -o cpu              # preview + open HTML
python -m cfb_coach ml offense-design -o cpu --stage      # stage + open checklist
python -m cfb_coach ml offense-design -o cpu --show       # staged or installed status
python -m cfb_coach ml offense-design -o cpu --no-open    # write HTML, don't launch
python -m cfb_coach ml offense-design -o cpu --text       # original JSON console output
```

The page is a **read-only local HTML file**. Checking play-install boxes
only helps the user prepare the exact verification command; it does not
claim to have physically installed content or write to SQLite. After
installing in the Madden editor and running the explicit CLI confirmation,
rerun `ml offense-design --show` to see the true updated state. The
live model continues to read the **applied** book and the separately
**verified-and-armed** macro registry only.

## Command workflow (Ubuntu)

First back up the database and retrain from any newly verified plays:

```bash
python -m cfb_coach ml backup-db
python -m cfb_coach ml experimental --retrain
```

**Preview** a complete suggested offense with provenance and deltas; read-only:

```bash
python -m cfb_coach ml offense-design -o cpu
```

**Stage** the model's latest proposal without touching the *applied* book:

```bash
python -m cfb_coach ml offense-design -o cpu --stage
python -m cfb_coach ml offense-design --show
```

The JSON includes a short `proposal_id`, new formations/plays and all
model-created macro blueprints. Inspect the changes. Build the proposed
*custom* offense in Madden's editor. You may need to select plays from
specified stock-book sources. **Do not claim installation until you have
verified they are actually present in Madden.**

When every proposed formation and play has been built and checked, run:

```bash
python -m cfb_coach ml offense-design \
  --confirm-installed YOUR_PROPOSAL_ID \
  --attest "I installed and individually checked every formation and play in the Madden custom editor."
```

Only then does the new offensive book become the in-DB applied/locked book,
callable by the model. Ordinary `prep --game madden27` preserves this
explicitly installed model-designed offense instead of quietly replacing it
with the old heuristic seed core. A future `ml offense-design --stage` or
explicit `--o-book` override may change it. Defense and existing game
outcomes are unchanged. The previous applied book is saved in the design
history.

You can later restore the old offense *after rebuilding it inside Madden*:

```bash
python -m cfb_coach ml offense-design \
  --rollback-design YOUR_PROPOSAL_ID \
  --attest "I restored and checked the prior custom offensive playbook inside Madden."
```

## Model-created Custom Adjustments

`macro_blueprints` are **new generated names and combinations of model-
selected play targets and sourced adjustment primitives**. They are not
silently added to your eight active offensive Custom Adjustment slots. Every
setting has an explicit source or an editor-verification warning; the model
does not fabricate missing route assignments or Xbox buttons.

After confirming a new playbook, create and verify one named macro using the
Madden editor. Then explicitly attest both settings and its *armed* slot:

```bash
python -m cfb_coach ml offense-design -o cpu \
  --verify-macro ML-MAN-HR \
  --attest "I built this named Custom Adjustment with the displayed settings and armed it in Madden."
```

If your eight slots are already occupied, intentionally remove one inside
Madden, and name the exact removed macro:

```bash
python -m cfb_coach ml offense-design -o cpu \
  --verify-macro ML-MAN-HR \
  --retire-existing MATCH \
  --attest "I removed MATCH, installed this named macro with the researched settings and armed the new slot in Madden."
```

The slot change requires an explicit user action. The verified macro is
stored per opponent, in a separate opt-in registry. Live offensive ML
may rank it **only** when the selected formation/play matches the approved
source pair and a corroborated pre-snap look warrants it. Unreviewed designs
are never available for live recommendations.

**No external automation edits Madden itself.** The repository cannot create
new native EA playbook entries or macros without editing in the game.
The HTML remains formation + play only, with optional actions behind a drawer;
you mark independently whether the recommended action was actually applied.

## Safety and next work

- No automatic in-game playbook switching; plan is offline and staged.
- Every recommended formation/play must be in the applied record.
- Changing the database record requires the matching proposal ID and an
  explicit installation attestation, with hash/revision checks for stale edits.
- Macro creation requires separate confirmation and a free active slot.
- All changes keep original executed-snap outcomes and research provenance.
- Postgame evaluation should compare verified execution/outcomes by formation,
  play concept, and used adjustments; unplayed alternatives remain unknown.
- Future: learn configuration-level action values once enough independently
  verified macros and hot routes have been used, and support automatic
  *proposal generation* following each CPU game with human editor confirmation.
