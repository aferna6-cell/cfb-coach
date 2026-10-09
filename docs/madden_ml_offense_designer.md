# Sprint 6C — ML offensive playbook and Custom Adjustment designer

**Status:** opt-in/offline designer; no unapproved live playbook or macro edits.
Stacked on Sprint 6B's model-primary offensive coach.

## Model controls the next game's offensive inventory

This is a *new design stage*, not merely a new name for heuristic prep.
It asks the installed experimental success model to score Madden 27
catalogued formation/play pairs over multiple generic pre-snap situations.
The designer may:

- Add, replace, or remove whole formations (up to 12 in this pilot).
- Include EVERY listed play from each chosen stock-book formation; no individual play trims.
- Select a formation's exact sourced plays from a *different* catalogued
  stock playbook; each formation has `formation_sources` provenance.
- Propose up to eight newly named offense Custom Adjustment **blueprints**
  derived from source-backed hot-route and protection primitives.
- Show formation-only ADD/REMOVE/REINSTALL_FULL_FORMATION deltas against your locked book.

The ML model's **play-concept** prediction and evidence drive the
formation's aggregate score; the current experimental artifact does NOT yet
learn an independent formation-specific effect. This distinction matters
until we collect verified results for multiple formations. The heuristic
does not choose the winning play. A slight incumbency preference
avoids pointless churn; diversified model concept scores rank a formation without
letting one high-scoring screen determine its entire value. At least one run is retained.
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
- New proposed whole formations and ALL their stock-sourced plays, source-book pages,
  formation-only deltas and one checkbox per full formation.
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

The page is a **read-only local HTML file**. Checking formation-install boxes
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

The JSON includes a short `proposal_id`, complete formations with their plays and all
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

## Sprint 6E — screen-lock fix and full-formation selection

The former model-primary live policy could choose the same HB Slip Screen
indefinitely because it ignored the computed anti-repeat adjustments.
The new `offense_selection_policy.model_primary_repetition_aware.v1`
uses the model's own predicted success scores and recent *recommended* calls
to reduce repeated exact-play and screen-family exposure when credible
alternatives exist. It does not treat unverified recommendations as executed
snap outcomes. The heuristic is still a failure fallback only, not the
live offensive play selector. Audited ranked candidates retain original
model probabilities and explicit `selection_score` penalties.

The offline designer now proposes **formations**, not hand-picked plays:
the coach may add/remove/reinstall entire formations, and the entire
catalogued play inventory from each selected *stock source* is available to
the live model. This is a source-specific Madden catalog (not an invented
cross-stock play union). The HTML install checklist has one checkbox per
complete formation rather than per individual play. To use new proposals,
regenerate/stage a fresh design and install/confirm it in the Madden editor.
Old staged or installed trimmed designs do not silently mutate.

## Sprint 7 — situation-aware, varied offensive model

The 28–7 CPU game exposed runs on third-and-long and repeated use of a small
set of plays. The earlier model's down/distance factor was COMMON TO ALL
candidates, so it could not distinguish a run from a pass on a long down.

- Train concept-by-down/distance and family-by-down/distance interactions
  from observed outcomes, with strong shrinkage for sparse matchups.
- On third/fourth-and-7+, exclude ground runs if any legal passing play exists.
  Do not invent plays when the applied book lacks a passing option.
- Add candidate-specific conversion suitability, short-yardage and two-minute
  scoring. Model probabilities remain separate from these policy costs.
- Live sessions forward the actual game ID and next snap sequence into
  inference; replayable exploration does not accidentally mix old CPU games.
- Keep up to 20 recommended calls PER SESSION; track recent exact plays,
  concepts, screen family and formations. A recommendation is not a
  statement that the user executed it.
- Explore among a reproducible, model-scored top set. CPU tests use wider
  exploration and human experiments tighter thresholds.
- Break repeated low-evidence call patterns, including screens, if credible
  alternatives are available. Heuristic remains emergency fallback only.
- Persist candidate counts, selection penalties and conversion-specific
  rationale in the sealed decision JSON for postgame analysis.

Learning provenance: The user attested to following all 63 final plays in
the 28–7 game 9f2ebdeb9d8f4e2d. Keep its missing outcomes missing;
do not fabricate yards, TDs or training labels for those snaps.

Next CPU test: after CI passes, backup DB, retrain the experimental model,
and play with the already installed book. Evaluate third-down calls,
concept and formation variety, execution rates, and live latency.

Commands:
    python -m cfb_coach ml backup-db
    python -m cfb_coach ml experimental --retrain
    python -m cfb_coach ml status
    python -m cfb_coach play --game madden27 -o cpu

## Sprint 8 — complete installed playbook tracking and live ML coverage

The approved/offense locked book is the authoritative inventory, **not**
the handful of plays in older heuristic gameplans or a capped candidate menu.
Each model-designed formation includes ALL catalogued plays from its named
stock source. The model considers every play that remains situationally
eligible; first it ranks them, and then a modest CPU exploration allowance
can sample from the **complete eligible playbook**, not merely 24 plays or
two plays per concept. Human-opponent exploration is more conservative.

The designer and confirmation step now record a stable `inventory_id` hash
for the exact (formation, source book, plays) state. A partial or drifted
source formation cannot be marked installed; staged drafts never become
live inventory until the user actually rebuilds the playbook in Madden.

Each sealed ML decision records the book fingerprint, all-play count,
formation count, eligible-play count and count excluded by situation,
plus its contextual sampling audit. Those counts do not invent outcomes.

The designer browser lists every play and, once installed, how often
it was **recommended** to the selected opponent. Unused inventory remains
listed at zero. A count is not confirmation of execution or causal success.

Read-only inspection:

```bash
python -m cfb_coach ml offense-inventory -o cpu
python -m cfb_coach ml offense-inventory -o cpu --summary
python -m cfb_coach ml offense-inventory --game-id YOUR_GAME_ID
python -m cfb_coach ml offense-design -o cpu --show
```

To use a newly suggested formation inventory, stage the complete design,
build the whole formations in Madden, and run the explicit installation
attestation. Simply viewing a proposal never makes it callable.

Situation protections remain: third/fourth-and-long ground runs are
removed when a legal pass is present, short-field-only plays need the
appropriate situation, and the model still ranks the normal plays.
An entire inventory being tracked does not mean every play should
be called in every situation or force equal usage of weak calls.


## Sprint 9 — Situation-specific offensive Custom Adjustments

The model has a complete playbook and is already able to choose one researched
pre-snap action for its selected formation/play. Sprint 9 addresses why action
recommendations have been sparse, overly generic or hard to troubleshoot.

The model's play selection is unchanged. A second, explicit action decision
compares **one researched action or verified armed macro** with doing nothing
before the snap. Action utility scores are *research-based and heuristic
calibration*, **not learned causal uplift**. The model must not assert that a
route or macro raises its predicted win probability without action-specific
outcome evidence.

- Source-backed hot routes and pass protection require a live or independently
  corroborated current defensive look (not just the last opponent snap).
- Existing Custom Adjustment macros are only eligible if selected in prep,
  grounded by editor settings and real researched controls, and compatible
  with the exact formation/play from the active custom playbook.
- Situation-only macros require relevant **red zone, goal line, or third/fourth
  and short** evidence. Being merely installed cannot cause constant firing.
- Model-created macros are never callable while drafts. Explicit approval
  after actual Madden editor installation and slot arming is still required.
- The no-action baseline becomes more conservative with an unknown defensive
  look and uncertain model evidence. No adjustment is a normal outcome.
- Macro compatibility checks now cover every play in every installed formation,
  not just the first 80 results from a large custom book.
- Live HTML's primary call stays **Formation — Play**. Any adjustment details
  remain in the collapsed secondary area, with independent applied-action
  confirmation. Existing postgame action recommended/applied counters remain
  available; they are not proof of a causal adjustment benefit.

### Before a CPU game

Run this **read-only** action readiness diagnostic (the output distinguishes
selected user-note macros from individually verified model-created macros):

```bash
python -m cfb_coach ml offense-actions -o cpu
python -m cfb_coach ml offense-inventory -o cpu --summary
```

If the report finds zero active/supported macros, check the active Custom
Adjustment slots in Madden and your last `prep` settings. The coach cannot
fire macros that have no installed compatible play or no researched settings.
A saved prep selection alone is not proof the editor entry is armed.

Next improvement: log verified *execution of the requested adjustment* and
its outcome by exact action/formation/play/coverage/situation before training
an action-value model. Until sufficient clean paired observations exist, no
action uplift claim or automatic macro creation/activation is justified.

## Sprint 10 — verified action learning + macro lab (consolidated)

Stacked on Sprint 9 situational actions. Sprint 10 had two alternate drafts
(PR #34 evidence-gated learning, PR #35 macro lab). This tree keeps **#34's**
learning contract (sealed `pre_snap_action_context`, explicit unchanged-play
confirmation, `--train` / `--promote` / `--rollback`) and **#35's**
`offense_macro_lab` draft generator. There is one learning module and one CLI.

```bash
python -m cfb_coach ml offense-action-learn --train
python -m cfb_coach ml offense-action-learn --promote   # only with ready groups
python -m cfb_coach ml offense-action-learn --rollback
python -m cfb_coach ml offense-macro-lab                # DRAFT preview
python -m cfb_coach ml offense-macro-lab --compose --stage
```

Learned action shifts stay shadow until promotion and remain bounded (±0.04).
Research eligibility, armed-slot checks and NO_ADJUSTMENT always win first.

## Sprint 11 — Adaptive AI Offensive Coordinator

**Status:** experimental, CPU-default; never auto-merges to `main`, never
mutates a live Franchise SQLite, never silently applies a staged playbook or
arms an unverified macro.

### Autonomy boundaries

| Actor | May do | Must not do |
|-------|--------|-------------|
| Model | Propose formation portfolio, joint play+adjustment, draft macros, update in-game memory from sealed decisions | Edit Madden, invent controller inputs, treat suggestions as executions, fabricate coverage |
| User | Install formations, attest install, verify/arm macros, confirm applied actions | — |
| Live path | Choose from **confirmed installed** book under 150 ms; heuristic fallback on timeout | Call draft/unarmed macros; use post-snap outcomes before the snap |

### Pregame → live → postgame walkthrough (CPU)

```bash
# 0. Optional: locate / backup laptop DB (read-only audit). Never rewrite history.
python -m cfb_coach ml find-db
python -m cfb_coach ml backup-db

# 1. Pregame: model designs whole formations (ALL source plays) + rationale
python -m cfb_coach ml offense-design -o cpu --max-formations 5 --text
python -m cfb_coach ml offense-design -o cpu --stage
# User installs in Madden, then:
# python -m cfb_coach ml offense-design -o cpu --confirm-installed <PROPOSAL_ID> \
#   --attest "Installed all formations/plays in Madden custom editor"

# 2. Macro drafts (optional compositions still DRAFT)
python -m cfb_coach ml offense-macro-lab --compose
python -m cfb_coach ml offense-macro-lab --compose --stage
# After building + arming a slot in Madden:
# python -m cfb_coach ml offense-design -o cpu --verify-macro NAME --attest "..."

# 3. Readiness + inventory
python -m cfb_coach ml offense-actions -o cpu
python -m cfb_coach ml offense-inventory -o cpu --summary

# 4. Enable experimental ML (CPU offense). Heuristic is emergency fallback only.
python -m cfb_coach ml experimental
# Live HTML / play window: compact FORMATION — PLAY; adjustments below.
# python -m cfb_coach play --opponent cpu --game madden27

# 5. After games: train play model + shadow action evidence
python -m cfb_coach ml experimental --retrain
python -m cfb_coach ml offense-action-learn --train
python -m cfb_coach ml postgame-experimental

# Rollback live ML anytime:
python -m cfb_coach ml heuristic
```

### Architecture (model-primary, not a second playcaller)

1. **`football_situation`** — explicit observations vs tendencies vs unknowns;
   labeled conversion / clock / pressure priors.
2. **`offense_designer` (v2)** — scores the full catalogued formation set across
   a situation matrix; retains every source-book play per chosen formation;
   returns rationale, audibles, deltas, provenance, inventory fingerprint.
3. **`offense_joint_decision`** — searches legal `(formation, play, adjustment_plan)`
   including **NO_ADJUSTMENT**, hot routes, protections, armed macros; bounded
   for the 150 ms budget; never invents settings.
4. **`offense_macro_lab`** — single-primitive and optional multi-primitive
   **DRAFT** compositions with conflict checks; VERIFIED/ARMED only via existing
   attestations.
5. **`offense_game_memory`** — per-game recommendations, live-look tendencies,
   verified executions kept separate; tendencies are not current coverage.
6. **`offense_action_learning`** — observational, evidence-gated; sealed
   pre-snap context only.

### Blowout-game evidence

Two user-reported CPU blowout wins were cited in earlier sprints. This cloud
environment has **no** laptop Franchise SQLite (`CFB_COACH_MADDEN_DB` unset;
no local `.db` with those game IDs). Regression tests therefore use **explicitly
labeled synthetic fixtures** only. Do not treat synthetic success rates as
claims about those games. On the laptop, audit read-only with
`ml find-db` / `ml report` / inventory before importing any sealed snaps.

### Capability matrix

| Capability | Status |
|------------|--------|
| Pregame formation portfolio + full plays | **Implemented** (staged; user installs) |
| Football situation evaluator | **Implemented** |
| Joint play+adjustment decision | **Implemented** (experimental path) |
| NO_ADJUSTMENT can win | **Implemented** |
| Multi-primitive macro drafts | **Implemented** (DRAFT only) |
| In-game opponent tendency memory | **Implemented** (live looks only) |
| Evidence-gated action learning | **Implemented** (shadow default) |
| Human-opponent ML offense | **Opt-in only** (existing flag) |
| Defense ML control | **Shadow-only** (unchanged) |
| Causal adjustment uplift | **Unverified** — observational only |
| Auto-install / auto-arm | **Forbidden** |
| CFB 27 reuse | **Architecture ready**; game packs unchanged |

### What is still future work

- Richer roster-verified personnel fit when Madden roster exports are available.
- Offline policy evaluation / holdout games for joint decisions.
- Broader multi-action editor field inventories once user-verified.
- Importing laptop blowout logs as sealed regression fixtures (user-mediated).
