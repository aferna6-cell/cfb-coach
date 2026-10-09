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


## Sprint 10 — Learn WHICH adjustments to call, using verified execution evidence

A completed pre-snap action decision is not an observation that the
adjustment was applied. The live Madden outcome panel can separately record:

1. **Adjustment offered and explicitly applied**, while the exact
   recommended formation/play was verified executed and has a labeled outcome.
2. **No action offered and explicitly confirmed unchanged**, including
   no manual hot route, protection change or Custom Adjustment, on a verified
   executed play with a labeled result.

Unchecked boxes, changed plays, missing outcomes and prior-snap coverage
are **not** upgraded into action-training examples. A successful play does
not, by itself, prove any adjustment was responsible for that success.

The offline model groups verified examples by *play concept + down/distance
bucket* and compares each action against confirmed unchanged plays in the
same group. It reports shrunk observational associations rather than
inventing counterfactual uplift. The evidence is still subject to
selection bias (e.g. pressure causes both more protections and worse
outcomes). Therefore it defaults to **shadow**.

The normal `ml experimental --retrain` command also refreshes the
**shadow action model** from eligible logged outcomes after each CPU game.
It never auto-promotes; if a bounded-active action artifact is installed,
routine retraining preserves it unchanged pending deliberate review.

When coverage is available from a **live pre-snap read**, the action learner
also groups that credible look with the play concept/down-distance. A
single last-snap hint and post-snap coverage labels are excluded from
decision-time action features to prevent look-ahead leakage.

The explicit promotion gate for a particular action/context requires:
at least 12 verified action applications, 12 explicitly confirmed unchanged
executions, 3 distinct games represented in **each** group, and a
sufficiently large shrunk observational association. Promotion is reversible
and can shift pre-snap research action scores by at most **0.04** in either
direction; it never permits unverified, incompatible or unarmed actions.
This is **not** a causal-effect model. No action preference is learned from
recommendations alone.

Run from the repo venv, outside a live game:

\`\`\`bash
python -m cfb_coach ml offense-action-learn --train     # recompute SHADOW evidence
python -m cfb_coach ml offense-action-learn --details   # inspect all counts
python -m cfb_coach ml offense-action-learn --promote   # refused until gate passes
python -m cfb_coach ml offense-action-learn --rollback  # immediately disable learned shifts
\`\`\`

The model can already propose new source-backed macro **blueprints**, but
these remain uninstalled until the user explicitly builds and verifies the
Madden Custom Adjustment. The next development increment should combine
action-model errors and the full-playbook inventory to prioritize **new
macro drafts**, validate every editor setting/source and legal formation/
play pair, then compare the proposed macro in CPU play. Unverified macros
must never enter a live action loadout.

Future agent objective: joint ranked **(formation, play, optional verified
action)** decisions, game-by-game evidence, audited on/offline evaluation
and rollback; no claim of an autonomous Xbox editor without an actual
authorized computer-use/device controller.
