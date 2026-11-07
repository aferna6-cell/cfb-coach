# Madden 27 route-art features: offense 93-play pilot

## Implemented and how it is used

`offense_play_art_features.json` is derived from the ten AceMadden formation-page **screenshots the user uploaded**, using offline graphics-only segmentation (yellow/red/cyan route ink). The 93 cropped tiles are **not published in the GitHub repo**. They remain in the user's private capture bundle.

Each play has 16 compact numeric features describing the visible paths' depth, lateral position, color, horizontal tendency, density, and spread. The model also derives 10 pre-snap coverage interaction features from the visible stock route geometry and the opponent's predicted coverage probabilities. These are learned inputs, not hard-coded rules saying that a particular play beats a coverage. The model **does not require OL assignments**. A screenshot can establish shape and run direction, but it cannot always establish player-specific labels, dynamic option routes, blocking reads, or exact Xbox hot-route legality.

The existing `offense_structural_model.py` now consumes these features through `offense_play_features.full_vector` alongside the previous experimental coordinator, opponent tendencies and existing route annotations. These features are a regularized additional input group, **not a hard-coded rule for choosing a play**. Every stock installed play may be scored, and only the uploaded formation/play matches contribute layout data.

### Coverage

- All **93/93** installed play tiles captured and represented in the numerical JSON.
- **91/93** pass the cross-source identity check and are used by the model.
- Two unmatched aliases, `Gun 5WR Tight / L Spot` (image says `Z Spot`) and `Gun 5WR Tight / Zl Spot Shake` (image says `Z Spot Shake`), are stored for review but return zeros when scored. They are not silently mapped.
- All 93 have visual features; **0/93** have all eleven player assignments verified by these feature vectors.

This is meaningful *route-layout knowledge*, not a full typed route tree. For example, the model now sees that Mesh and Mesh Post have different downfield occupancy rather than treating them as unrelated string names. When a route is adjusted, the visual features continue to mean the **stock play's visual layout**, and player-specific adjustments are still handled through the separate assignment-aware system, subject to actual control confirmation. No code pretends that an adjusted screenshot was observed.

## Rebuild from the user's capture bundle

```
python -m pip install Pillow
python scripts/madden_ml/build_offense_art_features.py \
  --captures /path/to/Madden27_93_offense_play_art_captures \
  --snapshot cfb_coach/data/madden27/installed_playbook_snapshot.json \
  --out /tmp/rebuilt_offense_play_art_features.json
```

The runtime uses **only** the committed JSON and standard library; no website connection, image processing or computer vision during a game. The data file digest is included in model artifact validation. Modifying play-art features requires retraining. The feature schema version is bumped to prevent previously trained artifacts from silently scoring a different vector.

## CPU research pilot (manual sidecar; not wired into live coach)

PR #108's CLI can train/evaluate on a **copy** of your existing Madden database:

```
python scripts/madden_ml/offense_structural_eval.py --db /path/to/COPY_of_coach.db --out /tmp/offense-eval
python scripts/madden_ml/offense_pilot_cli.py --workdir ~/.cfb-coach-pilot config-init
python scripts/madden_ml/offense_pilot_cli.py --workdir ~/.cfb-coach-pilot train --db /path/to/COPY_of_coach.db
python scripts/madden_ml/offense_pilot_cli.py --workdir ~/.cfb-coach-pilot validate
python scripts/madden_ml/offense_pilot_cli.py --workdir ~/.cfb-coach-pilot shadow --down 3 --distance 7 --yardline 45
```

Use `shadow` first. The pilot remains **off by default**, and you must separately approve it before any CPU-only manual pilot. `pilot` here is a **manual CLI recommender**, not a controller integration. Because no Xbox individual-hot-route options were confirmed, the live executable adjustment output is currently `NO ADJUSTMENT` unless you verify those controls.

This PR does not claim to win more games. Compare the original experimental model against the visual-feature residual on real, game-separated held-out CPU play data before activation; if no improvement, keep baseline. A screenshot-derived feature set is not supervised proof of play effectiveness.

### Acceptance and limitations

- Every numeric visual record has a crop hash and source page URL, but the source site's images are not redistributed in Git.
- Coverage match: 91 model-eligible / 2 withheld; 93 images analyzed.
- No fabricated per-player route or blocking assignments.
- Strict 150 ms existing decision budget remains unchanged; real-time inference is JSON lookup + arithmetic.
- CPU-only; no production database or live caller changes.
- The existing 40 third-party `source_documented` receiver assignments in PR #108 are a separate evidence source; do not treat their licensing as granted for ML training merely because this route-art feature bank exists. Audit that data channel before a live pilot.