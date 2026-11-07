# CPU-only previous-play coverage feedback bridge

This research feature consumes **the existing previous-play result + coverage form**. It does not add a second defensive snap logger, nor change the running coach.

## Existing data path

1. The browser's previous-play form calls `LastSnapBook.close_from_form`.
2. `LastSnapBook._record` writes `snaps.coverage_seen`, `result`, `opponent_id`, `session_id`, and `snap_seq`.
3. `offense_live_coverage.coverage_for_next_snap` reads only **closed earlier offense snaps** for that same session/opponent. The next/current snap is excluded by `snap_seq < next_snap_seq`.
4. It combines the available historical model's prior with the current game's named coverages (8-snap shrinkage) and preserves Cover 2 Man, Tampa 2, Palms, Quarters and other distinct labels. Generic "pressure" is tracked separately, **not** assumed to mean Cover 0.
5. `PilotSession.recommend_from_logged_game` supplies the prior+current-game mix to the structural ML ranking for a CPU-only recommendation. It records `coverage_history` in its *separate* pilot JSONL log.

## Opt-in CLI shadow test (after training a new artifact)

```bash
python scripts/madden_ml/offense_pilot_cli.py --workdir ~/.cfb-coach-pilot config-init
python scripts/madden_ml/offense_pilot_cli.py --workdir ~/.cfb-coach-pilot train --db /path/to/COPY-of-madden27.db
python scripts/madden_ml/offense_pilot_cli.py --workdir ~/.cfb-coach-pilot validate

python scripts/madden_ml/offense_pilot_cli.py --workdir ~/.cfb-coach-pilot shadow \
  --down 2 --distance 7 --yardline 45 --opponent cpu --opponent-type cpu \
  --history-db ~/.cfb-coach/madden27.db --session-id GAME_SESSION_ID --next-snap-seq NEXT_SNAP_SEQUENCE
```

The session ID, opponent ID and next snap sequence must match the user's *current* game's records. The SQL reads the game DB in SQLite `mode=ro`, never writes it.

## To be genuinely ready to test in the normal in-game overlay

- **Not done:** wire the `recommend_from_logged_game` bridge into the existing on-screen game loop behind an explicitly approved CPU-only feature flag. This CLI currently requires typing situation, session ID and sequence.
- Run the full repository suite and latency checks in the user's actual checkout. Six new isolated bridge tests passed locally, but there is no full-suite/CI result for this draft.
- Retrain/evaluate the actual offensive structural artifact on a read-only copy of the game DB, test against the existing coordinator, and validate source/fingerprint hashes.
- Test that entering Cover 2 Man on prior snap N changes *next* snap N+1's saved `coverage_history`, not N itself. Verify the resulting legal play, with adjustment either specific and confirmed or NO ADJUSTMENT.
- Confirm the pilot is CPU-only and off until user approval; do not merge or auto-promote this research branch.

Historical coverage labels are manually entered post-play, not automatically verified by screenshot. They are valid **user observations**, not infallible ground truth. Predictions are probabilities, not knowledge of the hidden next play.

No claim of improved win rate; until a game test shows a gain, this is an adaptation plumbing result only.
