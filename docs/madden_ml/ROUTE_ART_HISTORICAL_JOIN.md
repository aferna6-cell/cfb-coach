# Madden 27 route-art: read-only historical compatibility check

**Source:** the user-provided November 6 Madden database backup, queried in read-only mode. No private snap rows or raw histories were committed.

```sql
SELECT COALESCE(executed_formation,formation), COALESCE(executed_play,play),
       session_id, opponent_id
FROM snaps
WHERE side='offense' AND executed_status='identified'
  AND executed_verification='verified';
```

- 655 verified offense snap records in 16 session IDs / 3 opponent IDs.
- 655/655 exactly match a recorded installed formation/play identity.
- 637/655 match one of the 91 model-eligible AceMadden screenshot-derived art feature records.
- The other 18 are the quarantined cross-source names: Gun 5WR Tight / L Spot (13) and Zl Spot Shake (5).
- The visuals cover 93/93 named plays but neither these stats nor the images establish each player's exact route or a model improvement.

This is a **data-join smoke test**, not an offline model benchmark. Run the whole-game held-out evaluation from the full checkout against a copy of the DB before approving the CPU-only pilot.
