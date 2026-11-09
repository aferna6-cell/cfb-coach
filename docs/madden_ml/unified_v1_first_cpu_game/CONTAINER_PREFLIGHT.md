# CPU preflight: SHADOW ONLY

artifact: `cold_start_model.json:5d8f95f4e6`
blocking: ['offense_installed_book_confirmed_by_owner', 'defense_installed_book_confirmed_by_owner', 'history_evaluation_consistent_with_model', 'owner_database_audit']

| gate | status | critical |
|---|---|---|
| offense_installed_book_confirmed_by_owner | NOT_RUN | True |
| offense_book_fingerprint_matches | PASS | True |
| offense_formation_limit | PASS | True |
| defense_installed_book_confirmed_by_owner | NOT_RUN | True |
| defense_book_fingerprint_matches | PASS | True |
| defense_formation_limit | PASS | True |
| model_artifact_file | PASS | True |
| history_evaluation_consistent_with_model | NOT_RUN | True |
| owner_database_audit | NOT_RUN | True |
| local_game_state_parser | PASS | True |
| live_loop_ran_clean | PASS | True |
| live_latency_p95_le_150ms | PASS | False |
| live_no_timeouts | PASS | False |
| live_no_fallback_or_heuristic_substitution | PASS | True |
| live_no_uninstalled_calls | PASS | True |
| live_no_duplicate_snap_ids | PASS | True |
| live_recommended_vs_executed_logged_exactly | PASS | True |
| live_model_attribution_correct | PASS | True |
| live_only_legal_adjustments | PASS | True |
| live_side_follows_possession | PASS | True |
| live_previous_play_learning | PASS | True |
| protected_assets_unchanged | PASS | True |

## Evidence
```json
[
 {
  "gate": "offense_installed_book_confirmed_by_owner",
  "status": "NOT_RUN",
  "critical": true,
  "evidence": {
   "source": "repo snapshot (NOT this machine's installed book)",
   "fingerprint": "OFF-INST-14E6E7E688",
   "formations": 5
  },
  "how_to_run": "unified confirm-install --proposal <proposal_offense.json> --attested-by <you> --confirmed-books <file>"
 },
 {
  "gate": "offense_book_fingerprint_matches",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "fingerprint": "OFF-INST-14E6E7E688"
  }
 },
 {
  "gate": "offense_formation_limit",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "formations": 5,
   "limit": 15
  }
 },
 {
  "gate": "defense_installed_book_confirmed_by_owner",
  "status": "NOT_RUN",
  "critical": true,
  "evidence": {
   "source": "repo snapshot (NOT this machine's installed book)",
   "fingerprint": "DEF-INST-58E3DED020",
   "formations": 6
  },
  "how_to_run": "unified confirm-install --proposal <proposal_defense.json> --attested-by <you> --confirmed-books <file>"
 },
 {
  "gate": "defense_book_fingerprint_matches",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "fingerprint": "DEF-INST-58E3DED020"
  }
 },
 {
  "gate": "defense_formation_limit",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "formations": 6,
   "limit": 15
  }
 },
 {
  "gate": "model_artifact_file",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "path": "docs/madden_ml/unified_v1_first_cpu_game/cold_start_model.json",
   "sha256": "5d8f95f4e69c7d331f18d5434f3350082dc412c3852f892ac1c52804c53f2b85",
   "records": 0,
   "provenance": {},
   "mode": "UNCALIBRATED COLD-START PRIOR (no trained weights)"
  }
 },
 {
  "gate": "history_evaluation_consistent_with_model",
  "status": "NOT_RUN",
  "critical": true,
  "evidence": {
   "note": "historical import/evaluation not run on the owner's database"
  },
  "how_to_run": "unified history --db <isolated test-franchise db> --out <dir>"
 },
 {
  "gate": "owner_database_audit",
  "status": "NOT_RUN",
  "critical": true,
  "evidence": {
   "note": "no owner database was supplied"
  },
  "how_to_run": "unified preflight --db <isolated test-franchise coach.db> ..."
 },
 {
  "gate": "local_game_state_parser",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "cases": 7,
   "failures": []
  }
 },
 {
  "gate": "live_loop_ran_clean",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "snaps": 200,
   "rc": 0,
   "wall_s": 9.64
  }
 },
 {
  "gate": "live_latency_p95_le_150ms",
  "status": "PASS",
  "critical": false,
  "evidence": {
   "n": 200,
   "p50": 38.278488500054664,
   "p95": 59.5907869999337,
   "max": 63.65091899988329,
   "scope": "this machine, real cmd_game loop, persisted store; repo snapshot books",
   "finish_ms": {
    "p50": 3.657694000025913,
    "p95": 5.382475000033082
   }
  },
  "how_to_run": "unified preflight ... on the owner's laptop"
 },
 {
  "gate": "live_no_timeouts",
  "status": "PASS",
  "critical": false,
  "evidence": {
   "timeouts": 0
  }
 },
 {
  "gate": "live_no_fallback_or_heuristic_substitution",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "fallbacks": 0,
   "decision_models": [
    "unified_v1"
   ]
  }
 },
 {
  "gate": "live_no_uninstalled_calls",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "problems": []
  }
 },
 {
  "gate": "live_no_duplicate_snap_ids",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "unique": 200,
   "snaps": 200
  }
 },
 {
  "gate": "live_recommended_vs_executed_logged_exactly",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "mismatches": []
  }
 },
 {
  "gate": "live_model_attribution_correct",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "artifact": "cold_start_model.json:5d8f95f4e6",
   "bad": []
  }
 },
 {
  "gate": "live_only_legal_adjustments",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "bad": []
  }
 },
 {
  "gate": "live_side_follows_possession",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "bad": []
  }
 },
 {
  "gate": "live_previous_play_learning",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "expected_world_model_rows": {
    "offense": 68,
    "defense": 66
   },
   "replayed_rows_by_provenance": {
    "our_offense_executed": 68,
    "our_defense_executed": 66
   },
   "matches": true,
   "belief_sources_final": {
    "offense": "general_prior+game(99)",
    "defense": "general_prior+game(99)"
   }
  }
 },
 {
  "gate": "protected_assets_unchanged",
  "status": "PASS",
  "critical": true,
  "evidence": {
   "protected_files_checksummed": 25,
   "changed": [],
   "removed": [],
   "added": []
  }
 }
]
```

GO would mean ready for an experimental CPU test, not proven strategic superiority. Latency/legality are measured through the real continuous loop, never a bare in-process call.