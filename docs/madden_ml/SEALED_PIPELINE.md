# Sealed decision pipeline — status and follow-up plan

**Status:** Compatibility wrapper only. **Not production-ready.** Flag stays off.

Module: `cfb_coach/madden/model/decision_pipeline.py`  
Flag: `CFB_COACH_SEALED_PIPELINE=1` or meta `ml_sealed_pipeline=1`

## What exists today

| Piece | Status |
| --- | --- |
| Feature flag gate (`pipeline_enabled`) | Implemented |
| `assemble_decision` calling the legacy heuristic | Implemented |
| Optional VOD prior constrained to legal candidates | Partial (selects candidate identity only) |
| Parity report (formation/play/macro/side) | Implemented |
| Heuristic compatibility mode (still displays legacy call) | Implemented |
| Optional PR #19 `load_mappings` / `lookup` boundary | Stub (`optional_vod_lookup` returns None until resolved via applied book) |

## What is **not** implemented

1. **Reads and Custom Adjustments for the finally selected play** — the wrapper copies the legacy call's macro/rationale; it does not regenerate reads/adjustments after a VOD prior changes the candidate.
2. **Macro validity for the sealed play** — no check that the armed Active-8 macro still contains the sealed play; legacy post-hoc VOD swap still clears macros outside this module.
3. **Replacement of the live post-hoc VOD swap** — `cfb_coach.vod_model.live.apply_madden_call` still mutates formation/play after reads are built. The sealed pipeline is **not wired** into `madden/cli.py` / `playcaller.make_call` / HTML `LivePlayController`.
4. **Deterministic replay parity suite** — no automated comparison of old vs new across a library of situations for reads, macros, CA settings, book legality, anti-repeat, or CPU offense-only.
5. **PR #19 full integration** — `optional_vod_lookup` does not yet resolve mapped names through `candidate_in_book` against the applied book.
6. **Live call-path connection** — enabling the env flag does nothing in live play until a caller invokes `assemble_decision`.

## Safe for Franchise logging?

Yes for **shadow logging and verified data collection** with the flag **off** (default). The sealed pipeline does not need to be complete for that.

Do **not** set `CFB_COACH_SEALED_PIPELINE=1` during real games.

## Follow-up implementation plan (do not start in Sprint 3.1)

1. Wire `assemble_decision` behind the flag at a single site inside the Madden live caller, after `eligible()` candidates are built and before reads/macros are attached.
2. For the sealed candidate only: generate reads, select/validate macro, build Custom Adjustments.
3. Remove (flag-gated) the post-hoc `apply_madden_call` swap when the sealed path is active.
4. Resolve PR #19 mappings via `load_mappings`/`lookup` → `candidate_in_book` only.
5. Golden-file deterministic replay: formation, play, reads, macro, CA, book legality, anti-repeat, CPU offense-only — emit an explicit difference report.
6. Keep hybrid selection off until promotion + shadow validation + explicit approval.
