# Lions Ravens two-formation defensive ML coordinator (`ravens_defense_ml`)

Status: **NOT ENABLED. NO-GO for live use.** The outcome model is `UNTRAINED_PRIOR_ONLY` because there are zero
verified defensive executions. The installed defensive book has neither formation.

## Scope

- Formations: Baltimore Ravens **Nickel 2-4 Wide** and **Dime 2-3 Odd**. No other formation can be called.
- Calls (12, from `datasets/madden27/lions_ravens_two_formation_defense_v1.json`):
  - 6 Match calls (no macro). Zone It is optional on Quarters and Palms only.
  - 4 Nickel stock calls (24 Base macro required).
  - Stock Cover 4 Zone fallback.
  - Dime DB Fire 2 (blitz only, DB Fire macro required).
- Candidates: 14 complete candidates. A candidate is the play plus its macros; Quarters and Palms appear with and without Zone It. Every candidate is scored. None is forced, and that includes Tampa 2 Contain in the red zone.
- Match Cover 2 (no macro) and Nickel Cover 2 (24 Base) share a base play but stay separate candidates. Each has its own learned terms.
- Offense is untouched: `civil_scheme_only` files are hash-pinned in the tests. The offense and defense modes, restrictions and labels are independent.

## Pipeline (per defensive snap)

1. **Legal set (fail closed).** Each candidate is checked in `legality.check`:
   - approved candidate;
   - Dime only for DB Fire 2;
   - required macro present;
   - base play in the Ravens stock inventory;
   - formation and play in the applied (installed) defensive book;
   - two-formation fingerprint matches the owner-confirmed one;
   - each macro owner-confirmed;
   - coaching profile `COMPATIBLE`;
   - no dataset block (Match Cover 9 Show 2 is live-eligible since owner t358u; USER stays UNVERIFIED until confirmed in game).

   If no candidate is legal, the result is `NO_LEGAL_RAVENS_CALL`. There is never a fallback to another caller or formation.
2. **Opponent belief.** P(concept class) over 8 classes, using only strictly earlier snaps in this game plus finished earlier games. The live model is chosen by forward whole-game evaluation. A challenger replaces the Defensive V2 Dirichlet belief only if its game-clustered 95% CI of log-loss gain is entirely above 0.
3. **Outcome model.** `bayes_structural_logit.v1`:
   - The prior offense-success logit comes from the base rate plus the sourced coverage-vs-concept lift table (`football_priors`).
   - Candidate terms have a Laplace posterior fitted **only** on verified Ravens executions with outcomes.
   - Status is `UNTRAINED_PRIOR_ONLY`, `PARTIAL_UNDER_EVIDENCE_BAR` or `TRAINED` (at least 200 verified examples across at least 8 games).
4. **Decision.** Pick the lowest expected defensive loss (the Defensive V2 utility weights), integrated over the belief.
   - Zone It carries a small execution cost, so it must earn its place.
   - Zone It is never triggered by a single explosive play. Explosive and TD heads are candidate-agnostic until verified data exists.
5. **Re-validation, then USER.** The pick is re-checked, then USER is resolved.
   - USER is shown only if the owner verified it in game, in Nickel 2-4 Wide, for that call, Zone It state and offense shape. Otherwise it reads `USER: UNVERIFIED`.
   - Zone It never inherits the Match USER.
6. **Card** (all that shows live):
   ```
   PLAY: Nickel 2-4 Wide — Cover 3 Match
   USER: UNVERIFIED
   MACRO: 24 Base        (only when a macro applies)
   ```
   The full trace (belief, ranked scores, margins, uncertainty, latency) stays on `call.ravens_defense` and in the additive table `ravens_defense_decisions`.

## Commands

```
python3 -m cfb_coach ml ravens-defense audit        --db <mode-644 backup copy>
python3 -m cfb_coach ml ravens-defense train        --db <mode-644 backup copy> --out <artifact root>
python3 -m cfb_coach ml ravens-defense evaluate     --db <mode-644 backup copy> --out eval.json
python3 -m cfb_coach ml ravens-defense verify-book  --db <db>            # exit 1 when nothing is live-legal
python3 -m cfb_coach ml ravens-defense shadow       --db <copy> --artifact <dir> --down 3 --distance 8 --yardline 55 [--assume-setup]
python3 -m cfb_coach ml ravens-defense shadow       --db <copy> --session <id> --assume-setup   # historical replay
python3 -m cfb_coach ml ravens-defense postgame     --db <copy> --artifact <champion>          # never promotes
python3 -m cfb_coach ml ravens-defense enable       --db <db> --artifact <dir>  # refuses unless every release gate passes
python3 -m cfb_coach ml ravens-defense disable      --db <db> [--revert-artifact]              # revert
python3 -m cfb_coach ml ravens-defense coaching-audit | users | status
python3 -m cfb_coach play ... --ravens-defense | --previous-defense        # per-session override (CPU also needs --cpu-defense)
```

- `train`, `evaluate`, `audit`, `shadow` and `postgame` open `--db` with SQLite `mode=ro` + `query_only`. They prove the sha256 is unchanged, and they refuse mode-444 official copies.
- `enable` and `disable` are the only DB writers. Each backs up first.
- Release gates for `enable` (all must pass):
  - artifact sha256;
  - outcome model `TRAINED`;
  - held-out candidate-outcome gate `PASS`;
  - latency gate;
  - every non-dataset-blocked candidate live-legal (installed book, fingerprint, macros, coaching);
  - a valid owner confirmation file.

## Owner confirmation file (private, not committed)

The file lives at `~/.cfb-coach/ravens_defense/owner_confirmations.json`, or at the path in `CFB_COACH_RAVENS_DEFENSE_CONFIRMATIONS`. It uses schema `madden27.ravens_defense.owner_confirmations.v1`. Fields:

- `installed_book_fingerprint`, which must equal `verify-book`'s `RAV-DEF-…`;
- `macros.{24_BASE,DB_FIRE,ZONE_IT}`: `{exists, executes, verified_on}`;
- `coaching_profile`: `{profile_id, verified_on, advanced_nickel_adjustments_active: false, compatible_call_ids, incompatible_call_ids}`;
- `cover9_show2_alias_confirmed`;
- `standing_macro_attestation`;
- `user_verifications[]`: `{call_id, zone_it, offense_shape, formation, user, check, verified_on, source: "owner_in_game"}`.

A malformed file confirms nothing.

## Online league profile (Chiefs, owner t351u/t352u)

On `ml/lions-civil-scheme-only` (PR #153), `play --franchise online --opponent-team <team>` runs this coordinator as the session's defensive caller by default. It does that with no `ml ravens-defense enable` and no DB write:

- **Scheme book.** The two-formation dataset is the legal book (`legality.check(..., scheme_book=True)`). The installed-book and fingerprint items are satisfied by the dataset itself. A macro or coaching-profile item blocks a call only when the owner-confirmations file explicitly reports it missing or INCOMPATIBLE. Since owner t358u, Match Cover 9 Show 2 is live-eligible, so all 14 candidates are callable. USER for that call stays UNVERIFIED until confirmed in game.
- **Model.** A trained artifact is used when one exists. Otherwise an in-memory `UNTRAINED_PRIOR_ONLY` model is used, and nothing is written. The card's label says so.
- **Mixer.** This applies only with the online scheme book: the pick is drawn among candidates within `MIX_BAND` (0.015) of the best expected loss, with weight `exp(-Δ/0.006)`, and never repeats the previous call of the session. The Lions path stays a pure argmin.
- **Fail closed.** Any error gives `NO_LEGAL_RAVENS_CALL` with its reason, and the previous defensive caller never runs.
- **Logging.** `ravens_defense_decisions` gains `league` and `opponent_team`. HTML snaps carry the standing execution attestation for a displayed CALL, covering formation, play, and the shown macro.

See `docs/madden_ml/CIVIL_SCHEME_ONLY.md` § Online league.
