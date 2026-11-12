"""Lions ``civil_scheme_only`` offensive coordinator (26 approved scheme variants).

Separate, explicitly selectable mode. When it is OFF (the default) nothing in this
package runs and the existing unrestricted offensive coordinator is unchanged.

Modules
-------
catalog      load + validate the 26-variant dataset (datasets/madden27/...)
adjustments  parse each variant's presnap adjustments into a structured, auditable form
design       base-play geometry + adjusted-design representation and route features
beliefs      opponent coverage / pressure beliefs (history prior + same-game, recency weighted)
features     the model's input vector (situation x belief x adjusted design x prior knowledge)
learners     stdlib learners: offset logistic, gradient-boosted trees, NN gate
model        training, artifact, prediction, variant-level residual learning
legality     the ONE legal-candidate filter (before scoring) and re-validation (before display)
coordinator  live decision: score legal variants, pick, explain, or NO_LEGAL_SCHEME_PLAY
evidence     variant-level decision / execution logging and training-evidence assembly
evaluate     chronological, game-grouped offline evaluation and policy comparison
mode         enable / disable / resolve the mode (default OFF)
cli          ``python -m cfb_coach ml civil-scheme ...``
"""
