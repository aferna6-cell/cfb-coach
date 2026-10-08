"""Trained VOD model as a call-quality prior (Madden 27 and CFB 27).

Defaults (override with env):

* ``CFB_COACH_VOD_QUALITY_WEIGHT`` = 1.0 — VOD shrunk success is the primary
  signal for how good a call is against the look we expect.
* ``CFB_COACH_VOD_LOG_NUDGE`` = 0.25 — Aidan's own logged games. This can
  break a near-tie (two VOD rates within one nudge of each other). It does
  not override a wider VOD gap.
* Logged tendencies (for example James's Cover 3 rate) pick *which* defensive
  look to expect, and therefore which VOD beater applies. They do not outrank
  the VOD success rate of a call against that look.

Sample bar: n < 8 is silent. n >= 8 but still tentative or short of the med
bar may be mentioned and must not change the called play or the book. Med
(n >= 15, 3+ VODs, lower bound above the base rate, model not tentative) and
high (n >= 30, 5+ VODs, lower bound above the base rate by 0.05) may change
the call and the book. Weight grows with that tier (med 0.65, high 1.0).

``CFB_COACH_VOD_MODELS`` is the directory that contains ``LAST_TRAINED.json``
and ``vX.Y/``. Nothing here hardcodes a machine path. Missing or corrupt
output is a no-op.

``CFB_COACH_NO_VOD_PRIOR=1`` or ``--no-vod-prior`` turns the prior off (no
call changes, no book changes). ``CFB_COACH_VOD_FREEZE_BOOK=1`` or
``--freeze-vod-book`` still allows in-book VOD calls and blocks book edits.
"""

from cfb_coach.vod_model.flags import book_frozen, prior_enabled
from cfb_coach.vod_model.loader import load_model
from cfb_coach.vod_model.prior import LOG_NUDGE_DEFAULT, VOD_QUALITY_WEIGHT_DEFAULT

__all__ = [
    "LOG_NUDGE_DEFAULT",
    "VOD_QUALITY_WEIGHT_DEFAULT",
    "book_frozen",
    "load_model",
    "prior_enabled",
]
