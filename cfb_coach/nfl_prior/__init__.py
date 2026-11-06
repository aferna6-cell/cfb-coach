"""Offline NFL research prior (play-by-play from nflreadpy).

Research only: nothing in this package is imported by the live Madden caller,
and no artifact it writes is read by live code. Heavy dependencies
(``pip install -e .[nfl-research]``) are imported lazily.
"""

PRIOR_VERSION = "nfl-prior.v1"
