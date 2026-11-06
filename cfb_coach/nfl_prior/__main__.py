"""``python -m cfb_coach.nfl_prior --out DIR`` — offline NFL prior comparison."""

from __future__ import annotations

import argparse
import sys

from cfb_coach.nfl_prior import data as data_mod
from cfb_coach.nfl_prior import evaluate


def _seasons(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",") if s]


def main(argv: list[str] | None = None) -> int:
    cfg = evaluate.default_config()
    p = argparse.ArgumentParser(prog="python -m cfb_coach.nfl_prior", description=__doc__)
    p.add_argument("--out", required=True, help="Run directory root (must be outside ~/.cfb-coach)")
    p.add_argument("--cache", default=str(data_mod.DEFAULT_CACHE))
    p.add_argument("--train", default=f"{cfg['train_seasons'][0]}-{cfg['train_seasons'][-1]}")
    p.add_argument("--val", type=int, default=cfg["val_season"])
    p.add_argument("--test", default=",".join(map(str, cfg["test_seasons"])))
    p.add_argument("--seed", type=int, default=cfg["seed"])
    p.add_argument("--threads", type=int, default=cfg["threads"])
    p.add_argument("--quick", action="store_true", help="Tiny models for a smoke run")
    args = p.parse_args(argv)
    cfg.update(train_seasons=_seasons(args.train), val_season=args.val, test_seasons=_seasons(args.test),
               seed=args.seed, threads=args.threads, quick=args.quick)
    try:
        run_dir = evaluate.run(cfg, args.out, cache_dir=args.cache)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"Wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
