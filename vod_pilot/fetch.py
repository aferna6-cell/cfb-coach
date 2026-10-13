"""Attempt a public YouTube download with yt-dlp.

This environment's YouTube edge answers LOGIN_REQUIRED / "Sign in to confirm
you're not a bot" for watch pages and Innertube clients. This module reports
that failure. It does not try alternate front-ends or challenge solvers.
Clips for the pilot were obtained separately and are not part of this package.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        print("yt-dlp is not on PATH. Install it, then retry. YouTube still blocks this environment.", file=sys.stderr)
        return 2
    url = argv[0] if argv else "https://www.youtube.com/watch?v=z5ne-uBM0YI"
    cmd = [
        yt_dlp,
        "--no-playlist",
        "--skip-download",
        "--print",
        "%(id)s %(title)s",
        url,
    ]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(
            "YouTube refused the request from this environment. "
            "The pilot did not ship a workaround. Pass a local file to `python -m vod_pilot run`.",
            file=sys.stderr,
        )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
