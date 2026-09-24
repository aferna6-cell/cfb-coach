"""WSL-aware URL open helpers (live HTML play + shared with prep)."""

from __future__ import annotations

import unittest
from unittest import mock

from cfb_coach import browser_open


class TestOpenUrl(unittest.TestCase):
    def test_wsl_prefers_wslview_skips_webbrowser(self) -> None:
        with (
            mock.patch.object(browser_open, "is_wsl", return_value=True),
            mock.patch.object(browser_open, "try_cmd", side_effect=[True]) as try_cmd,
            mock.patch.object(browser_open.webbrowser, "open") as wb_open,
        ):
            browser_open.open_url("http://127.0.0.1:8765/")
        try_cmd.assert_called_once_with(["wslview", "http://127.0.0.1:8765/"])
        wb_open.assert_not_called()

    def test_wsl_falls_back_to_cmd_start(self) -> None:
        calls = [
            False,  # wslview
            True,  # cmd.exe start
        ]

        def _try(argv: list[str]) -> bool:
            return calls.pop(0)

        with (
            mock.patch.object(browser_open, "is_wsl", return_value=True),
            mock.patch.object(browser_open, "try_cmd", side_effect=_try) as try_cmd,
            mock.patch.object(browser_open.webbrowser, "open") as wb_open,
        ):
            browser_open.open_url("http://127.0.0.1:8765/")
        self.assertEqual(
            try_cmd.call_args_list[1].args[0],
            ["cmd.exe", "/c", "start", "", "http://127.0.0.1:8765/"],
        )
        wb_open.assert_not_called()

    def test_wsl_prints_manual_url_when_all_fail(self) -> None:
        with (
            mock.patch.object(browser_open, "is_wsl", return_value=True),
            mock.patch.object(browser_open, "try_cmd", return_value=False),
            mock.patch("builtins.print") as pr,
        ):
            browser_open.open_url("http://127.0.0.1:8765/")
        pr.assert_called_once_with("Open in Windows browser: http://127.0.0.1:8765/")

    def test_non_wsl_uses_webbrowser(self) -> None:
        with (
            mock.patch.object(browser_open, "is_wsl", return_value=False),
            mock.patch.object(browser_open.webbrowser, "open", return_value=True) as wb_open,
            mock.patch.object(browser_open, "try_cmd") as try_cmd,
        ):
            browser_open.open_url("http://127.0.0.1:8765/")
        wb_open.assert_called_once_with("http://127.0.0.1:8765/")
        try_cmd.assert_not_called()


if __name__ == "__main__":
    unittest.main()
