"""Optional Windows SAPI TTS behind --tts. Lazy import; no-op elsewhere."""

from __future__ import annotations

from typing import Any


class TipTTS:
    """Speak tip lines via Windows SAPI when available."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._voice: Any = None
        if enabled:
            self._voice = _try_sapi()

    @property
    def available(self) -> bool:
        return self._voice is not None

    def speak(self, text: str) -> None:
        if not self.enabled or not text:
            return
        if self._voice is None:
            self._voice = _try_sapi()
        if self._voice is None:
            return
        try:
            self._voice.Speak(str(text)[:200])
        except Exception:
            pass

    def speak_tips(self, tips: list[str]) -> None:
        if not tips:
            return
        self.speak(". ".join(tips[:3]))


def _try_sapi() -> Any | None:
    try:
        import win32com.client  # type: ignore

        return win32com.client.Dispatch("SAPI.SpVoice")
    except Exception:
        return None
