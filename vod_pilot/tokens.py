"""OCR tokens. A bare string is a token with no geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Tok:
    text: str
    x: float | None = None
    y: float | None = None
    conf: float | None = None


Token = Union[str, Tok]


def token_text(item: Token) -> str:
    return item.text if isinstance(item, Tok) else str(item)


def as_tok(item: Token) -> Tok:
    if isinstance(item, Tok):
        return item
    return Tok(str(item))


def as_texts(items: list[Token]) -> list[str]:
    return [token_text(item) for item in items]
