"""Move representation."""

from __future__ import annotations

from typing import NamedTuple

from .constants import LETTER_TO_TYPE, PIECE_LETTERS, parse_square, square_name


class Move(NamedTuple):
    """A move as (from, to, promotion piece type or 0). Castling is the king moving two files."""

    from_sq: int
    to_sq: int
    promotion: int = 0

    def uci(self) -> str:
        s = square_name(self.from_sq) + square_name(self.to_sq)
        return s + PIECE_LETTERS[self.promotion].lower() if self.promotion else s

    @classmethod
    def from_uci(cls, text: str) -> Move:
        if len(text) not in (4, 5):
            raise ValueError(f"invalid UCI move: {text!r}")
        promo = 0
        if len(text) == 5:
            promo = LETTER_TO_TYPE.get(text[4].upper(), 0)
            if not promo:
                raise ValueError(f"invalid promotion piece in {text!r}")
        return cls(parse_square(text[:2]), parse_square(text[2:4]), promo)

    def __str__(self) -> str:
        return self.uci()
