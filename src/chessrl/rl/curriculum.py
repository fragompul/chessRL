"""Random endgame positions for curriculum learning (RL-2).

Self-play from the opening gives almost no learning signal at first (random games rarely
end decisively). Starting from simple winning endgames (KQK, KRK, KPK...) gives the network
dense, learnable outcomes; the share of such starts is annealed away during training.
"""

from __future__ import annotations

import random

from ..engine import Board
from ..engine.constants import PIECE_LETTERS, WHITE, file_of, rank_of

# name -> (strong side pieces, weak side pieces), kings implicit
ENDGAMES: dict[str, tuple[str, str]] = {
    "KQK": ("Q", ""),
    "KRK": ("R", ""),
    "KPK": ("P", ""),
    "KBBK": ("BB", ""),
    "KRRK": ("RR", ""),
    "KQKR": ("Q", "R"),
    "KRKP": ("R", "P"),
    "KRPKR": ("RP", "R"),
    "KQPKQ": ("QP", "Q"),
}


def _kings_adjacent(a: int, b: int) -> bool:
    return abs(file_of(a) - file_of(b)) <= 1 and abs(rank_of(a) - rank_of(b)) <= 1


def random_endgame(kind: str, rng: random.Random | None = None, max_tries: int = 500) -> Board:
    """A legal, non-terminal position of the given endgame; the strong side is random."""
    rng = rng or random.Random()
    strong_pieces, weak_pieces = ENDGAMES[kind]
    for _ in range(max_tries):
        strong_white = rng.random() < 0.5
        free = list(range(64))
        rng.shuffle(free)
        pieces = [("K", strong_white), ("K", not strong_white)]
        pieces += [(c, strong_white) for c in strong_pieces]
        pieces += [(c, not strong_white) for c in weak_pieces]
        squares = {free.pop(): (c if white else c.lower()) for c, white in pieces}
        wk = next(s for s, c in squares.items() if c == "K")
        bk = next(s for s, c in squares.items() if c == "k")
        if _kings_adjacent(wk, bk):
            continue
        if any(c in "Pp" and rank_of(s) in (0, 7) for s, c in squares.items()):
            continue
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for file in range(8):
                c = squares.get(rank * 8 + file)
                if c is None:
                    empty += 1
                else:
                    row += (str(empty) if empty else "") + c
                    empty = 0
            rows.append(row + (str(empty) if empty else ""))
        turn = rng.choice("wb")
        board = Board(f"{'/'.join(rows)} {turn} - - 0 1")
        # The side that just "moved" must not leave its own king capturable.
        mover = 1 - board.turn
        if board.is_attacked(board.king_sq[mover], board.turn):
            continue
        if board.outcome() is not None:
            continue
        return board
    raise RuntimeError(f"could not generate a {kind} position")


def curriculum_start(kinds: list[str], rng: random.Random) -> Board:
    return random_endgame(rng.choice(kinds), rng)


__all__ = ["ENDGAMES", "PIECE_LETTERS", "WHITE", "curriculum_start", "random_endgame"]
