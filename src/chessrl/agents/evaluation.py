"""Hand-crafted static evaluation (material + piece-square tables).

Tables are from the "Simplified Evaluation Function" (Tomasz Michniewski), written from
White's point of view with rank 8 on the first row.
"""

from __future__ import annotations

from ..engine.board import Board
from ..engine.constants import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, file_of, rank_of

PIECE_VALUES = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 0}

# fmt: off
_PST = {
    PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0],
    KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50],
    BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20],
    ROOK: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0],
    QUEEN: [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20],
    KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20],
}
_KING_END = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50]
# fmt on


def _by_square(table: list[int], white: bool) -> list[int]:
    """Re-index a rank-8-first table by our square numbering (a1 = 0)."""
    out = [0] * 64
    for sq in range(64):
        r = rank_of(sq)
        row = 7 - r if white else r
        out[sq] = table[row * 8 + file_of(sq)]
    return out


_W = {t: _by_square(tab, True) for t, tab in _PST.items()}
_B = {t: _by_square(tab, False) for t, tab in _PST.items()}
_W_KING_END, _B_KING_END = _by_square(_KING_END, True), _by_square(_KING_END, False)


def evaluate(board: Board) -> int:
    """Static score in centipawns from the side-to-move's point of view."""
    white = black = 0
    minors = 0
    queens = 0
    total_non_king = 0
    for p in board.squares:
        if not p:
            continue
        t = abs(p)
        if t != KING:
            total_non_king += 1
        if t == QUEEN:
            queens += 1
        elif t in (KNIGHT, BISHOP):
            minors += 1
    if total_non_king <= 3 and board.is_insufficient_material():
        return 0
    endgame = queens == 0 or (minors <= 2 and total_non_king <= 8)
    for sq, p in enumerate(board.squares):
        if not p:
            continue
        t = abs(p)
        if p > 0:
            pst = _W_KING_END if (t == KING and endgame) else _W[t]
            white += PIECE_VALUES[t] + pst[sq]
        else:
            pst = _B_KING_END if (t == KING and endgame) else _B[t]
            black += PIECE_VALUES[t] + pst[sq]
    score = white - black
    return score if board.turn == 0 else -score
