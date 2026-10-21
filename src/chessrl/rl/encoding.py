"""Board -> network input.

Everything is expressed from the point of view of the side to move: when Black is to move
the board is mirrored vertically and the colours are swapped, so the network always sees
"my pieces" moving up the board. That halves what it has to learn.

Positions are stored compactly (`CompactState`: 64 int8 pieces + 7 int16 scalars, ~80 bytes)
and expanded to float planes only when a batch is built.
"""

from __future__ import annotations

import numpy as np

from ..engine import Board
from ..engine.constants import (
    BLACK,
    CASTLE_BK,
    CASTLE_BQ,
    CASTLE_WK,
    CASTLE_WQ,
    WHITE,
)

NUM_PLANES = 20
# Plane layout
#  0-5   own P N B R Q K        6-11  opponent P N B R Q K
#  12-13 own castling K / Q     14-15 opponent castling K / Q
#  16    en passant target      17    half-move clock / 100
#  18    position already seen  19    constant ones
META_LEN = 7  # castle(own K, own Q, opp K, opp Q), ep square (-1 = none), halfmove, repeated


def mirror(sq: int) -> int:
    """Flip a square vertically (a1 <-> a8)."""
    return sq ^ 56


def encode_compact(board: Board) -> tuple[np.ndarray, np.ndarray]:
    """Return (pieces int8[64], meta int16[7]) from the side to move's perspective."""
    sqs = np.asarray(board.squares, dtype=np.int8)
    us = board.turn
    if us == WHITE:
        rel = sqs
        own_k, own_q, opp_k, opp_q = CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ
        ep = board.ep_square if board.ep_square is not None else -1
    else:
        rel = -sqs.reshape(8, 8)[::-1].reshape(64)  # mirror ranks and swap colours
        own_k, own_q, opp_k, opp_q = CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ
        ep = mirror(board.ep_square) if board.ep_square is not None else -1
    c = board.castling
    meta = np.array(
        [
            bool(c & own_k), bool(c & own_q), bool(c & opp_k), bool(c & opp_q),
            ep, board.halfmove, board.repetition_count() >= 2,
        ],
        dtype=np.int16,
    )
    return np.ascontiguousarray(rel), meta


def planes_from_compact(pieces: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Expand a batch: pieces [N,64] int8, meta [N,7] int16 -> float32 [N, 20, 8, 8]."""
    pieces = np.atleast_2d(pieces)
    meta = np.atleast_2d(meta)
    n = pieces.shape[0]
    out = np.zeros((n, NUM_PLANES, 64), dtype=np.float32)
    for t in range(1, 7):
        out[:, t - 1] = pieces == t
        out[:, 5 + t] = pieces == -t
    for i in range(4):
        out[:, 12 + i] = meta[:, i, None]
    ep = meta[:, 4]
    has_ep = ep >= 0
    out[np.nonzero(has_ep)[0], 16, ep[has_ep]] = 1.0
    out[:, 17] = (meta[:, 5] / 100.0)[:, None]
    out[:, 18] = meta[:, 6, None]
    out[:, 19] = 1.0
    return out.reshape(n, NUM_PLANES, 8, 8)


def encode_board(board: Board) -> np.ndarray:
    """Convenience: one board -> float32 [20, 8, 8]."""
    pieces, meta = encode_compact(board)
    return planes_from_compact(pieces[None], meta[None])[0]


__all__ = ["BLACK", "META_LEN", "NUM_PLANES", "encode_board", "encode_compact", "mirror",
           "planes_from_compact"]
