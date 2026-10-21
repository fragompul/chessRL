"""Board constants and precomputed lookup tables.

Squares are indexed 0..63 with a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
Pieces are signed ints: +type for white, -type for black, 0 for an empty square.
"""

from __future__ import annotations

import random

WHITE, BLACK = 0, 1

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(1, 7)

PIECE_LETTERS = {PAWN: "P", KNIGHT: "N", BISHOP: "B", ROOK: "R", QUEEN: "Q", KING: "K"}
LETTER_TO_TYPE = {v: k for k, v in PIECE_LETTERS.items()}

FILE_NAMES = "abcdefgh"
RANK_NAMES = "12345678"

# Named squares used by castling logic.
A1, B1, C1, D1, E1, F1, G1, H1 = range(0, 8)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)

# Castling rights bitmask.
CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def square(file: int, rank: int) -> int:
    return rank * 8 + file


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


def square_name(sq: int) -> str:
    return FILE_NAMES[sq & 7] + RANK_NAMES[sq >> 3]


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] not in FILE_NAMES or name[1] not in RANK_NAMES:
        raise ValueError(f"invalid square: {name!r}")
    return square(FILE_NAMES.index(name[0]), RANK_NAMES.index(name[1]))


def _build_ray(sq: int, df: int, dr: int) -> tuple[int, ...]:
    f, r = file_of(sq) + df, rank_of(sq) + dr
    out = []
    while 0 <= f < 8 and 0 <= r < 8:
        out.append(square(f, r))
        f, r = f + df, r + dr
    return tuple(out)


def _jumps(sq: int, deltas: list[tuple[int, int]]) -> tuple[int, ...]:
    f0, r0 = file_of(sq), rank_of(sq)
    return tuple(
        square(f0 + df, r0 + dr)
        for df, dr in deltas
        if 0 <= f0 + df < 8 and 0 <= r0 + dr < 8
    )


_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KING_DELTAS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

KNIGHT_TARGETS = tuple(_jumps(s, _KNIGHT_DELTAS) for s in range(64))
KING_TARGETS = tuple(_jumps(s, _KING_DELTAS) for s in range(64))

# PAWN_ATTACKS[color][sq]: squares attacked by a pawn of `color` standing on `sq`.
PAWN_ATTACKS = (
    tuple(_jumps(s, [(-1, 1), (1, 1)]) for s in range(64)),
    tuple(_jumps(s, [(-1, -1), (1, -1)]) for s in range(64)),
)

DIAG_RAYS = tuple(
    tuple(_build_ray(s, df, dr) for df, dr in [(1, 1), (-1, 1), (-1, -1), (1, -1)])
    for s in range(64)
)
ORTHO_RAYS = tuple(
    tuple(_build_ray(s, df, dr) for df, dr in [(1, 0), (0, 1), (-1, 0), (0, -1)])
    for s in range(64)
)

# Rights that SURVIVE a move touching a square: rights &= KEEP[from] & KEEP[to].
_keep = [0b1111] * 64
_keep[A1] &= ~CASTLE_WQ
_keep[H1] &= ~CASTLE_WK
_keep[E1] &= ~(CASTLE_WK | CASTLE_WQ)
_keep[A8] &= ~CASTLE_BQ
_keep[H8] &= ~CASTLE_BK
_keep[E8] &= ~(CASTLE_BK | CASTLE_BQ)
CASTLE_KEEP = tuple(_keep)

# Zobrist keys (fixed seed -> reproducible hashes across runs).
_rng = random.Random(0xC4E55)
# Indexed by piece + 6 (pieces run from -6 to +6).
ZOBRIST_PIECE = tuple(tuple(_rng.getrandbits(64) for _ in range(64)) for _ in range(13))
ZOBRIST_CASTLING = tuple(_rng.getrandbits(64) for _ in range(16))
ZOBRIST_TURN = _rng.getrandbits(64)
