"""AlphaZero-style action space: 64 from-squares x 73 move types = 4672 actions.

Move types (always in the mover's perspective, "forward" = towards rank 8):
  0..55   queen-like moves: 8 directions x 7 distances (also queen promotions, and all pawn moves)
  56..63  knight moves
  64..72  under-promotions: 3 capture directions (left, straight, right) x (knight, bishop, rook)
Castling is the king moving two files, i.e. a distance-2 east/west queen-like move.
"""

from __future__ import annotations

import numpy as np

from ..engine import Board, Move
from ..engine.constants import BISHOP, KNIGHT, QUEEN, ROOK, file_of, rank_of, square

NUM_ACTIONS = 64 * 73

# direction index -> (dfile, drank); N, NE, E, SE, S, SW, W, NW
_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_DIR_INDEX = {d: i for i, d in enumerate(_DIRS)}
_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KNIGHT_INDEX = {d: i for i, d in enumerate(_KNIGHT_DELTAS)}
_UNDERPROMO = [KNIGHT, BISHOP, ROOK]


def _flip(sq: int) -> int:
    return sq ^ 56


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _plane(df: int, dr: int, promotion: int, is_pawn_promo: bool) -> int:
    if is_pawn_promo and promotion and promotion != QUEEN:
        return 64 + (df + 1) * 3 + _UNDERPROMO.index(promotion)
    if (df, dr) in _KNIGHT_INDEX:
        return 56 + _KNIGHT_INDEX[(df, dr)]
    dist = max(abs(df), abs(dr))
    return _DIR_INDEX[(_sign(df), _sign(dr))] * 7 + (dist - 1)


def move_to_index(move: Move, board: Board) -> int:
    """Index of `move` (a legal move of `board.turn`) in the 4672-action space."""
    frm, to = move.from_sq, move.to_sq
    if board.turn == 1:  # Black: mirror into the mover's perspective
        frm, to = _flip(frm), _flip(to)
    df, dr = file_of(to) - file_of(frm), rank_of(to) - rank_of(frm)
    return frm * 73 + _plane(df, dr, move.promotion, bool(move.promotion))


# Decode tables: (from_rel, plane) -> to_rel or -1
def _build_decode() -> tuple[np.ndarray, np.ndarray]:
    to_table = np.full((64, 73), -1, dtype=np.int16)
    promo_table = np.zeros((64, 73), dtype=np.int8)
    for frm in range(64):
        f, r = file_of(frm), rank_of(frm)
        for d, (dx, dy) in enumerate(_DIRS):
            for dist in range(1, 8):
                tf, tr = f + dx * dist, r + dy * dist
                if 0 <= tf < 8 and 0 <= tr < 8:
                    to_table[frm, d * 7 + dist - 1] = square(tf, tr)
        for k, (dx, dy) in enumerate(_KNIGHT_DELTAS):
            tf, tr = f + dx, r + dy
            if 0 <= tf < 8 and 0 <= tr < 8:
                to_table[frm, 56 + k] = square(tf, tr)
        if r == 6:  # under-promotions only from the 7th rank
            for dx in (-1, 0, 1):
                tf = f + dx
                if 0 <= tf < 8:
                    for i, piece in enumerate(_UNDERPROMO):
                        plane = 64 + (dx + 1) * 3 + i
                        to_table[frm, plane] = square(tf, 7)
                        promo_table[frm, plane] = piece
    return to_table, promo_table


_TO, _PROMO = _build_decode()


def index_to_move(index: int, board: Board) -> Move | None:
    """Decode an action index for `board`. Returns None if it names no on-board move.

    The result is not guaranteed legal; compare against `board.legal_moves()` for that.
    """
    frm, plane = divmod(index, 73)
    to = int(_TO[frm, plane])
    if to < 0:
        return None
    promo = int(_PROMO[frm, plane])
    if board.turn == 1:
        frm, to = _flip(frm), _flip(to)
    if not promo:
        piece = board.squares[frm]
        last = 7 if board.turn == 0 else 0
        if abs(piece) == 1 and rank_of(to) == last:
            promo = QUEEN
    return Move(frm, to, promo)


def legal_actions(board: Board) -> tuple[list[Move], np.ndarray]:
    """(legal moves, their action indices)."""
    moves = board.legal_moves()
    idx = np.fromiter((move_to_index(m, board) for m in moves), dtype=np.int64, count=len(moves))
    return moves, idx


def legal_mask(board: Board) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[legal_actions(board)[1]] = True
    return mask


__all__ = ["NUM_ACTIONS", "index_to_move", "legal_actions", "legal_mask", "move_to_index"]


def _build_flip_action() -> np.ndarray:
    """Permutation of action indices under a left-right mirror of the board (a-file <-> h-file)."""
    plane_flip = np.zeros(73, dtype=np.int64)
    for d, (dx, dy) in enumerate(_DIRS):
        for dist in range(7):
            plane_flip[d * 7 + dist] = _DIR_INDEX[(-dx, dy)] * 7 + dist
    for k, (dx, dy) in enumerate(_KNIGHT_DELTAS):
        plane_flip[56 + k] = 56 + _KNIGHT_INDEX[(-dx, dy)]
    for dx_idx in range(3):
        for i in range(3):
            plane_flip[64 + dx_idx * 3 + i] = 64 + (2 - dx_idx) * 3 + i
    frm = np.arange(NUM_ACTIONS) // 73
    plane = np.arange(NUM_ACTIONS) % 73
    return ((frm ^ 7) * 73 + plane_flip[plane]).astype(np.int64)


FLIP_ACTION = _build_flip_action()
