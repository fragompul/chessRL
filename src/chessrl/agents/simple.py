"""Trivial baseline agents."""

from __future__ import annotations

import random

from ..engine import Board, Move
from .base import SearchLimits
from .evaluation import PIECE_VALUES


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_move(self, board: Board, limits: SearchLimits | None = None) -> Move:
        return self.rng.choice(board.legal_moves())


class GreedyAgent:
    """One-ply material maximiser with random tie-breaking (avoids immediate mate-in-1 blunders)."""

    name = "greedy"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_move(self, board: Board, limits: SearchLimits | None = None) -> Move:
        us = board.turn
        best, best_val = [], -1e9
        for m in board.legal_moves():
            gain = abs(board.squares[m.to_sq]) and PIECE_VALUES[abs(board.squares[m.to_sq])]
            if m.promotion:
                gain += PIECE_VALUES[m.promotion] - PIECE_VALUES[1]
            board.push(m)
            if board.outcome() is not None and board.outcome().winner == us:
                gain += 10**6
            board.pop()
            if gain > best_val:
                best, best_val = [m], gain
            elif gain == best_val:
                best.append(m)
        return self.rng.choice(best)
