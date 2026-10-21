"""Alpha-beta (negamax) search agent: iterative deepening, quiescence, transposition table."""

from __future__ import annotations

import random
import time

from ..engine import Board, Move
from .base import SearchLimits
from .evaluation import PIECE_VALUES, evaluate

MATE = 100_000
INF = 10**9


class _Stop(Exception):
    pass


class AlphaBetaAgent:
    """Classical engine used as baseline opponent, Elo anchor and move-ranking oracle."""

    def __init__(self, depth: int = 3, seed: int | None = None, name: str | None = None,
                 quiescence: bool = True):
        self.depth = depth
        self.rng = random.Random(seed)
        self.name = name or f"alphabeta-d{depth}"
        self.quiescence = quiescence
        self.nodes = 0
        self._tt: dict[int, tuple[int, int, int, Move | None]] = {}
        self._deadline: float | None = None
        self._node_limit: int | None = None

    # ------------------------------------------------------------ public
    def select_move(self, board: Board, limits: SearchLimits | None = None) -> Move:
        scores = self.score_moves(board, limits)
        best = max(s for _, s in scores)
        return self.rng.choice([m for m, s in scores if s == best])

    def score_moves(
        self, board: Board, limits: SearchLimits | None = None
    ) -> list[tuple[Move, float]]:
        """Exact-ish score of every legal move (centipawns, side-to-move POV).

        Searches each root move with a full window, which is slower than a normal search but
        gives the complete ranking that humanised play needs.
        """
        limits = limits or SearchLimits()
        depth = limits.depth or self.depth
        self._set_budget(limits)
        moves = self._order(board, board.legal_moves(), None)
        if not moves:
            return []
        scores: list[tuple[Move, float]] = []
        result: dict[Move, int] = {}
        try:
            for d in range(1, depth + 1):
                this_depth: dict[Move, int] = {}
                for m in moves:
                    board.push(m)
                    try:
                        this_depth[m] = -self._negamax(board, d - 1, -INF, INF, 1)
                    finally:
                        board.pop()
                result = this_depth
                moves = sorted(moves, key=lambda x: -result[x])
        except _Stop:
            if not result:  # not even depth 1 finished: fall back to the static ordering
                result = {m: 0 for m in moves}
        finally:
            self._deadline = self._node_limit = None
        scores = [(m, float(result.get(m, -MATE))) for m in moves]
        return sorted(scores, key=lambda x: -x[1])

    def search_value(self, board: Board, depth: int | None = None) -> int:
        """Alpha-beta value of the position (side-to-move POV)."""
        self._set_budget(SearchLimits())
        return self._negamax(board, depth or self.depth, -INF, INF, 0)

    # ----------------------------------------------------------- internal
    def _set_budget(self, limits: SearchLimits) -> None:
        self.nodes = 0
        self._deadline = time.perf_counter() + limits.time if limits.time else None
        self._node_limit = limits.nodes

    def _check_budget(self) -> None:
        if self._node_limit is not None and self.nodes >= self._node_limit:
            raise _Stop
        if self._deadline is not None and self.nodes % 256 == 0 and time.perf_counter() > self._deadline:
            raise _Stop

    @staticmethod
    def _order(board: Board, moves: list[Move], tt_move: Move | None) -> list[Move]:
        sqs = board.squares

        def key(m: Move) -> int:
            if m == tt_move:
                return 10_000
            score = 0
            victim = sqs[m.to_sq]
            if victim:
                score += 10 * PIECE_VALUES[abs(victim)] - PIECE_VALUES[abs(sqs[m.from_sq])] + 1000
            elif abs(sqs[m.from_sq]) == 1 and m.to_sq == board.ep_square:
                score += 1000 + 900
            if m.promotion:
                score += PIECE_VALUES[m.promotion]
            return score

        return sorted(moves, key=key, reverse=True)

    def _negamax(self, board: Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._check_budget()
        if ply > 0 and (board.halfmove >= 100 or board.repetition_count() >= 2):
            return 0
        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply) if self.quiescence else evaluate(board)

        alpha0 = alpha
        entry = self._tt.get(board.hash)
        tt_move = None
        if entry is not None:
            e_depth, e_flag, e_score, tt_move = entry
            if e_depth >= depth and ply > 0:
                if e_flag == 0:
                    return e_score
                if e_flag == 1 and e_score >= beta:  # lower bound
                    return e_score
                if e_flag == 2 and e_score <= alpha:  # upper bound
                    return e_score

        moves = board.legal_moves()
        if not moves:
            return -MATE + ply if board.is_check() else 0
        best, best_move = -INF, None
        for m in self._order(board, moves, tt_move):
            board.push(m)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best, best_move = score, m
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        flag = 0
        if best <= alpha0:
            flag = 2
        elif best >= beta:
            flag = 1
        if len(self._tt) > 500_000:
            self._tt.clear()
        self._tt[board.hash] = (depth, flag, best, best_move)
        return best

    def _quiesce(self, board: Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._check_budget()
        in_check = board.is_check()
        if not in_check:
            stand = evaluate(board)
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
        else:
            stand = -MATE + ply
        moves = board.legal_moves()
        if not moves:
            return -MATE + ply if in_check else 0
        sqs = board.squares
        if not in_check:
            moves = [
                m for m in moves
                if sqs[m.to_sq] or m.promotion or (abs(sqs[m.from_sq]) == 1 and m.to_sq == board.ep_square)
            ]
        best = stand
        for m in self._order(board, moves, None):
            board.push(m)
            try:
                score = -self._quiesce(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
