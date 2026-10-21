"""Human-like play on top of any move-ranking agent.

A strong engine at reduced depth still plays "engine-like" (rarely blunders, then suddenly
collapses). People are different: mostly sensible moves, occasional oversights that depend on
how hard the position is and how much time is left. `HumanLikeAgent` reproduces that:

  1. out-of-book -> rank all moves with the base agent (alpha-beta or MCTS network)
  2. most of the time sample from a softmax over the scores (temperature rises as Elo falls)
  3. sometimes (rate depends on Elo, complexity and clock) play a plausible-but-bad move
  4. forced moves, book moves and mate-in-one are handled like a human would
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..agents.base import SearchLimits
from ..engine import Board, Move
from ..engine.constants import KING, PAWN, rank_of
from .book import OpeningBook, default_book
from .thinktime import ThinkTimeModel, complexity_from_scores

MATE_SCORE = 90_000
SCORE_CLIP = 1_500  # cp: scores beyond this are "winning" for sampling purposes


def interp(points: list[tuple[float, float]], x: float, log: bool = False) -> float:
    """Piecewise-linear interpolation (optionally in log space) clamped at the ends."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            return math.exp(math.log(y0) * (1 - t) + math.log(y1) * t) if log else y0 + (y1 - y0) * t
    return points[-1][1]


_TEMP = [(600, 400), (1000, 250), (1400, 140), (1800, 70), (2200, 35), (2600, 15)]
_BLUNDER = [(600, 0.20), (1000, 0.12), (1400, 0.07), (1800, 0.035), (2200, 0.012), (2600, 0.003)]
_BOOK = [(600, 2), (1000, 6), (1400, 10), (1800, 14), (2200, 20), (2600, 26)]


def temperature_for_elo(elo: float) -> float:
    return interp(_TEMP, elo, log=True)


def blunder_rate_for_elo(elo: float) -> float:
    return interp(_BLUNDER, elo, log=True)


def book_plies_for_elo(elo: float) -> int:
    return int(round(interp(_BOOK, elo)))


@dataclass
class Decision:
    complexity: float = 0.0
    forced: bool = False
    in_book: bool = False
    recapture: bool = False
    blundered: bool = False
    loss_cp: float = 0.0


def _style_bonus(board: Board, move: Move, style: str) -> float:
    if style == "balanced":
        return 0.0
    sqs = board.squares
    piece = abs(sqs[move.from_sq])
    capture = sqs[move.to_sq] != 0
    bonus = 0.0
    if style == "aggressive":
        bonus += 35 if capture else 0
        board.push(move)
        if board.is_check():
            bonus += 25
        board.pop()
        if piece == PAWN and rank_of(move.to_sq if board.turn == 0 else 7 - move.to_sq) >= 4:
            bonus += 10
    elif style == "solid":
        if piece == KING and abs(move.to_sq - move.from_sq) == 2:
            bonus += 30
        if piece == PAWN and rank_of(move.to_sq if board.turn == 0 else 7 - move.to_sq) >= 5:
            bonus -= 20
        if not capture:
            bonus += 8
    return bonus


class HumanLikeAgent:
    def __init__(
        self,
        base,
        elo: float,
        *,
        name: str | None = None,
        temperature: float | None = None,
        blunder_rate: float | None = None,
        book: OpeningBook | None | bool = True,
        book_plies: int | None = None,
        style: str = "balanced",
        think_speed: float = 1.0,
        seed: int | None = None,
    ):
        self.base = base
        self.elo = elo
        self.name = name or f"human~{int(elo)}"
        self.temperature = temperature if temperature is not None else temperature_for_elo(elo)
        self.blunder_rate = blunder_rate if blunder_rate is not None else blunder_rate_for_elo(elo)
        self.book = default_book() if book is True else (book or None)
        self.book_plies = book_plies if book_plies is not None else book_plies_for_elo(elo)
        self.style = style
        self.rng = random.Random(seed)
        self.think = ThinkTimeModel(elo, think_speed, seed=seed)
        self.last = Decision()

    # ----------------------------------------------------------------- move
    def select_move(self, board: Board, limits: SearchLimits | None = None) -> Move:
        legal = board.legal_moves()
        d = self.last = Decision()
        if len(legal) == 1:
            d.forced = True
            return legal[0]
        if self.book is not None:
            m = self.book.choose(board, self.rng, self.book_plies)
            if m is not None:
                d.in_book = True
                return m

        scores = self.base.score_moves(board, limits)
        if not scores:
            return self.rng.choice(legal)
        d.complexity = complexity_from_scores([min(max(s, -SCORE_CLIP), SCORE_CLIP) for _, s in scores])
        best_score = scores[0][1]

        # Humans nearly always spot a mate in one.
        mates = [m for m, s in scores if s >= MATE_SCORE - 2]
        if mates and self.rng.random() < (0.97 if self.elo >= 1000 else 0.6):
            return self.rng.choice(mates)

        move = self._maybe_blunder(board, scores, best_score, d)
        if move is None:
            move = self._sample(board, scores, best_score, d)
        d.recapture = self._is_recapture(board, move)
        return move

    def _clock_factor(self, limits: SearchLimits | None) -> float:
        if limits is not None and limits.time is not None and limits.time < 0.5:
            return 1.5
        return 1.0

    def _maybe_blunder(self, board, scores, best, d: Decision) -> Move | None:
        rate = self.blunder_rate * (0.6 + 1.2 * d.complexity)
        if self.rng.random() >= rate:
            return None
        # plausible mistakes: clearly worse than the best move, but not absurd
        cands = [(m, best - s) for m, s in scores if 120 <= best - s <= 900 and s > -MATE_SCORE / 2]
        if not cands:
            return None
        weights = [math.exp(-loss / 350.0) for _, loss in cands]
        m, loss = self.rng.choices(cands, weights)[0]
        d.blundered, d.loss_cp = True, loss
        return m

    def _sample(self, board, scores, best, d: Decision) -> Move:
        t = max(self.temperature, 1.0)
        vals = []
        for m, s in scores:
            s = max(min(s, SCORE_CLIP), -SCORE_CLIP)
            vals.append(s + _style_bonus(board, m, self.style))
        top = max(vals)
        weights = [math.exp((v - top) / t) for v in vals]
        i = self.rng.choices(range(len(scores)), weights)[0]
        d.loss_cp = max(0.0, best - scores[i][1])
        return scores[i][0]

    @staticmethod
    def _is_recapture(board: Board, move: Move) -> bool:
        if not board._undo:
            return False
        last_move, _piece, captured, cap_sq, *_ = board._undo[-1]
        return bool(captured) and move.to_sq == cap_sq and board.squares[move.to_sq] != 0

    # ------------------------------------------------------------ think time
    def think_seconds(self, board: Board, clock_left: float | None, increment: float = 0.0) -> float:
        d = self.last
        return self.think.sample(
            complexity=d.complexity, clock_left=clock_left, increment=increment,
            move_number=board.fullmove, forced=d.forced, in_book=d.in_book, recapture=d.recapture,
        )

    def score_moves(self, board: Board, limits: SearchLimits | None = None):
        return self.base.score_moves(board, limits)
