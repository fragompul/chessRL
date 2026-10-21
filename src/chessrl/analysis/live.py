"""Live annotation while a game is being played (runs on a background worker)."""

from __future__ import annotations

from ..agents.base import SearchLimits
from ..agents.minimax import AlphaBetaAgent
from ..engine import Board, Move
from .annotate import MoveAnalysis, analyse_position


class LiveAnnotator:
    """Analyses each move as it is played. Not thread-safe: the session feeds it from a
    single worker thread. Uses its own search engine so it never disturbs the opponent's."""

    def __init__(self, scorer=None, depth: int = 3, seconds_per_move: float = 1.5):
        self.scorer = scorer or AlphaBetaAgent(depth=depth, seed=0, name="annotator")
        self.limits = SearchLimits(time=seconds_per_move)

    def annotate(self, board_before: Board, move: Move, ply: int) -> MoveAnalysis:
        from ..levels.book import default_book

        book = default_book()
        after = board_before.copy()
        after.push(move)
        hit = book.name_at(after) if ply < 30 else None
        return analyse_position(self.scorer, board_before, move, self.limits, ply,
                                in_book=hit[1] if hit else None)
