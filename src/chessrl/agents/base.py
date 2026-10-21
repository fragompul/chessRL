"""Agent interface shared by every player (baselines, neural agents, humans, engines)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..engine import Board, Move


@dataclass(frozen=True)
class SearchLimits:
    """Optional resource limits handed to an agent for a single move."""

    time: float | None = None  # seconds available for this move
    nodes: int | None = None
    depth: int | None = None


class Agent(Protocol):
    name: str

    def select_move(self, board: Board, limits: SearchLimits | None = None) -> Move: ...


class Scorer(Protocol):
    """An agent that can also rank every legal move (needed for humanised play).

    Scores are from the side-to-move's point of view, in centipawn-like units.
    """

    def score_moves(
        self, board: Board, limits: SearchLimits | None = None
    ) -> list[tuple[Move, float]]: ...
