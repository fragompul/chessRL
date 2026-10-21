"""Headless games between two agents (the building block of arenas and Elo estimation)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..agents.base import Agent, SearchLimits
from ..engine import STARTING_FEN, WHITE, Board, Game

LimitsFn = Callable[[Board], SearchLimits | None]


@dataclass
class GameResult:
    game: Game
    result: str  # "1-0", "0-1", "1/2-1/2"
    termination: str
    plies: int

    def score_for(self, white: bool) -> float:
        if self.result == "1/2-1/2":
            return 0.5
        return 1.0 if (self.result == "1-0") == white else 0.0


def play_game(
    white: Agent,
    black: Agent,
    *,
    start_fen: str = STARTING_FEN,
    max_plies: int = 400,
    limits: LimitsFn | None = None,
    headers: dict[str, str] | None = None,
) -> GameResult:
    """Play one game to completion. Reaching `max_plies` is adjudicated as a draw."""
    board = Board(start_fen)
    game = Game(headers={"White": white.name, "Black": black.name, **(headers or {})},
                start_fen=start_fen)
    termination = "move limit"
    result = "1/2-1/2"
    while True:
        outcome = board.outcome()
        if outcome is not None:
            result, termination = outcome.result, outcome.termination.value
            break
        if len(game.moves) >= max_plies:
            break
        agent = white if board.turn == WHITE else black
        move = agent.select_move(board.copy(), limits(board) if limits else None)
        if move not in board.legal_moves():
            raise RuntimeError(f"{agent.name} produced an illegal move {move} in {board.fen()}")
        game.moves.append(move)
        board.push(move)
    game.result = result
    game.headers["Termination"] = termination
    return GameResult(game, result, termination, len(game.moves))
