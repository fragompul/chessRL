"""Gymnasium-style chess environment.

Observation: float32 [20, 8, 8] planes from the side-to-move's perspective.
Action: int in [0, 4672) (see `actions.py`); `info["legal_mask"]` marks the legal ones.
Reward: +1 for the player who just delivered mate, -1 for the player who just got mated
(only possible via timeout / illegal action), 0 otherwise.

Two modes:
  * self-play (`opponent=None`): every `step` is one ply; whoever is to move acts.
  * vs. an opponent agent: the env plays the opponent's reply internally, and rewards are
    from the learner's point of view.
"""

from __future__ import annotations

import random

import numpy as np

from ..engine import STARTING_FEN, Board
from .actions import NUM_ACTIONS, index_to_move, legal_mask
from .curriculum import curriculum_start
from .encoding import NUM_PLANES, encode_board

try:  # optional dependency: act as a real gymnasium.Env when it is installed
    import gymnasium as gym
    from gymnasium import spaces

    _Base = gym.Env
except ImportError:  # pragma: no cover
    gym = spaces = None
    _Base = object


class ChessEnv(_Base):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent=None,
        *,
        agent_color: int | None = None,
        max_plies: int = 300,
        start_fen: str = STARTING_FEN,
        endgames: list[str] | None = None,
        illegal_action: str = "raise",
        seed: int | None = None,
    ):
        self.opponent = opponent
        self.agent_color = agent_color
        self.max_plies = max_plies
        self.start_fen = start_fen
        self.endgames = endgames
        self.illegal_action = illegal_action
        self.rng = random.Random(seed)
        self.board = Board(start_fen)
        self.plies = 0
        if spaces is not None:
            self.observation_space = spaces.Box(0.0, 1.0, (NUM_PLANES, 8, 8), np.float32)
            self.action_space = spaces.Discrete(NUM_ACTIONS)

    # ----------------------------------------------------------------- api
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self.board = (
            curriculum_start(self.endgames, self.rng) if self.endgames else Board(self.start_fen)
        )
        self.plies = 0
        if self.agent_color is None and self.opponent is not None:
            self._learner = self.rng.choice([0, 1])
        else:
            self._learner = self.agent_color if self.agent_color is not None else 0
        if self.opponent is not None and self.board.turn != self._learner:
            self._opponent_move()
        return self._obs(), self._info()

    def step(self, action: int):
        move = index_to_move(int(action), self.board)
        if move is None or move not in self.board.legal_moves():
            if self.illegal_action == "raise":
                raise ValueError(f"illegal action {action} in {self.board.fen()}")
            return self._obs(), -1.0, True, False, {**self._info(), "illegal": True}
        mover = self.board.turn
        reward, terminated = self._apply(move, mover)
        if not terminated and self.opponent is not None:
            r2, terminated = self._opponent_move()
            reward -= r2  # the opponent's reward is the learner's loss
        truncated = not terminated and self.plies >= self.max_plies
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        return str(self.board)

    # ------------------------------------------------------------ internals
    def _apply(self, move, mover: int) -> tuple[float, bool]:
        self.board.push(move)
        self.plies += 1
        outcome = self.board.outcome()
        if outcome is None:
            return 0.0, False
        return (0.0 if outcome.winner is None else (1.0 if outcome.winner == mover else -1.0)), True

    def _opponent_move(self) -> tuple[float, bool]:
        mover = self.board.turn
        move = self.opponent.select_move(self.board.copy())
        return self._apply(move, mover)

    def _obs(self) -> np.ndarray:
        return encode_board(self.board)

    def _info(self) -> dict:
        return {"legal_mask": legal_mask(self.board), "to_play": self.board.turn, "plies": self.plies}
