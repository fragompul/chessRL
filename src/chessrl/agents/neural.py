"""Agent that plays with a trained network + MCTS."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..engine import Board, Move
from .base import SearchLimits


def q_to_cp(q: float) -> float:
    """Map a value in (-1, 1) to centipawn-like units (Leela Chess Zero's formula)."""
    return 111.714640912 * math.tan(1.5620688421 * max(-0.999, min(0.999, q)))


@dataclass
class Analysis:
    moves: list[Move]
    visits: np.ndarray
    q: np.ndarray  # mean value of each move for the side to move
    prior: np.ndarray
    value: float  # network value of the root position (side to move)
    forced: bool


class MCTSAgent:
    def __init__(
        self,
        evaluator,
        sims: int = 100,
        temperature: float = 0.0,
        c_puct: float = 1.5,
        leaves_per_round: int = 8,
        name: str = "mcts",
        seed: int | None = None,
    ):
        self.evaluator = evaluator
        self.sims = sims
        self.temperature = temperature
        self.c_puct = c_puct
        self.leaves_per_round = leaves_per_round
        self.name = name
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto", **kw) -> MCTSAgent:
        from ..rl.checkpoint import load_net
        from ..rl.evaluator import NetEvaluator
        from ..rl.network import get_device

        dev = get_device(device)
        return cls(NetEvaluator(load_net(path, dev), dev), **kw)

    def analyse(self, board: Board, limits: SearchLimits | None = None) -> Analysis:
        from ..rl.mcts import Search, SearchConfig, run_search

        limits = limits or SearchLimits()
        sims = limits.nodes or self.sims
        cfg = SearchConfig(sims=sims, c_puct=self.c_puct, leaves_per_round=self.leaves_per_round,
                           virtual_loss=1.0)
        search = Search(board.copy(), cfg, self.rng)
        deadline = time.perf_counter() + limits.time if limits.time else None
        run_search(search, self.evaluator, sims, leaves_per_round=self.leaves_per_round,
                   deadline=deadline)
        root = search.root
        if not root.expanded or not root.moves:
            raise ValueError("no legal moves")
        return Analysis(root.moves, root.N.copy(), search.q_values(), root.P_raw.copy(), root.v,
                        forced=len(root.moves) == 1)

    def select_move(self, board: Board, limits: SearchLimits | None = None) -> Move:
        a = self.analyse(board, limits)
        if a.forced:
            return a.moves[0]
        if self.temperature <= 1e-3:
            return a.moves[int(np.argmax(a.visits))]
        w = a.visits.astype(np.float64) ** (1.0 / self.temperature)
        return a.moves[int(self.rng.choice(len(w), p=w / w.sum()))]

    def score_moves(self, board: Board, limits: SearchLimits | None = None) -> list[tuple[Move, float]]:
        """Moves ranked by search value, in centipawn-like units (side-to-move POV)."""
        a = self.analyse(board, limits)
        # Unvisited moves have no evidence: put them below every visited one, ordered by prior.
        scores = [
            (m, q_to_cp(float(q)) if v > 0 else -10_000.0 + float(p))
            for m, q, v, p in zip(a.moves, a.q, a.visits, a.prior)
        ]
        return sorted(scores, key=lambda x: -x[1])
