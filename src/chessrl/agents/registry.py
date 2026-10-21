"""Build agents from short names, so CLIs and UIs can say `--opponent ab3`."""

from __future__ import annotations

from .base import Agent
from .minimax import AlphaBetaAgent
from .simple import GreedyAgent, RandomAgent


def make_agent(spec: str, seed: int | None = None) -> Agent:
    """`random`, `greedy`, `abN` (alpha-beta of depth N). Extended by levels / checkpoints later."""
    if spec == "random":
        return RandomAgent(seed)
    if spec == "greedy":
        return GreedyAgent(seed)
    if spec.startswith("ab") and spec[2:].isdigit():
        return AlphaBetaAgent(depth=int(spec[2:]), seed=seed)
    raise ValueError(f"unknown agent spec: {spec!r}")


def list_opponents() -> list[dict]:
    """Opponents offered by the UIs. Rough Elo values are placeholders until calibrated (phase 5/6)."""
    return [
        {"id": "random", "label": "Random mover", "elo": None},
        {"id": "greedy", "label": "Greedy (captures)", "elo": None},
        {"id": "ab1", "label": "Alpha-beta, depth 1", "elo": None},
        {"id": "ab2", "label": "Alpha-beta, depth 2", "elo": None},
        {"id": "ab3", "label": "Alpha-beta, depth 3", "elo": None},
        {"id": "ab4", "label": "Alpha-beta, depth 4", "elo": None},
    ]
