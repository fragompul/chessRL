"""Build agents from short names, so CLIs and UIs can say `--opponent ab3`."""

from __future__ import annotations

from .base import Agent
from .minimax import AlphaBetaAgent
from .simple import GreedyAgent, RandomAgent


def make_agent(spec: str, seed: int | None = None) -> Agent:
    """`random`, `greedy`, `abN` (alpha-beta depth N), `ckpt:<path>[@sims]` (trained net + MCTS),
    `sf[:elo]` (Stockfish), `level:<name|elo>` (humanised level)."""
    if spec == "random":
        return RandomAgent(seed)
    if spec == "greedy":
        return GreedyAgent(seed)
    if spec.startswith("ab") and spec[2:].isdigit():
        return AlphaBetaAgent(depth=int(spec[2:]), seed=seed)
    if spec.startswith("ckpt:"):
        path, _, sims = spec[5:].partition("@")
        from .neural import MCTSAgent

        agent = MCTSAgent.from_checkpoint(path, sims=int(sims or 100), seed=seed)
        agent.name = f"net:{path.rsplit('/', 1)[-1]}@{agent.sims}"
        return agent
    if spec.startswith("level:"):
        from ..levels import find_level, make_level_agent

        return make_level_agent(find_level(spec[6:]), seed=seed)
    if spec.startswith("sf"):
        from ..eval.uci import stockfish_agent

        _, _, elo = spec.partition(":")
        return stockfish_agent(int(elo) if elo else None)
    raise ValueError(f"unknown agent spec: {spec!r}")


def list_opponents() -> list[dict]:
    """Opponents offered by the UIs: humanised levels first, then the raw baseline engines."""
    from ..levels.levels import default_levels

    levels = [
        {"id": f"level:{lv.name}", "label": f"{lv.name} (~{int(lv.display_elo)})", "elo": lv.display_elo,
         "group": "Levels"}
        for lv in default_levels()
    ]
    return levels + [
        {"id": "random", "label": "Random mover", "elo": None},
        {"id": "greedy", "label": "Greedy (captures)", "elo": None},
        {"id": "ab1", "label": "Alpha-beta, depth 1", "elo": None},
        {"id": "ab2", "label": "Alpha-beta, depth 2", "elo": None},
        {"id": "ab3", "label": "Alpha-beta, depth 3", "elo": None},
        {"id": "ab4", "label": "Alpha-beta, depth 4", "elo": None},
    ]
