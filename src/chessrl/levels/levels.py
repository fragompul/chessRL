"""Difficulty levels: named Elo targets mapped to concrete agent settings.

`LevelSpec`s start from nominal Elos; `scripts/calibrate_levels.py` plays every level against
Stockfish and stores the *measured* Elo in `levels.json`, which is what the UIs then display.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..agents.minimax import AlphaBetaAgent
from .humanize import HumanLikeAgent, blunder_rate_for_elo, book_plies_for_elo, temperature_for_elo

LEVELS_FILE = Path(__file__).with_name("levels.json")

NAMED = [
    ("Beginner", 600), ("Casual", 800), ("Novice", 1000), ("Club", 1200), ("Intermediate", 1400),
    ("Advanced", 1600), ("Expert", 1800), ("Candidate Master", 2000), ("Master", 2200),
]


@dataclass
class LevelSpec:
    name: str
    elo: float  # nominal target
    engine: str = "ab"  # "ab" (alpha-beta) | "net" (trained network + MCTS)
    depth: int = 2
    sims: int = 100
    checkpoint: str | None = None
    temperature: float = 100.0
    blunder_rate: float = 0.05
    book_plies: int = 10
    think_speed: float = 1.0
    style: str = "balanced"
    error_scale: float = 1.0  # multiplies temperature and blunder rate (set by tune.py)
    acpl: float | None = None  # average centipawn loss achieved at tuning time
    measured_elo: float | None = None  # filled in by calibration

    @property
    def display_elo(self) -> float:
        return self.measured_elo if self.measured_elo is not None else self.elo


def _depth_for(elo: float) -> int:
    return 1 if elo < 950 else 2 if elo < 1450 else 3 if elo < 2100 else 4


def spec_for_elo(elo: float, name: str | None = None, **overrides) -> LevelSpec:
    spec = LevelSpec(
        name=name or f"Elo {int(elo)}", elo=elo, depth=_depth_for(elo),
        temperature=temperature_for_elo(elo), blunder_rate=blunder_rate_for_elo(elo),
        book_plies=book_plies_for_elo(elo),
    )
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec


def default_levels() -> list[LevelSpec]:
    levels = [spec_for_elo(elo, name) for name, elo in NAMED]
    if LEVELS_FILE.exists():
        stored = {d["name"]: d for d in json.loads(LEVELS_FILE.read_text())}
        for lv in levels:
            if lv.name in stored:
                for k, v in stored[lv.name].items():
                    setattr(lv, k, v)
    return levels


def save_levels(levels: list[LevelSpec], path: str | Path = LEVELS_FILE) -> None:
    Path(path).write_text(json.dumps([asdict(lv) for lv in levels], indent=2))


def find_level(query: str | float) -> LevelSpec:
    """A named level ("Club"), or any Elo number ("1350", 1350) -> an interpolated spec."""
    levels = default_levels()
    if isinstance(query, str):
        for lv in levels:
            if lv.name.lower() == query.lower():
                return lv
        query = float(query)
    exact = [lv for lv in levels if abs(lv.elo - query) < 1e-6]
    return exact[0] if exact else spec_for_elo(float(query))


def make_level_agent(spec: LevelSpec, seed: int | None = None) -> HumanLikeAgent:
    if spec.engine == "net":
        from ..agents.neural import MCTSAgent

        base = MCTSAgent.from_checkpoint(spec.checkpoint, sims=spec.sims, seed=seed)
    else:
        base = AlphaBetaAgent(depth=spec.depth, seed=seed)
    return HumanLikeAgent(
        base, spec.display_elo, name=f"{spec.name} ({int(spec.display_elo)})",
        temperature=spec.temperature * spec.error_scale,
        blunder_rate=min(0.9, spec.blunder_rate * spec.error_scale), book_plies=spec.book_plies,
        style=spec.style, think_speed=spec.think_speed, seed=seed,
    )


class AdaptiveElo:
    """Keeps the opponent near the player's level: raise after wins, lower after losses (LVL-5)."""

    def __init__(self, start: float = 1200, k: float = 60, lo: float = 500, hi: float = 2400):
        self.elo, self.k, self.lo, self.hi = start, k, lo, hi

    def update(self, player_score: float) -> float:
        """`player_score` in {0, 0.5, 1}; a win for the player raises the next opponent's Elo."""
        self.elo = min(self.hi, max(self.lo, self.elo + self.k * (player_score - 0.5) * 2))
        return self.elo
