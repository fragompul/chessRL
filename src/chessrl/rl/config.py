"""Training configuration (JSON-serialisable) and presets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from .mcts import SearchConfig
from .network import NetConfig
from .selfplay import SelfPlayConfig


@dataclass
class TrainConfig:
    name: str = "run"
    seed: int = 0
    device: str = "auto"
    net: NetConfig = field(default_factory=NetConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    # data
    games_per_iter: int = 32
    buffer_capacity: int = 300_000
    min_buffer: int = 2_000  # no gradient steps until the buffer holds this many positions
    # optimisation
    batch_size: int = 256
    replay_ratio: float = 4.0  # positions sampled per newly generated position
    min_steps_per_iter: int = 20
    max_steps_per_iter: int = 400
    lr_schedule: list[list[float]] = field(default_factory=lambda: [[0, 2e-3], [300, 1e-3], [1000, 3e-4]])
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    value_loss_weight: float = 1.0
    # endgame curriculum: probability of an endgame start decays linearly to `endgame_floor`
    endgame_prob_start: float = 0.0
    endgame_floor: float = 0.0
    endgame_decay_iters: int = 100
    # league (RL-4)
    league_prob_start: float = 0.0
    league_pool: int = 5
    # housekeeping
    snapshot_every: int = 10  # weights-only files used for levels, league and Elo ladders
    buffer_save_every: int = 5
    pgn_games_per_iter: int = 4
    eval_every: int = 10
    eval_games: int = 6
    eval_sims: int = 50
    eval_depth: int = 2
    eval_baselines: list[str] = field(default_factory=lambda: ["random", "greedy", "ab1"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return _build(cls, d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def lr_at(self, iteration: int) -> float:
        lr = self.lr_schedule[0][1]
        for start, value in self.lr_schedule:
            if iteration >= start:
                lr = value
        return lr

    def endgame_prob_at(self, iteration: int) -> float:
        if not self.selfplay.endgames:
            return 0.0
        t = min(1.0, iteration / max(1, self.endgame_decay_iters))
        return self.endgame_prob_start + (self.endgame_floor - self.endgame_prob_start) * t


def _build(cls, d: dict):
    """Recursively build nested dataclasses from a dict, ignoring unknown keys."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        sub = {"net": NetConfig, "selfplay": SelfPlayConfig, "search": SearchConfig}.get(f.name)
        kwargs[f.name] = _build(sub, d[f.name]) if sub and isinstance(d[f.name], dict) else d[f.name]
    return cls(**kwargs)


def apply_overrides(cfg: TrainConfig, overrides: list[str]) -> TrainConfig:
    """Apply `dotted.key=value` overrides (value parsed as JSON when possible)."""
    for item in overrides:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        obj = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            obj = getattr(obj, p)
        if not hasattr(obj, leaf):
            raise KeyError(f"unknown config key: {key}")
        setattr(obj, leaf, value)
    assert is_dataclass(cfg)
    return cfg


def preset(name: str) -> TrainConfig:
    """Ready-made configurations. Sizes assume: smoke = CI test, mac = M-series laptop,
    colab = a single cloud GPU."""
    if name == "smoke":
        return TrainConfig(
            name="smoke", net=NetConfig(channels=16, blocks=1, value_hidden=16),
            selfplay=SelfPlayConfig(parallel=4, search=SearchConfig(sims=6, leaves_per_round=2),
                                    max_plies=40, endgames=["KQK"], endgame_prob=0.5),
            games_per_iter=4, buffer_capacity=5_000, min_buffer=50, batch_size=32,
            min_steps_per_iter=2, max_steps_per_iter=4, snapshot_every=1, buffer_save_every=1,
            eval_every=1, eval_games=2, eval_sims=6, eval_depth=1, pgn_games_per_iter=1,
        )
    if name == "mac":
        return TrainConfig(
            name="mac", net=NetConfig(channels=64, blocks=6),
            selfplay=SelfPlayConfig(
                parallel=24, search=SearchConfig(sims=100, leaves_per_round=4),
                temp_moves=24, max_plies=250,
                endgames=["KQK", "KRK", "KPK", "KBBK", "KRRK", "KQKR", "KRKP", "KRPKR"],
                endgame_prob=0.6),
            games_per_iter=48, endgame_prob_start=0.6, endgame_floor=0.05, endgame_decay_iters=150,
            league_prob_start=0.0,
        )
    if name == "colab":
        return TrainConfig(
            name="colab", net=NetConfig(channels=128, blocks=10),
            selfplay=SelfPlayConfig(
                parallel=64, search=SearchConfig(sims=200, leaves_per_round=8),
                temp_moves=30, max_plies=300,
                endgames=["KQK", "KRK", "KPK", "KBBK", "KRRK", "KQKR", "KRKP", "KRPKR"],
                endgame_prob=0.5),
            games_per_iter=128, buffer_capacity=600_000, batch_size=512, endgame_prob_start=0.5,
            endgame_floor=0.03, endgame_decay_iters=200, snapshot_every=10, eval_every=20,
        )
    raise KeyError(f"unknown preset {name!r}")
