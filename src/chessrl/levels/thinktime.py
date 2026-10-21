"""Human-like thinking time.

A person does not move in constant time: forced moves and book moves are near-instant, quiet
positions are quick, sharp ones take longer, and everyone budgets against the clock. This
model turns (Elo, position complexity, clock) into a plausible delay.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def complexity_from_scores(scores: list[float]) -> float:
    """0..1: how many near-equal candidate moves are there? (from side-to-move centipawn scores)

    Many reasonable moves within ~50cp of the best -> a hard decision; one clearly best move
    (or a lost/won position where everything else is terrible) -> an easy one.
    """
    if len(scores) <= 1:
        return 0.0
    top = sorted(scores, reverse=True)
    close = sum(1 for s in top if top[0] - s <= 50)
    spread = min(1.0, (close - 1) / 6.0)
    margin = min(1.0, (top[0] - top[1]) / 200.0) if len(top) > 1 else 1.0
    sharp = 1.0 - margin  # small margin between best and second best -> hard to pick
    return max(0.0, min(1.0, 0.6 * spread + 0.4 * sharp))


@dataclass
class ThinkTimeModel:
    """`speed` > 1 makes the player slower, < 1 faster (a personality knob)."""

    elo: float = 1500
    speed: float = 1.0
    min_time: float = 0.25
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def sample(
        self,
        *,
        complexity: float,
        clock_left: float | None,
        increment: float = 0.0,
        move_number: int = 1,
        forced: bool = False,
        in_book: bool = False,
        recapture: bool = False,
    ) -> float:
        rng = self.rng
        noise = math.exp(rng.gauss(0.0, 0.35))
        if forced:
            t = rng.uniform(0.3, 0.9)
        elif in_book:
            t = rng.uniform(0.4, 1.6) * (1.2 - min(self.elo, 2400) / 4000)
        else:
            if clock_left is None or clock_left > 1e8:
                base = 2.5
            else:
                # budget: spread the remaining time over the moves a game usually still has
                moves_left = max(18.0, 42.0 - move_number * 0.8)
                base = clock_left / moves_left + 0.7 * increment
            elo_factor = 0.75 + (min(max(self.elo, 600), 2600) - 1200) / 2800.0
            t = base * (0.25 + 1.6 * complexity) * elo_factor
            if recapture:
                t *= 0.4
        t *= noise * self.speed
        t = max(t, self.min_time)
        if clock_left is not None and clock_left < 1e8:
            t = min(t, 0.35 * clock_left)  # never burn the whole clock on one move
            if clock_left < 8:
                t = min(t, max(0.2, 0.12 * clock_left))
        return max(0.0, t)
