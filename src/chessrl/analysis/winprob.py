"""Centipawns <-> win probability, and per-move accuracy (Lichess' published formulas)."""

from __future__ import annotations

import math


def win_percent(cp: float) -> float:
    """Win chance in [0, 100] for the side whose evaluation is `cp` centipawns."""
    cp = max(-1000.0, min(1000.0, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def move_accuracy(win_before: float, win_after: float) -> float:
    """Accuracy of one move from the mover's win% before / after it (100 = no loss)."""
    loss = max(0.0, win_before - win_after)
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * loss) - 3.1669))


def game_accuracy(accuracies: list[float]) -> float:
    """Aggregate: mean of the arithmetic and harmonic means (punishes isolated blunders)."""
    if not accuracies:
        return 100.0
    arith = sum(accuracies) / len(accuracies)
    harm = len(accuracies) / sum(1.0 / max(a, 1.0) for a in accuracies)
    return (arith + harm) / 2.0
