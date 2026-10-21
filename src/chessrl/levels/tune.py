"""Tune human-likeness against a target *average centipawn loss* (ACPL), without playing games.

Strength in Elo is slow to measure (hundreds of games); ACPL is cheap: pick positions, rank the
moves once with the base engine and once with a deeper reference, then simulate what a level
would choose. The `error_scale` that makes the mean loss match the Elo's typical ACPL is stored
on the level; final Elo is then confirmed with games (scripts/calibrate_levels.py).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..agents.base import SearchLimits
from ..agents.minimax import AlphaBetaAgent
from ..engine import Board, Move
from .humanize import HumanLikeAgent, interp

# Typical average centipawn loss per move by rating (approximate, from public Lichess/engine
# statistics; used as a *target shape*, not a measured truth).
_ACPL = [(600, 150.0), (800, 125.0), (1000, 105.0), (1200, 88.0), (1400, 72.0), (1600, 58.0),
         (1800, 46.0), (2000, 36.0), (2200, 28.0), (2400, 22.0)]
LOSS_CAP = 500.0  # a single move never costs more than this in the average (as is customary)


def target_acpl(elo: float) -> float:
    return interp(_ACPL, elo)


@dataclass
class Position:
    fen: str
    ref: dict[str, float]  # uci -> reference score (cp, side to move)
    base: dict[int, dict[str, float]]  # depth -> uci -> base-engine score


def _clip(x: float) -> float:
    return max(-1500.0, min(1500.0, x))


def sample_positions(n: int, seed: int = 0, min_ply: int = 8, max_ply: int = 60) -> list[str]:
    rng = random.Random(seed)
    fens: list[str] = []
    while len(fens) < n:
        b = Board()
        for _ in range(rng.randint(min_ply, max_ply)):
            moves = b.legal_moves()
            if not moves or b.outcome() is not None:
                break
            # a mix of random and "sensible" (capture-biased) moves gives playable positions
            caps = [m for m in moves if b.squares[m.to_sq]]
            b.push(rng.choice(caps if caps and rng.random() < 0.3 else moves))
        if b.outcome() is None and len(b.legal_moves()) >= 6:
            fens.append(b.fen())
    return fens


def prepare(fens: list[str], depths: list[int], ref_depth: int = 3, seconds: float = 2.0,
            progress=None) -> list[Position]:
    ref = AlphaBetaAgent(depth=ref_depth, seed=0, name="ref")
    bases = {d: AlphaBetaAgent(depth=d, seed=0, name=f"base{d}") for d in depths}
    out = []
    for i, fen in enumerate(fens):
        b = Board(fen)
        lim = SearchLimits(time=seconds)
        out.append(Position(
            fen,
            {m.uci(): _clip(s) for m, s in ref.score_moves(b, lim)},
            {d: {m.uci(): _clip(s) for m, s in a.score_moves(b, lim)} for d, a in bases.items()},
        ))
        if progress:
            progress(i + 1, len(fens))
    return out


class _Cached:
    """A scorer that returns precomputed move scores for one position."""

    def __init__(self) -> None:
        self.scores: list[tuple[Move, float]] = []

    def score_moves(self, board: Board, limits=None):
        return self.scores


def acpl(positions: list[Position], depth: int, elo: float, error_scale: float, style: str = "balanced",
         repeats: int = 4, seed: int = 0) -> float:
    """Mean centipawn loss of a level (error parameters x `error_scale`) on the cached positions."""
    from .humanize import blunder_rate_for_elo, temperature_for_elo

    cache = _Cached()
    agent = HumanLikeAgent(cache, elo, book=False, style=style,
                           temperature=temperature_for_elo(elo) * error_scale,
                           blunder_rate=min(0.9, blunder_rate_for_elo(elo) * error_scale), seed=seed)
    total, n = 0.0, 0
    for pos in positions:
        board = Board(pos.fen)
        base = pos.base[depth]
        cache.scores = sorted(((board.parse_uci(u), s) for u, s in base.items()), key=lambda x: -x[1])
        best_ref = max(pos.ref.values())
        for _ in range(repeats):
            m = agent.select_move(board)
            total += min(LOSS_CAP, max(0.0, best_ref - pos.ref.get(m.uci(), min(pos.ref.values()))))
            n += 1
    return total / max(n, 1)


def fit_error_scale(positions: list[Position], depth: int, elo: float, style: str = "balanced",
                    lo: float = 0.02, hi: float = 6.0, iters: int = 14, seed: int = 0) -> tuple[float, float]:
    """Bisection (log scale) for the error scale that hits `target_acpl(elo)`.

    Returns (scale, achieved ACPL). If even the smallest scale is already too lossy (base engine
    too weak for this Elo) the minimum scale is returned.
    """
    target = target_acpl(elo)
    if acpl(positions, depth, elo, lo, style, seed=seed) >= target:
        return lo, acpl(positions, depth, elo, lo, style, seed=seed)
    if acpl(positions, depth, elo, hi, style, seed=seed) <= target:
        return hi, acpl(positions, depth, elo, hi, style, seed=seed)
    a, b = math.log(lo), math.log(hi)
    for _ in range(iters):
        mid = (a + b) / 2
        if acpl(positions, depth, elo, math.exp(mid), style, seed=seed) < target:
            a = mid
        else:
            b = mid
    scale = math.exp((a + b) / 2)
    return scale, acpl(positions, depth, elo, scale, style, seed=seed)
