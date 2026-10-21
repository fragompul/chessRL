"""Elo maths: expected score, rating differences with confidence intervals, rating estimation
from many pairwise results with fixed anchors, and SPRT.

Scale note: absolute Elo only means something relative to a reference. Ratings estimated here
are on the scale of whatever the anchors are (e.g. Stockfish `UCI_Elo` levels). Without an
external anchor they are an *internal* scale, comparable only within the same pool.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


def expected_score(rating_diff: float) -> float:
    """Expected score of a player who is `rating_diff` Elo above the opponent."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


def elo_diff(score: float, eps: float = 1e-3) -> float:
    """Rating difference implied by a score fraction in [0, 1] (clamped to avoid +-inf)."""
    s = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


@dataclass(frozen=True)
class MatchStats:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.5

    def elo(self) -> float:
        return elo_diff(self.score)

    def elo_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Approximate (default 95%) confidence interval on the Elo difference."""
        n = self.games
        if n == 0:
            return (-math.inf, math.inf)
        s = self.score
        var = (self.wins * (1 - s) ** 2 + self.draws * (0.5 - s) ** 2 + self.losses * s**2) / n
        se = math.sqrt(var / n)
        return (elo_diff(s - z * se), elo_diff(s + z * se))

    def __str__(self) -> str:
        lo, hi = self.elo_interval()
        return (f"+{self.wins} ={self.draws} -{self.losses}  score {self.score:.3f}  "
                f"Elo {self.elo():+.0f} [{lo:+.0f}, {hi:+.0f}]")


def estimate_ratings(
    results: Iterable[tuple[str, str, float]],
    anchors: dict[str, float] | None = None,
    iterations: int = 500,
    prior_games: float = 0.5,
) -> dict[str, float]:
    """Maximum-likelihood ratings from pairwise results.

    `results`: (player_a, player_b, score_a) with score in {0, 0.5, 1} (or an average over
    several games: weights are then given by repeating the tuple). `anchors` fixes some
    players' ratings; without any, the mean rating is pinned at 0.

    A weak prior (`prior_games` virtual draws against the pool average) keeps ratings finite
    when a player has won or lost every game.
    """
    results = list(results)
    players = sorted({p for a, b, _ in results for p in (a, b)} | set(anchors or {}))
    anchors = anchors or {}
    rating = {p: anchors.get(p, 0.0) for p in players}
    free = [p for p in players if p not in anchors]
    if not free:
        return rating
    games: dict[str, list[tuple[str, float]]] = {p: [] for p in players}
    for a, b, s in results:
        games[a].append((b, s))
        games[b].append((a, 1.0 - s))
    k = math.log(10.0) / 400.0
    for _ in range(iterations):
        biggest = 0.0
        for p in free:
            g = games[p]
            grad = sum(s - expected_score(rating[p] - rating[o]) for o, s in g)
            hess = sum(expected_score(rating[p] - rating[o]) * (1 - expected_score(rating[p] - rating[o]))
                       for o, _ in g)
            ref = sum(rating[o] for o, _ in g) / len(g) if g else 0.0
            grad += prior_games * (0.5 - expected_score(rating[p] - ref))
            hess += prior_games * 0.25
            if hess <= 0:
                continue
            step = max(-100.0, min(100.0, grad / (k * hess)))
            rating[p] += step
            biggest = max(biggest, abs(step))
        if not anchors:
            mean = sum(rating[p] for p in free) / len(free)
            for p in free:
                rating[p] -= mean
        if biggest < 1e-4:
            break
    return rating


# ------------------------------------------------------------------------------ SPRT
def sprt_llr(stats: MatchStats, elo0: float, elo1: float) -> float:
    """Log-likelihood ratio of H1 (rating gap elo1) over H0 (elo0), normal approximation."""
    n = stats.games
    if n == 0:
        return 0.0
    s = stats.score
    var = (stats.wins * (1 - s) ** 2 + stats.draws * (0.5 - s) ** 2 + stats.losses * s**2) / n
    if var <= 0:
        return 0.0
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return (s1 - s0) * (2 * s - s0 - s1) * n / (2 * var)


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple[float, float]:
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def sprt_status(stats: MatchStats, elo0: float, elo1: float, alpha: float = 0.05,
                beta: float = 0.05) -> str:
    """'H1' (improvement proven), 'H0' (no improvement) or 'continue'."""
    llr = sprt_llr(stats, elo0, elo1)
    lo, hi = sprt_bounds(alpha, beta)
    return "H1" if llr >= hi else "H0" if llr <= lo else "continue"
