"""Matches between agents: fair openings, alternating colours, PGN output, Elo estimates."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..agents.base import Agent, SearchLimits
from ..engine import STARTING_FEN, Board, Game
from ..play.match import GameResult, play_game
from .elo import MatchStats, estimate_ratings, sprt_status


def random_opening(plies: int, rng: random.Random) -> str:
    """FEN after `plies` random legal moves (kept non-terminal): a cheap source of variety."""
    for _ in range(50):
        b = Board()
        ok = True
        for _ in range(plies):
            moves = b.legal_moves()
            if not moves:
                ok = False
                break
            b.push(rng.choice(moves))
        if ok and b.outcome() is None:
            return b.fen()
    return STARTING_FEN


@dataclass
class MatchResult:
    a: str
    b: str
    stats: MatchStats
    games: list[GameResult] = field(default_factory=list)

    def pgn(self) -> str:
        return "\n".join(g.game.to_pgn() for g in self.games)

    def save_pgn(self, path: str | Path) -> None:
        Path(path).write_text(self.pgn())

    def __str__(self) -> str:
        return f"{self.a} vs {self.b}: {self.stats}"


def play_match(
    a: Agent,
    b: Agent,
    games: int = 10,
    *,
    opening_plies: int = 4,
    max_plies: int = 300,
    seed: int = 0,
    limits: Callable[[Board], SearchLimits | None] | None = None,
    on_game: Callable[[int, GameResult], None] | None = None,
    sprt: tuple[float, float] | None = None,
) -> MatchResult:
    """Play `games` games between `a` and `b`. Openings are played twice with colours swapped
    so that luck in the opening cancels out. `sprt=(elo0, elo1)` stops early when decided."""
    rng = random.Random(seed)
    results: list[GameResult] = []
    w = d = losses = 0
    fen = STARTING_FEN
    for i in range(games):
        if i % 2 == 0:
            fen = random_opening(opening_plies, rng) if opening_plies else STARTING_FEN
        a_white = i % 2 == 0
        white, black = (a, b) if a_white else (b, a)
        r = play_game(white, black, start_fen=fen, max_plies=max_plies, limits=limits,
                      headers={"Round": str(i + 1)})
        results.append(r)
        score = r.score_for(a_white)
        w, d, losses = w + (score == 1), d + (score == 0.5), losses + (score == 0)
        if on_game:
            on_game(i, r)
        if sprt and sprt_status(MatchStats(w, d, losses), *sprt) != "continue":
            break
    return MatchResult(a.name, b.name, MatchStats(w, d, losses), results)


@dataclass
class Tournament:
    """Round-robin results turned into ratings."""

    pairs: dict[tuple[str, str], MatchStats] = field(default_factory=dict)

    def add(self, m: MatchResult) -> None:
        self.pairs[(m.a, m.b)] = m.stats

    def ratings(self, anchors: dict[str, float] | None = None) -> dict[str, float]:
        results = []
        for (a, b), s in self.pairs.items():
            results += [(a, b, 1.0)] * s.wins + [(a, b, 0.5)] * s.draws + [(a, b, 0.0)] * s.losses
        return estimate_ratings(results, anchors)


def round_robin(
    agents: list[Agent],
    games_per_pair: int = 10,
    anchors: dict[str, float] | None = None,
    **match_kwargs,
) -> tuple[Tournament, dict[str, float]]:
    t = Tournament()
    for i, a in enumerate(agents):
        for b in agents[i + 1 :]:
            t.add(play_match(a, b, games_per_pair, **match_kwargs))
    return t, t.ratings(anchors)


def game_summary(g: Game) -> str:
    return f"{g.headers.get('White')} vs {g.headers.get('Black')}: {g.result}"
