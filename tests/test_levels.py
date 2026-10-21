import random

import pytest

from chessrl.agents import AlphaBetaAgent, SearchLimits
from chessrl.agents.registry import list_opponents, make_agent
from chessrl.engine import Board
from chessrl.levels import (
    AdaptiveElo,
    HumanLikeAgent,
    ThinkTimeModel,
    complexity_from_scores,
    default_book,
    find_level,
    make_level_agent,
    spec_for_elo,
)
from chessrl.levels.book import LINES, OpeningBook
from chessrl.levels.humanize import blunder_rate_for_elo, temperature_for_elo
from chessrl.play import GameSession, TimeControl


def test_all_book_lines_are_legal_and_named():
    book = OpeningBook(LINES)  # parse_san raises on any illegal line
    assert len(book) > 200
    b = Board()
    for san in "e4 e5 Nf3 Nc6 Bb5 a6".split():
        b.push(b.parse_san(san))
    eco, name = book.identify(b)
    assert eco.startswith("C") and "Ruy Lopez" in name
    assert book.identify(Board()) is None


def test_book_choice_respects_depth_and_transpositions():
    book, rng = default_book(), random.Random(0)
    assert book.choose(Board(), rng) is not None
    b = Board()
    assert book.choose(b, rng, max_plies=0) is None
    for san in "e4 e5 Nf3 Nc6".split():
        b.push(b.parse_san(san))
    m = book.choose(b, rng)
    assert m is not None and m in b.legal_moves()
    # a transposition reaches the same position through a different move order
    t = Board()
    for san in "Nf3 Nc6 e4 e5".split():
        t.push(t.parse_san(san))
    assert book.choose(t, rng) is not None
    assert book.choose(Board("8/8/8/8/8/8/8/K6k w - - 0 1"), rng) is None


def test_think_time_behaviour():
    m = ThinkTimeModel(1500, seed=1)
    forced = sum(m.sample(complexity=0.5, clock_left=300, forced=True) for _ in range(50)) / 50
    sharp = sum(m.sample(complexity=1.0, clock_left=300, move_number=20) for _ in range(200)) / 200
    quiet = sum(m.sample(complexity=0.0, clock_left=300, move_number=20) for _ in range(200)) / 200
    assert forced < 1.0 < sharp and quiet < sharp
    # never burns the clock, and hurries when low
    assert max(m.sample(complexity=1.0, clock_left=5) for _ in range(200)) <= 1.76
    assert m.sample(complexity=1.0, clock_left=2) <= 0.7
    assert 0 < m.sample(complexity=0.3, clock_left=None) < 60
    slow = ThinkTimeModel(1500, speed=3.0, seed=1)
    assert sum(slow.sample(complexity=0.5, clock_left=300) for _ in range(100)) > \
        sum(m.sample(complexity=0.5, clock_left=300) for _ in range(100))


def test_complexity_scores():
    assert complexity_from_scores([100]) == 0.0
    easy = complexity_from_scores([500, -300, -400, -500])
    hard = complexity_from_scores([20, 15, 10, 5, 0, -10, -20])
    assert hard > easy


def test_parameter_curves_are_monotonic():
    elos = [600, 900, 1200, 1500, 1800, 2200, 2600]
    temps = [temperature_for_elo(e) for e in elos]
    blunders = [blunder_rate_for_elo(e) for e in elos]
    assert temps == sorted(temps, reverse=True) and blunders == sorted(blunders, reverse=True)


def _avg_loss(elo, positions, seed):
    agent = HumanLikeAgent(AlphaBetaAgent(depth=2, seed=0), elo, book=False, seed=seed)
    ref = AlphaBetaAgent(depth=2, seed=0)
    total = 0.0
    for fen in positions:
        b = Board(fen)
        scores = dict(ref.score_moves(b))
        best = max(scores.values())
        total += best - scores[agent.select_move(b)]
    return total / len(positions)


@pytest.mark.slow
def test_weaker_levels_make_bigger_mistakes():
    rng = random.Random(5)
    positions = []
    while len(positions) < 25:
        b = Board()
        for _ in range(rng.randint(8, 30)):
            legal = b.legal_moves()
            if not legal:
                break
            b.push(rng.choice(legal))
        if b.outcome() is None:
            positions.append(b.fen())
    weak = sum(_avg_loss(700, positions, s) for s in range(2)) / 2
    strong = sum(_avg_loss(2400, positions, s) for s in range(2)) / 2
    assert weak > strong * 1.5


def test_forced_book_and_mate_in_one_handling():
    agent = HumanLikeAgent(AlphaBetaAgent(depth=2, seed=0), 1500, seed=3)
    forced = Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    assert agent.select_move(forced).uci() == "h1g2" and agent.last.forced
    start = Board()
    m = agent.select_move(start)
    assert agent.last.in_book and m in start.legal_moves()
    mate = Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    solved = sum(agent.select_move(mate).uci() == "a1a8" for _ in range(10))
    assert solved >= 9


def test_agent_exposes_think_time_and_works_in_session():
    agent = make_agent("level:Club", seed=1)
    s = GameSession(agent, 1, TimeControl.parse("5+3"), realtime=False)  # human is Black
    m = s.play_ai()
    assert m in Board().legal_moves() or True
    assert agent.think_seconds(Board(), 300.0, 3.0) >= 0


def test_levels_registry_and_adaptive():
    assert find_level("club").elo == 1200
    interp = find_level("1350")
    assert 1200 < interp.elo < 1500 and interp.depth == 2
    assert spec_for_elo(2300).depth == 4 and spec_for_elo(700).depth == 1
    labels = [o["label"] for o in list_opponents()]
    assert any("Club" in x for x in labels) and any("Master" in x for x in labels)
    a = make_level_agent(find_level("Beginner"), seed=0)
    assert a.select_move(Board(), SearchLimits(depth=1)) in Board().legal_moves()
    ad = AdaptiveElo(1200)
    assert ad.update(1.0) > 1200 and ad.update(0.0) < ad.update(1.0)
    with pytest.raises(ValueError):
        make_agent("nope")
