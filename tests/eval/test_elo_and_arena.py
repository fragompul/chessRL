import math
import random
import sys
import textwrap

import pytest

from chessrl.agents import AlphaBetaAgent, RandomAgent
from chessrl.eval.arena import play_match, random_opening, round_robin
from chessrl.eval.elo import (
    MatchStats,
    elo_diff,
    estimate_ratings,
    expected_score,
    sprt_bounds,
    sprt_llr,
    sprt_status,
)
from chessrl.eval.plots import line_chart_svg
from chessrl.eval.uci import UCIAgent, find_stockfish, stockfish_agent


def test_expected_score_and_inverse():
    assert expected_score(0) == 0.5
    assert abs(expected_score(400) - 10 / 11) < 1e-9
    for d in (-300, -50, 0, 120, 640):
        assert abs(elo_diff(expected_score(d)) - d) < 1e-6
    assert math.isfinite(elo_diff(1.0)) and math.isfinite(elo_diff(0.0))


def test_match_stats_interval_contains_truth():
    s = MatchStats(60, 20, 20)  # 70% score ~ +147 Elo
    lo, hi = s.elo_interval()
    assert lo < s.elo() < hi and 90 < s.elo() < 200
    assert "+60 =20 -20" in str(s)


def test_estimate_ratings_recovers_known_gaps():
    rng = random.Random(0)
    truth = {"a": 0.0, "b": 200.0, "c": 400.0}
    results = []
    for x, y in (("a", "b"), ("b", "c"), ("a", "c")):
        p = expected_score(truth[x] - truth[y])
        results += [(x, y, 1.0 if rng.random() < p else 0.0) for _ in range(400)]
    est = estimate_ratings(results, anchors={"a": 0.0})
    assert abs(est["b"] - 200) < 40 and abs(est["c"] - 400) < 50
    free = estimate_ratings(results)  # no anchor: mean pinned to 0
    assert abs(sum(free.values())) < 1e-6
    assert abs((free["c"] - free["a"]) - 400) < 60


def test_estimate_ratings_finite_for_shutout():
    est = estimate_ratings([("w", "l", 1.0)] * 10, anchors={"l": 1000.0})
    assert math.isfinite(est["w"]) and est["w"] > 1000


def test_sprt():
    lo, hi = sprt_bounds()
    assert lo < 0 < hi
    strong = MatchStats(300, 100, 100)
    assert sprt_status(strong, 0, 30) == "H1"
    assert sprt_status(MatchStats(100, 300, 100), 0, 30) in ("H0", "continue")
    assert sprt_llr(MatchStats(0, 0, 0), 0, 30) == 0.0
    assert sprt_status(MatchStats(1, 1, 1), 0, 30) == "continue"


def test_random_opening_is_legal_and_varied():
    rng = random.Random(1)
    fens = {random_opening(4, rng) for _ in range(10)}
    assert len(fens) > 3


def test_match_alphabeta_beats_random_with_sprt():
    m = play_match(AlphaBetaAgent(depth=2, seed=0), RandomAgent(0), games=8, max_plies=200, sprt=(0, 100))
    assert m.stats.score > 0.8
    assert "[Round" in m.pgn() and "alphabeta-d2" in str(m)


@pytest.mark.slow
def test_round_robin_orders_strength():
    agents = [RandomAgent(0), AlphaBetaAgent(depth=1, seed=0), AlphaBetaAgent(depth=2, seed=0)]
    _, ratings = round_robin(agents, games_per_pair=4, anchors={"random": 0.0}, max_plies=150)
    assert ratings["alphabeta-d2"] > ratings["random"] and ratings["alphabeta-d1"] > ratings["random"]


FAKE_ENGINE = textwrap.dedent('''
    import sys
    from chessrl.engine import Board
    board = Board()
    opts = {}
    for line in sys.stdin:
        parts = line.split()
        if not parts: continue
        if parts[0] == "uci":
            print("id name FakeFish"); print("uciok")
        elif parts[0] == "isready": print("readyok")
        elif parts[0] == "setoption": opts[parts[2]] = parts[-1]
        elif parts[0] == "position":
            board = Board(" ".join(parts[2:8]))
        elif parts[0] == "go":
            print("bestmove", sorted(m.uci() for m in board.legal_moves())[0])
        elif parts[0] == "quit": break
        sys.stdout.flush()
''')


def test_uci_wrapper_with_fake_engine(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(FAKE_ENGINE)
    agent = UCIAgent([sys.executable, str(script)], {"UCI_Elo": 1500}, movetime=0.01)
    assert agent.name == "FakeFish"
    from chessrl.engine import Board

    b = Board()
    assert agent.select_move(b).uci() == "a2a3"
    r = play_match(agent, RandomAgent(0), games=2, max_plies=60)
    assert r.stats.games == 2
    agent.close()


@pytest.mark.skipif(find_stockfish() is None, reason="stockfish not installed")
def test_real_stockfish_plays_and_is_stronger_than_alphabeta():
    sf = stockfish_agent(1500, movetime=0.02)
    from chessrl.engine import Board

    assert sf.select_move(Board()) in Board().legal_moves()
    m = play_match(AlphaBetaAgent(depth=1, seed=0), sf, games=2, max_plies=150, opening_plies=2)
    assert m.stats.score <= 0.5
    sf.close()


def test_svg_chart():
    svg = line_chart_svg({"net": [(0, 800), (10, 900), (20, 1000)]}, title="Elo", xlabel="it", ylabel="Elo",
                         bands={"net": [(0, 780, 820), (10, 880, 920), (20, 970, 1030)]})
    assert svg.startswith("<svg") and "polyline" in svg and "Elo" in svg
    assert line_chart_svg({}).startswith("<svg")
