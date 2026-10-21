import time

from chessrl.agents import AlphaBetaAgent, GreedyAgent, RandomAgent, SearchLimits, evaluate
from chessrl.engine import Board


def test_random_and_greedy_play_legal_moves():
    for agent in (RandomAgent(0), GreedyAgent(0)):
        b = Board()
        for _ in range(40):
            if b.is_game_over():
                break
            m = agent.select_move(b)
            assert m in b.legal_moves()
            b.push(m)


def test_greedy_takes_free_queen():
    b = Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")
    assert GreedyAgent(0).select_move(b).uci() == "d1d5"


def test_eval_symmetry_and_start():
    assert evaluate(Board()) == 0
    a = evaluate(Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    assert a == 0
    assert evaluate(Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")) > 800


def test_alphabeta_finds_mate_in_one_and_two():
    b = Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert AlphaBetaAgent(depth=2, seed=0).select_move(b).uci() == "a1a8"
    # Mate in two (Qh5+ style ladder): Kh8 would be mated by Ra7 then Rb8 etc.
    b = Board("7k/8/5K2/8/8/8/8/R7 w - - 0 1")
    agent = AlphaBetaAgent(depth=4, seed=0)
    m = agent.select_move(b)
    b.push(m)
    assert not b.is_game_over()
    assert agent.search_value(b, 3) <= -50_000  # opponent is getting mated


def test_alphabeta_avoids_hanging_queen_and_beats_random():
    wins = 0
    for seed in range(4):
        b = Board()
        ab, rnd = AlphaBetaAgent(depth=2, seed=seed), RandomAgent(seed)
        ab_white = seed % 2 == 0
        for _ in range(160):
            if b.is_game_over():
                break
            agent = ab if (b.turn == 0) == ab_white else rnd
            b.push(agent.select_move(b))
        out = b.outcome()
        if out and out.winner is not None and (out.winner == 0) == ab_white:
            wins += 1
    assert wins >= 3


def test_search_respects_node_and_time_limits():
    b = Board()
    agent = AlphaBetaAgent(depth=6, seed=0)
    t = time.perf_counter()
    m = agent.select_move(b, SearchLimits(time=0.3))
    assert m in b.legal_moves() and time.perf_counter() - t < 2.0
    agent.select_move(b, SearchLimits(nodes=500))
    assert agent.nodes <= 600


def test_score_moves_ranking():
    b = Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")
    scores = AlphaBetaAgent(depth=2, seed=0).score_moves(b)
    assert scores[0][0].uci() == "d1d5"
    assert len(scores) == len(b.legal_moves())
