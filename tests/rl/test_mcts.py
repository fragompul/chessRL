import numpy as np
import torch

from chessrl.agents import MCTSAgent
from chessrl.engine import Board
from chessrl.rl.actions import FLIP_ACTION, legal_actions
from chessrl.rl.buffer import ReplayBuffer
from chessrl.rl.evaluator import NetEvaluator
from chessrl.rl.mcts import Search, SearchConfig, run_search
from chessrl.rl.network import ChessNet, NetConfig
from chessrl.rl.selfplay import SelfPlayConfig, SelfPlayEngine

torch.manual_seed(0)


def tiny_eval():
    return NetEvaluator(ChessNet(NetConfig(channels=16, blocks=1, value_hidden=16)), "cpu")


def test_network_shapes():
    net = ChessNet(NetConfig(channels=16, blocks=2))
    logits, v = net(torch.zeros(3, 20, 8, 8))
    assert logits.shape == (3, 4672) and v.shape == (3,) and v.abs().max() <= 1


def test_flip_action_matches_mirrored_position():
    # No castling rights, so a left-right mirror is a chess symmetry.
    fen = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 9"
    mirrored = "1kr1qb1r/ppp1bnpp/2n1pn2/4pp2/4P3/2NBP1P1/PPP1NP1P/1KRQ1B1R w - - 0 9"
    assert FLIP_ACTION[FLIP_ACTION] .tolist() == list(range(4672))  # involution
    b = Board(fen)
    # mirror by hand: reverse every rank string
    rows = fen.split()[0].split("/")
    m = Board("/".join(r[::-1] for r in rows) + " w - - 0 9")
    _, idx = legal_actions(b)
    _, idx_m = legal_actions(m)
    assert sorted(FLIP_ACTION[idx].tolist()) == sorted(idx_m.tolist())
    del mirrored


def test_search_visit_counts_and_board_restored():
    board = Board()
    fen = board.fen()
    search = Search(board, SearchConfig(sims=40, leaves_per_round=4), np.random.default_rng(0))
    run_search(search, tiny_eval(), 40, leaves_per_round=4)
    assert board.fen() == fen and len(board._undo) == 0
    assert search.root_visits >= 40
    assert abs(search.root.N.sum() - round(search.root.N.sum())) < 1e-6  # no stray virtual loss


def test_search_finds_mate_in_one_even_with_random_net():
    board = Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    search = Search(board, SearchConfig(sims=200, leaves_per_round=4), np.random.default_rng(0))
    run_search(search, tiny_eval(), 200, leaves_per_round=4)
    moves, _, probs = search.policy(temperature=0)
    assert moves[int(np.argmax(probs))].uci() == "a1a8"


def test_tree_reuse():
    board = Board()
    search = Search(board, SearchConfig(sims=30), np.random.default_rng(0))
    run_search(search, tiny_eval(), 30)
    move = search.root.moves[int(np.argmax(search.root.N))]
    visits_child = search.root.children[int(np.argmax(search.root.N))].N.sum()
    board.push(move)
    search.advance(move)
    assert search.root_visits == visits_child


def test_mcts_agent_plays_legal_and_handles_forced():
    agent = MCTSAgent(tiny_eval(), sims=20, seed=0)
    b = Board()
    assert agent.select_move(b) in b.legal_moves()
    forced = Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")  # only Kxg2 is legal
    m = agent.select_move(forced)
    assert m.uci() == "h1g2"
    scores = agent.score_moves(b)
    assert len(scores) == 20


def test_selfplay_produces_consistent_training_data():
    cfg = SelfPlayConfig(parallel=4, search=SearchConfig(sims=8, leaves_per_round=2), max_plies=30,
                         temp_moves=10, endgames=["KQK"], endgame_prob=0.5)
    engine = SelfPlayEngine(cfg, tiny_eval(), seed=1)
    games = engine.run(4)
    assert len(games) == 4
    for g in games:
        n = len(g.z)
        assert n == g.plies  # self-play records both sides
        assert g.pieces.shape == (n, 64) and g.pol_idx.shape == (n, 64)
        assert np.allclose(g.pol_p.astype(np.float32).sum(axis=1), 1.0, atol=2e-2)
        # z alternates in sign with the mover unless the game was drawn
        if g.result != "1/2-1/2":
            assert set(np.unique(g.z)) <= {-1.0, 1.0}
            assert g.game.board_at().outcome() is not None or g.termination == "move limit"
    assert engine.stats["nn_evals"] > 0


def test_buffer_roundtrip_and_sampling(tmp_path):
    cfg = SelfPlayConfig(parallel=2, search=SearchConfig(sims=6), max_plies=20)
    games = SelfPlayEngine(cfg, tiny_eval(), seed=2).run(2)
    buf = ReplayBuffer(30)
    for g in games:
        buf.add(g.pieces, g.meta, g.pol_idx, g.pol_p, g.z)
    assert len(buf) == min(30, sum(g.plies for g in games))
    rng = np.random.default_rng(0)
    planes, idx, p, z = buf.sample(8, rng)
    assert planes.shape == (8, 20, 8, 8) and idx.shape == (8, 64) and z.shape == (8,)
    buf.save(tmp_path / "b.npz")
    again = ReplayBuffer.load(tmp_path / "b.npz")
    assert len(again) == len(buf) and np.array_equal(np.sort(again.z), np.sort(buf.z[: buf.size]))
    again.add(games[0].pieces, games[0].meta, games[0].pol_idx, games[0].pol_p, games[0].z)


def test_selfplay_never_stalls_on_repetition_prone_endgames():
    # Regression: re-rooting onto a tree-terminal (2-fold repetition) node used to hang forever.
    for seed in range(6):
        cfg = SelfPlayConfig(parallel=3, search=SearchConfig(sims=10, leaves_per_round=3),
                             max_plies=80, endgames=["KRK", "KQK"], endgame_prob=1.0)
        games = SelfPlayEngine(cfg, tiny_eval(), seed=seed).run(3)
        assert len(games) == 3
