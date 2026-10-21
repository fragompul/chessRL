import random

import numpy as np

from chessrl.engine import Board
from chessrl.rl.actions import NUM_ACTIONS, index_to_move, legal_actions, legal_mask, move_to_index
from chessrl.rl.curriculum import ENDGAMES, random_endgame
from chessrl.rl.encoding import NUM_PLANES, encode_board, encode_compact, planes_from_compact
from chessrl.rl.env import ChessEnv


def _random_positions(n, seed=0, plies=60):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = Board()
        for _ in range(rng.randint(0, plies)):
            legal = b.legal_moves()
            if not legal:
                break
            b.push(rng.choice(legal))
        if b.legal_moves():
            out.append(b)
    return out


def test_actions_are_a_bijection_on_legal_moves():
    for b in _random_positions(150):
        moves, idx = legal_actions(b)
        assert len(set(idx.tolist())) == len(moves), b.fen()  # unique
        assert idx.max() < NUM_ACTIONS and idx.min() >= 0
        for m, i in zip(moves, idx):
            assert index_to_move(int(i), b) == m, (b.fen(), m)


def test_underpromotion_and_castling_indices():
    b = Board("8/P6k/8/8/8/8/8/R3K2R w KQ - 0 1")
    ucis = {m.uci(): move_to_index(m, b) for m in b.legal_moves()}
    promos = [ucis[u] for u in ("a7a8q", "a7a8r", "a7a8b", "a7a8n")]
    assert len(set(promos)) == 4
    assert ucis["e1g1"] != ucis["e1c1"]
    # Black's mirrored view produces the same index for the mirrored move.
    w = Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    bl = Board("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")
    assert move_to_index(w.parse_uci("e2e4"), w) == move_to_index(bl.parse_uci("e7e5"), bl)


def test_encoding_perspective_symmetry():
    w = Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    mirrored = Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1")
    assert np.array_equal(encode_board(w), encode_board(mirrored))
    assert encode_board(w).shape == (NUM_PLANES, 8, 8)


def test_planes_content():
    b = Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 7 1")
    p = encode_board(b)
    assert p[3, 0, 0] == 1 and p[3, 0, 7] == 1 and p[5, 0, 4] == 1  # own rooks, own king
    assert p[9, 7, 0] == 1 and p[11, 7, 4] == 1  # opponent rook, king
    assert p[12].min() == 1 and p[13].max() == 0 and p[14].max() == 0 and p[15].min() == 1
    assert abs(p[17, 0, 0] - 0.07) < 1e-6 and p[19].min() == 1


def test_batch_expansion_matches_single():
    boards = _random_positions(8, seed=3)
    comp = [encode_compact(b) for b in boards]
    batch = planes_from_compact(np.stack([c[0] for c in comp]), np.stack([c[1] for c in comp]))
    for i, b in enumerate(boards):
        assert np.array_equal(batch[i], encode_board(b))


def test_legal_mask_counts():
    b = Board()
    assert legal_mask(b).sum() == 20


def test_endgame_generator_yields_legal_nonterminal_positions():
    rng = random.Random(0)
    for kind in ENDGAMES:
        for _ in range(15):
            b = random_endgame(kind, rng)
            assert b.outcome() is None and b.legal_moves()


def test_env_selfplay_episode():
    env = ChessEnv(max_plies=60, seed=0)
    obs, info = env.reset()
    assert obs.shape == (NUM_PLANES, 8, 8)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        action = rng.choice(np.nonzero(info["legal_mask"])[0])
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    assert env.plies <= 60


def test_env_vs_opponent_and_illegal_action():
    from chessrl.agents import RandomAgent

    env = ChessEnv(opponent=RandomAgent(0), agent_color=1, seed=1)
    obs, info = env.reset()
    assert env.board.turn == 1 and env.plies == 1  # opponent (white) already moved
    try:
        env.step(0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    env2 = ChessEnv(illegal_action="lose")
    env2.reset()
    _, r, term, _, info = env2.step(0)
    assert r == -1.0 and term and info["illegal"]


def test_env_endgame_curriculum_reset():
    env = ChessEnv(endgames=["KRK"], seed=5)
    env.reset()
    assert len(env.board.fen().split()[0].replace("/", "").replace("8", "")) >= 3
