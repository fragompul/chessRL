"""Differential test: play random games in both our engine and python-chess (the oracle)
and require identical legal moves, FEN, SAN, hashes and game status at every ply."""

import random

import chess

from chessrl.engine import Board, Move


def _uci_set(moves):
    return sorted(m.uci() for m in moves)


def test_random_games_match_python_chess():
    rng = random.Random(1234)
    for _ in range(60):
        ours, ref = Board(), chess.Board()
        for _ply in range(300):
            legal = ours.legal_moves()
            assert _uci_set(legal) == sorted(m.uci() for m in ref.legal_moves), ours.fen()
            assert ours.fen() == ref.fen(en_passant="fen")
            assert ours.is_check() == ref.is_check()
            assert ours.is_checkmate() == ref.is_checkmate()
            assert ours.is_stalemate() == ref.is_stalemate()
            assert ours.is_insufficient_material() == ref.is_insufficient_material(), ours.fen()
            if not legal or ref.is_insufficient_material():
                break
            move = rng.choice(legal)
            ref_move = chess.Move.from_uci(move.uci())
            assert ours.san(move) == ref.san(ref_move), ours.fen()
            assert ours.parse_san(ref.san(ref_move)) == move
            h_before = ours.hash
            ours.push(move)
            ref.push(ref_move)
            assert ours.hash == ours._compute_hash()
            # undo / redo must restore the exact state
            ours.pop()
            assert ours.hash == h_before
            ours.push(move)


def test_push_pop_restores_everything():
    rng = random.Random(7)
    board = Board()
    snapshots = []
    for _ in range(120):
        legal = board.legal_moves()
        if not legal:
            break
        snapshots.append((board.fen(), board.hash))
        board.push(rng.choice(legal))
    while snapshots:
        board.pop()
        assert (board.fen(), board.hash) == snapshots.pop()


def test_move_roundtrip_uci():
    assert Move.from_uci("e7e8q") == Move(52, 60, 5)
    assert Move.from_uci("e2e4").uci() == "e2e4"
