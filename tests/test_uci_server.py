import io
import sys

from chessrl.agents import AlphaBetaAgent
from chessrl.engine import Board
from chessrl.eval.arena import play_match
from chessrl.eval.uci import UCIAgent
from chessrl.uci_server import parse_position, run_uci


def test_parse_position_variants():
    b = parse_position("position startpos moves e2e4 e7e5".split())
    assert b.fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3") and b.turn == 0
    fen = "8/8/8/8/8/8/R7/K6k w - - 0 1"
    assert parse_position(["position", "fen", *fen.split()]).fen() == fen
    b = parse_position(["position", "fen", *fen.split(), "moves", "a2a8"])
    assert b.turn == 1 and b.squares[56] == 4


def test_protocol_transcript():
    stdin = io.StringIO("uci\nisready\nucinewgame\nposition startpos moves e2e4\ngo depth 1\nquit\n")
    stdout = io.StringIO()
    run_uci(AlphaBetaAgent(depth=1, seed=0), stdin, stdout)
    lines = stdout.getvalue().splitlines()
    assert lines[0].startswith("id name chessRL") and "uciok" in lines and "readyok" in lines
    best = [ln for ln in lines if ln.startswith("bestmove")][0].split()[1]
    after_e4 = Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert after_e4.parse_uci(best) in after_e4.legal_moves()


def test_no_legal_moves_returns_null_move():
    stdout = io.StringIO()
    stalemate = "position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1\ngo\nquit\n"
    run_uci(AlphaBetaAgent(depth=1), io.StringIO(stalemate), stdout)
    assert "bestmove 0000" in stdout.getvalue()


def test_round_trip_through_our_own_uci_client():
    cmd = [sys.executable, "-m", "chessrl", "uci", "--agent", "ab1", "--seed", "0"]
    ours = UCIAgent(cmd, movetime=0.05, name="uci-ab1")
    from chessrl.agents import RandomAgent

    m = play_match(ours, RandomAgent(0), games=2, max_plies=80, opening_plies=2)
    ours.close()
    assert m.stats.games == 2 and "uci-ab1" in str(m)
