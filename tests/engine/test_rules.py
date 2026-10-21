import pytest

from chessrl.engine import BLACK, WHITE, Board, Termination, parse_game, read_pgn
from chessrl.engine.constants import STARTING_FEN


def play(board, *sans):
    for s in sans:
        board.push(board.parse_san(s))
    return board


def test_fen_roundtrip():
    for fen in [
        STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "8/8/8/8/8/8/8/K6k b - - 12 70",
    ]:
        assert Board(fen).fen() == fen


@pytest.mark.parametrize("bad", ["", "8/8/8 w - - 0 1", "8/8/8/8/8/8/8/8 w - - 0 1", "x"])
def test_bad_fen_rejected(bad):
    with pytest.raises(ValueError):
        Board(bad)


def test_fools_mate():
    b = play(Board(), "f3", "e5", "g4", "Qh4#")
    out = b.outcome()
    assert out.termination is Termination.CHECKMATE and out.winner == BLACK
    assert out.result == "0-1"


def test_stalemate():
    b = Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    out = b.outcome()
    assert out.termination is Termination.STALEMATE and out.winner is None


def test_castling_rights_and_blocking():
    b = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    ucis = {m.uci() for m in b.legal_moves()}
    assert {"e1g1", "e1c1"} <= ucis
    # Rook attacks f1: cannot castle kingside, and cannot castle out of / through check.
    b = Board("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1")
    assert "e1g1" not in {m.uci() for m in b.legal_moves()}
    # Moving the rook loses that side's right.
    b = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    b.push(b.parse_san("Rh2"))
    assert not b.castling & 1 and b.castling & 2


def test_en_passant_and_its_expiry():
    b = play(Board(), "e4", "a6", "e5", "d5")
    assert b.ep_square is not None
    assert "e5d6" in {m.uci() for m in b.legal_moves()}
    play(b, "Nf3", "a5")
    assert "e5d6" not in {m.uci() for m in b.legal_moves()}


def test_promotion_and_san():
    b = Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert {m.uci() for m in b.legal_moves() if m.from_sq == 48} == {
        "a7a8q", "a7a8r", "a7a8b", "a7a8n"
    }
    assert b.san(b.parse_uci("a7a8q")) == "a8=Q"


def test_san_disambiguation():
    b = Board("4k3/8/8/8/8/8/4K3/R6R w - - 0 1")
    assert b.san(b.parse_uci("a1d1")) == "Rad1"
    b = Board("4k3/8/8/8/8/2N3N1/8/4K3 w - - 0 1")
    assert b.san(b.parse_uci("c3e4")) == "Nce4"


@pytest.mark.parametrize(
    "fen,expected",
    [
        ("8/8/8/8/8/8/8/K6k w - - 0 1", True),
        ("8/8/8/8/8/8/8/KN5k w - - 0 1", True),
        ("8/8/8/8/8/8/8/KB5k w - - 0 1", True),
        ("8/8/8/8/8/8/8/KNN4k w - - 0 1", False),
        ("8/8/8/8/8/8/8/KP5k w - - 0 1", False),
        ("8/8/8/8/8/8/4b3/K1B4k w - - 0 1", False),  # opposite-colour bishops
        ("8/8/8/8/8/8/5b2/K1B4k w - - 0 1", True),  # same-colour bishops
    ],
)
def test_insufficient_material(fen, expected):
    assert Board(fen).is_insufficient_material() == expected


def test_threefold_repetition():
    b = Board()
    assert b.repetition_count() == 1
    for _ in range(2):
        play(b, "Nf3", "Nf6", "Ng1", "Ng8")
    assert b.repetition_count() == 3
    assert b.outcome().termination is Termination.THREEFOLD_REPETITION


def test_fifty_move_rule():
    b = Board("8/8/8/8/8/8/R7/K6k w - - 100 80")
    assert b.outcome().termination is Termination.FIFTY_MOVES
    assert Board("8/8/8/8/8/8/R7/K6k w - - 99 80").outcome() is None


def test_pgn_roundtrip_with_comments():
    b = Board()
    game = parse_game('[White "A"]\n[Black "B"]\n\n1. e4 {best by test} e5 2. Nf3 Nc6 *\n')
    assert [m.uci() for m in game.moves] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert game.comments == {0: "best by test"}
    again = parse_game(game.to_pgn())
    assert again.moves == game.moves and again.comments == game.comments
    assert again.headers["White"] == "A"
    assert game.board_at(2).fen() == play(b, "e4", "e5").fen()


def test_pgn_variations_and_nags_ignored_and_multiple_games():
    text = (
        '[Event "x"]\n\n1. e4 $1 e5 (1... c5 2. Nf3) 2. Nf3 1-0\n\n'
        '[Event "y"]\n\n1. d4 d5 1/2-1/2\n'
    )
    games = read_pgn(text)
    assert len(games) == 2
    assert [m.uci() for m in games[0].moves] == ["e2e4", "e7e5", "g1f3"]
    assert games[0].result == "1-0" and games[1].result == "1/2-1/2"


def test_copy_is_independent():
    a = Board()
    b = a.copy()
    b.push(b.parse_san("e4"))
    assert a.fen() == STARTING_FEN and b.turn == BLACK and a.turn == WHITE
