import pytest

from chessrl.agents import AlphaBetaAgent
from chessrl.analysis import (
    analyse_game,
    analyse_position,
    annotate_game,
    comment_for,
    eval_graph_svg,
    game_accuracy,
    move_accuracy,
    win_percent,
)
from chessrl.analysis.annotate import eval_text, is_sacrifice, mate_in
from chessrl.engine import BLACK, WHITE, Board, Game, parse_game


@pytest.fixture(scope="module")
def scorer():
    return AlphaBetaAgent(depth=2, seed=0)


def game_from(sans: str) -> Game:
    b, g = Board(), Game()
    for s in sans.split():
        m = b.parse_san(s)
        g.moves.append(m)
        b.push(m)
    return g


def test_win_percent_and_accuracy_curves():
    assert win_percent(0) == 50.0
    assert win_percent(300) > 70 > 30 > win_percent(-300)
    assert win_percent(5000) <= 100 and win_percent(-5000) >= 0
    assert move_accuracy(60, 60) > 99 and move_accuracy(60, 30) < 40
    assert move_accuracy(50, 70) > 99  # improving your position is never a loss
    assert game_accuracy([100, 100, 100]) > 99
    assert game_accuracy([100, 100, 5]) < game_accuracy([100, 100, 80])


def test_mate_helpers():
    assert mate_in(100_000 - 1) == 1 and mate_in(100_000 - 3) == 2 and mate_in(150) is None
    assert eval_text(35) == "+0.35" and eval_text(-10_000, 3) == "-M3"


def test_blunder_and_best_move_classification(scorer):
    b = Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")  # Re8# is mate in one
    best = analyse_position(scorer, b, b.parse_san("Re8#"))
    assert best.label in ("best", "great", "brilliant") and best.cp_loss == 0
    blunder = analyse_position(scorer, b, b.parse_san("Re6"))  # hangs the rook to ...fxe6
    assert blunder.label == "blunder" and blunder.best_san == "Re8#" and blunder.win_loss > 40
    assert comment_for(blunder).startswith("?? Blunder") and "Re8#" in comment_for(blunder)


def test_sacrifice_detection():
    # Knight lands on a square covered by a pawn, undefended: a sacrifice.
    b = Board("4k3/8/3p4/8/8/2N5/8/4K3 w - - 0 1")
    assert is_sacrifice(b.copy(), b.parse_san("Ne4")) is False  # e4 is not attacked by d6
    b = Board("4k3/8/5p2/8/8/3N4/8/4K3 w - - 0 1")
    assert is_sacrifice(b, b.parse_san("Ne5")) is True  # attacked by the f6 pawn, undefended
    assert is_sacrifice(Board(), Board().parse_san("e4")) is False


def test_full_game_analysis_and_annotation(scorer):
    # Fool's-mate style: white blunders with 2.f3?? and 3.g4??, black mates.
    g = game_from("f3 e5 g4 Qh4#")
    g.result = "0-1"
    a = analyse_game(g, scorer)
    assert len(a.moves) == 4 and a.moves[3].label in ("best", "great", "brilliant")
    assert a.moves[0].label in ("mistake", "inaccuracy", "blunder", "good", "excellent")
    assert a.moves[2].label == "blunder"
    assert a.accuracy_white < a.accuracy_black
    assert a.final_eval < -9000
    assert a.critical_moments()[0].ply == 2
    series = a.eval_series()
    assert len(series) == 5 and series[-1] < -9000
    d = a.to_dict()
    assert d["accuracy"]["white"] < d["accuracy"]["black"] and d["critical"]
    annotate_game(g, a)
    assert 2 in g.comments and "Blunder" in g.comments[2]
    assert g.headers["AccuracyWhite"] and "?? Blunder" in g.to_pgn()
    reread = parse_game(g.to_pgn())
    assert reread.comments[2] == g.comments[2]
    assert eval_graph_svg(a).startswith("<svg")


def test_book_moves_are_recognised_and_excluded_from_accuracy(scorer):
    g = game_from("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6")
    a = analyse_game(g, scorer)
    assert all(m.label == "book" for m in a.moves)
    assert a.opening and "Ruy Lopez" in a.opening[1]
    annotate_game(g, a)
    assert g.headers["Opening"].startswith("Ruy Lopez") and g.headers["ECO"].startswith("C")
    assert a.accuracy_white == 100.0
    # without the book every move is analysed
    a2 = analyse_game(g, scorer, use_book=False)
    assert all(m.label != "book" for m in a2.moves)


def test_only_move_is_great(scorer):
    # Only Kxg2 avoids losing the king... any legal single reply is trivially forced, so use a
    # position where one move saves a queen and everything else loses it.
    b = Board("4k3/8/8/8/8/2n5/1q6/K7 w - - 0 1")  # white must play Kxb2
    assert len(b.legal_moves()) == 1  # forced: analysis still works
    a = analyse_position(scorer, b, b.parse_san("Kxb2"))
    assert a.cp_loss == 0
    assert WHITE != BLACK
