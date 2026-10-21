import pytest

from chessrl.agents import AlphaBetaAgent, RandomAgent
from chessrl.engine import BLACK, WHITE
from chessrl.play import Clock, GameSession, TimeControl, play_game


class FakeTime:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_time_control_parse_and_category():
    assert TimeControl.parse("5+3") == TimeControl(300, 3)
    assert TimeControl.parse("1+0").category == "bullet"
    assert TimeControl.parse("3+2").category == "blitz"
    assert TimeControl.parse("10+0").category == "rapid"
    assert TimeControl.parse("30+0").category == "classical"


def test_clock_runs_increments_and_flags():
    now = FakeTime()
    c = Clock(TimeControl(60, 2), now)
    c.start(WHITE)
    now.t = 10
    assert c.remaining(WHITE) == 50 and c.remaining(BLACK) == 60
    assert c.press(WHITE) == 10
    assert c.remaining(WHITE) == 52
    now.t = 80
    assert c.flagged(BLACK) and not c.flagged(WHITE)


def test_play_game_runs_to_completion():
    r = play_game(RandomAgent(1), RandomAgent(2), max_plies=300)
    assert r.result in ("1-0", "0-1", "1/2-1/2")
    assert r.game.board_at().fen()  # replayable
    assert r.game.headers["White"] == "random"


def test_session_human_and_ai_flow():
    s = GameSession(AlphaBetaAgent(depth=1, seed=0), WHITE, realtime=False)
    s.play_human("e2e4")
    assert s.is_ai_turn()
    s.play_ai()
    assert s.board.turn == WHITE and len(s.moves) == 2
    with pytest.raises(ValueError):
        s.play_human("e2e5")
    assert s.takeback() == 2 and len(s.moves) == 0
    s.play_human("e4")  # SAN accepted too
    assert s.hint() is not None


def test_session_black_human_ai_moves_first():
    s = GameSession(AlphaBetaAgent(depth=1, seed=0), BLACK, realtime=False)
    assert s.is_ai_turn()
    s.play_ai()
    assert not s.is_ai_turn() and s.board.turn == BLACK


def test_session_resign_flag_and_draw():
    s = GameSession(RandomAgent(0), WHITE, realtime=False)
    s.resign(WHITE)
    assert s.to_game().result == "0-1"

    now = FakeTime()
    s = GameSession(RandomAgent(0), WHITE, TimeControl(10, 0), realtime=False, now=now)
    now.t = 11
    with pytest.raises(ValueError):
        s.play_human("e2e4")
    assert s.ending.reason == "timeout" and s.ending.result == "0-1"

    s = GameSession(None, None, realtime=False)
    s.offer_draw(WHITE)
    assert s.accept_draw(BLACK) and s.ending.result == "1/2-1/2"


def test_session_ai_accepts_draw_only_when_not_better():
    s = GameSession(AlphaBetaAgent(depth=1, seed=0), WHITE, realtime=False)
    assert not s.offer_draw(WHITE)  # too early in the game
    for u in ["g1f3", "g8f6", "f3g1", "f6g8"] * 5:
        s.board.push(s.board.parse_uci(u))
        s.moves.append(s.board._undo[-1][0])
    assert s.offer_draw(WHITE) is True  # equal and long enough


def test_session_claim_threefold():
    s = GameSession(None, None, realtime=False)
    for u in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        s.board.push(s.board.parse_uci(u))
    assert s.board.repetition_count() == 3
    assert s.claim_draw()
