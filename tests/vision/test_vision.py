import cv2
import numpy as np
import pytest

from chessrl.agents import AlphaBetaAgent
from chessrl.engine import Board, Game
from chessrl.vision import DetectionEstimator, MoveTracker, OccupancyEstimator, find_board
from chessrl.vision.board_detect import find_board as find
from chessrl.vision.detectors import OracleDetector
from chessrl.vision.geometry import SQ, BoardCalibration, order_corners, rotate_corners_for_white
from chessrl.vision.session import VisionSession
from chessrl.vision.synth import (
    BoardStyle,
    CameraSpec,
    game_frames,
    render_frame,
    write_video,
    yolo_label_text,
)

ITALIAN = "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O Bg4 h3 h5 hxg4 hxg4 d4 exd4 Nxd4 Qh4 Nf5 Qh7".split()


def make_game(sans: str | list[str], fen: str | None = None) -> Game:
    b = Board(fen) if fen else Board()
    g = Game(start_fen=b.fen())
    for s in sans.split() if isinstance(sans, str) else sans:
        m = b.parse_san(s)
        g.moves.append(m)
        b.push(m)
    return g


def run_tracker(frames, calibration, estimator, board=None):
    board = board or Board()
    tracker = MoveTracker(board, estimator, calibration)
    events = []
    for f in frames:
        if isinstance(estimator, DetectionEstimator):
            estimator.detector.current = f
        ev = tracker.update(f.image)
        if ev is not None:
            events.append(ev)
    return board, events


def test_calibration_round_trip_and_square_mapping():
    f = render_frame(Board(), CameraSpec(roll=0, tilt=0.1, seed=1), BoardStyle())
    cal = BoardCalibration(f.corners)
    for sq in range(64):
        x, y = cal.square_center(sq)
        assert cal.point_to_square((x, y)) == sq
    rect = cal.rectify(f.image)
    assert rect.shape == (512, 512, 3)
    # a1 is dark, h1 light; the white rook sits bottom-left
    a1 = rect[7 * SQ + 4 : 8 * SQ - 4, 4:SQ - 4].reshape(-1, 3).mean(axis=0)
    assert order_corners(f.corners).shape == (4, 2) and a1.mean() > 0
    assert cal.point_to_square((0, 0)) is None


@pytest.mark.parametrize("seed", range(6))
def test_find_board_any_camera_and_rotation(seed):
    rng = np.random.default_rng(seed)
    spec = CameraSpec.random(rng, any_rotation=True)
    f = render_frame(Board(), spec, BoardStyle.random(rng), rng=rng)
    cal = find_board(f.image)
    assert cal is not None
    assert np.abs(cal.corners - f.corners).max() < 8, "corners (and therefore orientation) must match"


def test_find_board_with_known_orientation_mid_game():
    b = Board()
    for s in "e4 e5 Nf3 Nc6".split():
        b.push(b.parse_san(s))
    f = render_frame(b, CameraSpec.random(np.random.default_rng(5)), BoardStyle(), rng=np.random.default_rng(5))
    cal = find(f.image, white_side=0)
    assert cal is not None and np.abs(cal.corners - f.corners).max() < 8
    assert rotate_corners_for_white(order_corners(f.corners), 0).shape == (4, 2)


@pytest.mark.parametrize("seed,rotated", [(3, False), (11, True)])
def test_tracks_a_whole_game_with_hand_occlusion(seed, rotated):
    rng = np.random.default_rng(seed)
    spec = CameraSpec.random(rng, any_rotation=rotated)
    style = BoardStyle.random(rng)
    frames = list(game_frames(make_game(ITALIAN), spec, style, idle_frames=5, seed=seed))
    cal = find_board(frames[0].image)
    board, events = run_tracker(frames, cal, OccupancyEstimator())
    got = [e.san[0] for e in events if e.kind == "move"]
    assert got == ITALIAN
    assert not [e for e in events if e.kind == "unexplained"]
    assert board.fen() == make_game(ITALIAN).board_at().fen()


def test_en_passant_castling_and_queen_promotion_assumed():
    g = make_game("e4 a6 e5 d5 exd6 Nf6 dxc7 Qxc7 Nf3 e6 Be2 Bd6 O-O O-O".split())
    spec, style = CameraSpec.random(np.random.default_rng(2)), BoardStyle()
    frames = list(game_frames(g, spec, style, idle_frames=5, seed=2))
    cal = find_board(frames[0].image)
    _, events = run_tracker(frames, cal, OccupancyEstimator())
    assert [e.san[0] for e in events] == ["e4", "a6", "e5", "d5", "exd6", "Nf6", "dxc7", "Qxc7", "Nf3", "e6",
                                          "Be2", "Bd6", "O-O", "O-O"]

    fen = "r3k2r/1P5p/8/8/8/8/6P1/R2QK2R w KQkq - 0 1"
    g2 = make_game(["bxa8=Q+", "Kf7"], fen)
    frames = list(game_frames(g2, CameraSpec(seed=4), BoardStyle(), idle_frames=5, seed=4))
    cal = find_board(frames[0].image, white_side=0)
    board = Board(fen)
    est = OccupancyEstimator()
    _, events = run_tracker(frames, cal, est, board)
    assert events[0].san == ["bxa8=Q+"] and events[0].promotion_assumed
    assert events[1].san == ["Kf7"]


def test_piece_level_estimator_distinguishes_underpromotion():
    fen = "r3k2r/1P5p/8/8/8/8/6P1/R2QK2R w KQkq - 0 1"
    g = make_game(["bxa8=N"], fen)
    frames = list(game_frames(g, CameraSpec(seed=4), BoardStyle(), idle_frames=5, seed=4))
    cal = find_board(frames[0].image, white_side=0)
    oracle = OracleDetector()
    est = DetectionEstimator(oracle, cal)
    _, events = run_tracker(frames, cal, est, Board(fen))
    assert [e.san[0] for e in events] == ["bxa8=N"] and not events[0].promotion_assumed


def test_oracle_detection_tracks_game():
    frames = list(game_frames(make_game(ITALIAN[:10]), CameraSpec(seed=7), BoardStyle(), idle_frames=5, seed=7))
    cal = find_board(frames[0].image)
    board, events = run_tracker(frames, cal, DetectionEstimator(OracleDetector(), cal))
    assert [e.san[0] for e in events] == ITALIAN[:10]
    assert board.fen() == make_game(ITALIAN[:10]).board_at().fen()


def test_two_moves_between_frames_and_unexplained_change():
    g = make_game("e4 e5")
    spec, style = CameraSpec(seed=6), BoardStyle()
    frames = list(game_frames(g, spec, style, idle_frames=5, seed=6, with_hand=False))
    cal = find_board(frames[0].image)
    # only the start frame and the final one: two plies happened in between (a fast player)
    board, events = run_tracker([frames[0]] * 5 + [frames[-1]] * 8, cal, OccupancyEstimator())
    assert [e.san for e in events] == [["e4", "e5"]] and events[0].kind == "two_moves"

    # a piece disappears (knocked off the board): no legal move explains it
    b = Board()
    start = render_frame(b, spec, style, rng=np.random.default_rng(1))
    missing = render_frame(b, spec, style, hide={6}, rng=np.random.default_rng(1))
    _, events = run_tracker([start] * 5 + [missing] * 8, find_board(start.image), OccupancyEstimator())
    assert any(e.kind == "unexplained" for e in events) or not events


def test_yolo_labels_are_valid():
    f = render_frame(Board(), CameraSpec(seed=2), BoardStyle())
    lines = yolo_label_text(f).splitlines()
    assert len(lines) == 32
    for line in lines:
        c, x, y, w, h = line.split()
        assert 0 <= int(c) < 12 and all(0 <= float(v) <= 1 for v in (x, y, w, h))


def test_session_on_video_file_records_pgn_and_suggests(tmp_path):
    g = make_game(ITALIAN[:8])
    frames = game_frames(g, CameraSpec.random(np.random.default_rng(9)), BoardStyle.random(np.random.default_rng(9)),
                         idle_frames=5, seed=9)
    video = write_video(frames, tmp_path / "demo.mp4")
    assert video.stat().st_size > 1000
    seen = []
    sess = VisionSession(agent=AlphaBetaAgent(depth=1, seed=0), on_move=lambda ev, b: seen.append(ev.san[0]),
                         think_seconds=0.2)
    game = sess.run(str(video), out_dir=tmp_path / "out", save_video=True, log=lambda *_: None)
    assert [m.uci() for m in game.moves] == [m.uci() for m in g.moves]
    assert seen == ITALIAN[:8]
    assert "1. e4 e5" in (tmp_path / "out" / "game.pgn").read_text()
    assert (tmp_path / "out" / "annotated.mp4").exists() and (tmp_path / "out" / "calibration.json").exists()
    assert sess.suggestion is not None and sess.suggestion.move in sess.board.legal_moves()
    cap = cv2.VideoCapture(str(tmp_path / "out" / "annotated.mp4"))
    assert cap.isOpened()
