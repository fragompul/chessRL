"""Measure move-recognition accuracy of the appearance-based vision tracker on synthetic games.

Plays full games through random cameras / board styles / orientations and reports how many are
tracked perfectly and the confidence margins. Used to choose the estimator constants
(`estimator.TEMPERATURE`, sigmas) and the tracker's `min_margin`.

    python scripts/tune_vision.py --seeds 8
"""

import argparse

import numpy as np

import chessrl.vision.estimator as est
from chessrl.engine import Board, Game
from chessrl.vision import MoveTracker, OccupancyEstimator, find_board
from chessrl.vision.synth import BoardStyle, CameraSpec, game_frames

GAMES = {
    "italian": "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O Bg4 h3 h5 hxg4 hxg4 d4 exd4 Nxd4 Qh4 Nf5 Qh7",
    "ep+castle": "e4 a6 e5 d5 exd6 Nf6 dxc7 Qxc7 Nf3 e6 Be2 Bd6 O-O O-O",
}


def make_game(sans: str) -> Game:
    board, game = Board(), Game()
    for san in sans.split():
        move = board.parse_san(san)
        game.moves.append(move)
        board.push(move)
    return game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=est.TEMPERATURE)
    args = ap.parse_args()
    est.TEMPERATURE = args.temperature
    perfect = total = 0
    margins: list[float] = []
    for name, sans in GAMES.items():
        game = make_game(sans)
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            spec = CameraSpec.random(rng, any_rotation=seed % 2 == 1)
            frames = list(game_frames(game, spec, BoardStyle.random(rng), idle_frames=5, seed=seed))
            cal = find_board(frames[0].image)
            board = Board()
            tracker = MoveTracker(board, OccupancyEstimator(), cal)
            events = [e for f in frames if (e := tracker.update(f.image)) is not None]
            got = [e.san[0] for e in events if e.kind == "move"]
            ok = got == sans.split() and all(e.kind == "move" for e in events)
            perfect += ok
            total += 1
            margins += [e.confidence for e in events if e.kind == "move"]
            print(f"{name:10s} seed {seed}: {'ok' if ok else 'FAIL'}  min margin "
                  f"{min((e.confidence for e in events), default=0):.1f}")
    print(f"\nperfectly tracked: {perfect}/{total}; margins min {min(margins):.1f} "
          f"median {float(np.median(margins)):.1f}")


if __name__ == "__main__":
    main()
