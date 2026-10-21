"""Infer the moves played on a physical board from a stream of camera frames.

Principle: the game state is known, so the only question is *which legal move happened*.
For every legal move we know exactly which squares change and how; we pick the move whose
predicted board best explains what the camera sees. This is far more robust than recognising
the whole position from scratch every frame, and illegal moves can never be produced.

Frames are only judged when the picture has *settled* (no motion, no hand over the board), so
the moving hand and the lifted piece never confuse it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..engine import Board, Move
from .estimator import Estimator
from .geometry import BoardCalibration


@dataclass
class TrackerEvent:
    kind: str  # "move" | "two_moves" | "unexplained"
    moves: list[Move] = field(default_factory=list)
    san: list[str] = field(default_factory=list)
    confidence: float = 0.0  # log-likelihood margin over the next best explanation
    promotion_assumed: bool = False
    frame: int = 0


class MoveTracker:
    def __init__(
        self,
        board: Board,
        estimator: Estimator,
        calibration: BoardCalibration,
        *,
        settle_frames: int = 4,
        motion_thresh: float = 0.003,
        min_margin: float = 2.0,
        max_unknown: int = 1,
        allow_two_ply: bool = True,
    ):
        self.board = board
        self.est = estimator
        self.calib = calibration
        self.settle_frames = settle_frames
        self.motion_thresh = motion_thresh
        self.min_margin = min_margin
        self.max_unknown = max_unknown
        self.allow_two_ply = allow_two_ply
        self.frame_index = 0
        self._prev: np.ndarray | None = None
        self._still = 0
        self._judged: np.ndarray | None = None
        self.started = False
        self.last_unexplained = 0

    @staticmethod
    def _small(rect: np.ndarray) -> np.ndarray:
        return cv2.resize(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def _changed_fraction(a: np.ndarray, b: np.ndarray, level: float = 18.0) -> float:
        """Share of pixels (on the 64x64 board map) that changed noticeably between two frames.
        A hand or a moving piece changes a few pixels a lot; sensor noise changes none."""
        return float((np.abs(a - b) > level).mean())

    def start(self, frame: np.ndarray) -> None:
        """Learn the board's appearance from a frame showing the current (known) position."""
        rect = self.calib.rectify(frame)
        self.est.calibrate(frame, rect, self.board)
        self._prev = self._small(rect)
        self._judged = self._prev.copy()
        self.started = True

    # ------------------------------------------------------------------ main
    def update(self, frame: np.ndarray) -> TrackerEvent | None:
        """Feed one frame. Returns an event when a move (or two) has been recognised."""
        if not self.started:
            self.start(frame)
            return None
        self.frame_index += 1
        rect = self.calib.rectify(frame)
        small = self._small(rect)
        motion = self._changed_fraction(small, self._prev)
        self._prev = small
        self._still = self._still + 1 if motion < self.motion_thresh else 0
        if self._still < self.settle_frames:
            return None
        if self._changed_fraction(small, self._judged) < self.motion_thresh:
            return None  # nothing new since the last judgement
        obs = self.est.observe(frame, rect)
        if obs.unknown > self.max_unknown:
            return None  # a hand (or something else) is still over the board
        self._judged = small.copy()
        return self._judge(frame, rect, obs)

    # --------------------------------------------------------------- judging
    @staticmethod
    def _loglik(obs, expected: np.ndarray) -> float:
        return obs.loglik(expected)

    def _judge(self, frame, rect, probs) -> TrackerEvent | None:
        board = self.board
        base = self._loglik(probs, self.est.expected(board))
        scored: list[tuple[float, list[Move], np.ndarray]] = []
        legal = board.legal_moves()
        for m in legal:
            board.push(m)
            exp = self.est.expected(board)
            ll = self._loglik(probs, exp)
            if self.allow_two_ply:
                scored.append((ll, [m], exp))
            else:
                scored.append((ll, [m], exp))
            board.pop()
        scored.sort(key=lambda t: -t[0])
        best_ll, best_moves, best_exp = scored[0]

        # Promotions to different pieces are one *move* (same from/to); the picture of the other
        # squares cannot tell them apart, so judge the move first and the piece afterwards.
        key = (best_moves[0].from_sq, best_moves[0].to_sq)
        variants = [s for s in scored if (s[1][0].from_sq, s[1][0].to_sq) == key]
        promotion_assumed = len(variants) > 1
        if promotion_assumed:
            queen = next((v for v in variants if v[1][0].promotion == 5), variants[0])
            if variants[0][0] - queen[0] < 0.5 * self.min_margin:
                best_moves = queen[1]  # no clear evidence for another piece: assume a queen
            else:
                best_moves, promotion_assumed = variants[0][1], False
        rivals = [s[0] for s in scored if (s[1][0].from_sq, s[1][0].to_sq) != key]
        second = max(rivals + [base])  # next best distinct explanation (incl. "nothing happened")

        single_ok = best_ll - second >= self.min_margin and best_ll > base
        if self.allow_two_ply:
            # Two plies (a fast player) must explain the picture clearly better than the best
            # single move; with only one real change they never do, since the extra ply would
            # have to invent a change that is not there.
            two = self._best_two_ply(probs, max(base, best_ll))
            if two is not None:
                (m1, m2), margin = two
                if margin >= self.min_margin:
                    ev = self._commit(frame, rect, [m1, m2], margin, False)
                    ev.kind = "two_moves"
                    return ev
        if single_ok:
            return self._commit(frame, rect, best_moves, best_ll - second, promotion_assumed)
        if best_ll > base + 0.5 * self.min_margin:
            self.last_unexplained += 1
            return TrackerEvent("unexplained", [], [], best_ll - second, frame=self.frame_index)
        return None

    def _best_two_ply(self, probs: np.ndarray, base: float):
        board = self.board
        best = (-1e18, None)
        second = -1e18
        results: list[tuple[float, tuple[Move, Move], np.ndarray]] = []
        for m1 in board.legal_moves():
            board.push(m1)
            for m2 in board.legal_moves():
                board.push(m2)
                exp = self.est.expected(board)
                results.append((self._loglik(probs, exp), (m1, m2), exp))
                board.pop()
            board.pop()
        if not results:
            return None
        results.sort(key=lambda t: -t[0])
        top_ll, pair, top_exp = results[0]
        for ll, _, exp in results[1:]:
            if not np.array_equal(exp, top_exp):
                second = ll
                break
        del best
        margin = top_ll - max(second, base)
        return (pair, margin) if top_ll > base else None

    def _commit(self, frame, rect, moves: list[Move], margin: float, promo: bool) -> TrackerEvent:
        sans = []
        for m in moves:
            sans.append(self.board.san(m))
            self.board.push(m)
        self.est.confirm(frame, rect, self.board)
        self._judged = self._small(rect)
        return TrackerEvent("move", moves, sans, margin, promo, self.frame_index)

    # ------------------------------------------------------------ manual fixes
    def force_move(self, move: Move, frame: np.ndarray | None = None) -> None:
        """Apply a move decided by the user (e.g. after an 'unexplained' event or a promotion prompt)."""
        self.board.push(move)
        if frame is not None:
            self.est.confirm(frame, self.calib.rectify(frame), self.board)

    def undo(self) -> Move:
        """Take back the last recognised move (a mis-recognition the user spotted)."""
        return self.board.pop()
