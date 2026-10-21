"""Turn a camera frame into per-square beliefs about what stands on the board.

Two interchangeable estimators feed the move tracker:

* `OccupancyEstimator` needs no trained model. It learns what every (square colour, piece)
  combination looks like straight from the first frame (the position is known) and keeps
  adapting as confirmed moves arrive.
* `DetectionEstimator` turns detections from a piece detector (YOLO) into per-square beliefs
  over 13 classes (empty + 12 pieces).

Both expose: calibrate / observe / expected / confirm, where `observe` returns an object that
can score any hypothesised board (`loglik`) and counts squares it cannot explain (`unknown`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..engine import Board
from .geometry import FEATURE_DIM, BoardCalibration, is_light_square, square_features_rel


class Observation(Protocol):
    """What the camera shows, able to score any hypothesised board."""

    unknown: int  # squares that look like nothing we know (hand, glare...)

    def loglik(self, expected: np.ndarray) -> float: ...


class Estimator(Protocol):
    def calibrate(self, frame: np.ndarray, rect: np.ndarray, board: Board) -> None: ...
    def observe(self, frame: np.ndarray, rect: np.ndarray) -> Observation: ...
    def expected(self, board: Board) -> np.ndarray: ...
    def confirm(self, frame: np.ndarray, rect: np.ndarray, board: Board) -> None: ...


# 13-class piece map: 0 empty, 1..6 white P N B R Q K, 7..12 black p n b r q k
def piece_classes_of(board: Board) -> np.ndarray:
    sq = np.asarray(board.squares)
    return np.where(sq == 0, 0, np.where(sq > 0, sq, 6 - sq)).astype(np.int64)


_LIGHT = np.array([int(is_light_square(s)) for s in range(64)])
_SQ = np.arange(64)
PENALTY_CAP = 7.0  # max log-likelihood loss charged to one square (robust to outliers)
TEMPERATURE = FEATURE_DIM / 6.0  # tuned on synthetic games (scripts/tune_vision.py)  # the features are strongly correlated: soften the Gaussian


class _AppearanceObservation:
    def __init__(self, feats: np.ndarray, mu: np.ndarray, var: np.ndarray, unknown_threshold: float,
                 floor: np.ndarray):
        self.feats = feats
        self.mu, self.var = mu, var
        # Occam term: a class we know little about (wide variance) must not win by being vague.
        self.pen = 0.5 * (np.log(var).sum(axis=-1) - np.log(floor).sum()) / TEMPERATURE  # [2, 13]
        # a square is "unknown" if even its best-matching class is far away
        d_all = (((feats[:, None, None, :] - mu[None]) ** 2) / var[None]).sum(axis=-1)  # [64, 2, 13]
        d_same = d_all[_SQ, _LIGHT]  # [64, 13] distances to the classes of this square's colour
        self.unknown = int((d_same.min(axis=1) > unknown_threshold).sum())

    def loglik(self, expected: np.ndarray) -> float:
        mu = self.mu[_LIGHT, expected]  # [64, F]
        var = self.var[_LIGHT, expected]
        d = (((self.feats - mu) ** 2) / var).sum(axis=1)
        return float(-(np.minimum(0.5 * d / TEMPERATURE, PENALTY_CAP) + self.pen[_LIGHT, expected]).sum())


@dataclass
class OccupancyEstimator:
    """Learns the look of every (square colour, piece) combination from the known game state.

    Calibrated on the first frame (the position is known), then refined after every confirmed
    move, so it adapts to lighting drift and to pieces it had not met on that colour of square.
    No trained model is needed; it is the baseline next to the YOLO-based estimator.
    """

    color_sigma: float = 12.0
    patch_sigma: float = 22.0
    unknown_threshold: float = 900.0

    @property
    def var_floor(self) -> np.ndarray:
        return np.array([self.color_sigma**2] * 3 + [self.patch_sigma**2] * (FEATURE_DIM - 3), dtype=np.float32)

    def __post_init__(self) -> None:
        self.mu = np.zeros((2, 13, FEATURE_DIM), dtype=np.float32)  # [square colour, piece code, feature]
        self.var = np.tile(self.var_floor, (2, 13, 1))
        self.seen = np.zeros((2, 13), dtype=bool)

    def calibrate(self, frame: np.ndarray, rect: np.ndarray, board: Board) -> None:
        feats = square_features_rel(rect)
        codes = piece_classes_of(board)
        floor = self.var_floor
        self.seen[:] = False
        for c in (0, 1):
            for code in range(13):
                sel = (_LIGHT == c) & (codes == code)
                if sel.any():
                    self.mu[c, code] = feats[sel].mean(axis=0)
                    self.var[c, code] = (feats[sel].var(axis=0) if sel.sum() > 1 else 0) + floor
                    self.seen[c, code] = True
        # classes not on the board yet (e.g. a white queen on a dark square): borrow from the
        # other colour of square (features are relative to the ring, so they carry over), with
        # extra uncertainty; failing that, from the same side's other pieces.
        for c in (0, 1):
            for code in range(13):
                if not self.seen[c, code] and self.seen[1 - c, code]:
                    self.mu[c, code] = self.mu[1 - c, code]
                    self.var[c, code] = self.var[1 - c, code] * 4
        for c in (0, 1):
            for code in range(13):
                if not self.seen[c, code] and not self.seen[1 - c, code]:
                    pool = [k for k in (range(1, 7) if 1 <= code <= 6 else range(7, 13) if code > 6 else [0])
                            if self.seen[c, k] or self.seen[1 - c, k]]
                    if pool:
                        mus = [self.mu[c, k] if self.seen[c, k] else self.mu[1 - c, k] for k in pool]
                        self.mu[c, code] = np.mean(mus, axis=0)
                        self.var[c, code] = np.var(mus, axis=0) + floor * 6

    def confirm(self, frame: np.ndarray, rect: np.ndarray, board: Board, alpha: float = 0.2) -> None:
        """All 64 squares are known now: pull each exact class's centroid towards what we see."""
        feats = square_features_rel(rect)
        codes = piece_classes_of(board)
        for sq in range(64):
            c, code = int(_LIGHT[sq]), int(codes[sq])
            if not self.seen[c, code]:
                self.mu[c, code] = feats[sq]
                self.var[c, code] = self.var_floor * 2
                self.seen[c, code] = True
            else:
                self.mu[c, code] += alpha * (feats[sq] - self.mu[c, code])

    def observe(self, frame: np.ndarray, rect: np.ndarray) -> _AppearanceObservation:
        return _AppearanceObservation(square_features_rel(rect), self.mu, self.var, self.unknown_threshold,
                                      self.var_floor)

    def expected(self, board: Board) -> np.ndarray:
        return piece_classes_of(board)


@dataclass
class Detection:
    cls: int  # 0..11 (YOLO class: P N B R Q K p n b r q k)
    conf: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels


class _DetectionObservation:
    def __init__(self, probs: np.ndarray):
        self.probs = probs
        self.unknown = 0

    def loglik(self, expected: np.ndarray) -> float:
        return float(np.log(np.clip(self.probs[_SQ, expected], 1e-3, 1.0)).sum())


class DetectionEstimator:
    """Beliefs from a piece detector. Detections are mapped to squares through the board
    homography using the *foot* of each box (its bottom centre), which lies on the board plane."""

    def __init__(self, detector, calibration: BoardCalibration, foot: float = 0.08):
        self.detector = detector
        self.calib = calibration
        self.foot = foot

    def calibrate(self, frame, rect, board) -> None:
        return None

    def confirm(self, frame, rect, board) -> None:
        return None

    def observe(self, frame: np.ndarray, rect: np.ndarray) -> _DetectionObservation:
        probs = np.zeros((64, 13), dtype=np.float64)
        probs[:, 0] = 0.9
        best = np.zeros(64)
        for d in self.detector.detect(frame):
            x0, y0, x1, y1 = d.box
            foot = ((x0 + x1) / 2, y1 - self.foot * (y1 - y0))
            sq = self.calib.point_to_square(foot)
            if sq is None or d.conf < best[sq]:
                continue
            best[sq] = d.conf
            probs[sq] = 0.02 / 12
            probs[sq, 1 + d.cls] = d.conf
            probs[sq, 0] = max(0.02, 1 - d.conf)
        probs /= probs.sum(axis=1, keepdims=True)
        return _DetectionObservation(probs)

    def expected(self, board: Board) -> np.ndarray:
        return piece_classes_of(board)
