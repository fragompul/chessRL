"""Board geometry: corners, homography, rectification, orientation.

Conventions
-----------
* Image coordinates: x right, y down (OpenCV).
* A *calibration* stores the four playing-area corners in the order **a1, h1, h8, a8** as they
  appear in the image. Rectification maps a1 -> bottom-left, so the rectified board always has
  White at the bottom, rank 8 at the top, file a on the left: square (file, rank) occupies
  pixels x in [file*S, (file+1)*S), y in [(7-rank)*S, (8-rank)*S).
* Square index = rank * 8 + file (a1 = 0), as everywhere else in the code base.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BOARD_PX = 512  # side of the rectified board image
SQ = BOARD_PX // 8


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left (image view)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


_DST = np.array([[0, BOARD_PX], [BOARD_PX, BOARD_PX], [BOARD_PX, 0], [0, 0]], dtype=np.float32)


@dataclass
class BoardCalibration:
    """Camera <-> board mapping for a fixed camera."""

    corners: np.ndarray  # (4,2) float32: a1, h1, h8, a8 in image pixels

    def __post_init__(self) -> None:
        self.corners = np.asarray(self.corners, dtype=np.float32).reshape(4, 2)
        self.H = cv2.getPerspectiveTransform(self.corners, _DST)  # image -> rectified board
        self.H_inv = np.linalg.inv(self.H)

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.H, (BOARD_PX, BOARD_PX), flags=cv2.INTER_AREA)

    def image_to_board(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    def board_to_image(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H_inv).reshape(-1, 2)

    def point_to_square(self, pt: tuple[float, float]) -> int | None:
        """Square index under an image point (e.g. the foot of a detected piece), or None."""
        x, y = self.image_to_board(np.array([pt]))[0]
        if not (0 <= x < BOARD_PX and 0 <= y < BOARD_PX):
            return None
        return (7 - int(y // SQ)) * 8 + int(x // SQ)

    def square_polygon(self, sq: int) -> np.ndarray:
        """Image-space polygon (4,2) of a square (for overlays)."""
        f, r = sq % 8, sq // 8
        x0, y0 = f * SQ, (7 - r) * SQ
        quad = np.array([[x0, y0], [x0 + SQ, y0], [x0 + SQ, y0 + SQ], [x0, y0 + SQ]], dtype=np.float32)
        return self.board_to_image(quad)

    def square_center(self, sq: int) -> tuple[float, float]:
        f, r = sq % 8, sq // 8
        c = self.board_to_image(np.array([[f * SQ + SQ / 2, (7 - r) * SQ + SQ / 2]]))[0]
        return float(c[0]), float(c[1])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"corners": self.corners.tolist()}))

    @classmethod
    def load(cls, path: str | Path) -> BoardCalibration:
        return cls(np.array(json.loads(Path(path).read_text())["corners"], dtype=np.float32))


def square_crop(board_img: np.ndarray, sq: int, frac: float = 0.5) -> np.ndarray:
    """Central part (`frac` of the side) of a square in a rectified board image."""
    f, r = sq % 8, sq // 8
    m = int(SQ * (1 - frac) / 2)
    x0, y0 = f * SQ + m, (7 - r) * SQ + m
    return board_img[y0 : y0 + SQ - 2 * m, x0 : x0 + SQ - 2 * m]


def square_features(board_img: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """(64, 4) per-square features: mean B, G, R and grey std of the central crop."""
    out = np.zeros((64, 4), dtype=np.float32)
    for sq in range(64):
        c = square_crop(board_img, sq, frac)
        out[sq, :3] = c.reshape(-1, 3).mean(axis=0)
        out[sq, 3] = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).std()
    return out


# Reference ring: the top and side margins of the square. The bottom margin is left out
# because a piece's wide base reaches into it.
_RING = np.zeros((SQ, SQ), dtype=bool)
_RING[4:12, 4 : SQ - 4] = True
_RING[4:46, 4:10] = True
_RING[4:46, SQ - 10 : SQ - 4] = True
_CENTER = np.zeros((SQ, SQ), dtype=bool)
_CENTER[16 : SQ - 16, 16 : SQ - 16] = True


PATCH_ROWS, PATCH_COLS = 8, 6
FEATURE_DIM = 3 + PATCH_ROWS * PATCH_COLS


def square_features_rel(board_img: np.ndarray) -> np.ndarray:
    """(64, 51) lighting-robust descriptor of each square, relative to its own border ring.

    [mean B,G,R of the centre minus the ring's] + [an 8x6 luminance patch of the piece zone
    minus the ring's median]. A black piece is a dark blob whatever the square; a white piece
    is bright on dark squares and nearly invisible (outline only) on light ones; an empty
    square is flat zero. The patch also encodes the silhouette.
    """
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = np.zeros((64, FEATURE_DIM), dtype=np.float32)
    for sq in range(64):
        f, r = sq % 8, sq // 8
        y0, x0 = (7 - r) * SQ, f * SQ
        cell = board_img[y0 : y0 + SQ, x0 : x0 + SQ]
        g = gray[y0 : y0 + SQ, x0 : x0 + SQ]
        ring_med = float(np.median(g[_RING]))
        out[sq, :3] = cell[_CENTER].mean(axis=0) - cell[_RING].mean(axis=0)
        zone = g[8:60, 10:54]
        patch = cv2.resize(zone, (PATCH_COLS, PATCH_ROWS), interpolation=cv2.INTER_AREA)
        out[sq, 3:] = (patch - ring_med).ravel()
    return out


def is_light_square(sq: int) -> bool:
    return (sq % 8 + sq // 8) % 2 == 1  # a1 (0) is dark


def detect_white_side(rect: np.ndarray) -> int:
    """Given a rectified board (any rotation) showing the *starting position*, return the side
    White occupies: 0 = bottom, 1 = left, 2 = top, 3 = right.

    White pieces are brighter than empty squares of the same colour, black pieces darker; the
    side whose two outer rows are brightest relative to the (empty) centre is White's.
    """
    lum = square_features(rect)[:, :3].mean(axis=1).reshape(8, 8)[::-1]  # [row, col], row 0 = top
    base = {}
    for light in (0, 1):
        vals = [lum[r, c] for r in range(3, 5) for c in range(8) if (r + c) % 2 == light]
        base[light] = float(np.median(vals))
    dev = np.array([[lum[r, c] - base[(r + c) % 2] for c in range(8)] for r in range(8)])
    sides = {
        0: dev[6:8, :].mean(),
        1: dev[:, 0:2].mean(),
        2: dev[0:2, :].mean(),
        3: dev[:, 6:8].mean(),
    }
    return max(sides, key=sides.get)


def rotate_corners_for_white(corners_tl_tr_br_bl: np.ndarray, white_side: int) -> np.ndarray:
    """Corners ordered (tl, tr, br, bl) as seen in the image -> canonical (a1, h1, h8, a8).

    `white_side`: 0 bottom, 1 left, 2 top, 3 right. With White at the bottom, a1 is the
    bottom-left corner; for the other sides the same four corners are just cyclically shifted
    (e.g. White on the left: a1 is the top-left corner as seen from White's seat).
    """
    tl, tr, br, bl = corners_tl_tr_br_bl
    cycle = [bl, br, tr, tl]  # a1, h1, h8, a8 when White sits at the bottom
    shift = (-white_side) % 4
    return np.array(cycle[shift:] + cycle[:shift], dtype=np.float32)
