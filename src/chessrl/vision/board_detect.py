"""Find the playing area in a camera frame and work out the board's orientation.

Strategies, tried in order:
  1. `findChessboardCornersSB` on the inner 7x7 corner grid (exact; works on empty boards and
     usually with all pieces on),
  2. the largest convex quadrilateral (board frame) refined to the checker pattern,
  3. manual corners supplied by the user (CLI `--corners`), always possible.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import (
    BOARD_PX,
    SQ,
    BoardCalibration,
    detect_white_side,
    order_corners,
    rotate_corners_for_white,
    square_features,
)


def corners_from_grid(gray: np.ndarray) -> np.ndarray | None:
    """Outer corners (tl, tr, br, bl) of the 8x8 area from the inner 7x7 corner grid."""
    found, pts = cv2.findChessboardCornersSB(gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not found:
        return None
    pts = pts.reshape(-1, 2)
    idx = np.array([(c + 1, r + 1) for r in range(7) for c in range(7)], dtype=np.float32)
    H, _ = cv2.findHomography(idx, pts, 0)
    if H is None:
        return None
    outer = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32).reshape(-1, 1, 2)
    return order_corners(cv2.perspectiveTransform(outer, H).reshape(4, 2))


def corner_candidates(bgr: np.ndarray, min_area: float = 0.10, max_candidates: int = 4) -> list[np.ndarray]:
    """Plausible board quadrilaterals (tl, tr, br, bl), largest first.

    Edges are closed morphologically, each large contour is replaced by its convex hull (so
    clutter overlapping the frame does not break it) and simplified to 4 points.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    h, w = gray.shape
    found: list[tuple[float, np.ndarray]] = []
    for lo, hi in ((40, 120), (20, 60), (60, 160)):
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < min_area * h * w * 0.5:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            quad = None
            for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    quad = approx.reshape(4, 2)
                    break
            if quad is None:
                quad = cv2.boxPoints(cv2.minAreaRect(hull))
            area = cv2.contourArea(quad.astype(np.float32))
            if area >= min_area * h * w:
                found.append((area, order_corners(quad)))
    found.sort(key=lambda t: -t[0])
    out: list[np.ndarray] = []
    for _, q in found:
        if all(np.abs(q - o).max() > 12 for o in out):
            out.append(q)
        if len(out) >= max_candidates:
            break
    return out


def corners_from_contour(bgr: np.ndarray, min_area: float = 0.10) -> np.ndarray | None:
    """Best board quadrilateral, chosen among candidates by checker-grid alignment."""
    cands = corner_candidates(bgr, min_area)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    return max(cands, key=lambda q: _grid_alignment(bgr, refine_to_playing_area(bgr, q, ((12.0, 9.0), (6.0, 5.0))), 5.0))


def _grid_alignment(bgr: np.ndarray, quad: np.ndarray, sigma: float = 1.2) -> float:
    """How well the rectified image's strong edges sit on the expected 8x8 grid lines.

    Checker boundaries produce sharp gradients exactly at multiples of SQ; a quad that is off by
    a few pixels (or includes the board frame) smears them. Pieces mostly sit on ranks 1-2 / 7-8,
    so vertical lines are scored on the empty middle ranks only.
    """
    cal = BoardCalibration(rotate_corners_for_white(quad, 0))
    g = cv2.cvtColor(cal.rectify(bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    gx = np.abs(g[:, 2:] - g[:, :-2])  # response of vertical edges, shifted by 1 px
    gy = np.abs(g[2:, :] - g[:-2, :])
    r = max(1, int(round(sigma)))
    mid = slice(2 * SQ, 6 * SQ)
    score = 0.0
    for k in range(0, 9):
        x = min(max(k * SQ - 1, 0), gx.shape[1] - 1)
        score += gx[mid, max(x - r, 0) : x + r + 1].max(axis=1).mean()
    for k in (0, 2, 3, 4, 5, 6, 8):
        y = min(max(k * SQ - 1, 0), gy.shape[0] - 1)
        score += gy[max(y - r, 0) : y + r + 1, SQ : 7 * SQ].max(axis=0).mean()
    return float(score)


def refine_to_playing_area(bgr: np.ndarray, quad: np.ndarray,
                           schedule=((12.0, 9.0), (6.0, 5.0), (3.0, 3.0), (1.5, 1.8), (0.75, 1.2))) -> np.ndarray:
    """Coarse-to-fine coordinate descent on the 8 corner coordinates (step px, blur sigma)."""
    best = quad.astype(np.float32).copy()
    for step, sigma in schedule:
        best_score = _grid_alignment(bgr, best, sigma)
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for axis in (0, 1):
                    for sign in (-1.0, 1.0):
                        trial = best.copy()
                        trial[i, axis] += sign * step
                        sc = _grid_alignment(bgr, trial, sigma)
                        if sc > best_score * 1.0005:
                            best, best_score, improved = trial, sc, True
    return best


def find_board(frame: np.ndarray, white_side: int | None = None) -> BoardCalibration | None:
    """Locate the board and decide orientation.

    `white_side` (0 bottom, 1 left, 2 top, 3 right) can be given when the frame is not the
    start position; otherwise it is detected from the piece brightness (needs the start position).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    quad = corners_from_grid(gray)
    if quad is None:
        outer = corners_from_contour(frame)
        if outer is None:
            return None
        quad = refine_to_playing_area(frame, outer)
    if white_side is None:
        white_side = pick_white_side(frame, quad)
    return BoardCalibration(rotate_corners_for_white(quad, white_side))


def pick_white_side(frame: np.ndarray, quad_tl_tr_br_bl: np.ndarray) -> int:
    """Orientation from a start-position frame: brightest outer side that also puts a dark a1
    in the corner (the checker parity rules out half of the options)."""
    rect = BoardCalibration(rotate_corners_for_white(quad_tl_tr_br_bl, 0)).rectify(frame)
    lum = square_features(rect)[:, :3].mean(axis=1).reshape(8, 8)[::-1]  # [row, col], row 0 = top
    base = {c: float(np.median([lum[r, x] for r in range(3, 5) for x in range(8) if (r + x) % 2 == c]))
            for c in (0, 1)}
    dev = np.array([[lum[r, c] - base[(r + c) % 2] for c in range(8)] for r in range(8)])
    side_score = {0: dev[6:8, :].mean(), 1: dev[:, 0:2].mean(), 2: dev[0:2, :].mean(), 3: dev[:, 6:8].mean()}
    # parity: with the right orientation, a3 (rank 3, file a) is a dark square
    def parity_ok(k: int) -> bool:
        cal = BoardCalibration(rotate_corners_for_white(quad_tl_tr_br_bl, k))
        r = square_features(cal.rectify(frame), frac=0.6)[:, :3].mean(axis=1)
        a3, b3 = r[16], r[17]
        return a3 < b3

    candidates = [k for k in range(4) if parity_ok(k)] or list(range(4))
    return max(candidates, key=lambda k: side_score[k])


__all__ = ["BOARD_PX", "SQ", "corners_from_contour", "corners_from_grid", "detect_white_side",
           "find_board", "pick_white_side", "refine_to_playing_area"]
