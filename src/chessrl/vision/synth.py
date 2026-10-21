"""Synthetic board photos with ground truth, for tests, demos and YOLO training data.

Boards are drawn flat in "board space" (pieces as simple silhouettes that lean slightly upward,
mimicking height seen from an oblique camera), then warped by a random perspective camera,
lit unevenly, blurred and noised, over a cluttered background. A moving "hand" can occlude
squares, as in real recordings.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..engine import Board, Game
from ..engine.constants import KING, PIECE_LETTERS, square_name

CLASS_NAMES = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]  # YOLO class order
S = 80  # square size in board space
M = 28  # frame width
CANVAS = 8 * S + 2 * M

_HEIGHT = {1: 0.45, 2: 0.7, 3: 0.8, 4: 0.62, 5: 0.95, 6: 1.05}


def piece_class(p: int) -> int:
    t = abs(p) - 1
    return t if p > 0 else 6 + t


@dataclass
class CameraSpec:
    size: tuple[int, int] = (960, 720)  # (w, h)
    scale: float = 0.78  # board height as a fraction of the image height
    tilt: float = 0.12  # perspective strength (0 = top-down)
    roll: float = 0.0  # rotation about the optical axis, degrees (0 = white at the bottom)
    shift: tuple[float, float] = (0.0, 0.0)  # fraction of image size
    noise: float = 4.0
    blur: float = 0.8
    light: float = 0.12  # strength of the lighting gradient
    seed: int = 0

    @classmethod
    def random(cls, rng: np.random.Generator, any_rotation: bool = False) -> CameraSpec:
        return cls(
            scale=float(rng.uniform(0.62, 0.86)), tilt=float(rng.uniform(0.0, 0.18)),
            roll=float(rng.choice([0, 90, 180, 270]) + rng.uniform(-6, 6)) if any_rotation
            else float(rng.uniform(-6, 6)),
            shift=(float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.04, 0.04))),
            noise=float(rng.uniform(2, 6)), blur=float(rng.uniform(0.4, 1.1)),
            light=float(rng.uniform(0.05, 0.2)), seed=int(rng.integers(1 << 30)),
        )


@dataclass
class BoardStyle:
    light: tuple[int, int, int] = (190, 214, 238)  # BGR
    dark: tuple[int, int, int] = (96, 140, 176)
    frame: tuple[int, int, int] = (40, 70, 110)

    @classmethod
    def random(cls, rng: np.random.Generator) -> BoardStyle:
        hue = rng.integers(0, 3)
        presets = [
            ((190, 214, 238), (96, 140, 176)),  # classic wood
            ((200, 225, 205), (95, 140, 100)),  # green
            ((225, 215, 200), (150, 110, 80)),  # blue-ish
        ]
        light, dark = presets[int(hue)]
        j = rng.integers(-12, 13, size=3)
        return cls(tuple(int(np.clip(c + d, 0, 255)) for c, d in zip(light, j)),
                   tuple(int(np.clip(c + d, 0, 255)) for c, d in zip(dark, j)), (40, 70, 110))


@dataclass
class SynthFrame:
    image: np.ndarray  # BGR uint8
    corners: np.ndarray  # (4,2) a1, h1, h8, a8 in image pixels (white-at-bottom order)
    fen: str
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)  # YOLO
    hand: bool = False


# --------------------------------------------------------------------------- drawing
def _square_origin(sq: int) -> tuple[int, int]:
    f, r = sq % 8, sq // 8
    return M + f * S, M + (7 - r) * S


def _draw_piece(img: np.ndarray, piece: int, sq: int) -> tuple[int, int, int, int]:
    """Draw a piece; returns its bounding box (x0, y0, x1, y1) in board space."""
    x0, y0 = _square_origin(sq)
    cx, cy = x0 + S / 2, y0 + S / 2
    white = piece > 0
    t = abs(piece)
    fill = (236, 236, 236) if white else (36, 36, 36)
    edge = (70, 70, 70) if white else (205, 205, 205)
    h = _HEIGHT[t] * S * 0.55
    base_y = cy + 0.28 * S
    top_y = base_y - h
    pts_all: list[tuple[float, float]] = []

    def poly(points, close=True):
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [pts], fill)
        cv2.polylines(img, [pts], close, edge, 2, cv2.LINE_AA)
        pts_all.extend(map(tuple, points))

    def circ(c, r):
        cv2.circle(img, (int(c[0]), int(c[1])), int(r), fill, -1, cv2.LINE_AA)
        cv2.circle(img, (int(c[0]), int(c[1])), int(r), edge, 2, cv2.LINE_AA)
        pts_all.extend([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)])

    # base disc
    cv2.ellipse(img, (int(cx), int(base_y)), (int(0.34 * S), int(0.13 * S)), 0, 0, 360, fill, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(cx), int(base_y)), (int(0.34 * S), int(0.13 * S)), 0, 0, 360, edge, 2, cv2.LINE_AA)
    pts_all.extend([(cx - 0.34 * S, base_y - 0.13 * S), (cx + 0.34 * S, base_y + 0.13 * S)])
    w0 = 0.30 * S
    if t == 1:  # pawn
        poly([(cx - w0 * 0.8, base_y), (cx + w0 * 0.8, base_y), (cx + 0.1 * S, top_y + 0.1 * S), (cx - 0.1 * S, top_y + 0.1 * S)])
        circ((cx, top_y + 0.02 * S), 0.13 * S)
    elif t == 4:  # rook
        poly([(cx - w0, base_y), (cx + w0, base_y), (cx + w0, top_y), (cx - w0, top_y)])
        for k in (-1, 0, 1):
            poly([(cx + k * 0.2 * S - 0.07 * S, top_y), (cx + k * 0.2 * S + 0.07 * S, top_y),
                  (cx + k * 0.2 * S + 0.07 * S, top_y - 0.1 * S), (cx + k * 0.2 * S - 0.07 * S, top_y - 0.1 * S)])
    elif t == 2:  # knight
        poly([(cx - w0, base_y), (cx + w0, base_y), (cx + w0 * 0.7, base_y - h * 0.5), (cx + w0 * 0.3, top_y),
              (cx - 0.05 * S, top_y - 0.05 * S), (cx - 0.38 * S, top_y + h * 0.35), (cx - 0.15 * S, base_y - h * 0.45)])
        cv2.circle(img, (int(cx - 0.02 * S), int(top_y + 0.1 * S)), 2, edge, -1)
    elif t == 3:  # bishop
        poly([(cx - w0 * 0.85, base_y), (cx + w0 * 0.85, base_y), (cx + 0.12 * S, base_y - h * 0.55), (cx + 0.1 * S, top_y + 0.1 * S),
              (cx, top_y - 0.04 * S), (cx - 0.1 * S, top_y + 0.1 * S), (cx - 0.12 * S, base_y - h * 0.55)])
        cv2.line(img, (int(cx - 0.05 * S), int(top_y + 0.15 * S)), (int(cx + 0.07 * S), int(top_y + 0.05 * S)), edge, 2)
    elif t == 5:  # queen
        poly([(cx - w0, base_y), (cx + w0, base_y), (cx + 0.2 * S, top_y + 0.05 * S), (cx - 0.2 * S, top_y + 0.05 * S)])
        for k in (-2, -1, 0, 1, 2):
            circ((cx + k * 0.1 * S, top_y - 0.02 * S - (0.03 * S if k == 0 else 0)), 0.05 * S)
    else:  # king
        poly([(cx - w0, base_y), (cx + w0, base_y), (cx + 0.18 * S, top_y + 0.05 * S), (cx - 0.18 * S, top_y + 0.05 * S)])
        poly([(cx - 0.035 * S, top_y + 0.05 * S), (cx + 0.035 * S, top_y + 0.05 * S), (cx + 0.035 * S, top_y - 0.16 * S), (cx - 0.035 * S, top_y - 0.16 * S)])
        poly([(cx - 0.11 * S, top_y - 0.08 * S), (cx + 0.11 * S, top_y - 0.08 * S), (cx + 0.11 * S, top_y - 0.13 * S), (cx - 0.11 * S, top_y - 0.13 * S)])
    xs, ys = zip(*pts_all)
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def render_board_space(board: Board, style: BoardStyle, hide: set[int] | None = None,
                       hand_at: tuple[float, float] | None = None, empty: bool = False):
    """Top-down canvas of a position. Returns (canvas BGR, {sq: bbox})."""
    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    img[:] = style.frame
    for sq in range(64):
        x0, y0 = _square_origin(sq)
        light = (sq % 8 + sq // 8) % 2 == 1
        img[y0 : y0 + S, x0 : x0 + S] = style.light if light else style.dark
    boxes: dict[int, tuple[int, int, int, int]] = {}
    if not empty:
        # draw far pieces (high rank) first so nearer pieces overlap them
        for sq in sorted(range(64), key=lambda s: -(s // 8)):
            p = board.squares[sq]
            if p and not (hide and sq in hide):
                boxes[sq] = _draw_piece(img, p, sq)
    if hand_at is not None:
        hx, hy = hand_at
        pts = np.array([(hx - 0.28 * S, hy + 0.9 * S), (hx - 0.3 * S, hy - 0.1 * S), (hx - 0.1 * S, hy - 0.4 * S),
                        (hx + 0.15 * S, hy - 0.4 * S), (hx + 0.32 * S, hy - 0.05 * S), (hx + 0.3 * S, hy + 0.9 * S)],
                       dtype=np.int32)
        cv2.fillPoly(img, [pts], (140, 170, 215))
        cv2.polylines(img, [pts], True, (100, 130, 175), 2, cv2.LINE_AA)
    return img, boxes


def _camera_matrix(spec: CameraSpec) -> np.ndarray:
    w, h = spec.size
    side = spec.scale * h
    cx, cy = w / 2 + spec.shift[0] * w, h / 2 + spec.shift[1] * h
    half = side / 2
    # destination quad for the *whole canvas*, then perspective + roll
    sc = side / CANVAS
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)
    squeeze = spec.tilt * half  # the far edge (top) is narrower
    corners[0, 0] += squeeze
    corners[1, 0] -= squeeze
    corners[0, 1] += squeeze * 0.6
    corners[1, 1] += squeeze * 0.6
    a = math.radians(spec.roll)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
    dst = corners @ rot.T + np.array([cx, cy], dtype=np.float32)
    src = np.array([[0, 0], [CANVAS, 0], [CANVAS, CANVAS], [0, CANVAS]], dtype=np.float32)
    del sc
    return cv2.getPerspectiveTransform(src, dst)


def _background(spec: CameraSpec, rng: np.random.Generator) -> np.ndarray:
    w, h = spec.size
    base = rng.integers(60, 200, size=3).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 1 + 0.25 * ((xx / w) - 0.5) * rng.uniform(-1, 1) + 0.25 * ((yy / h) - 0.5) * rng.uniform(-1, 1)
    bg = np.clip(base[None, None, :] * grad[..., None], 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(2, 6))):  # clutter: a few rectangles near the borders
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(bg, (x0, y0), (x0 + int(rng.integers(30, 160)), y0 + int(rng.integers(20, 120))),
                      tuple(int(c) for c in rng.integers(30, 230, size=3)), -1)
    return bg


def render_frame(
    board: Board,
    spec: CameraSpec | None = None,
    style: BoardStyle | None = None,
    hide: set[int] | None = None,
    hand_at: tuple[float, float] | None = None,
    empty: bool = False,
    rng: np.random.Generator | None = None,
) -> SynthFrame:
    spec = spec or CameraSpec()
    rng = rng or np.random.default_rng(spec.seed)
    style = style or BoardStyle()
    canvas, boxes = render_board_space(board, style, hide, hand_at, empty)
    Hm = _camera_matrix(spec)
    w, h = spec.size
    bg = _background(spec, np.random.default_rng(spec.seed))  # fixed per camera: a static scene
    warped = cv2.warpPerspective(canvas, Hm, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((CANVAS, CANVAS), 255, np.uint8), Hm, (w, h), flags=cv2.INTER_NEAREST)
    img = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lrng = np.random.default_rng(spec.seed + 1)
    ang = lrng.uniform(0, 2 * math.pi)
    img *= (1 + spec.light * np.cos(ang) * (xx / w - 0.5) * 2 + spec.light * np.sin(ang) * (yy / h - 0.5) * 2)[..., None]
    if spec.blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), spec.blur)
    img += rng.normal(0, spec.noise, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    corner_board = np.array([[M, M + 8 * S], [M + 8 * S, M + 8 * S], [M + 8 * S, M], [M, M]], dtype=np.float32)
    corners = cv2.perspectiveTransform(corner_board.reshape(-1, 1, 2), Hm).reshape(4, 2)
    yolo = []
    for sq, (bx0, by0, bx1, by1) in boxes.items():
        quad = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], dtype=np.float32).reshape(-1, 1, 2)
        pq = cv2.perspectiveTransform(quad, Hm).reshape(4, 2)
        x0, y0 = np.clip(pq.min(axis=0), 0, [w, h])
        x1, y1 = np.clip(pq.max(axis=0), 0, [w, h])
        if x1 - x0 > 4 and y1 - y0 > 4:
            yolo.append((piece_class(board.squares[sq]), (x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                         (x1 - x0) / w, (y1 - y0) / h))
    return SynthFrame(img, corners.astype(np.float32), board.fen(), yolo, hand_at is not None)


# ---------------------------------------------------------------------------- videos
def _sq_center(sq: int) -> tuple[float, float]:
    x0, y0 = _square_origin(sq)
    return x0 + S / 2, y0 + S / 2


def game_frames(
    game: Game,
    spec: CameraSpec | None = None,
    style: BoardStyle | None = None,
    idle_frames: int = 8,
    seed: int = 0,
    with_hand: bool = True,
) -> Iterator[SynthFrame]:
    """Frames of a game being played by hand: rest, hand reaches in, piece lifted, placed, rest."""
    spec = spec or CameraSpec()
    style = style or BoardStyle()
    rng = np.random.default_rng(seed)
    board = Board(game.start_fen)

    def frame(b, **kw):
        return render_frame(b, spec, style, rng=rng, **kw)

    for _ in range(idle_frames):
        yield frame(board)
    for move in game.moves:
        frm, to = move.from_sq, move.to_sq
        a, b = _sq_center(frm), _sq_center(to)
        if with_hand:
            for t in (0.3, 0.7):  # hand approaches the piece
                yield frame(board, hand_at=(a[0] + (1 - t) * 90, a[1] + (1 - t) * 120))
            yield frame(board, hand_at=a)
            lifted = {frm}
            if board.squares[to]:
                lifted.add(to)
            for t in (0.25, 0.5, 0.75):  # carrying the piece
                yield frame(board, hide=lifted, hand_at=(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        board.push(move)
        if with_hand:
            yield frame(board, hand_at=b)
            yield frame(board, hand_at=(b[0] + 70, b[1] + 110))
        for _ in range(idle_frames):
            yield frame(board)


def write_video(frames: Iterator[SynthFrame], path: str | Path, fps: int = 10) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for f in frames:
        if writer is None:
            h, w = f.image.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(f.image)
    if writer is not None:
        writer.release()
    return path


def yolo_label_text(frame: SynthFrame) -> str:
    return "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in frame.boxes)


__all__ = ["CLASS_NAMES", "BoardStyle", "CameraSpec", "SynthFrame", "game_frames", "render_frame",
           "write_video", "yolo_label_text", "KING", "PIECE_LETTERS", "square_name"]
