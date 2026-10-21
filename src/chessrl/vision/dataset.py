"""Synthetic YOLO dataset: random positions, cameras, lighting and board styles, with labels."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from ..engine import Board
from .synth import CLASS_NAMES, BoardStyle, CameraSpec, render_frame, yolo_label_text


def random_position(rng: random.Random, max_plies: int = 90) -> Board:
    """A plausible mid-game position: random legal moves, biased towards keeping material."""
    board = Board()
    for _ in range(rng.randint(0, max_plies)):
        moves = board.legal_moves()
        if not moves:
            break
        quiet = [m for m in moves if not board.squares[m.to_sq]]
        board.push(rng.choice(quiet if quiet and rng.random() < 0.7 else moves))
        if board.outcome() is not None:
            board.pop()
            break
    return board


def make_yolo_dataset(out: str | Path, images: int = 500, seed: int = 0, val_fraction: float = 0.1,
                      any_rotation: bool = False, progress=None) -> Path:
    """Write `out/images/{train,val}`, `out/labels/{train,val}` and `out/data.yaml` (Ultralytics format)."""
    out = Path(out)
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(images):
        split = "val" if rng.random() < val_fraction else "train"
        board = random_position(rng)
        spec = CameraSpec.random(nrng, any_rotation=any_rotation)
        frame = render_frame(board, spec, BoardStyle.random(nrng), rng=nrng)
        name = f"synth_{seed}_{i:06d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / split / f"{name}.txt").write_text(yolo_label_text(frame))
        if progress:
            progress(i + 1, images)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return out / "data.yaml"
