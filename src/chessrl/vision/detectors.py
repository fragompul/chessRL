"""Piece detectors: YOLO wrapper (optional dependency) and an oracle for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .estimator import Detection
from .synth import CLASS_NAMES, SynthFrame


class YoloPieceDetector:
    """Ultralytics YOLO model trained on the 12 piece classes (see scripts/train_yolo.py).

    Requires `pip install ultralytics`; imported lazily so the rest of the package does not.
    """

    def __init__(self, weights: str | Path, conf: float = 0.35, imgsz: int = 960, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("YoloPieceDetector needs `pip install ultralytics`") from e
        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover - needs weights
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        for box, c, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.cpu().numpy(), strict=False):
            out.append(Detection(int(cls), float(c), tuple(float(v) for v in box)))
        return out


class OracleDetector:
    """Returns the ground-truth boxes of the synthetic frame being processed (test double)."""

    def __init__(self) -> None:
        self.current: SynthFrame | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        sf = self.current
        h, w = frame.shape[:2]
        out = []
        for cls, cx, cy, bw, bh in sf.boxes:
            out.append(Detection(cls, 0.95, ((cx - bw / 2) * w, (cy - bh / 2) * h,
                                             (cx + bw / 2) * w, (cy + bh / 2) * h)))
        return out


__all__ = ["CLASS_NAMES", "OracleDetector", "YoloPieceDetector"]
