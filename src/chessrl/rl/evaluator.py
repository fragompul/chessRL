"""Batched network inference."""

from __future__ import annotations

import numpy as np
import torch

from .encoding import planes_from_compact
from .network import ChessNet


class NetEvaluator:
    """Wraps a network for fast, no-grad, batched evaluation of compact positions."""

    def __init__(self, net: ChessNet, device: str | torch.device = "cpu", max_batch: int = 256):
        self.net = net
        self.device = torch.device(device)
        self.max_batch = max_batch
        self.net.to(self.device)

    def __call__(self, pieces: np.ndarray, meta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """pieces [N,64] int8, meta [N,7] int16 -> (logits [N,4672] float32, values [N] float32)."""
        was_training = self.net.training
        self.net.eval()
        logits_out, values_out = [], []
        with torch.inference_mode():
            for s in range(0, len(pieces), self.max_batch):
                x = torch.from_numpy(planes_from_compact(pieces[s : s + self.max_batch],
                                                         meta[s : s + self.max_batch]))
                logits, values = self.net(x.to(self.device))
                logits_out.append(logits.float().cpu().numpy())
                values_out.append(values.float().cpu().numpy())
        if was_training:
            self.net.train()
        return np.concatenate(logits_out), np.concatenate(values_out)
