"""Replay buffer of self-play positions, stored compactly and persisted to disk.

Per position: 64 int8 pieces + 7 int16 meta (see encoding.py), a sparse policy target
(up to K=64 action indices + float16 probabilities) and the game outcome z for the player
to move. ~340 bytes each, so 500k positions fit in ~170 MB.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .actions import FLIP_ACTION, NUM_ACTIONS
from .encoding import META_LEN, planes_from_compact

K = 64


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.pieces = np.zeros((capacity, 64), dtype=np.int8)
        self.meta = np.zeros((capacity, META_LEN), dtype=np.int16)
        self.pol_idx = np.full((capacity, K), -1, dtype=np.int16)
        self.pol_p = np.zeros((capacity, K), dtype=np.float16)
        self.z = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    def add(self, pieces, meta, pol_idx, pol_p, z) -> None:
        n = len(z)
        if n == 0:
            return
        if n > self.capacity:  # keep the most recent positions
            sl = slice(n - self.capacity, n)
            pieces, meta, pol_idx, pol_p, z = (a[sl] for a in (pieces, meta, pol_idx, pol_p, z))
            n = self.capacity
        pos = (self.ptr + np.arange(n)) % self.capacity
        self.pieces[pos], self.meta[pos] = pieces, meta
        self.pol_idx[pos], self.pol_p[pos], self.z[pos] = pol_idx, pol_p, z
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.total_added += n

    def sample(self, batch_size: int, rng: np.random.Generator, augment: bool = True):
        """-> (planes [B,20,8,8], policy idx [B,K] int64, policy prob [B,K], z [B]) as tensors."""
        ix = rng.integers(0, self.size, size=batch_size)
        pieces, meta = self.pieces[ix].copy(), self.meta[ix].copy()
        idx = self.pol_idx[ix].astype(np.int64)
        if augment:
            # A left-right mirror is a symmetry of chess when nobody can castle.
            flip = (rng.random(batch_size) < 0.5) & (meta[:, :4].sum(axis=1) == 0)
            if flip.any():
                pieces[flip] = pieces[flip][:, np.arange(64) ^ 7]
                ep = meta[flip, 4]
                meta[flip, 4] = np.where(ep >= 0, ep ^ 7, ep)
                f = idx[flip]
                idx[flip] = np.where(f >= 0, FLIP_ACTION[np.clip(f, 0, NUM_ACTIONS - 1)], -1)
        planes = torch.from_numpy(planes_from_compact(pieces, meta))
        return (planes, torch.from_numpy(idx), torch.from_numpy(self.pol_p[ix].astype(np.float32)),
                torch.from_numpy(self.z[ix]))

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, pieces=self.pieces[: self.size], meta=self.meta[: self.size],
                         pol_idx=self.pol_idx[: self.size], pol_p=self.pol_p[: self.size],
                         z=self.z[: self.size],
                         state=np.array([self.capacity, self.size, self.ptr, self.total_added]))
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: str | Path, capacity: int | None = None) -> ReplayBuffer:
        with np.load(path) as d:
            cap0, size, ptr, total = (int(x) for x in d["state"])
            buf = cls(capacity or cap0)
            n = min(size, buf.capacity)
            # Saved arrays are in ring order; keep the newest `n` entries in chronological order.
            order = (np.arange(size) + (ptr if size == cap0 else 0)) % max(size, 1)
            order = order[size - n:] if size else order
            buf.pieces[:n], buf.meta[:n] = d["pieces"][order], d["meta"][order]
            buf.pol_idx[:n], buf.pol_p[:n], buf.z[:n] = d["pol_idx"][order], d["pol_p"][order], d["z"][order]
            buf.size, buf.ptr, buf.total_added = n, n % buf.capacity, total
        return buf
