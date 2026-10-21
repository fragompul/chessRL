"""Atomic checkpoint I/O.

A *full* checkpoint (`latest.pt`) holds everything needed to resume training exactly:
network, optimiser, iteration counter, RNG state and running statistics. A *weights* file
(`iter_000050.pt`) holds just the network + its config and is what agents / levels load.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from .network import ChessNet, NetConfig


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_weights(path: str | Path, net: ChessNet, meta: dict | None = None) -> None:
    _atomic_save(
        {"net_config": net.config.to_dict(), "state_dict": net.state_dict(), "meta": meta or {}},
        Path(path),
    )


def load_net(path: str | Path, device: str | torch.device = "cpu") -> ChessNet:
    """Load a network from a weights file *or* a full checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = ChessNet(NetConfig(**blob["net_config"]))
    net.load_state_dict(blob["state_dict"])
    return net.to(device).eval()


def read_meta(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False).get("meta", {})


def save_full(path: str | Path, net: ChessNet, optimizer, state: dict) -> None:
    _atomic_save(
        {
            "net_config": net.config.to_dict(),
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state,
            "meta": {"iteration": state.get("iteration", 0)},
        },
        Path(path),
    )


def load_full(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
