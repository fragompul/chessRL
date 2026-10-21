"""Export a trained network to ONNX (for the in-browser demo, edge devices, other runtimes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .encoding import NUM_PLANES
from .network import ChessNet


def export_onnx(net: ChessNet, path: str | Path, opset: int = 17) -> Path:
    """Write `net` as ONNX: input `planes` [N,20,8,8] -> outputs `policy` [N,4672], `value` [N]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = net.cpu().eval()
    dummy = torch.zeros(2, NUM_PLANES, 8, 8)
    torch.onnx.export(
        net, dummy, str(path), input_names=["planes"], output_names=["policy", "value"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    return path


def verify_onnx(net: ChessNet, path: str | Path, atol: float = 1e-4) -> float:
    """Max abs difference between PyTorch and ONNX Runtime outputs on random inputs."""
    import onnxruntime as ort

    net = net.cpu().eval()
    x = torch.rand(3, NUM_PLANES, 8, 8)
    with torch.no_grad():
        logits, value = net(x)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    o_logits, o_value = sess.run(None, {"planes": x.numpy()})
    err = max(float(np.abs(o_logits - logits.numpy()).max()), float(np.abs(o_value - value.numpy()).max()))
    if err > atol:
        raise AssertionError(f"ONNX output differs from PyTorch by {err:.2e}")
    return err
