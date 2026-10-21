import numpy as np

from chessrl.engine import Board
from chessrl.rl.checkpoint import save_weights
from chessrl.rl.encoding import encode_board
from chessrl.rl.export import export_onnx, verify_onnx
from chessrl.rl.network import ChessNet, NetConfig


def test_onnx_export_matches_pytorch_and_handles_batches(tmp_path):
    import onnxruntime as ort
    import torch

    net = ChessNet(NetConfig(channels=16, blocks=2, value_hidden=16)).eval()
    path = export_onnx(net, tmp_path / "m.onnx")
    assert verify_onnx(net, path) < 1e-4
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for n in (1, 5):
        x = np.stack([encode_board(Board())] * n)
        logits, value = sess.run(None, {"planes": x})
        assert logits.shape == (n, 4672) and value.shape == (n,)
    save_weights(tmp_path / "w.pt", net)
    assert torch.is_tensor(next(iter(net.parameters())))
