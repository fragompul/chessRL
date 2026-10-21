"""Policy/value residual network (AlphaZero-style), sized to run on a laptop."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .actions import NUM_ACTIONS
from .encoding import NUM_PLANES


@dataclass
class NetConfig:
    channels: int = 64
    blocks: int = 6
    value_hidden: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


class _ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class ChessNet(nn.Module):
    """planes [B,20,8,8] -> (policy logits [B,4672], value [B] in (-1, 1))."""

    def __init__(self, config: NetConfig | None = None):
        super().__init__()
        self.config = config or NetConfig()
        c = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU()
        )
        self.tower = nn.Sequential(*[_ResBlock(c) for _ in range(self.config.blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 73, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.Flatten(),
            nn.Linear(8 * 64, self.config.value_hidden), nn.ReLU(),
            nn.Linear(self.config.value_hidden, 1), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))
        p = self.policy_head(h)  # [B, 73, 8, 8]
        b = p.shape[0]
        # action index = from_square * 73 + move_type
        logits = p.reshape(b, 73, 64).permute(0, 2, 1).reshape(b, NUM_ACTIONS)
        return logits, self.value_head(h).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
