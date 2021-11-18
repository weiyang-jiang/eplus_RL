"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 09:21:21
"""
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)