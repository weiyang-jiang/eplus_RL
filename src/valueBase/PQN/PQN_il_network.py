"""
   @Author: Weiyang Jiang
   @Date: 2021-11-18 23:51:17
"""
import torch
from torch import nn
import torch.nn.functional as F


class IL_Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int):
        """Initialization."""
        super(IL_Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = self.layers(x)
        return x

    # Calculates the log probability of an action a with the policy π(·|s) given state s
    def log_prob(self, state, action):
        x = self.forward(state)  # (32, 25)
        x = F.softmax(x, dim=1)
        res = torch.log(x.gather(1, action))
        return res