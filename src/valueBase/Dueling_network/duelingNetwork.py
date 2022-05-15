"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 13:46:26
"""
import torch
from torch import nn
import torch.nn.functional as F


class Dueling_Network(nn.Module):
    # 这里面有两个神经网络
    def __init__(self, in_dim: int, out_dim: int, hidden_size):
        """Initialization."""
        super(Dueling_Network, self).__init__()

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
        )

        # set value layer
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)
        # advantage 和 value network都在使用同一个特征层
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q

    # Calculates the log probability of an action a with the policy π(·|s) given state s
    def log_prob(self, state, action):
        x = self.forward(state)  # (32, 25)
        x = F.softmax(x, dim=1)
        res = torch.log(x.gather(1, action))
        return res