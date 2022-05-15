"""
   @Author: Weiyang Jiang
   @Date: 2021-11-17 13:33:01
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)  # (73, 128)
        self.mu_layer = nn.Linear(128, out_dim)  # (128, 22)
        self.log_std_layer = nn.Linear(128, out_dim)  # (128, 22)
        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))  # (1, 73)*(73, 128)=(1, 128)
        mu = torch.tanh(self.mu_layer(x)) * 2  # (1, 128)*(128, 22)=(1, 22)
        log_std = F.softplus(self.log_std_layer(x))  # 相当于一个平滑版本的Relu函数  (1, 22)
        std = torch.exp(log_std)  # (1, 22)

        dist = Normal(mu, std)  # (1, 22) 的正态分布， 让整个数据呈现正太分布 mu为均值，
        # std为标准差 mu - 2*std, mu - std, mu , mu + std, mu+2*std
        action = dist.sample()  # (1, 22)

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.out = nn.Linear(128, 1)

        initialize_uniformly(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        value = self.out(x)

        return value
