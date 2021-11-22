"""
   @Author: Weiyang Jiang
   @Date: 2021-11-18 12:47:12
"""
import os

import h5py
from torch import optim
from torch.utils import data
import numpy as np
import glob

from torch.utils.data import DataLoader
from tqdm import tqdm

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


def behavioural_cloning_update(agent, expert_trajectories, agent_optimiser, batch_size):
    expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True,
                                   num_workers=4)
    expert_dataloader = tqdm(expert_dataloader)
    for expert_transition in expert_dataloader:
        expert_state, expert_action = expert_transition  # (28, 32)   (8, 32)
        expert_state, expert_action = expert_state.to(device), expert_action.to(device)  # (28, 32)   (8, 32)
        agent_optimiser.zero_grad(set_to_none=True)
        behavioural_cloning_loss = -agent.log_prob(expert_state, expert_action).mean()  # Maximum likelihood objective
        behavioural_cloning_loss.backward()
        agent_optimiser.step()





class IL_dataset(data.Dataset):
    def __init__(self):
        self.state = []
        self.action = []
        hd5_list = glob.glob("./data_expert/Part1-Light-Pit-Train-v1.h5")

        for hd5_path in hd5_list:
            h5 = h5py.File(hd5_path, "r")
            state = list(np.squeeze(h5["state"]))
            action = list(np.array(h5["action"]))
            self.state.extend(state)
            self.action.extend(action)
            h5.close()

    def __getitem__(self, index):
        action = self.action[index-1]
        state = self.state[index-1]
        return np.array(state, dtype=np.float32), np.array(action, dtype=np.int64)

    def __len__(self):
        return len(self.state)

def save_model(dqn, window_len):
    model_file_path = os.path.join("./expert_model")
    if not os.path.exists(model_file_path):
        os.mkdir(model_file_path)
    torch.save(dqn.state_dict(), os.path.join(model_file_path, f'Dueling_expert_dqn_{window_len}.pth'))


if __name__ == '__main__':
    window_len = 1
    ilDateset = IL_dataset()
    device = torch.device("cpu")
    agent = Dueling_Network(73*window_len, 25, 128).to(device)
    agent_optimiser = optim.RMSprop(agent.parameters(), lr=3.0e-05, alpha=0.9)
    behavioural_cloning_update(agent, ilDateset, agent_optimiser, batch_size=16)
    save_model(agent, window_len)