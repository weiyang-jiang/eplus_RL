"""
   @Author: Weiyang Jiang
   @Date: 2021-11-18 12:47:12
"""
import os

import h5py
import torch
from torch import optim
from torch.utils import data
import numpy as np
import pandas as pd
import glob

from torch.utils.data import DataLoader
from tqdm import tqdm

from valueBase.PQN.PQN_il_network import IL_Network


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
        hd5_list = glob.glob("./data_expert/*.h5")

        for hd5_path in hd5_list:
            h5 = h5py.File(hd5_path, "r")
            state = list(np.squeeze(h5["state"]))
            action = list(np.array(h5["action"]))
            self.state.extend(state)
            self.action.extend(action)
            h5.close()

    def __getitem__(self, index):
        action = list(self.action)[index-1]
        state = torch.Tensor(self.state)[index-1]
        return state, action

    def __len__(self):
        return len(self.state)

def save_model(dqn, window_len):
    model_file_path = os.path.join("./expert_model")
    if not os.path.exists(model_file_path):
        os.mkdir(model_file_path)
    torch.save(dqn.state_dict(), os.path.join(model_file_path, f'expert_dqn_{window_len}.pth'))

if __name__ == '__main__':
    window_len = 35
    ilDateset = IL_dataset()
    device = torch.device("cpu")
    agent = IL_Network(73*window_len, 25, 128).to(device)
    agent_optimiser = optim.RMSprop(agent.parameters(), lr=3.0e-05, alpha=0.9)
    behavioural_cloning_update(agent, ilDateset, agent_optimiser, batch_size=2)
    save_model(agent, window_len)