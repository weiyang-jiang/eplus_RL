"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 16:00:25
"""

from typing import Dict
import torch.nn.functional as F
import numpy as np
import torch

from valueBase.Asyn_agent_main import AgentMain, AsynAgentMain
from valueBase.Asyn_agent_test_main import Agent_test
from valueBase.n_step_DQN.n_step_replaybuffer import ReplayBuffer



class AsynNstep_Agent(AsynAgentMain):

    def complie_agent(self):
        self.Agent_test = Agent_test
        self.agent = AgentMain
        self.memory = ReplayBuffer(
            self.hist_state_dim, self.memory_size, self.batch_size, n_step=1
        )
        self.memory_n = ReplayBuffer(
            self.hist_state_dim, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma
        )
        self.add_hparams_dict["n_step"] = self.n_step

    def compute_dqn_loss_n_step(self, samples: Dict[str, np.ndarray], gamma) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(
            dim=1, keepdim=True
        )[0].detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()
        indices = samples["indices"]
        loss = self.compute_dqn_loss_n_step(samples, self.gamma)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance.

        samples = self.memory_n.sample_batch_from_idxs(indices)
        gamma = self.gamma ** self.n_step
        n_loss = self.compute_dqn_loss_n_step(samples, gamma)
        loss += n_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def memory_func(self, transitions):
        one_step_transition = [self.memory_n.store(*transition) for transition in transitions]
        if one_step_transition[0]:
            [self.memory.store(*transition) for transition in transitions]