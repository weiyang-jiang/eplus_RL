"""
   @Author: Weiyang Jiang
   @Date: 2021-10-24 00:33:53
"""

import os, sys, glob


from valueBase.rainbow.agent_v1_test import Rainbow_Agent_test

srcPath = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
src_list = glob.glob(srcPath + "/*")
sys.path.append(srcPath)
for src_path in src_list:
    sys.path.append(src_path)


from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
from PrioritizedReplayBuffer_v1 import PrioritizedReplayBuffer
from ReplayBuffer_v1 import ReplayBuffer
from Network_v1 import Network


from valueBase.Asyn_agent_main import AgentMain, AsynAgentMain

class Rainbow_Agent(AgentMain):

    def select_action(self, state: np.array, epsilon, dqn) -> Dict:
        """Select an action from the input state."""
        # epsilon greedy policy
        action_raw_idx = dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()  # 把state值传入dqn神经网络中
        action_raw_idx = action_raw_idx.detach().cpu().numpy()
        action_raw_tup = self.action_space[action_raw_idx]


        action_stpt_prcd, action_effect_idx = self.action_func(action_raw_tup, action_raw_idx, self._raw_state_limits,
                                                               self.action_limits, state, self.local_logger,
                                                               is_show_debug=False)
        selected_action = action_stpt_prcd

        return {self.env_name: selected_action}

class AsynRainbow_Agent(AsynAgentMain):

    def epsilon_update(self, frame_idx):
        return self.epsilon

    def complie_agent(self):
        self.agent = Rainbow_Agent
        self.Agent_test = Rainbow_Agent_test
        self.memory = PrioritizedReplayBuffer(
            self.hist_state_dim, self.memory_size, self.batch_size, alpha=self.alpha
        )
        self.memory_n = ReplayBuffer(
            self.hist_state_dim, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma
        )
        # Categorical 1_DQN_relpayBuffer_target parameters
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        self.add_hparams_dict["Noise net std"] = self.noise_net_std
        self.add_hparams_dict["beta"] = self.beta
        self.add_hparams_dict["prior_eps"] = self.prior_eps
        self.add_hparams_dict["v_min"] = self.v_min
        self.add_hparams_dict["v_max"] = self.v_max
        self.add_hparams_dict["atom_size"] = self.atom_size
        self.add_hparams_dict["n_step"] = self.n_step

    def complie_dqn(self):
        self.dqn = Network(
            self.hist_state_dim, self.action_dim, self.atom_size, self.hidden_size, self.noise_net_std, self.support
        ).to(self.device)
        self.dqn_target = Network(
            self.hist_state_dim, self.action_dim, self.atom_size, self.hidden_size, self.noise_net_std, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=self.eps)

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self.compute_dqn_loss_rainbow(samples, self.gamma)

        # PER: importance sampling before average
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.

        gamma = self.gamma ** self.n_step
        samples = self.memory_n.sample_batch_from_idxs(indices)
        elementwise_loss_n_loss = self.compute_dqn_loss_rainbow(samples, gamma)
        elementwise_loss += elementwise_loss_n_loss

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def compute_dqn_loss_rainbow(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical 1_DQN_relpayBuffer_target algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double 1_DQN_relpayBuffer_target
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def Prioritized(self, frame_idx, num_frames):
        fraction = min(frame_idx / num_frames, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

    def memory_func(self, transitions):
        one_step_transition = [self.memory_n.store(*transition) for transition in transitions]
        if one_step_transition[0]:
            [self.memory.store(*transition) for transition in transitions]