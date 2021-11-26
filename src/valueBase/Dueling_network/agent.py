"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 13:47:23
"""

from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from valueBase.Asyn_agent_main import AgentMain, AsynAgentMain
from valueBase.Dueling_network.agent_test import Dueling_Agent_test
from valueBase.Dueling_network.duelingNetwork import Dueling_Network
from valueBase.util.PrioritizedReplayBuffer import PrioritizedReplayBuffer


class AsynDuelingAgent(AsynAgentMain):

    def complie_agent(self):
        self.Agent_test = Dueling_Agent_test
        self.agent = AgentMain
        self.memory = PrioritizedReplayBuffer(
            self.hist_state_dim, self.memory_size, self.batch_size, self.alpha
        )
        self.add_hparams_dict["beta"] = self.beta
        self.add_hparams_dict["prior_eps"] = self.prior_eps

    def complie_dqn(self):
        # networks: dqn, dqn_target
        self.dqn = Dueling_Network(self.hist_state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn_target = Dueling_Network(self.hist_state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=self.eps)

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)  # 按照优先顺序从buffer中拿出transition
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)  # 输出这个batch中的权重参数
        indices = samples["indices"]  # 输出这个batch的所有索引列表

        # PER: importance sampling before average
        elementwise_loss = self.compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)  # 这里用权重值代表lr

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps  # pt = |loss| + e, e是一个很小的参数为了避免新的pt更新为0了
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
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

        a_star = self.dqn(next_state).argmax(dim=1, keepdim=True)
        next_q_value = self.dqn_target(next_state).gather(1, a_star).detach()

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss

    def Prioritized(self, frame_idx, num_frames):
        fraction = min(frame_idx / num_frames, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
