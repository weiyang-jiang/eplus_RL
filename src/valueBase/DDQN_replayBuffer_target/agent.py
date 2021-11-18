"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 09:21:04
"""
from typing import Dict

import numpy as np
import torch

import torch.nn.functional as F


from valueBase.agent_main import AgentMain




class DDQNAgent(AgentMain):

    def compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        # 算法最关键的部分， 传入了transition的batch
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # y_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)  # q(s,a;w) 这个就是预测的q值
        a_star = self.dqn(next_state).argmax(dim=1, keepdim=True)
        next_q_value = self.dqn_target(next_state).gather(1, a_star).detach() # 这里使用target network 来预测, 并不参与更新，detach()的含义就是让其中的参数避免更新

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)
        """
        y_predict = q(s,a;w)
        a* = Q(s_, :;w)
        y_target = r_t + gamma * argmax(Q_target(s_, a*; w_target))

        error = y_predict - y_target
        loss = (error)^2/2
        """
        return loss




