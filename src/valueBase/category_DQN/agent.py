"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 14:22:16
"""
from typing import Dict

import numpy as np
import torch
from valueBase.Asyn_agent_main import AgentMain, AsynAgentMain
import torch.optim as optim
from valueBase.category_DQN.agent_test import C51_Agent_test
from valueBase.category_DQN.C51Network import Network
from valueBase.util.replaybuffer import ReplayBuffer


class AsynC51Agent(AsynAgentMain):


    def complie_agent(self):
        self.Agent_test = C51_Agent_test
        self.agent = AgentMain
        self.memory = ReplayBuffer(self.hist_state_dim, self.memory_size, self.batch_size)
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)  # 生成从v_min到v_max的atom_size的等差数列

        self.add_hparams_dict["v_min"] = self.v_min
        self.add_hparams_dict["v_max"] = self.v_max
        self.add_hparams_dict["atom_size"] = self.atom_size

    def complie_dqn(self):
        # networks: dqn, dqn_target
        self.dqn = Network(self.hist_state_dim, self.action_dim, self.atom_size, self.hidden_size, self.support).to(self.device)
        self.dqn_target = Network(self.hist_state_dim, self.action_dim, self.atom_size, self.hidden_size, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=self.eps)


    def compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)  # 每个atom之间的间隔

        with torch.no_grad():
            next_action = self.dqn_target(next_state).argmax(1)  # [32] 选择使用哪个action
            next_dist = self.dqn_target.dist(next_state)  # [32, 2, 51]  列出来所有action的分布值
            next_dist = next_dist[range(self.batch_size), next_action]  # [32, 51] 把action代入选出合适的Z值 Z(s,a)

            t_z = reward + (1 - done) * self.gamma * self.support  # [32, 51]
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)  # [32, 51]
            b = (t_z - self.v_min) / delta_z # [32, 51]
            l = b.floor().long() # [32, 51]
            u = b.ceil().long() # [32, 51]
            """
            就是把 r + gamma * Z_0 分摊到 Z_0 和 Z_1 上，把  r + gamma * Z_1 分摊到 Z_1 和 Z_2 上，以此类推。
            """

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
            )  # [32, 51]

            proj_dist = torch.zeros(next_dist.size(), device=self.device) # [32, 51]
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            """
            torch.Tensor.index_add_能实现指定行或列的内容相加的功能
                dim：这个参数表明你要沿着哪个维度索引；
                index：包含索引的tensor；
                tensor：被索引出来去相加的tensor；
            """
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        # 这里用到了KL散度  https://baike.baidu.com/item/%E7%9B%B8%E5%AF%B9%E7%86%B5?fromtitle=KL%E6%95%A3%E5%BA%A6&fromid=23238109
        # loss = - sum(m_i * log(pi(s_t, a_t)))
        m_i = proj_dist
        loss = -(m_i * log_p).sum(1).mean()

        return loss



