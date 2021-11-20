"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 14:12:40
"""


import torch

from valueBase.agent_test_main import Agent_test
from valueBase.noiseNet.noiseNetwork import Network




class Noise_Agent_test(Agent_test):

    def complie_dqn(self):
        # networks: dqn, dqn_target
        self.dqn = Network(
            self.hist_dim, self.action_dim,
        self.hidden_size, self.noise_net_std).to(self.device)
        self.dqn.load_state_dict(torch.load(self.model_path))

