"""
   @Author: Weiyang Jiang
   @Date: 2021-11-03 10:45:30
"""
import torch

from Network_v1 import Network
from valueBase.Asyn_agent_test_main import Agent_test



class Rainbow_Agent_test(Agent_test):

        # Categorical 1_DQN_relpayBuffer_target parameters
    def complie_test_agent(self):
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

    def complie_dqn(self):
        self.dqn = Network(
            self.hist_dim, self.action_dim, self.atom_size, self.hidden_size, self.noise_net_std, self.support
        ).to(self.device)
        self.dqn.load_state_dict(torch.load(self.model_path))


