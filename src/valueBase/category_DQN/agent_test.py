"""
   @Author: Weiyang Jiang
   @Date: 2021-11-03 10:45:30
"""
import torch
from valueBase.category_DQN.C51Network import Network


from valueBase.agent_test_main import Agent_test


class C51_Agent_test(Agent_test):


    def complie_test_agent(self):
        # Categorical 1_DQN_relpayBuffer_target parameters
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)



    def complie_dqn(self):
        self.dqn = Network(
            self.obs_dim, self.action_dim, self.atom_size, self.hidden_size, self.support
        ).to(self.device)
        self.dqn.load_state_dict(torch.load(self.model_path))




