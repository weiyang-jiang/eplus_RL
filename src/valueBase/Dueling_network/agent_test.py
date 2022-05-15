"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 13:47:29
"""
import torch

from valueBase.Dueling_network.duelingNetwork import Dueling_Network
from valueBase.Asyn_agent_test_main import Agent_test


class Dueling_Agent_test(Agent_test):

    def complie_dqn(self):
        # networks: dqn, dqn_target
        self.dqn = Dueling_Network(self.hist_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn.load_state_dict(torch.load(self.model_path))

