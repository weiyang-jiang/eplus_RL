"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 13:47:23
"""
import os

import torch

from torch import optim

from valueBase.Dueling_network.agent import AsynDuelingAgent
from valueBase.Dueling_network.duelingNetwork import Dueling_Network



class DuelingAgent_v2(AsynDuelingAgent):

    def complie_dqn(self):
        self.expert_dqn = os.path.join(self.visual_main_path + "/valueBase/util/expert_model/Dueling_expert_dqn_35.pth")
        # networks: dqn, dqn_target
        self.dqn = Dueling_Network(self.hist_state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn.load_state_dict(torch.load(self.expert_dqn, map_location=torch.device("cpu")))
        self.dqn_target = Dueling_Network(self.hist_state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=self.eps)








