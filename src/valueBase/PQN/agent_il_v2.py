"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 12:48:20
"""
import os
import torch


from torch import optim
from valueBase.PQN.PQN_il_network import IL_Network
from valueBase.PQN.agent import PQNAgent



class IL_PQNAgent_v2(PQNAgent):

    def complie_dqn(self):
        self.expert_dqn = os.path.join(self.visual_main_path + "/valueBase/PQN/expert_model/expert_dqn.pth")
        # networks: dqn, dqn_target
        self.dqn = IL_Network(self.hist_state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn.load_state_dict(torch.load(self.expert_dqn, map_location=torch.device("cpu")))
        self.dqn_target = IL_Network(self.hist_state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=self.eps)



