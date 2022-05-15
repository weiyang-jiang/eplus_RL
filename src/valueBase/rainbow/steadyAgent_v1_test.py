"""
   @Author: Weiyang Jiang
   @Date: 2021-11-03 10:45:30
"""
from typing import Tuple

import numpy as np
import torch

from Network_v1 import Network
from valueBase.Asyn_agent_test_main import Agent_test


class Steady_Agent_test(Agent_test):

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


    def reset(self):
        time_this, state_raw, done = self.env.reset()  # 初始化环境参数
        state = self.process_raw_state_cmbd(state_raw, [time_this],
                                       self._env_st_yr, self._env_st_mn,
                                       self._env_st_dy, self._env_st_wd,
                                       self._pcd_state_limits, self.is_add_time_to_state)  # 1-D list
        return time_this, state, done

    def step(self, action: list) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""

        time_next, next_state_raw, done = self.env.step(action)
        next_state = self.process_raw_state_cmbd(next_state_raw, [time_next],
                                            self._env_st_yr, self._env_st_mn,
                                            self._env_st_dy, self._env_st_wd,
                                            self._pcd_state_limits, self.is_add_time_to_state)  # 1-D list

        return time_next, next_state, done

    def test(self):
        """Train the agent."""

        time_this, state, done = self.reset()  # 初始化环境参数
        # hist_state = self.histProcessor. \
        #     process_state_for_network(state)  # 2-D array

        self.hist_dim = len(state)
        self.complie_dqn()

        energy_total = 0
        comfort_total = 0
        score = 0

        # eplus
        this_ep_energy = 0
        this_ep_comfort = 0

        # eplus
        frame_idx = 1
        while not done:
            action = self.select_action(state)
            # eplus
            time_next, next_state_raw, done = self.env.step(action)  # 把预测出来的action代入到环境当中，得到下一步的状态和奖励
            next_state = self.process_raw_state_cmbd(next_state_raw, [time_next],
                                                self._env_st_yr, self._env_st_mn,
                                                self._env_st_dy, self._env_st_wd,
                                                self._pcd_state_limits,
                                                self.is_add_time_to_state)  # 1-D list
            # next_hist_state = self.histProcessor. \
            #     process_state_for_network(next_state)  # 2-D array
            this_ep_reward = self.reward_func(state, action, next_state_raw, self._pcd_state_limits,
                                              self._e_weight, self._p_weight, *self.rewardArgs)

            score += this_ep_reward

            this_ep_energy, this_ep_comfort, iats, clgssp, htgssp = self.metric_func(next_state_raw, this_ep_energy, this_ep_comfort)

            comfort_total += this_ep_comfort
            energy_total += this_ep_energy
            state = next_state

            frame_idx += 1
            if self.is_test and frame_idx >= 100:
                break

        energy_Baseline, comfort_baseline = self.resultParser.compare(energy_total, self.env_name)

        list_current = ["Energy Total", "Energy Baseline", "Temperature Not Met", "Temperature Not Met Baseline"]
        self.write_data(list_current,
                         energy_total, energy_Baseline, comfort_total, comfort_baseline)
        self.histProcessor.reset()
        self._save_json(energy_total, comfort_total)
        self._local_logger.info(f"{self.env_name} energy:{energy_total} comfort:{comfort_total}")
        self.env.close()
        return self.add_hparams_dict