"""
   @Author: Weiyang Jiang
   @Date: 2021-11-17 19:20:04
"""


# Performs a behavioural cloning update
import os
from typing import Tuple

import gym
import numpy as np
import pandas as pd
import torch
from torch import optim
from valueBase.PQN.PQN_il_network import IL_Network
from valueBase.util.IL_parse_data import IlParse
from valueBase.customized.action_funcs import act_func_dict
from valueBase.customized.actions import action_map
from valueBase.customized.raw_state_processors import raw_state_process_map
from valueBase.env_interaction import IWEnvInteract
from valueBase.util.preprocessors import process_raw_state_cmbd
from valueBase.util.logger import Logger

LOG_LEVEL = 'INFO'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"

class IL_learning(object):
    """1_DQN_relpayBuffer_target Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment 这个是环境
        memory (ReplayBuffer): replay memory to store transitions 采用了经验回放的方式来减少经验浪费和相邻transition之间的相关性
        batch_size (int): batch size for sampling 这个表示每一次训练取出多少个batch_size
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
            self,
            env,
            path,
            path_name,
            # Adam
            lr=6.5e-05,
            adam_eps=0.00015,

            # Q network
            hidden_size=128,

            device="cpu",

            batch_size=32,
            raw_stateLimit_process_func=raw_state_process_map["cslDx_1"][0],
            action_func=act_func_dict["cslDxActCool_1"][0],
            action_space=action_map["part1_v1"],
            action_limits=act_func_dict["cslDxActCool_1"][1],
            # test
            raw_state_process_func=raw_state_process_map["cslDx_1"][1],
            state_dim=71,

            gamma=0.99,
            output_file="./",
            is_add_time_to_state=True,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            raw_stateLimit_process_func: stateLimit process function
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
            e_weight=1.0: reward_para The penalty weight on energy.
            p_weight=1.0, reward_para The penalty weight on comfort.

        """
        self.action_dim = env.action_space.n  # 表示动作action是几个的
        self._env_st_yr = env.start_year
        self._env_st_mn = env.start_mon
        self._env_st_dy = env.start_day
        self._env_st_wd = env.start_weekday
        self.raw_stateLimit_process_func = raw_stateLimit_process_func
        env_state_limits = raw_stateLimit_process_func(env.min_max_limits)
        self._raw_state_limits = np.transpose(np.copy(env_state_limits))

        self.is_add_time_to_state = is_add_time_to_state
        self.obs_dim = env.observation_space.shape[0]  # 表示输入的state是几个的

        if self.is_add_time_to_state == True:
            env_state_limits.insert(0, (0, 23))  # Add hour limit
            env_state_limits.insert(0, (0, 6))  # Add weekday limit
            self.obs_dim += 2

        self._pcd_state_limits = np.transpose(env_state_limits)
        self.env = env

        # reward
        # action
        self.action_space = action_space
        self.action_func = action_func
        self.action_limits = action_limits
        self.action_size = len(self.action_space)
        # test
        self.method = "IL_learning"
        self.env_name = self.env.env_name
        self._local_logger = Logger().getLogger('%s_Train-%s'
                                                % (self.method, self.env_name),
                                                LOG_LEVEL, LOG_FMT, output_file + '/main.log')

        self.batch_size = batch_size

        self.gamma = gamma

        # device: cpu / gpu
        self.device_name = device
        self.device = torch.device(
            self.device_name
        )

        print(self.device)

        self.hidden_size = hidden_size
        self.lr = lr
        self.eps = adam_eps
        # transition to store in memory

        # mode: train / test



        self.raw_state_process_func = raw_state_process_func
        self.state_dim = state_dim
        self.dir_path = self.env.model_path


        il = IlParse(path)
        self.expert_actions = il.parse_csv(path_name)

        self.h5 = pd.HDFStore(f'./data/{self.env_name}.h5', 'w', complevel=4, complib='blosc')

    def complie_dqn(self):
        # networks: dqn, dqn_target

        self.dqn = IL_Network(self.obs_dim, self.action_dim, self.hidden_size).to(self.device)
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=self.eps)


    def select_action(self, state: np.array):
        """Select an action from the input state."""
        # epsilon greedy policy
        expert_action = float(self.expert_actions[self.inmitation_step])
        expert_action = [round(expert_action, 1)]
        action_raw_idx = self.action_space.index(expert_action)
        action_raw_tup = expert_action

        action_stpt_prcd, action_effect_idx = self.action_func(action_raw_tup, action_raw_idx, self._raw_state_limits,
                                                               self.action_limits, state, self._local_logger,
                                                               is_show_debug=False)
        selected_action = action_stpt_prcd

        return selected_action, [action_raw_idx]

    def reset(self):
        time_this, state_raw, done = self.env.reset()  # 初始化环境参数
        state = process_raw_state_cmbd(state_raw, [time_this],
                                       self._env_st_yr, self._env_st_mn,
                                       self._env_st_dy, self._env_st_wd,
                                       self._pcd_state_limits, self.is_add_time_to_state)  # 1-D list
        return time_this, state, done

    def step(self, action: list) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""

        time_next, next_state_raw, done = self.env.step(action)

        next_state = process_raw_state_cmbd(next_state_raw, [time_next],
                                            self._env_st_yr, self._env_st_mn,
                                            self._env_st_dy, self._env_st_wd,
                                            self._pcd_state_limits, self.is_add_time_to_state)  # 1-D list

        return time_next, next_state, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()  # 在buffer里面随机抽取batch_size个transition

        loss = self.compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train(self):

        """Train the agent."""
        state_DF = pd.DataFrame()
        action_DF = pd.DataFrame()

        for epoch in range(1, 6):

            time_this, state, done = self.reset()  # 初始化环境参数

            state_DF = state_DF.append([state])
            action_DF = action_DF.append([20])
            self.inmitation_step = 0
            # eplus

            while True:
                action, action_idx = self.select_action(state)  # 输入state输出一个action

                time_next, next_state, done = self.step(action)  # 把预测出来的action代入到环境当中，得到下一步的状态和奖励

                state_DF = state_DF.append([next_state])
                action_DF = action_DF.append(action_idx)
                self.inmitation_step += 1

                # visual

                state = next_state


                # if episode ends
                if done:
                    self._local_logger.info(f"{epoch} epoch is finished")
                    self.inmitation_step = 0
                    break

        self.env.close()

        self.h5["state"] = state_DF
        self.h5["action"] = action_DF
        self.h5.close()

    def save_model(self):
        model_file_path = os.path.join(self.dir_path, "model")
        os.mkdir(model_file_path)
        torch.save(self.dqn.state_dict(), os.path.join(model_file_path, 'dqn.pth'))





if __name__ == '__main__':
    env_name = "Part1-Light-Pit-Train-v1"
    spec = gym.spec(env_name)
    _env = spec.make(method="IL_learning")

    env_interact_wrapper = IWEnvInteract(
        env=_env,
        ob_state_process_func=raw_state_process_map["cslDx_1"][0],
        action_space=action_map["part1_v1"],
        state_dim=71
    )

    il = IL_learning(env_interact_wrapper, "/home/weiyang/eplus_RL/A3C_data", "Train_v1")
    il.train()