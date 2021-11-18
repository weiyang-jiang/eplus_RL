"""
   @Author: Weiyang Jiang
   @Date: 2021-11-10 17:15:55
"""

import json
import os.path
import time
from typing import Tuple

import gym
import numpy as np
import torch


from valueBase.util.network import Network
from valueBase.util.preprocessors import process_raw_state_cmbd
from valueBase.util.logger import Logger

LOG_LEVEL = 'INFO'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"


class Agent_test(object):
    """1_DQN_relpayBuffer_target Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
            self,
            add_hparams_dict,
            train_writer,
            resultParser,
            # Q network
            train_env_name,
            str_time,
            visual_main_path,
            hidden_size,

            device,
            Train_dir,
            env,
            raw_stateLimit_process_func,
            reward_func,
            action_func,
            action_space,
            action_limits,
            metric_func,
            method,

            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,

            e_weight=1.0,
            p_weight=1.0,
            rewardArgs=10,
            output_file=".",
            is_add_time_to_state=True,
            is_show=False,
            noise_net_std=0.5,
            is_test=False,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        self.env = env
        self.env_name = env.env_name
        self.noise_net_std = noise_net_std
        self.hidden_size = hidden_size

        # eplus
        self.action_dim = env.action_space.n  # 表示动作action是几个的
        self._env_st_yr = env.start_year
        self._env_st_mn = env.start_mon
        self._env_st_dy = env.start_day
        self._env_st_wd = env.start_weekday
        env_state_limits = raw_stateLimit_process_func(env.min_max_limits)
        self._raw_state_limits = np.transpose(np.copy(env_state_limits))
        self.is_add_time_to_state = is_add_time_to_state
        self.obs_dim = env.observation_space.shape[0]
        if self.is_add_time_to_state == True:
            env_state_limits.insert(0, (0, 23))  # Add hour limit
            env_state_limits.insert(0, (0, 6))  # Add weekday limit
            self.obs_dim += 2
        # reward
        self.reward_func = reward_func
        self.rewardArgs = rewardArgs
        self._e_weight = e_weight
        self._p_weight = p_weight
        # action
        self.action_space = action_space
        self.action_func = action_func
        self.action_limits = action_limits
        self.action_size = len(self.action_space)
        self._pcd_state_limits = np.transpose(env_state_limits)
        # test
        self.metric_func = metric_func
        self.method = method
        # Create a logger
        self._local_logger = Logger().getLogger('%s_Test-%s'
                                                % (self.method, self.env.env_name),
                                                LOG_LEVEL, LOG_FMT, output_file + '/main.log')
        self._local_logger.debug('Evaluation worker starts!')
        # eplus

        # device: cpu / gpu
        self.device = torch.device(
            device
        )
        self.is_show = is_show

        # networks: dqn, dqn_target
        self.train_dir = Train_dir
        self.model_path = os.path.join(os.path.join(Train_dir, "model"), "dqn.pth")
        self.visual_main_path = visual_main_path
        self.str_time = str_time
        self.train_env_name = train_env_name
        self.resultParser = resultParser

        self.is_test = is_test
        self.train_writer = train_writer
        self.add_hparams_dict = add_hparams_dict

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.complie_test_agent()

    def complie_test_agent(self):
        pass

    def complie_dqn(self):
        # networks: dqn, dqn_target
        self.dqn = Network(self.obs_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn.load_state_dict(torch.load(self.model_path))

    def select_action(self, state: np.array) -> list:
        """Select an action from the input state."""
        # epsilon greedy policy

        action_raw_idx = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()  # 把state值传入dqn神经网络中
        action_raw_idx = action_raw_idx.detach().cpu().numpy()
        action_raw_tup = self.action_space[action_raw_idx]

        action_stpt_prcd, action_effect_idx = self.action_func(action_raw_tup, action_raw_idx, self._raw_state_limits,
                                                               self.action_limits, state, self._local_logger,
                                                               is_show_debug=False)
        selected_action = action_stpt_prcd

        return selected_action

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



    def write_data(self, metric_list, *args):
        for index, arg in enumerate(args):
            self.add_hparams_dict[f"{self.env_name}/{metric_list[index]}"] = arg


    def test(self):
        """Train the agent."""

        self.complie_dqn()


        time_this, state, done = self.reset()  # 初始化环境参数

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
            next_state = process_raw_state_cmbd(next_state_raw, [time_next],
                                                self._env_st_yr, self._env_st_mn,
                                                self._env_st_dy, self._env_st_wd,
                                                self._pcd_state_limits,
                                                self.is_add_time_to_state)  # 1-D list
            this_ep_reward = self.reward_func(state, action, next_state, self._pcd_state_limits,
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
        self._save_json(energy_total, comfort_total)
        self._local_logger.info(f"{self.env_name} energy:{energy_total} comfort:{comfort_total}")
        self.env.close()
        return self.add_hparams_dict

    def _save_json(self, energy, comfort):
        str_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        data_dir = os.path.join(self.train_dir, f"test_data/{str_time}_{self.env_name}_result")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(os.path.join(data_dir, "data.json"), "w", encoding="utf-8") as file:
            dict = {
                "total energy": float(energy),
                "comfort": float(comfort)
            }
            file.write(json.dumps(dict, ensure_ascii=False, indent=4))



