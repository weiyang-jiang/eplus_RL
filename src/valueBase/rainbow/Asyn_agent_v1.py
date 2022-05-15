"""
   @Author: Weiyang Jiang
   @Date: 2021-11-10 17:15:42
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
import pandas as pd
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import gym
import torch
import torch.optim as optim
from valueBase.rainbow.PrioritizedReplayBuffer_v1 import PrioritizedReplayBuffer
from valueBase.rainbow.agent_v1_test import Rainbow_Agent_test
from Network_v1 import Network
from valueBase.rainbow.ReplayBuffer_v1 import *
from valueBase.env_interaction import IWEnvInteract
from valueBase.util.preprocessors import HistoryPreprocessor
from valueBase.util.logger import Logger
from visualdl import LogWriter
from valueBase.util import ResultEvaluation
from valueBase.util import Evaluation

LOG_LEVEL = 'INFO'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"
CWD = os.getcwd()


class Rainbow_Agent(object):
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
            # Adam
            local_logger,
            add_hparams_dict,
            feature,
            visual_main_path,
            lr,
            adam_eps,
            history_size,

            # noise
            noise_net_std,
            # Q network
            hidden_size,

            device,
            env,
            memory_size,
            batch_size,
            target_update,

            epsilon_decay,
            raw_stateLimit_process_func,
            reward_func,
            action_func,
            action_space,
            action_limits,
            metric_func,
            process_raw_state_cmbd,
            RL_method,
            # test
            raw_state_process_func,
            state_dim,
            test_envs,
            memory,
            memory_n,

            # PER parameters
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            # Categorical 1_DQN_relpayBuffer_target parameters
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 3,

            max_epsilon=1.0,
            min_epsilon=0.01,
            gamma=0.99,
            e_weight=1.0,
            p_weight=1.0,
            rewardArgs=10,
            output_file="./",
            is_add_time_to_state=True,
            is_on_server=True,
            is_test=False,
            window_len=35,
            forecast_len=0,
            prcdState_dim=1,

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
        self.POOL_TEST = ThreadPoolExecutor(2)

        self.feature = feature
        self.action_dim = env.action_space.n  # 表示动作action是几个的
        self._env_st_yr = env.start_year
        self._env_st_mn = env.start_mon
        self._env_st_dy = env.start_day
        self._env_st_wd = env.start_weekday
        self.raw_stateLimit_process_func = raw_stateLimit_process_func
        env_state_limits = raw_stateLimit_process_func(env.min_max_limits)
        self._raw_state_limits = np.transpose(np.copy(env_state_limits))
        self.process_raw_state_cmbd = process_raw_state_cmbd

        self.is_add_time_to_state = is_add_time_to_state
        self.obs_dim = env.observation_space.shape[0]  # 表示输入的state是几个的
        self.memory_size = memory_size
        if self.is_add_time_to_state == True:
            env_state_limits.insert(0, (0, 23))  # Add hour limit
            env_state_limits.insert(0, (0, 6))  # Add weekday limit
            self.obs_dim += 2

        self._pcd_state_limits = np.transpose(env_state_limits)
        self.env = env

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
        # test
        self.metric_func = metric_func
        # Create a logger
        self.method = RL_method
        self.local_logger = local_logger
        """
        在经验回放中，需要将所有的transition存储到buffer里面，然后这个buffer有一个memory_size，
        每一次要计算TD-target的时候都要随机挑选batch_size个transition，然后进行求解。
        """
        self.add_hparams_dict = add_hparams_dict
        self.batch_size = batch_size
        self.history_size = history_size

        self.epsilon = max_epsilon
        self.target_update = target_update
        self.gamma = gamma

        # device: cpu / gpu
        self.device_name = device
        self.device = device

        self.hidden_size = hidden_size
        self.lr = lr
        self.eps = adam_eps
        # transition to store in memory
        self.transition = list()

        # mode: train / test

        self.is_on_server = is_on_server

        self.window_len = window_len
        self.forecast_len = forecast_len
        self.prcdState_dim = prcdState_dim
        self.histProcessor = HistoryPreprocessor(window_len, forecast_len, prcdState_dim)

        self.raw_state_process_func = raw_state_process_func
        self.state_dim = state_dim + 2
        self.dir_path = self.env.model_path
        self.env_name = self.env.env_name
        # test
        if isinstance(test_envs, str):
            self.test_envs = [test_envs]
        else:
            self.test_envs = test_envs
        self.test_envs.insert(0, self.env_name)

        self.visual_main_path = visual_main_path

        self.str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.is_test = is_test
        self.list_main = [f"train/{i}" for i in
                          ["Score", "Loss", "Epsilon", "Energy", "Comfort", "Action", "Temperature"]]

        # complie
        # Categorical 1_DQN_relpayBuffer_target parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size

        # memory for N-step Learning
        self.n_step = n_step
        # PER
        self.beta = beta
        self.prior_eps = prior_eps
        self.alpha = alpha
        # noise
        self.noise_net_std = noise_net_std
        self.model_path = os.path.join(os.path.join(self.dir_path, "model"), "dqn.pth")
        self.test_csv = os.path.join(os.path.join(self.dir_path, "test_csv"), "test.csv")
        self.hist_state_dim = window_len * self.state_dim
        self.memory = memory
        self.memory_n = memory_n

        self.complie_agent()
        self.complie_dqn()

    def complie_agent(self):
        self.Agent_test = Rainbow_Agent_test
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

    def complie_dqn(self):
        self.dqn = Network(
            self.hist_state_dim, self.action_dim, self.atom_size, self.hidden_size, self.noise_net_std, self.support
        ).to(self.device)
        self.dqn_target = Network(
            self.hist_state_dim, self.action_dim, self.atom_size, self.hidden_size, self.noise_net_std, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=self.eps)

    def select_action(self, state: np.array) -> List:
        """Select an action from the input state."""
        # epsilon greedy policy
        action_raw_idx = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()  # 把state值传入dqn神经网络中
        action_raw_idx = action_raw_idx.detach().cpu().numpy()
        action_raw_tup = self.action_space[action_raw_idx]

        action_stpt_prcd, action_effect_idx = self.action_func(action_raw_tup, action_raw_idx, self._raw_state_limits,
                                                               self.action_limits, state, self.local_logger,
                                                               is_show_debug=False)
        selected_action = action_stpt_prcd

        return selected_action

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

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self.compute_dqn_loss_rainbow(samples, self.gamma)

        # PER: importance sampling before average
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.

        gamma = self.gamma ** self.n_step
        samples = self.memory_n.sample_batch_from_idxs(indices)
        elementwise_loss_n_loss = self.compute_dqn_loss_rainbow(samples, gamma)
        elementwise_loss += elementwise_loss_n_loss

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def complie_visual(self):
        visual_path = self.visual_main_path + f"/visual/Asyn_training/{self.env_name}/{self.method}/{self.str_time}_{self.feature}_run"
        writer = LogWriter(visual_path, display_name=self.feature)
        return writer

    def write_data(self, writer, metric_list, num, *args):
        for index, arg in enumerate(args):
            writer.add_scalar(tag=f"train/{metric_list[index]}", value=arg, step=num)

    def train(self, num_frames):
        print(self.model_path)
        """Train the agent."""
        self.local_logger.info(f"{self.env_name} thread start")
        self.add_hparams_dict["Number frames"] = num_frames
        self.train_writer = self.complie_visual()
        time_this, state, done = self.reset()  # 初始化环境参数
        hist_state = self.histProcessor. \
            process_state_for_network(state)  # 2-D array

        update_cnt = 0

        # visual
        epoch = 1
        score = 0
        scores = []
        energy_total_eps = 0
        comfort_total_eps = 0
        this_ep_energy = 0
        this_ep_comfort = 0

        # test
        test_score = 0
        test_i = 0
        test_result = pd.DataFrame(columns=["Test Reward", "Energy Total", "Temperature Not Met"])


        # eplus
        iter_tqdm = tqdm(range(1, num_frames + 1))
        iter_tqdm.set_description(f"{self.env_name}")
        for frame_idx in iter_tqdm:  # 开启训练
            action = self.select_action(hist_state)  # 输入state输出一个action
            self.transition = [hist_state, action]  # 把当前的transition添加到列表当中去
            time_next, next_state_raw, done = self.env.step(action)  # 把预测出来的action代入到环境当中，得到下一步的状态和奖励
            next_state = self.process_raw_state_cmbd(next_state_raw, [time_next],
                                                     self._env_st_yr, self._env_st_mn,
                                                     self._env_st_dy, self._env_st_wd,
                                                     self._pcd_state_limits,
                                                     self.is_add_time_to_state)  # 1-D list
            next_hist_state = self.histProcessor. \
                process_state_for_network(next_state)  # 2-D array
            # Process and normalize the raw observation

            this_ep_reward = self.reward_func(state, action, next_state, self._pcd_state_limits,
                                              self._e_weight, self._p_weight, *self.rewardArgs)
            score += this_ep_reward

            # visual
            this_ep_energy, this_ep_comfort, iats, clgssp, htgssp = self.metric_func(next_state_raw, this_ep_energy,
                                                                                     this_ep_comfort)
            list_current = ["Action", "Temperature"]
            self.write_data(self.train_writer, list_current, frame_idx,
                            action[0], iats)

            self.transition += [this_ep_reward, next_hist_state, done]  # 将整体的一个小的transition存储到大的list当中
            # 这一步是将当前的transition存到buffer里面
            # 一个transition中包含(state, selected_action, reward, next_state, done)

            one_step_transition = self.memory_n.store(*self.transition)
            if one_step_transition:
                self.memory.store(*self.transition)

            hist_state = next_hist_state

            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # visual
            comfort_total_eps += this_ep_comfort
            energy_total_eps += this_ep_energy

            # if episode ends
            if done:
                time_this, state, done = self.reset()  # 初始化环境参数
                self.histProcessor.reset()
                hist_state = self.histProcessor. \
                    process_state_for_network(state)  # 2-D array

                scores.append(score)
                score = 0

                # visual
                score_plot = np.mean(scores) if len(scores) <= 4 else np.mean(scores[-4])
                list_current = ["Score", "Energy", "Comfort"]
                self.write_data(self.train_writer, list_current, epoch,
                                score_plot, energy_total_eps, comfort_total_eps)
                comfort_total_eps = 0
                energy_total_eps = 0
                epoch += 1

            # if training is ready
            if len(self.memory) >= self.history_size:  # 首先初始化一段时间，然replay_buffer里面有足够的数据。
                if len(self.memory) == self.history_size:
                    self.local_logger.debug("进入训练")
                loss = self.update_model()  # 当初始化结束之后，开始更新模型
                update_cnt += 1  # 记录一下开始更新参数的轮数

                # linearly decrease epsilon
                """
                epsilon是用来决策探索率的超参数
                epsilon的初始值是最大的为1,之后的参数在不断减小，一开始有很大的探索率，随着训练更新，探索率就要不断缩小。
                """

                # if hard update is needed
                """
                每隔target_update次就要更新一次target network
                """
                if update_cnt % self.target_update == 0:
                    self.target_hard_update()

                list_current = ["Loss"]
                self.write_data(self.train_writer, list_current, frame_idx,
                                loss)
                # plotting
                if self.is_test and frame_idx == num_frames:
                    self.save_model()

                if frame_idx % (num_frames / 15) == 0 and not self.is_test:
                    # data_list = {"Test Reward": score, "Energy Total": energy_total,
                    #              "Temperature Not Met": comfort_total}
                    data_list = self.test_task(test_env=self.env_name, resultParser=None, is_process=True, train_dqn=self.dqn)
                    test_result = pd.concat([test_result, pd.DataFrame(data_list, index=[0])])
                    if data_list["Test Reward"] > test_score:
                        test_score = data_list["Test Reward"]
                        self.save_model()


        self.env.close()
        test_result = test_result.reset_index(drop=True)
        test_csv_file_path = os.path.join(self.dir_path, "test_csv")
        if not os.path.exists(test_csv_file_path):
            os.mkdir(test_csv_file_path)
        test_result.to_csv(self.test_csv)
        self.test()
        return self.env_name

    def save_model(self):
        model_file_path = os.path.join(self.dir_path, "model")
        if not os.path.exists(model_file_path):
            os.mkdir(model_file_path)
        torch.save(self.dqn.state_dict(), os.path.join(model_file_path, 'dqn.pth'))
        torch.save(self.dqn.state_dict(), os.path.join(model_file_path, 'dqn_target.pth'))
        self.local_logger.info("models are saved")

    def compute_dqn_loss_rainbow(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical 1_DQN_relpayBuffer_target algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double 1_DQN_relpayBuffer_target
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def target_hard_update(self):
        """Hard update: target <- local."""
        # 这里直接把当前的q网络的参数赋值给target network 了
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def save_csv(self):
        import shutil
        values_data = [f"{i}" for i in self.add_hparams_dict.values()]
        str_data = ",".join(values_data).replace("nan", "").replace("[[", '"[[').replace("]]", ']]"')
        with open(os.path.dirname(self.visual_main_path) + "/Q_learning_data/QLearnData.csv", "a") as f:
            f.write("\r\n" + str_data)
        shutil.copy(os.path.dirname(self.visual_main_path) + "/Q_learning_data/QLearnData.csv",
                    os.path.dirname(self.visual_main_path) + "/Q_learning_data/QLearnData_copy.csv")

    def test_task(self, test_env, resultParser, is_process=False, train_dqn=None):

        spec = gym.spec(test_env)
        env_ = spec.make(method=self.method, train_dir_path=self.dir_path)
        env_interact_wrapper = IWEnvInteract(
            env=env_,
            ob_state_process_func=self.raw_state_process_func,
            action_space=self.action_space,
            state_dim=self.state_dim
        )

        agent = self.Agent_test(
            is_process=is_process,
            train_writer=self.train_writer,
            resultParser=resultParser,

            # Q network
            str_time=self.str_time,
            visual_main_path=self.visual_main_path,
            hidden_size=self.hidden_size,

            device=self.device,
            model_path=self.model_path,
            Train_dir=self.dir_path,
            env=env_interact_wrapper,

            raw_stateLimit_process_func=self.raw_stateLimit_process_func,
            reward_func=self.reward_func,
            action_func=self.action_func,
            action_space=self.action_space,
            action_limits=self.action_limits,
            metric_func=self.metric_func,
            process_raw_state_cmbd=self.process_raw_state_cmbd,
            method=self.method,

            window_len=self.window_len,
            forecast_len=self.forecast_len,
            prcdState_dim=self.prcdState_dim,

            v_min=self.v_min,
            v_max=self.v_max,
            atom_size=self.atom_size,

            # eplus-parameter
            e_weight=self._e_weight,
            p_weight=self._p_weight,
            rewardArgs=self.rewardArgs,
            output_file="./",
            is_add_time_to_state=True,
            is_show=False,
            noise_net_std=self.noise_net_std,
            is_test=self.is_test,
            train_dqn=train_dqn
        )
        return agent.test()

    def test(self):
        """Test the agent."""
        test_envs = self.test_envs
        resultParser = ResultEvaluation.ResultParser(self.dir_path)
        future_list = [self.POOL_TEST.submit(self.test_task, test_env, resultParser) for test_env in test_envs]
        self.POOL_TEST.shutdown(True)
        [self.add_hparams_dict.update(fu.result()) for fu in future_list]
        self.train_writer.add_hparams(hparams_dict=self.add_hparams_dict,
                                      metrics_list=self.list_main)
        self.train_writer.close()
        self.save_csv()
        resultparser = Evaluation.ResultParser(self.dir_path)
        resultparser.plot_result()


class AsynAgentMainRainbow(object):
    def __init__(
            self,
            # Adam
            feature,
            visual_main_path,
            lr,
            adam_eps,
            history_size,

            # noise
            noise_net_std,
            # Q network
            hidden_size,

            device,
            env,
            memory_size,
            batch_size,
            target_update,

            epsilon_decay,
            raw_stateLimit_process_func,
            reward_func,
            action_func,
            action_space,
            action_limits,
            metric_func,
            process_raw_state_cmbd,
            RL_method,
            # test
            raw_state_process_func,
            state_dim,
            test_envs,

            # PER parameters
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            # Categorical 1_DQN_relpayBuffer_target parameters
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 3,

            max_epsilon=1.0,
            min_epsilon=0.01,
            gamma=0.99,
            e_weight=1.0,
            p_weight=1.0,
            rewardArgs=10,
            output_file="./",
            is_add_time_to_state=True,
            is_on_server=True,
            is_test=False,
            window_len=35,
            forecast_len=0,
            prcdState_dim=1,
            is_shuffle_envs=False
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
        self.feature = feature
        self.POOL_TRAIN = ThreadPoolExecutor(10)

        self._local_logger = Logger().getLogger('%s_Train'
                                                % (RL_method),
                                                LOG_LEVEL, LOG_FMT, output_file + '/main.log')
        self._local_logger.info('Evaluation worker starts!')

        """
        在经验回放中，需要将所有的transition存储到buffer里面，然后这个buffer有一个memory_size，
        每一次要计算TD-target的时候都要随机挑选batch_size个transition，然后进行求解。
        """

        if len(env) == 1:
            self.env_name = env[0]
        else:
            self.env_name = "Asyn_Train_environment"
        self.train_envs = env

        self.dir_path = self._get_eplus_working_folder(CWD, '-%s-%s-res' % (self.env_name, self.feature))

        self.train_env_num = len(self.train_envs)

        self.device_name = device
        self.device = torch.device(
            self.device_name
        )

        print(self.device)

        # test
        if isinstance(test_envs, str):
            self.test_envs = [test_envs]
        else:
            self.test_envs = test_envs

        self.test_envs = [i for i in set(self.test_envs + self.train_envs)]
        self.visual_main_path = visual_main_path

        self.add_hparams_dict = {
            "feature": feature,
            "Model Path": self.dir_path,
            "Reward function": reward_func.__name__,
            "Metric function": metric_func.__name__,
            "Number frames": "None",
            "learning rate": lr,
            "Adam eps": adam_eps,
            "History size": history_size,
            "Hidden layer size": hidden_size,
            "device": self.device_name,
            "method": RL_method,
            "Environment": " ".join(self.train_envs),
            "Test Environment": " ".join(self.test_envs),
            "Action Space": f"{action_space}",
            "State dim": state_dim,
            "e weight": e_weight,
            "p weight": p_weight,
            "rewardArgs": f"{rewardArgs}",
            "memory size": memory_size,
            "Batch size": batch_size,
            "target update": target_update,
            "epsilon": epsilon_decay,
            "gamma": gamma,
            "alpha": "None",
            "beta": "None",
            "prior_eps": "None",
            "v_min": "None",
            "v_max": "None",
            "atom_size": "None",
            "n_step": "None",
            "seed": "None",
            "is_on_server": "True",
            "Noise net std": "None",
            "window_len": window_len,
            "prcdState_dim": prcdState_dim,
            "process_raw_state_cmbd": process_raw_state_cmbd.__name__
        }
        self.noise_net_std = noise_net_std

        self.str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.is_test = is_test
        self.list_main = [f"train/{i}" for i in
                          ["Score", "Loss", "Epsilon", "Energy", "Comfort", "Action", "Temperature"]]

        # Categorical 1_DQN_relpayBuffer_target parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma

        # memory for N-step Learning
        self.n_step = n_step
        # PER
        self.beta = beta
        self.prior_eps = prior_eps
        self.alpha = alpha

        self.raw_state_process_func = raw_state_process_func
        self.action_space = action_space

        self.state_dim = state_dim
        if is_add_time_to_state == True:
            self.state_dim += 2

        self.hist_state_dim = window_len * self.state_dim
        self.complie_agent()

        self.agent_list = []
        for train_env in self.train_envs:
            spec = gym.spec(train_env)
            _env = spec.make(method=self.feature, train_dir_path=self.dir_path, is_Asyn=True)
            env_interact_wrapper = IWEnvInteract(
                env=_env,
                ob_state_process_func=self.raw_state_process_func,
                action_space=self.action_space,
                state_dim=self.state_dim
            )
            agent = self.agent(
                # Adam
                self._local_logger,
                self.add_hparams_dict,
                feature,
                visual_main_path,
                lr,
                adam_eps,
                history_size,

                # noise
                noise_net_std,
                # Q network
                hidden_size,

                self.device,
                env_interact_wrapper,
                memory_size,
                batch_size,
                target_update,

                epsilon_decay,
                raw_stateLimit_process_func,
                reward_func,
                action_func,
                action_space,
                action_limits,
                metric_func,
                process_raw_state_cmbd,
                RL_method,
                # test
                raw_state_process_func,
                state_dim,
                test_envs,
                self.memory,
                self.memory_n,

                # PER parameters
                alpha,
                beta,
                prior_eps,
                # Categorical 1_DQN_relpayBuffer_target parameters
                v_min,
                v_max,
                atom_size,
                # N-step Learning
                n_step,

                max_epsilon,
                min_epsilon,
                gamma,
                e_weight,
                p_weight,
                rewardArgs,
                output_file,
                is_add_time_to_state,
                is_on_server,
                is_test,
                window_len,
                forecast_len,
                prcdState_dim
            )
            self.agent_list.append(agent)

    def _get_eplus_working_folder(self, parent_dir, dir_sig='-run'):
        os.makedirs(parent_dir, exist_ok=True)
        experiment_id = 0
        for folder_name in os.listdir(parent_dir):
            if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split(dir_sig)[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass
        experiment_id += 1

        parent_dir = os.path.join(parent_dir, 'Eplus-env')
        parent_dir = parent_dir + '%s%d' % (dir_sig, experiment_id)
        return parent_dir

    def complie_agent(self):
        self.agent = Rainbow_Agent
        self.memory = PrioritizedReplayBuffer(
            self.hist_state_dim, self.memory_size, self.batch_size, alpha=self.alpha
        )
        self.memory_n = ReplayBuffer(
            self.hist_state_dim, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma
        )
        self.add_hparams_dict["Noise net std"] = self.noise_net_std
        self.add_hparams_dict["beta"] = self.beta
        self.add_hparams_dict["prior_eps"] = self.prior_eps
        self.add_hparams_dict["v_min"] = self.v_min
        self.add_hparams_dict["v_max"] = self.v_max
        self.add_hparams_dict["atom_size"] = self.atom_size
        self.add_hparams_dict["n_step"] = self.n_step

    def train(self, num_frames: int):
        """Train the agent."""
        rep = as_completed([self.POOL_TRAIN.submit(agent.train, num_frames) for agent in self.agent_list])
        info = [i.result() for i in rep]
        self.POOL_TRAIN.shutdown(True)
        self._local_logger.info("training end")

    def test(self):
        pass