"""
   @Author: Weiyang Jiang
   @Date: 2021-10-24 00:33:53
"""

import os, sys, glob

from valueBase.util.ResultEvaluation import ResultParser
from valueBase.env_interaction import IWEnvInteract
from valueBase.rainbow.agent_v1_test import Rainbow_Agent_test

srcPath = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
src_list = glob.glob(srcPath + "/*")
sys.path.append(srcPath)
for src_path in src_list:
    sys.path.append(src_path)

from tqdm import tqdm
from typing import Dict

import gym

import numpy as np
import torch
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
from PrioritizedReplayBuffer_v1 import PrioritizedReplayBuffer
from ReplayBuffer_v1 import ReplayBuffer
from Network_v1 import Network
from valueBase.util.preprocessors import process_raw_state_cmbd

from valueBase.agent_main import AgentMain


class Rainbow_Agent(AgentMain):
    
    def complie_agent(self):
        self.Agent_test = Rainbow_Agent_test

        # PER
        # memory for 1-step Learning
        self.memory = PrioritizedReplayBuffer(
            self.obs_dim, self.memory_size, self.batch_size, alpha=self.alpha
        )

        # memory for N-step Learning
        self.memory_n = ReplayBuffer(
            self.obs_dim, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma
        )

        # Categorical 1_DQN_relpayBuffer_target parameters
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        self.add_hparams_dict["Noise net std"] = self.noise_net_std
        self.add_hparams_dict["beta"] = self.beta
        self.add_hparams_dict["prior_eps"] = self.prior_eps
        self.add_hparams_dict["v_min"] = self.v_min
        self.add_hparams_dict["v_max"] = self.v_max
        self.add_hparams_dict["atom_size"] = self.atom_size
        self.add_hparams_dict["window_len"] = self.n_step

    def complie_dqn(self):
        self.dqn = Network(
            self.obs_dim, self.action_dim, self.atom_size, self.hidden_size, self.noise_net_std, self.support
        ).to(self.device)
        self.dqn_target = Network(
            self.obs_dim, self.action_dim, self.atom_size, self.hidden_size, self.noise_net_std, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=self.eps)
        # self.optimizer = optim.SGD(self.dqn.parameters(), lr=0.001)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        # selected_action = self.dqn(
        #     torch.FloatTensor(state).to(self.device)
        # ).argmax()
        # selected_action = selected_action.detach().cpu().numpy()

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


    def train(self, num_frames: int):

        """Train the agent."""


        # visual
        self.add_hparams_dict["Number frames"] = num_frames
        self.train_writer = self.complie_visual()

        self.complie_dqn()
        time_this, state, done = self.reset()  # 初始化环境参数
        update_cnt = 0

        # visual
        epoch = 1
        score = 0
        scores = []
        energy_total_eps = 0
        comfort_total_eps = 0
        this_ep_energy = 0
        this_ep_comfort = 0

        # eplus
        iter_tqdm = tqdm(range(1, num_frames + 1))
        for frame_idx in iter_tqdm:  # 开启训练
            action = self.select_action(state)  # 输入state输出一个action
            iter_tqdm.set_description(f"{self.env_name}  cooling temp setpoint:{np.squeeze(action)}")


            self.transition = [state, action]  # 把当前的transition添加到列表当中去

            time_next, next_state_raw, done = self.env.step(action)  # 把预测出来的action代入到环境当中，得到下一步的状态和奖励

            next_state = process_raw_state_cmbd(next_state_raw, [time_next],
                                                self._env_st_yr, self._env_st_mn,
                                                self._env_st_dy, self._env_st_wd,
                                                self._pcd_state_limits,
                                                self.is_add_time_to_state)  # 1-D list

            # Process and normalize the raw observation

            this_ep_reward = self.reward_func(state, action, next_state, self._pcd_state_limits,
                                              self._e_weight, self._p_weight, *self.rewardArgs)
            score += this_ep_reward

            # visual
            this_ep_energy, this_ep_comfort, iats, clgssp, htgssp = self.metric_func(next_state_raw, this_ep_energy, this_ep_comfort)

            list_current = ["Action", "Temperature"]
            self.write_data(self.train_writer, list_current, frame_idx,
                            action[0], iats)



            self.transition += [this_ep_reward, next_state, done]  # 将整体的一个小的transition存储到大的list当中
            # N-step transition

            one_step_transition = self.memory_n.store(*self.transition)

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

            state = next_state

            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # visual
            comfort_total_eps += this_ep_comfort
            energy_total_eps += this_ep_energy

            # if episode ends
            if done:
                time_this, state, done = self.reset()  # 初始化环境参数
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
                    self._local_logger.debug("进入训练")
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

                # visual
                list_current = ["Loss"]
                self.write_data(self.train_writer, list_current, update_cnt,
                                loss)

            # plotting

        self.save_model()
        self.env.close()


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


    def test(self):
        resultParser = ResultParser(self.dir_path)
        test_envs = self.test_envs
        iter_tqdm = tqdm(test_envs)
        for test_env in iter_tqdm:
            iter_tqdm.set_description(f"{test_env}")
            spec = gym.spec(test_env)
            env_ = spec.make(method=self.method, train_dir_path=self.dir_path)
            env_interact_wrapper = IWEnvInteract(
                env=env_,
                ob_state_process_func=self.raw_state_process_func,
                action_space=self.action_space,
                state_dim=self.state_dim
            )

            agent = Rainbow_Agent_test(
                add_hparams_dict=self.add_hparams_dict,
                # Q network
                train_writer=self.train_writer,
                # noise
                is_test=self.is_test,
                resultParser=resultParser,
                train_env_name=self.env_name,
                str_time=self.str_time,
                visual_main_path=self.visual_main_path,

                noise_net_std=self.noise_net_std,

                # Q network
                hidden_size=self.hidden_size,

                device=self.device_name,

                Train_dir=self.dir_path,
                env=env_interact_wrapper,

                raw_stateLimit_process_func=self.raw_stateLimit_process_func,
                reward_func=self.reward_func,
                action_func=self.action_func,
                action_space=self.action_space,
                action_limits=self.action_limits,
                metric_func=self.metric_func,
                method=self.method,

                # Categorical 1_DQN_relpayBuffer_target parameters
                v_min=self.v_min,
                v_max=self.v_max,
                atom_size=self.atom_size,

                # eplus-parameter
                e_weight=self._e_weight,
                p_weight=self._p_weight,
                rewardArgs=self.rewardArgs,
                output_file="./",
                is_add_time_to_state=True
            )

            self.add_hparams_dict = agent.test()

        self.train_writer.add_hparams(hparams_dict=self.add_hparams_dict,
                                      metrics_list=self.list_main)
        self.train_writer.close()
