"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 16:00:25
"""
from typing import Dict
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch

from valueBase.agent_main import AgentMain
from valueBase.agent_test_main import Agent_test
from valueBase.n_step_DQN.n_step_replaybuffer import ReplayBuffer
from valueBase.util.preprocessors import process_raw_state_cmbd


class Nstep_Agent(AgentMain):

    def complie_agent(self):
        self.Agent_test = Agent_test
        # memory for 1-step Learning
        self.memory = ReplayBuffer(
            self.obs_dim, self.memory_size, self.batch_size, n_step=1
        )

        # memory for N-step Learning
        self.memory_n = ReplayBuffer(
            self.obs_dim, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma
        )
        self.add_hparams_dict["window_len"] = self.n_step

    def compute_dqn_loss_n_step(self, samples: Dict[str, np.ndarray], gamma) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(
            dim=1, keepdim=True
        )[0].detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()
        indices = samples["indices"]
        loss = self.compute_dqn_loss_n_step(samples, self.gamma)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance.

        samples = self.memory_n.sample_batch_from_idxs(indices)
        gamma = self.gamma ** self.n_step
        n_loss = self.compute_dqn_loss_n_step(samples, gamma)
        loss += n_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train(self, num_frames: int):

        """Train the agent."""

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
            one_step_transition = self.memory_n.store(*self.transition)
            if one_step_transition:
                self.memory.store(*one_step_transition)

            state = next_state

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

                self.epsilon = self.actEpsilonScheduler.get(frame_idx)

                # if hard update is needed
                """
                每隔target_update次就要更新一次target network
                """
                if update_cnt % self.target_update == 0:
                    self.target_hard_update()

                list_current = ["Loss", "Epsilon"]
                self.write_data(self.train_writer, list_current, frame_idx,
                                loss, self.epsilon)
            # plotting

        self.save_model()
        self.env.close()

