"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 12:48:20
"""
import os

import numpy as np
import torch



from tqdm import tqdm

from valueBase.util.IL_parse_data import IlParse

from valueBase.PQN.agent import PQNAgent
from valueBase.util.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from valueBase.util.preprocessors import process_raw_state_cmbd


class IL_PQNAgent_v1(PQNAgent):

    def load_expert(self):
        csv_path = os.path.join(os.path.dirname(self.visual_main_path) + "/A3C_data")
        il = IlParse(csv_path)
        self.expert_actions = il.parse_csv("Train_v1")



    def select_action(self, state: np.array) -> list:
        """Select an action from the input state."""
        # epsilon greedy policy

        if self.inmitation_epoch <= 5:
            expert_action = float(self.expert_actions[self.inmitation_step])
            expert_action = [round(expert_action, 1)]
            action_raw_idx = self.action_space.index(expert_action)
            action_raw_tup = expert_action
        else:
            if self.epsilon > np.random.random():
                # Select action returns None, indicating the net work output is not valid
                random_act_idx = np.random.choice(self.action_size)
                action_raw_idx = random_act_idx
                action_raw_tup = self.action_space[random_act_idx]
            else:
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


    def train(self, num_frames: int):

        """Train the agent."""
        self.add_hparams_dict["Number frames"] = num_frames
        self.train_writer = self.complie_visual()
        time_this, state, done = self.reset()  # 初始化环境参数
        hist_state = self.histProcessor. \
            process_state_for_network(state)  # 2-D array
        self.hist_state_dim = hist_state.shape[1]
        self.memory = PrioritizedReplayBuffer(
            self.hist_state_dim, self.memory_size, self.batch_size, self.alpha
        )
        self.complie_dqn()
        self.load_expert()

        update_cnt = 0

        # visual
        epoch = 1
        score = 0
        scores = []
        energy_total_eps = 0
        comfort_total_eps = 0
        this_ep_energy = 0
        this_ep_comfort = 0
        self.inmitation_epoch = 1
        self.inmitation_step = 0
        # eplus
        iter_tqdm = tqdm(range(1, num_frames + 1))
        for frame_idx in iter_tqdm:  # 开启训练
            action = self.select_action(hist_state)  # 输入state输出一个action
            iter_tqdm.set_description(f"{self.env_name}  cooling temp setpoint:{np.squeeze(action)}")


            self.transition = [hist_state, action]  # 把当前的transition添加到列表当中去

            time_next, next_state_raw, done = self.env.step(action)  # 把预测出来的action代入到环境当中，得到下一步的状态和奖励

            next_state = process_raw_state_cmbd(next_state_raw, [time_next],
                                                self._env_st_yr, self._env_st_mn,
                                                self._env_st_dy, self._env_st_wd,
                                                self._pcd_state_limits,
                                                self.is_add_time_to_state)  # 1-D list

            # Process and normalize the raw observation
            next_hist_state = self.histProcessor. \
                process_state_for_network(next_state)  # 2-D array

            this_ep_reward = self.reward_func(state, action, next_state, self._pcd_state_limits,
                                              self._e_weight, self._p_weight, *self.rewardArgs)
            score += this_ep_reward

            self.inmitation_step += 1

            # visual
            this_ep_energy, this_ep_comfort, iats, clgssp, htgssp = self.metric_func(next_state_raw, this_ep_energy, this_ep_comfort)

            list_current = ["Action", "Temperature"]
            self.write_data(self.train_writer, list_current, frame_idx,
                            action[0], iats)


            self.transition += [this_ep_reward, next_state, done]  # 将整体的一个小的transition存储到大的list当中
            self.memory.store(*self.transition)  # 这一步是将当前的transition存到buffer里面
                # 一个transition中包含(state, selected_action, reward, next_state, done)

            hist_state = next_hist_state

            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # visual
            comfort_total_eps += this_ep_comfort
            energy_total_eps += this_ep_energy

            # if episode ends
            if done:
                self.inmitation_step = 0
                self.inmitation_epoch += 1
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
                    self._local_logger.debug("进入训练")
                loss = self.update_model()  # 当初始化结束之后，开始更新模型
                update_cnt += 1  # 记录一下开始更新参数的轮数

                # linearly decrease epsilon
                """
                epsilon是用来决策探索率的超参数
                epsilon的初始值是最大的为1,之后的参数在不断减小，一开始有很大的探索率，随着训练更新，探索率就要不断缩小。
                """
                if self.inmitation_epoch > 5:
                    self.epsilon = self.actEpsilonScheduler.get(frame_idx - len(self.expert_actions)*5)
                    list_current = ["Epsilon"]
                    self.write_data(self.train_writer, list_current, frame_idx,
                                    self.epsilon)

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

        self.save_model()
        self.env.close()

