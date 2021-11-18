"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 12:49:13
"""
from typing import Dict

import numpy as np


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        """
        一个transition中包含(state, selected_action, reward, next_state, done)
        :param obs:
        :param act:
        :param rew:
        :param next_obs:
        :param done:
        :return:
        """
        self.obs_buf[self.ptr] = obs  # 将当前的observation存储到obs_buf
        self.next_obs_buf[self.ptr] = next_obs  # 将下一步的observation存储到next_obs_buf
        act = np.squeeze(act)
        self.acts_buf[self.ptr] = act  # 将当前动作存入buffer中
        self.rews_buf[self.ptr] = rew  # 将当前reward存入buffer中
        self.done_buf[self.ptr] = done  # 将当前是否完成任务存入buffer中
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        # 随机抽取batch_size 个transition
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

