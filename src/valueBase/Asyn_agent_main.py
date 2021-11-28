"""
   @Author: Weiyang Jiang
   @Date: 2021-11-10 17:15:42
"""

import os
import random
import time
from tqdm import tqdm

import gym

import torch

import torch.nn.functional as F
import torch.optim as optim

from valueBase.Asyn_agent_test_main import Agent_test
from valueBase.util import Evaluation
from valueBase.util.network import Network
from valueBase.util.replaybuffer import *
from valueBase.env_interaction import IWEnvInteract
from valueBase.util.preprocessors import HistoryPreprocessor
from valueBase.util.logger import Logger
from valueBase.util.eps_scheduler import ActEpsilonScheduler
from visualdl import LogWriter
from valueBase.util.ResultEvaluation import ResultParser
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor


LOG_LEVEL = 'INFO'
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"

CWD = os.getcwd()


class AgentMain(object):
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
            device,
            env,
            raw_stateLimit_process_func,
            reward_func,
            action_func,
            action_space,
            action_limits,
            process_raw_state_cmbd,
            e_weight,
            p_weight,
            rewardArgs,
            is_add_time_to_state,
            window_len,
            forecast_len,
            prcdState_dim,
            local_logger,

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

        if self.is_add_time_to_state == True:
            env_state_limits.insert(0, (0, 23))  # Add hour limit
            env_state_limits.insert(0, (0, 6))  # Add weekday limit
            self.obs_dim += 2

        self._pcd_state_limits = np.transpose(env_state_limits)
        self.env = env
        self.env_name = self.env.env_name
        # reward
        self.reward_func = reward_func
        self.rewardArgs = rewardArgs
        self._e_weight = e_weight
        self._p_weight = p_weight

        self.local_logger = local_logger
        self.local_logger.info(f'{self.env_name} thread worker starts!')

        """
        在经验回放中，需要将所有的transition存储到buffer里面，然后这个buffer有一个memory_size，
        每一次要计算TD-target的时候都要随机挑选batch_size个transition，然后进行求解。
        """

        # device: cpu / gpu
        self.device = device
        # mode: train / test

        self.histProcessor = HistoryPreprocessor(window_len, forecast_len, prcdState_dim)

        # action
        self.action_space = action_space
        self.action_func = action_func
        self.action_limits = action_limits
        self.action_size = len(self.action_space)

    def select_action(self, state: np.array, epsilon, dqn):
        """Select an action from the input state."""
        # epsilon greedy policy
        if epsilon > np.random.random():
            # Select action returns None, indicating the net work output is not valid
            random_act_idx = np.random.choice(self.action_size)
            action_raw_idx = random_act_idx
            action_raw_tup = self.action_space[random_act_idx]
        else:
            action_raw_idx = dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()  # 把state值传入dqn神经网络中
            action_raw_idx = action_raw_idx.detach().cpu().numpy()
            action_raw_tup = self.action_space[action_raw_idx]

        action_stpt_prcd, action_effect_idx = self.action_func(action_raw_tup, action_raw_idx, self._raw_state_limits,
                                                               self.action_limits, state, self.local_logger,
                                                               is_show_debug=False)
        selected_action = action_stpt_prcd

        return {self.env_name: selected_action}

    def reset(self):
        time_this, state_raw, done = self.env.reset()  # 初始化环境参数
        state = self.process_raw_state_cmbd(state_raw, [time_this],
                                            self._env_st_yr, self._env_st_mn,
                                            self._env_st_dy, self._env_st_wd,
                                            self._pcd_state_limits, self.is_add_time_to_state)  # 1-D list
        hist_state = self.histProcessor. \
            process_state_for_network(state)  # 2-D array

        this_ep_reward = 0
        next_state_raw = []
        return {self.env_name: [time_this, this_ep_reward, next_state_raw, hist_state, done, state]}

    def step(self, action: list, state):
        """Take an action and return the response of the env."""

        time_next, next_state_raw, done = self.env.step(action)

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

        return {self.env_name: [time_next, this_ep_reward, next_state_raw, next_hist_state, done, state]}

    def close(self):
        self.env.close()

    def his_reset(self):
        self.histProcessor.reset()

class AsynAgentMain(object):
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
        self.POOL_TEST = ThreadPoolExecutor(10)
        self.POOL_TRAIN = ThreadPoolExecutor(10)

        self.process_raw_state_cmbd = process_raw_state_cmbd
        self.memory_size = memory_size

        # reward
        self.reward_func = reward_func
        self.rewardArgs = rewardArgs
        self._e_weight = e_weight
        self._p_weight = p_weight

        # test
        self.metric_func = metric_func

        # Create a logger
        self.method = RL_method
        self._local_logger = Logger().getLogger('%s_Train'
                                                % (self.method),
                                                LOG_LEVEL, LOG_FMT, output_file + '/main.log')
        self._local_logger.info('Evaluation worker starts!')

        """
        在经验回放中，需要将所有的transition存储到buffer里面，然后这个buffer有一个memory_size，
        每一次要计算TD-target的时候都要随机挑选batch_size个transition，然后进行求解。
        """

        self.batch_size = batch_size
        self.history_size = history_size
        self.actEpsilonScheduler = ActEpsilonScheduler(
            epsilon_start=max_epsilon, epsilon_final=min_epsilon, epsilon_decay=epsilon_decay,
            method="exponential", start_frame=self.history_size,
        )

        self.epsilon = max_epsilon
        self.target_update = target_update
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

        self.is_on_server = is_on_server

        self.raw_state_process_func = raw_state_process_func
        self.raw_stateLimit_process_func = raw_stateLimit_process_func
        self.action_func = action_func
        self.action_dim = len(action_space)
        self.action_space = action_space
        self.action_limits = action_limits

        self.state_dim = state_dim
        if is_add_time_to_state == True:
            self.state_dim += 2

        self.window_len = window_len
        self.forecast_len = forecast_len
        self.prcdState_dim = prcdState_dim

        if len(env) == 1:
            self.env_name = env[0]
        else:
            self.env_name = "Asyn_Train_environment"
        self.train_envs = env
        self.is_shuffle_envs = is_shuffle_envs
        self.dir_path = self._get_eplus_working_folder(CWD, '-%s-%s-res' % (self.env_name, self.feature))

        self.train_env_num = len(self.train_envs)

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
            "Reward function": self.reward_func.__name__,
            "Metric function": self.metric_func.__name__,
            "Number frames": "None",
            "learning rate": self.lr,
            "Adam eps": self.eps,
            "History size": self.history_size,
            "Hidden layer size": self.hidden_size,
            "device": self.device_name,
            "method": self.method,
            "Environment": " ".join(self.train_envs),
            "Test Environment": " ".join(self.test_envs),
            "Action Space": f"{action_space}",
            "State dim": self.state_dim,
            "e weight": self._e_weight,
            "p weight": self._p_weight,
            "rewardArgs": f"{self.rewardArgs}",
            "memory size": self.memory_size,
            "Batch size": self.batch_size,
            "target update": self.target_update,
            "epsilon": epsilon_decay,
            "gamma": self.gamma,
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
            "process_raw_state_cmbd": self.process_raw_state_cmbd.__name__
        }

        self.str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.is_test = is_test
        self.list_main = [f"train/{i}" for i in
                          ["Score", "Loss", "Epsilon", "Energy", "Comfort", "Action", "Temperature"]]

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

        self.hist_state_dim = window_len * self.state_dim
        self.complie_agent()
        self.complie_dqn()
        self.agent_list = []
        for train_env in self.train_envs:
            spec = gym.spec(train_env)
            _env = spec.make(method=self.feature, train_dir_path=self.dir_path)
            env_interact_wrapper = IWEnvInteract(
                env=_env,
                ob_state_process_func=self.raw_state_process_func,
                action_space=self.action_space,
                state_dim=self.state_dim
            )
            agent = self.agent(
                # Q network
                device=self.device,
                # epsilon
                env=env_interact_wrapper,
                raw_stateLimit_process_func=raw_stateLimit_process_func,
                reward_func=reward_func,
                action_func=action_func,
                action_space=action_space,
                action_limits=action_limits,
                process_raw_state_cmbd=process_raw_state_cmbd,
                # test
                # multi-state
                window_len=window_len,
                forecast_len=forecast_len,
                # eplus-parameter
                e_weight=e_weight,
                p_weight=p_weight,
                rewardArgs=rewardArgs,
                is_add_time_to_state=is_add_time_to_state,
                prcdState_dim=prcdState_dim,
                local_logger=self._local_logger
            )
            self.agent_list.append(agent)

    def _get_eplus_working_folder(self, parent_dir, dir_sig='-run'):
        """Return Eplus output folder. Author: CMU-10703 Spring 2017 TA

        Assumes folders in the parent_dir have suffix -run{run
        number}. Finds the highest run number and sets the output folder
        to that number + 1.

        Parameters
        ----------
        parent_dir: str
        Parent dir of the Eplus output directory.

        Returns
        -------
        parent_dir/run_dir
        Path to Eplus save directory.
        """

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
        self.agent = AgentMain
        self.memory = ReplayBuffer(self.hist_state_dim, self.memory_size, self.batch_size)
        self.Agent_test = Agent_test

    def complie_dqn(self):
        # networks: dqn, dqn_target
        self.dqn = Network(self.hist_state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn_target = Network(self.hist_state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=self.eps)

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()  # 在buffer里面随机抽取batch_size个transition

        loss = self.compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_model(self):
        model_file_path = os.path.join(self.dir_path, "model")
        if not os.path.exists(model_file_path):
            os.mkdir(model_file_path)
        torch.save(self.dqn.state_dict(), os.path.join(model_file_path, 'dqn.pth'))
        torch.save(self.dqn.state_dict(), os.path.join(model_file_path, 'dqn_target.pth'))
        self._local_logger.info("models are saved")

    def compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        # 算法最关键的部分， 传入了transition的batch
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        # y_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)  # q(s,a;w) 这个就是预测的q值
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()  # 这里使用target network 来预测, 并不参与更新，detach()的含义就是让其中的参数避免更新

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)
        """
        y_predict = q(s,a;w)

        y_target = r_t + gamma * argmax(Q_target(s_, :; w_target))

        error = y_predict - y_target
        loss = (error)^2/2
        """
        return loss

    def train(self, num_frames: int):
        self.add_hparams_dict["Number frames"] = num_frames
        self.train_writer = self.complie_visual()
        """Train the agent."""
        init_req = as_completed([self.POOL_TRAIN.submit(agent.reset) for agent in self.agent_list])
        init_info = {}
        [init_info.update(i.result()) for i in init_req]
        info = init_info
        iter_tqdm = tqdm(range(1, num_frames + 1))

        update_cnt = 0
        epoch = 1
        score = 0
        scores = []
        this_ep_energy = 0
        this_ep_comfort = 0
        energy_total_eps = 0
        comfort_total_eps = 0

        train_envs = self.train_envs.copy()

        for frame_idx in iter_tqdm:  # 开启训练
            if self.is_shuffle_envs:
                random.shuffle(train_envs)
            iter_tqdm.set_description(f"Asyncio Run")
            action_req = as_completed(
                [self.POOL_TRAIN.submit(self.agent_list[index].select_action, info[train_env][3], self.epsilon, self.dqn)
                 for index, train_env in enumerate(train_envs)])
            action_info = {}
            [action_info.update(i.result()) for i in action_req]

            transitions = [[info[train_env][3], action_info[train_env]] for train_env in train_envs]

            next_req = as_completed([self.POOL_TRAIN.submit(self.agent_list[index].step, action_info[train_env], info[train_env][5])
                                     for index, train_env in enumerate(train_envs)])  # 把预测出来的action代入到环境当中，得到下一步的状态和奖励
            next_info = {}
            [next_info.update(i.result()) for i in next_req]  # time_next, this_ep_reward, next_state_raw, next_hist_state, done

            #########################################################################
            # Mean Reward
            this_ep_reward = float(np.mean([next_info[train_env][1] for train_env in train_envs]))
            score += this_ep_reward
            #########################################################################

            this_ep_energy, this_ep_comfort, iats, clgssp, htgssp = \
                self.metric_func(np.mean([i[2] for i in next_info.values()], axis=0).tolist(), this_ep_energy, this_ep_comfort)

            list_current = ["Action", "Temperature"]
            self.write_data(self.train_writer, list_current, frame_idx,
                            float(np.mean([i[0] for i in action_info.values()])), iats)

            for index, train_env in enumerate(train_envs):
                transitions[index].extend([this_ep_reward, next_info[train_env][3], next_info[train_env][4]])

            self.memory_func(transitions)

            info = next_info

            self.Prioritized(frame_idx, num_frames)
            # visual
            comfort_total_eps += this_ep_comfort
            energy_total_eps += this_ep_energy

            done = list(next_info.values())[0][4]

            if done:
                init_req = as_completed([self.POOL_TRAIN.submit(agent.reset) for agent in self.agent_list])
                info = {}
                [info.update(i.result()) for i in init_req]
                as_completed([self.POOL_TRAIN.submit(agent.his_reset) for agent in self.agent_list])

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

                self.epsilon = self.epsilon_update(frame_idx)

                # if hard update is needed
                """
                每隔target_update次就要更新一次target network
                """
                if update_cnt % self.target_update == 0:
                    self.target_hard_update()

                list_current = ["Loss", "Epsilon"]
                self.write_data(self.train_writer, list_current, frame_idx,
                                loss, self.epsilon)

                if self.is_test and frame_idx == num_frames:
                    self.save_model()

                if frame_idx % (num_frames / 5) == 0 and not self.is_test:
                    self.save_model()

        as_completed([self.POOL_TRAIN.submit(agent.close) for agent in self.agent_list])
        self.POOL_TRAIN.shutdown(True)

    def epsilon_update(self, frame_idx):
        return self.actEpsilonScheduler.get(frame_idx - self.history_size)


    def target_hard_update(self):
        """Hard update: target <- local."""
        # 这里直接把当前的q网络的参数赋值给target network 了
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def complie_visual(self):
        visual_path = self.visual_main_path + f"/visual/{self.env_name}/{self.method}/{self.str_time}_{self.feature}_run"
        writer = LogWriter(visual_path, display_name=self.feature)
        return writer

    def write_data(self, writer, metric_list, num, *args):
        for index, arg in enumerate(args):
            writer.add_scalar(tag=f"train/{metric_list[index]}", value=arg, step=num)

    def save_csv(self):
        import shutil
        values_data = [f"{i}" for i in self.add_hparams_dict.values()]
        str_data = ",".join(values_data).replace("nan", "").replace("[[", '"[[').replace("]]", ']]"')
        with open(os.path.dirname(self.visual_main_path) + "/Q_learning_data/QLearnData.csv", "a") as f:
            f.write("\r\n" + str_data)
        shutil.copy(os.path.dirname(self.visual_main_path) + "/Q_learning_data/QLearnData.csv",
                    os.path.dirname(self.visual_main_path) + "/Q_learning_data/QLearnData_copy.csv")

    def test_task(self, test_env, resultParser):

        spec = gym.spec(test_env)
        env_ = spec.make(method=self.method, train_dir_path=self.dir_path)
        env_interact_wrapper = IWEnvInteract(
            env=env_,
            ob_state_process_func=self.raw_state_process_func,
            action_space=self.action_space,
            state_dim=self.state_dim
        )

        agent = self.Agent_test(
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
            is_test=self.is_test
        )
        return agent.test()

    def test(self):
        """Test the agent."""
        test_envs = self.test_envs
        resultParser = ResultParser(self.dir_path)
        future_list = [self.POOL_TEST.submit(self.test_task, test_env, resultParser) for test_env in test_envs]
        self.POOL_TEST.shutdown(True)
        [self.add_hparams_dict.update(fu.result()) for fu in future_list]
        self.train_writer.add_hparams(hparams_dict=self.add_hparams_dict,
                                      metrics_list=self.list_main)
        self.train_writer.close()
        self.save_csv()
        resultparser = Evaluation.ResultParser(self.dir_path)
        resultparser.plot_result()

    def Prioritized(self, frame_idx, num_frames):
        pass

    def memory_func(self, transitions):
        [self.memory.store(*transition) for transition in transitions]