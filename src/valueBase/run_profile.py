"""
   @Author: Weiyang Jiang
   @Date: 2021-11-19 00:26:22
"""
import os
import time

import torch
from tqdm import tqdm
from visualdl import LogWriter
from concurrent.futures import ThreadPoolExecutor
from valueBase.main_args_sh import *
from valueBase.util.ResultEvaluation import ResultParser



def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class RunProfileTrain(object):
    def __init__(self, agent, args):
        self.args_make = ArgsMake(args, is_test_env=False)
        self.env = self.args_make.train_envs
        self.agent = agent(
            feature=self.args_make.feature,
            # Adam
            visual_main_path=self.args_make.visual_main_path,
            lr=self.args_make.lr,
            adam_eps=self.args_make.adam_eps,
            history_size=self.args_make.history_size,

            # noise
            noise_net_std=self.args_make.noise_net_std,

            # Q network
            hidden_size=self.args_make.hidden_size,

            device=self.args_make.device,
            gamma=self.args_make.gamma,
            
            # epsilon
            max_epsilon=1.0,
            min_epsilon=0.01,
            epsilon_decay=self.args_make.epsilon_decay,
            
            env=self.env,
            memory_size=self.args_make.memory_size,
            batch_size=self.args_make.batch_size,
            target_update=self.args_make.target_update,

            raw_stateLimit_process_func=self.args_make.raw_stateLimit_process_func,
            reward_func=self.args_make.reward_func,
            action_func=self.args_make.train_action_func,
            action_space=self.args_make.action_space,
            action_limits=self.args_make.train_action_limits,
            metric_func=self.args_make.metric_func,
            process_raw_state_cmbd=self.args_make.process_raw_state_cmbd,

            RL_method=self.args_make.method,

            # test
            state_dim=self.args_make.state_dim,
            raw_state_process_func=self.args_make.raw_state_process_func,

            # PER parameters
            alpha=self.args_make.alpha,
            beta=self.args_make.beta,
            prior_eps=self.args_make.prior_eps,

            # Categorical 1_DQN_relpayBuffer_target parameters
            v_min=self.args_make.v_min,
            v_max=self.args_make.v_max,
            atom_size=self.args_make.atom_size,

            # N-step Learning
            n_step=self.args_make.n_step,

            # multi-state
            window_len=self.args_make.window_len,
            forecast_len=self.args_make.forecast_len,

            # eplus-parameter
            e_weight=self.args_make.e_weight,
            p_weight=self.args_make.p_weight,
            rewardArgs=self.args_make.rewardArgs,
            output_file="./",
            is_add_time_to_state=True,
            is_on_server=self.args_make.is_on_server,
            test_envs=self.args_make.test_envs,
            is_test=self.args_make.is_test,
            is_shuffle_envs=self.args_make.is_shuffle_envs

        )


    def run(self):
        self.agent.train(self.args_make.num_frames)
        # agent.train(10000)
        self.agent.test()
        return self.agent.dir_path


class RunProfileTest(object):
    def __init__(self, agent, args, str_time, dir_path, model_path, visual_path, model_name, save_path):
        self.args_make = ArgsMake(args, is_test_env=True)
        self.feature = self.args_make.feature
        self.train_env_name = args.env
        self.dir_path = dir_path
        self.model_path = model_path
        self.visual_main_path = visual_path
        self.save_path = save_path
        self.agent = agent
        self.str_time = str_time
        self.writer = self.complie_visual(model_name)
        self.add_hparams_dict = {
            "Model Path": self.model_path,
            "Model Name": model_name,
            "Reward function": self.args_make.reward_func.__name__,
            "Metric function": self.args_make.metric_func.__name__,
            "device": self.args_make.device,
            "method": self.args_make.method,
            "Test Environment": " ".join(self.args_make.test_envs),
            "Action Space": f"{self.args_make.action_space}",
            "State dim": self.args_make.state_dim,
            "e weight": self.args_make.e_weight,
            "p weight": self.args_make.p_weight,
            "rewardArgs": f"{self.args_make.rewardArgs}",
        }
        self.POOL_TEST = ThreadPoolExecutor(10)
    def complie_visual(self, model):
        visual_path = self.visual_main_path + f"/visual_test/{model}/{self.str_time}_{self.feature}_run"
        writer = LogWriter(visual_path, display_name=self.feature)
        return writer

    def test_task(self, test_env, resultParser):

        spec = gym.spec(test_env)
        env_ = spec.make(method=self.args_make.method, train_dir_path=self.save_path)
        env_interact_wrapper = IWEnvInteract(
            env=env_,
            ob_state_process_func=self.args_make.raw_state_process_func,
            action_space=self.args_make.action_space,
            state_dim=self.args_make.state_dim
        )
        test_agent = self.agent(
            train_writer=self.writer,
            resultParser=resultParser,

            # Q network
            str_time=self.str_time,
            visual_main_path=self.args_make.visual_main_path,
            hidden_size=self.args_make.hidden_size,

            device=self.args_make.device,
            model_path=self.model_path,
            Train_dir=self.save_path,
            env=env_interact_wrapper,

            raw_stateLimit_process_func=self.args_make.raw_stateLimit_process_func,
            reward_func=self.args_make.reward_func,
            action_func=self.args_make.train_action_func,
            action_space=self.args_make.action_space,
            action_limits=self.args_make.train_action_limits,
            metric_func=self.args_make.metric_func,
            process_raw_state_cmbd=self.args_make.process_raw_state_cmbd,
            method=self.args_make.method,

            # multi-state
            window_len=self.args_make.window_len,
            forecast_len=self.args_make.forecast_len,
            prcdState_dim=1,

            v_min=self.args_make.v_min,
            v_max=self.args_make.v_max,
            atom_size=self.args_make.atom_size,

            # eplus-parameter
            e_weight=self.args_make.e_weight,
            p_weight=self.args_make.p_weight,
            rewardArgs=self.args_make.rewardArgs,
            output_file="./",
            is_add_time_to_state=True,
            is_show=False,
            noise_net_std=self.args_make.noise_net_std,
            is_test=self.args_make.is_test
        )
        return test_agent.test()

    def run(self):
        resultParser = ResultParser(self.dir_path)
        test_envs = self.train_env_name + self.args_make.test_envs
        future_list = [self.POOL_TEST.submit(self.test_task, test_env, resultParser) for test_env in test_envs]
        self.POOL_TEST.shutdown(True)
        [self.add_hparams_dict.update(fu.result()) for fu in future_list]
        self.writer.add_hparams(hparams_dict=self.add_hparams_dict,
                                      metrics_list=["visual"])
        self.writer.add_scalar(tag="visual", value=0, step=1)
        self.writer.close()
        return self.save_path