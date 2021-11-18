"""
   @Author: Weiyang Jiang
   @Date: 2021-11-19 00:26:22
"""

import torch

from valueBase.main_args_sh import *




def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class RunProfile(object):
    def __init__(self, agent, args):
        self.args_make = ArgsMake(args, is_test_env=False)
        self.env = self.args_make.env_interact_wrapper
        self.agent = agent(
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
            n_step=self.args_make.window_len,

            # eplus-parameter
            e_weight=self.args_make.e_weight,
            p_weight=self.args_make.p_weight,
            rewardArgs=self.args_make.rewardArgs,
            output_file="./",
            is_add_time_to_state=True,
            is_on_server=self.args_make.is_on_server,
            test_envs=self.args_make.test_envs,
            is_test=self.args_make.is_test
        )


    def run(self):
        self.agent.train(self.args_make.num_frames)
        # agent.train(10000)
        self.agent.test()
        self.env.close()
        return self.agent.dir_path

