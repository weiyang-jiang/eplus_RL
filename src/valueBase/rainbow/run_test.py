"""
   @Author: Weiyang Jiang
   @Date: 2021-11-03 11:26:27
"""


import os, sys, glob

from tqdm import tqdm

srcPath = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
src_list = glob.glob(srcPath + "/*")
sys.path.append(srcPath)
for src_path in src_list:
    sys.path.append(src_path)

import gym
import torch

from valueBase.env_interaction import IWEnvInteract
from valueBase.main_args_json import ArgsMake
from valueBase.rainbow.agent_v1_test import Rainbow_Agent_test
from valueBase.util.Evaluation import ResultParser

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



def run(args_make, Train_dir = ""):
    test_envs = args_make.test_envs + [args_make.env_name]
    iter_tqdm = tqdm(test_envs)
    for test_env in iter_tqdm:
        iter_tqdm.set_description(f"{test_env}")
        spec = gym.spec(test_env)
        env_ = spec.make(method=args_make.method, train_dir_path=Train_dir)
        env_interact_wrapper = IWEnvInteract(
           env=env_,
            ob_state_process_func=args_make.raw_state_process_func,
            action_space=args_make.action_space,
            state_dim=args_make.state_dim
        )

        agent = Rainbow_Agent_test(
            # noise
            noise_net_std=args_make.noise_net_std,

            # Q network
            hidden_size=args_make.hidden_size,

            device=args_make.device,
            Train_dir=Train_dir,
            env=env_interact_wrapper,


            raw_stateLimit_process_func=args_make.raw_stateLimit_process_func,
            reward_func=args_make.reward_func,
            action_func=args_make.train_action_func,
            action_space=args_make.action_space,
            action_limits=args_make.train_action_limits,
            metric_func=args_make.metric_func,
            method=args_make.method,




            # Categorical 1_DQN_relpayBuffer_target parameters
            v_min=args_make.v_min,
            v_max=args_make.v_max,
            atom_size=args_make.atom_size,



            # eplus-parameter
            e_weight=args_make.e_weight,
            p_weight=args_make.p_weight,
            rewardArgs=args_make.rewardArgs,
            output_file="./",
            is_add_time_to_state=True,
            is_show=False
        )

        agent.test()

if __name__ == '__main__':

    args_make = ArgsMake("rl_parametric_part1_pit_light/2/run_value.json", is_test=True)

    Train_dir = "/home/weiyang/eplus_RL/src/run/rl_parametric_part1_pit_light/2/Eplus-env-Part1-Light-Pit-Train-v1-RAINBOW-res5"

    # seed = args_make.seed
    #
    # np.random.seed(seed)
    # seed_torch(seed)

    run(args_make, Train_dir)
    parser = ResultParser(Train_dir)
    parser.plot_result()
