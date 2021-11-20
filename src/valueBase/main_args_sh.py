"""
   @Author: Weiyang Jiang
   @Date: 2021-11-04 15:05:53
"""
import argparse

import gym

import eplus_env



from valueBase.env_interaction import IWEnvInteract
from valueBase.customized.reward_funcs import reward_func_dict, metric_func_dict
from valueBase.customized.action_funcs import act_func_dict
from valueBase.customized.raw_state_processors import raw_state_process_map
from valueBase.customized.actions import action_map



LOG_LEVEL = 'DEBUG'
LOG_FORMATTER = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"

def get_args():
    parser = argparse.ArgumentParser(description='Run Rainbow on EnergyPlus')
    parser.add_argument('--env', default='Eplus-v1', help='EnergyPlus env name')
    parser.add_argument(
        '-o', '--output', default='.', help='Directory to save data to')
    parser.add_argument('--num_frames', default=15000000, type=int,
                        help='The max number of interactions with the environment for A3C, default is 15000000.')
    parser.add_argument("--lr", default=0.000065, help="learning rate", type=float)


    parser.add_argument("--device", default="cpu", help="training device", type=str)

    parser.add_argument("--adam_eps", default=1.5e-4, help="adam eps", type=float)
    parser.add_argument("--history_size", default=80000, help="history size", type=int)
    parser.add_argument("--noise_net_std", default=0.5, help="noise network std", type=float)
    parser.add_argument("--hidden_size", default=64, help="hidden layer size", type=int)

    parser.add_argument('--state_dim', default=15, type=int,
                        help='The observation state length of one step, default is 15.')
    parser.add_argument('--seed', default=777, type=int, help='The random seed.')

    parser.add_argument('--e_weight', default=0.4, type=float,
                        help='Reward weight on HVAC energy consumption, default is 0.4.')
    parser.add_argument('--p_weight', default=0.6, type=float,
                        help='Reward wegith on PPD, default is 0.6.')
    parser.add_argument('--action_space', default='default', type=str,
                        help='The action space name, default is default.')
    parser.add_argument('--test_env', nargs='+', type=str)

    parser.add_argument('--batch_size', default=32, type=int, help='batch size for train.')
    parser.add_argument('--memory_size', default=40, type=int, help='The memory size for replayBuffer')
    parser.add_argument('--target_update', default=100, type=int, help='target Network update frequency')
    parser.add_argument('--gamma', default=0.99, type=float)

    parser.add_argument('--alpha', default=0.2, type=float, help='PER parameter')
    parser.add_argument('--beta', default=0.6, type=float, help='PER parameter')
    parser.add_argument('--prior_eps', default=1e-06, type=float, help='PER parameter')

    parser.add_argument('--v_min', default=10.0, type=float, help='Categorical 1_DQN_relpayBuffer_target parameters')
    parser.add_argument('--v_max', default=-10.0, type=float,
                        help='Categorical 1_DQN_relpayBuffer_target parameters')
    parser.add_argument('--atom_size', default=51, type=int,
                        help='Categorical 1_DQN_relpayBuffer_target parameters')

    parser.add_argument('--window_len', default=35, type=int, help='The state stacking window length, default is 35')
    parser.add_argument('--forecast_len', default=0, type=int, help='forecast_len')
    parser.add_argument('--n_step', default=3, type=int, help='N-step Learning.')

    parser.add_argument('--is_on_server', default=True, type=bool, help='whether run on the shell')

    # Specific args
    parser.add_argument('--rewardArgs', default=10.0, type=float,
                        help='Scale temperature setpoint violation error, default is 10.0.')
    parser.add_argument('--train_action_func', default='cslDxActCool_1', type=str,
                        help='The action function corresponding to the action space, default is cslDxActCool_1')
    parser.add_argument('--eval_action_func', default='cslDxActCool_1', type=str,
                        help='The action function corresponding to the action space, default is cslDxActCool_1')
    parser.add_argument('--reward_func', default='cslDxCool_1', type=str)
    parser.add_argument('--metric_func', default='cslDxCool_1', type=str)
    parser.add_argument('--raw_state_process_func', default='cslDx_1', type=str)

    return parser

# environment
class ArgsMake(object):

    def __init__(self, args, is_test_env=False):
        # Prepare case specific args
        self._args = args
        self.method = self._args.method.upper()
        self.visual_main_path = self._args.visual_main_path
        if not is_test_env:
            spec = gym.spec(self._args.env)
            self._env = spec.make(method=self.method)

        self.is_test = self._args.is_test
        self.lr = self._args.lr
        self.adam_eps = self._args.adam_eps
        self.history_size = self._args.history_size

        # noise
        self.noise_net_std = self._args.noise_net_std

        self.epsilon_decay = self._args.epsilon_decay
        # Q network
        self.hidden_size = self._args.hidden_size

        self.device = self._args.device

        self._train_action = self._args.train_action_func
        self._eval_action = self._args.eval_action_func
        self._raw_state_process = self._args.raw_state_process_func
        # 这些可以调用
        self.reward_func = reward_func_dict[self._args.reward_func]
        self.metric_func = metric_func_dict[self._args.metric_func]
        self.train_action_func = act_func_dict[self._train_action][0]
        self.train_action_limits = act_func_dict[self._train_action][1]
        self.eval_action_func = act_func_dict[self._eval_action][0]
        self.eval_action_limits = act_func_dict[self._eval_action][1]
        self.raw_state_process_func = raw_state_process_map[self._raw_state_process][0]
        self.raw_stateLimit_process_func = raw_state_process_map[self._raw_state_process][1]
        self.action_space = action_map[self._args.action_space]
        self.state_dim = self._args.state_dim
        self.e_weight = self._args.e_weight
        self.p_weight = self._args.p_weight
        self.n_step = self._args.n_step
        self.window_len = self._args.window_len
        self.forecast_len = self._args.forecast_len
        self.test_envs = self._args.test_env

        if not is_test_env:
            self.env_interact_wrapper = IWEnvInteract(
                env=self._env,
                ob_state_process_func=self.raw_state_process_func,
                action_space=self.action_space,
                state_dim=self.state_dim
            )

        self.rewardArgs = [self._args.rewardArgs]

        self.memory_size = self._args.memory_size
        self.batch_size = self._args.batch_size
        self.target_update = self._args.target_update
        self.gamma = self._args.gamma

        self.alpha = self._args.alpha
        self.beta = self._args.beta
        self.prior_eps = self._args.prior_eps

        self.v_min = self._args.v_min
        self.v_max = self._args.v_max
        self.atom_size = self._args.atom_size

        self.num_frames = self._args.num_frames
        self.seed = self._args.seed

        self.is_on_server = self._args.is_on_server
        # self.output_file = get_output_folder(".", self._args.env)
        # self.main_logger = Logger().getLogger(NAME, LOG_LEVEL, LOG_FORMATTER, self.output_file + '/main.log');



