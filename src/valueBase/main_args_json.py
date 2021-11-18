"""
   @Author: valueBase Jiang
   @Date: 2021-10-29 21:50:29
"""


import gym

import eplus_env



from valueBase.env_interaction import IWEnvInteract
from valueBase.customized.reward_funcs import reward_func_dict, metric_func_dict
from valueBase.customized.action_funcs import act_func_dict
from valueBase.customized.raw_state_processors import raw_state_process_map
from valueBase.customized.actions import action_map
from valueBase.util.jsonParse import json_parse

NAME = 'A3C_AGENT_MAIN'
LOG_LEVEL = 'INFO'
LOG_FORMATTER = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"

# environment
class ArgsMake(object):

    def __init__(self, run_name, is_test=False):
        # 这些都不允许调用
        self._args = json_parse(run_name)
        self.env_name = self._args["env"]
        if not is_test:
            self._env = gym.make(self._args["env"])
            
        self.lr = self._args["lr"]
        self.adam_eps = self._args["adam_eps"]
        self.history_size = self._args["history_size"]

        # noise
        self.noise_net_std = self._args["noise_net_std"]


        # Q network
        self.hidden_size = self._args["hidden_size"]

        self.device = self._args["device"]
        
        self._train_action = self._args["train_action_func"]
        self._eval_action = self._args["eval_action_func"]
        self._raw_state_process = self._args["raw_state_process_func"]
        # 这些可以调用
        self.reward_func = reward_func_dict[self._args["reward_func"]]
        self.metric_func = metric_func_dict[self._args["metric_func"]]
        self.train_action_func = act_func_dict[self._train_action][0]
        self.train_action_limits = act_func_dict[self._train_action][1]
        self.eval_action_func = act_func_dict[self._eval_action][0]
        self.eval_action_limits = act_func_dict[self._eval_action][1]
        self.raw_state_process_func = raw_state_process_map[self._raw_state_process][0]
        self.raw_stateLimit_process_func = raw_state_process_map[self._raw_state_process][1]
        self.action_space = action_map[self._args["action_space"]]
        self.state_dim = self._args["state_dim"]
        self.e_weight = self._args["e_weight"]
        self.p_weight = self._args["p_weight"]
        self.window_len = self._args["window_len"]
        self.test_envs = self._args["test_env"]

        if not is_test:
            self.env_interact_wrapper = IWEnvInteract(
                env=self._env,
                ob_state_process_func=self.raw_state_process_func,
                action_space=self.action_space,
                state_dim=self.state_dim
            )

        self.rewardArgs = [self._args["rewardArgs"]]

        self.memory_size = self._args["memory_size"]
        self.batch_size = self._args["batch_size"]
        self.target_update = self._args["target_update"]
        self.gamma = self._args["gamma"]

        self.alpha = self._args["alpha"]
        self.beta = self._args["beta"]
        self.prior_eps = self._args["prior_eps"]

        self.v_min = self._args["v_min"]
        self.v_max = self._args["v_max"]
        self.atom_size = self._args["atom_size"]

        self.num_frames = self._args["num_frames"]
        self.seed = self._args["seed"]
        self.method = self._args["method"]
        # self.output_file = get_output_folder(".", self._args["env"])
        # self.main_logger = Logger().getLogger(NAME, LOG_LEVEL, LOG_FORMATTER, self.output_file + '/main.log');