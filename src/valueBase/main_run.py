"""
   @Author: Weiyang Jiang
   @Date: 2021-11-05 08:55:03
"""
import logging
import os, sys, glob



dir_paths = glob.glob(os.path.dirname(__file__) + "/*")
for dir_path_ in dir_paths:
    if os.path.isdir(dir_path_):
        sys.path.append(dir_path_)
srcPath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_list = glob.glob(srcPath + "/*")
sys.path.append(srcPath)
for src_path in src_list:
    sys.path.append(src_path)


from main_args_sh import get_args
from valueBase.DQN_replayBuffer_target.agent import AsynDQNAgent
from valueBase.DDQN_replayBuffer_target.agent import AsynDDQNAgent
from valueBase.PQN.agent import AsynPQNAgent
from valueBase.PQN.agent_il_v1 import IL_PQNAgent_v1
from valueBase.PQN.agent_il_v2 import IL_PQNAgent_v2
from valueBase.Dueling_network.agent import AsynDuelingAgent
from valueBase.noiseNet.agent import AsynNoiseAgent
from valueBase.category_DQN.agent import AsynC51Agent
from valueBase.rainbow.agent_v1 import AsynRainbow_Agent
from valueBase.n_step_DQN.agent import AsynNstep_Agent
from valueBase.run_profile import RunProfileTrain
from valueBase.Dueling_network.agent_il_v1 import DuelingAgent_v1
from valueBase.Dueling_network.agent_il_v2 import DuelingAgent_v2
from valueBase.rainbow.Asyn_agent_v1 import AsynAgentMainRainbow

def run():
    parser = get_args()
    parser.add_argument('--method', default='rainbow', help='RL Algorithm: including ["DQN", "DDQN", "PQN","DUELING","NOISE","C51", "NSTEP", "RAINBOW"]', type=str)
    parser.add_argument('--visual_main_path', default=srcPath, help='/home/weiyang/eplus_RL/src', type=str)
    parser.add_argument('--is_test', default="False", help='is test', type=str)
    parser.add_argument('--epsilon_decay', default=30000, help='epsilon_decay', type=int)
    args = parser.parse_args()
    method = args.method.upper()

    if args.feature == None:
        logging.info(f"You did not prepare feature, using {args.method} as substitute feature")
        args.feature = args.method

    if args.is_test.upper() == "FALSE":
        args.is_test = False
    elif args.is_test.upper() == "TRUE":
        args.is_test = True
        args.num_frames = 100
        args.history_size = 32
    else:
        args.is_test = False

    if args.is_shuffle_envs.upper() == "FALSE":
        args.is_shuffle_envs = False
    elif args.is_shuffle_envs.upper() == "TRUE":
        args.is_shuffle_envs = True
    else:
        args.is_shuffle_envs = False

    if method == "DQN":
        agent = AsynDQNAgent
    elif method == "DDQN":
        agent = AsynDDQNAgent
    elif method == "PQN":
        agent = AsynPQNAgent
    elif method == "DUELING":
        agent = AsynDuelingAgent
    elif method == "NOISE":
        agent = AsynNoiseAgent
    elif method == "C51":
        agent = AsynC51Agent
    elif method == "NSTEP":
        agent = AsynNstep_Agent
    elif method == "RAINBOW":
        agent = AsynRainbow_Agent
    elif method == "ASYNRAINBOW":
        agent = AsynAgentMainRainbow
    elif method == "ILPQNV1":
        agent = IL_PQNAgent_v1
    elif method == "ILPQNV2":
        agent = IL_PQNAgent_v2
    elif method == "ILDUELINGV1":
        agent = DuelingAgent_v1
    elif method == "ILDUELINGV2":
        agent = DuelingAgent_v2
    else:
        logging.error("RL method is not exists.")
        return
    RunProfileTrain(agent=agent, args=args).run()


if __name__ == '__main__':
    run()

