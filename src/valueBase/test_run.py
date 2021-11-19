"""
   @Author: Weiyang Jiang
   @Date: 2021-11-19 13:36:33
"""
import logging
import os, sys, glob
import time

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
from valueBase.util.Evaluation import ResultParser
from valueBase.Dueling_network.agent_test import Dueling_Agent_test
from valueBase.noiseNet.agent_test import Noise_Agent_test
from valueBase.category_DQN.agent_test import C51_Agent_test
from valueBase.rainbow.agent_v1_test import Rainbow_Agent_test
from valueBase.agent_test_main import Agent_test
from valueBase.run_profile import RunProfileTest


str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

def test():
    parser = get_args()
    parser.add_argument('--method', default='rainbow', help='RL Algorithm: including ["DQN", "DDQN", "PQN","DUELING","NOISE","C51", "NSTEP", "RAINBOW"]', type=str)
    parser.add_argument('--visual_main_path', default=srcPath, help='/home/weiyang/eplus_RL/src', type=str)
    parser.add_argument('--epsilon_decay', default=30000, help='epsilon_decay', type=int)
    parser.add_argument('--is_test', default="False", help='is test', type=str)
    parser.add_argument('--dir_path', default="", help='dir_path with Eplus output', type=str)
    parser.add_argument('--model_path', default="", help='model_path', type=str)
    parser.add_argument('--model_name', default="ILPQNV2", help='model_name', type=str)
    parser.add_argument('--save_path', default=os.getcwd(), help='save_path', type=str)
    args = parser.parse_args()
    method = args.method.upper()
    args.dir_path = srcPath + "/run/" + args.dir_path
    if args.save_path == os.getcwd():
        args.save_path = os.getcwd() + f"/{args.env}-{str_time}-test"

    if f"Eplus-env-{args.env}-{method}" not in args.dir_path:
        logging.error("RL method or environment is not met.")
        return

    if args.model_path == "":
        args.model_path = os.path.join(os.path.join(args.dir_path, "model"), "dqn.pth")

    if args.is_test.upper() == "FALSE":
        args.is_test = False
    elif args.is_test.upper() == "TRUE":
        args.is_test = True
    else:
        args.is_test = False


    if method == "DQN":
        agent = Agent_test
    elif method == "DDQN":
        agent = Agent_test
    elif method == "PQN":
        agent = Agent_test
    elif method == "DUELING":
        agent = Dueling_Agent_test
    elif method == "NOISE":
        agent = Noise_Agent_test
    elif method == "C51":
        agent = C51_Agent_test
    elif method == "NSTEP":
        agent = Noise_Agent_test
    elif method == "RAINBOW":
        agent = Rainbow_Agent_test
    elif method == "ILPQNV1":
        agent = Agent_test
    elif method == "ILPQNV2":
        agent = Agent_test
    elif method == "ILDUELINGV1":
        agent = Dueling_Agent_test
    elif method == "ILDUELINGV2":
        agent = Dueling_Agent_test
    else:
        logging.error("RL method is not exists.")
        return
    dir_path = RunProfileTest(
        agent=agent,
        args=args,
        str_time=str_time,
        dir_path=args.dir_path,
        model_path=args.model_path,
        visual_path=args.visual_main_path,
        model_name=args.model_name,
        save_path=args.save_path
    ).run()
    resultparser = ResultParser(dir_path)
    resultparser.plot_result()

if __name__ == '__main__':
    test()