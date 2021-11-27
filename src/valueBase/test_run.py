"""
   @Author: Weiyang Jiang
   @Date: 2021-11-19 13:36:33
"""
import logging
import os, sys, glob
import time

import tqdm

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
from valueBase.Asyn_agent_test_main import Agent_test
from valueBase.run_profile import RunProfileTest

str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

def _get_eplus_working_folder(parent_dir, dir_sig='-run'):
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
        args.save_path = _get_eplus_working_folder(args.save_path, '-Test-%s-%s-res' % (args.env[0], args.feature))

    if f"Eplus-env-{args.env[0]}-{method}".upper() not in args.dir_path.upper()\
            and f"Eplus-env-{args.env[0]}-{args.feature}".upper() not in args.dir_path.upper()\
            and f"Eplus-env-Asyn_Train_environment-{args.feature}".upper() not in args.dir_path.upper():
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


