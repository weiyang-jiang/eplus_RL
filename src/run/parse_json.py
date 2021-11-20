"""
   @Author: Weiyang Jiang
   @Date: 2021-10-30 13:09:01
"""
import json
import os
import glob

dir_list_main = glob.glob("*")
for dir_list_small in dir_list_main:
    if os.path.isfile(dir_list_small):
       continue
    num_list = glob.glob(f"{dir_list_small}/*/")
    num_list += glob.glob(f"{dir_list_small}/*/*/")
    for sh_file in num_list:
        path = glob.glob(f"{sh_file}/run.sh")
        if path == []:
            continue
        with open(path[0], "r") as file:
            # print(file.read())

            data = file.read()
            data = data.replace("\\", "")
            data = data.replace("\n", "")
            list_data = data.split("--")[1:]

            dict_data = {}
            for i_data in list_data:
                i_data_list = i_data.strip().split(" ")
                key = i_data_list[0]
                if len(i_data_list) > 2:
                    value = [i for i in i_data_list[1:]]
                else:
                    value = i_data_list[1]
                dict_data[key] = value
        json_path = sh_file + "/run_value.json"
        with open(json_path, "w", encoding="utf-8") as file:
            dict = {
                "lr": float(dict_data.get("lr", 6.5e-05)),
                "adam_eps": float(dict_data.get("adam_eps", 0.00015)),
                "noise_net_std": float(dict_data.get("noise_net_std", 0.5)),
                "history_size": int(dict_data.get("history_size", 80000)),
                "hidden_size": 128,
                "device": dict_data.get("device", "cpu"),
                "method": dict_data.get("method", "rainbow"),
                "##########################强化学习环境名字": "###########################",
                "env": dict_data.get("env", 'Eplus-v1'),
                "##############################奖励函数": "#############################",
                "reward_func": dict_data.get("reward_func", "cslDxCool_1"),
                "###############序列化函数，用于处理comfort和energy": "#####################",
                "metric_func": dict_data.get("metric_func", "cslDxCool_1"),
                "#################测试用的action函数": "###################################",
                "eval_action_func": dict_data.get("eval_act_func", "cslDxActCool_1"),
                "##############测试用的环境": "############################################",
                "test_env": dict_data.get("test_env", None),
                "############ 训练用的动作函数": "##########################################",
                "train_action_func": dict_data.get("train_act_func", "cslDxActCool_1"),
                "########## 用于处理状态的序列化函数": "#####################################",
                "raw_state_process_func": dict_data.get("raw_state_prcs_func", "cslDx_1"),
                "###执行该环境时操作的动作": "例如：[[12.0][12.5][13.0][13.5][14.0][14.5]...]##",
                "action_space": dict_data.get("action_space", 'default'),
                "输出的状态的维度": "状态的具体信息可以从cfg文件中找出来，一般状态是一个大的列表包含[16, 77, 28 ...]",
                "state_dim": int(dict_data.get("state_dim", 15)),
                "###############The penalty weight on energy": "######################",
                "e_weight": float(dict_data.get("rwd_e_para", 0.4)),
                "#############The penalty weight on comfort": "#######################",
                "p_weight": float(dict_data.get("rwd_p_para", 0.6)),
                "rewardArgs": float(dict_data.get("violation_penalty_scl", 10.0)),

                "#################################": "################################",
                "memory_size": int(dict_data.get("memory_size", 100000)),
                "batch_size": int(dict_data.get("batch_size", 32)),
                "target_update": int(dict_data.get("target_update", 2000)),
                "gamma": float(dict_data.get("gamma", 0.99)),
                "##################### PER parameter": "###############################",
                "alpha": 0.2,
                "beta": 0.6,
                "prior_eps": 1e-6,
                "###### Categorical 1_DQN_relpayBuffer_target parameters": "###########",
                "v_min": float(dict_data.get("v_min", -10.0)),
                "v_max": float(dict_data.get("v_max", 10.0)),
                "atom_size": int(dict_data.get("atom_size", 51)),
                "############################### N-step Learning": "###################",
                "window_len": int(dict_data.get("window_len", 35)),
                "forecast_len": 0,
                "n_step": int(dict_data.get("n_step", 3)),

                "############################### train parameter": "###################",
                "num_frames": int(dict_data.get("num_frames", 1500000)),
                "seed": 777,
                "is_on_server": True,
                "is_test": dict_data.get("is_test", 'False'),
            }
            file.write(json.dumps(dict, ensure_ascii=False, indent=4))
            print("已经将信息写入"+json_path)
