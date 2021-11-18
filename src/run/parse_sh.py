"""
   @Author: Weiyang Jiang
   @Date: 2021-11-04 12:16:19
"""
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
    num_list = glob.glob(f"{dir_list_small}/*")
    for sh_file in num_list:
        path = glob.glob(f"{sh_file}/*.sh")
        json_path = os.path.join(sh_file, "run_value.json")
        with open(json_path, "r", encoding="utf-8") as file:
            json_dict = json.loads(file.read())
        if isinstance(json_dict['test_env'], list):
            test_env = ' '.join(json_dict['test_env'])
        else:
            test_env = json_dict['test_env']
        with open(path[0], "w", encoding="utf-8") as file:
            sh_train = f"python ../../../valueBase/main_run.py --method {'rainbow'} --window_len {42} --lr {0.000065} --adam_eps {1.5e-4} --history_size {80000} --noise_net_std {0.5} --hidden_size {128} --env {json_dict['env']} --device cpu --num_frames {1500000} --reward_func {json_dict['reward_func']} --metric_func {json_dict['metric_func']} --eval_action_func {json_dict['eval_action_func']} --action_space {json_dict['action_space']} --test_env {test_env} --train_action_func {json_dict['train_action_func']} --raw_state_process_func {json_dict['raw_state_process_func']} --action_space {json_dict['action_space']} --state_dim {json_dict['state_dim']} --e_weight {json_dict['e_weight']} --p_weight {json_dict['p_weight']} --rewardArgs {json_dict['rewardArgs']} --memory_size {100000} --batch_size {json_dict['batch_size']} --target_update {2000} --gamma {json_dict['gamma']} --alpha {0.5} --beta {0.6} --prior_eps {json_dict['prior_eps']} --v_min {-10} --v_max {10} --atom_size {51} --seed {json_dict['seed']} --is_on_server {True}"
            # sh_train = sh_train.replace("--", "\n --")
            file.write(sh_train)
        print(path[0])

            # print("已经将信息写入"+json_path)
