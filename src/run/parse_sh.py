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
    num_list = glob.glob(f"{dir_list_small}/*/")
    str_list = glob.glob(f"{dir_list_small}/*/*/")
    for sh_file in num_list:
        path = glob.glob(f"{sh_file}/run.sh")
        if path == []:
            continue
        json_path = os.path.join(sh_file, "run_value.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as file:
            json_dict = json.loads(file.read())
        if isinstance(json_dict['test_env'], list):
            test_env = ' '.join(json_dict['test_env'])
        else:
            test_env = json_dict['test_env']
        sh_train = f"python ../../../valueBase/main_run.py --method {json_dict['method']} --window_len {35} --n_step {json_dict['n_step']} --forecast_len {0} --lr {json_dict['lr']} --adam_eps {json_dict['adam_eps']} --history_size {json_dict['history_size']} --noise_net_std {json_dict['noise_net_std']} --hidden_size {128} --env {json_dict['env']} --device cpu --num_frames {json_dict['num_frames']} --reward_func {json_dict['reward_func']} --metric_func {json_dict['metric_func']} --eval_action_func {json_dict['eval_action_func']} --action_space {json_dict['action_space']} --test_env {test_env} --train_action_func {json_dict['train_action_func']} --raw_state_process_func {json_dict['raw_state_process_func']} --action_space {json_dict['action_space']} --state_dim {json_dict['state_dim']} --e_weight {json_dict['e_weight']} --p_weight {json_dict['p_weight']} --rewardArgs {json_dict['rewardArgs']} --memory_size {json_dict['memory_size']} --batch_size {json_dict['batch_size']} --target_update {json_dict['target_update']} --gamma {json_dict['gamma']} --alpha {0.5} --beta {0.6} --prior_eps {json_dict['prior_eps']} --v_min {json_dict['v_min']} --v_max {json_dict['v_max']} --atom_size {json_dict['atom_size']} --seed {json_dict['seed']} --is_on_server {True} --is_test {False}"
        with open(path[0], "w", encoding="utf-8") as file:
            file.write(sh_train)
        print(path[0])

    for sh_file in str_list:
        path = glob.glob(f"{sh_file}/run.sh")
        if path == []:
            continue
        json_path = os.path.join(sh_file, "run_value.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as file:
            json_dict = json.loads(file.read())
        if isinstance(json_dict['test_env'], list):
            test_env = ' '.join(json_dict['test_env'])
        else:
            test_env = json_dict['test_env']
        sh_train = f"python ../../../../valueBase/main_run.py --method {json_dict['method']} --window_len {35} --n_step {json_dict['n_step']} --forecast_len {0} --lr {json_dict['lr']} --adam_eps {json_dict['adam_eps']} --history_size {json_dict['history_size']} --noise_net_std {json_dict['noise_net_std']} --hidden_size {128} --env {json_dict['env']} --device cpu --num_frames {json_dict['num_frames']} --reward_func {json_dict['reward_func']} --metric_func {json_dict['metric_func']} --eval_action_func {json_dict['eval_action_func']} --action_space {json_dict['action_space']} --test_env {test_env} --train_action_func {json_dict['train_action_func']} --raw_state_process_func {json_dict['raw_state_process_func']} --action_space {json_dict['action_space']} --state_dim {json_dict['state_dim']} --e_weight {json_dict['e_weight']} --p_weight {json_dict['p_weight']} --rewardArgs {json_dict['rewardArgs']} --memory_size {json_dict['memory_size']} --batch_size {json_dict['batch_size']} --target_update {json_dict['target_update']} --gamma {json_dict['gamma']} --alpha {0.5} --beta {0.6} --prior_eps {json_dict['prior_eps']} --v_min {json_dict['v_min']} --v_max {json_dict['v_max']} --atom_size {json_dict['atom_size']} --seed {json_dict['seed']} --is_on_server {True} --is_test {False}"
        with open(path[0], "w", encoding="utf-8") as file:
            file.write(sh_train)
        print(path[0])
            # print("已经将信息写入"+json_path)
