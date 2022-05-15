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

        if isinstance(json_dict['env'], list):
            env = ' '.join(json_dict['env'])
        else:
            env = json_dict['env']
        sh_train = rf"python ../../../valueBase/main_run.py `# 训练执行文件(train main file)` \
--feature {json_dict['feature']} `# 此次训练的自定义特征名称：如使用rainbow方法采用rewardv1奖励时[rainbow_v1](The custom feature for this training)` \
--method {json_dict['method']} `# 此次训练的方法(Algorithm for this training)` \
--process_raw_state_cmbd {json_dict['process_raw_state_cmbd']} `# 正则化处理state的函数(normalized state function)` \
--window_len {35} `# state的滑动窗口大小(state window length)` \
--n_step {json_dict['n_step']} `# 多步算法可以提高训练收敛速度(multi-step algorithm for faster training)` \
--forecast_len {0} `# 用于给state增加预测项(forecast length for state)` \
--lr {json_dict['lr']} `# 梯度下降的学习率(learning rate for gradient descent)` \
--adam_eps {json_dict['adam_eps']} `# 梯度下降的探索率(epsilon for gradient descent)` \
--history_size {json_dict['history_size']} `# 训练开始的步数(training start step)` \
--noise_net_std {json_dict['noise_net_std']} `# 噪声值(noise net std)` \
--hidden_size {128} `# 隐藏层的大小(hidden layer size)` \
--env {env} `# 训练环境的名称(training environment name)` \
--device cpu `# 训练使用的设备名称(training device name)` \
--num_frames {json_dict['num_frames']} `# 总的训练步数(total training steps)` \
--reward_func {json_dict['reward_func']} `# 奖励函数(reward function)` \
--metric_func {json_dict['metric_func']} `# 评价函数(metric function)` \
--eval_action_func {json_dict['eval_action_func']} `# 测试动作函数(eval action function)` \
--action_space {json_dict['action_space']} `# 动作的大小(action space)` \
--test_env {test_env} `# 测试环境名称(test environment name)` \
--train_action_func {json_dict['train_action_func']} `# 训练动作函数(train action function)` \
--raw_state_process_func {json_dict['raw_state_process_func']} `# state处理函数(state process function)` \
--state_dim {json_dict['state_dim']} `# 状态大小(state dimension)` \
--e_weight {json_dict['e_weight']} `# 奖励函数中能耗的比重(energy weight in reward function)` \
--p_weight {json_dict['p_weight']} `# 奖励函数中舒适度的比重(comfort weight in reward function)` \
--rewardArgs {json_dict['rewardArgs']} `# 奖励函数中的参数(reward function parameters)` \
--memory_size {json_dict['memory_size']} `# 缓冲器的存储大小(Replaybuffer memory size)` \
--batch_size {json_dict['batch_size']} `# 训练batch的大小(batch size)` \
--target_update {json_dict['target_update']} `# 目标网络的更新间隔(update interval for target network)` \
--gamma {json_dict['gamma']} `# 折扣率(decaying rate)` \
--alpha {0.5} --beta {0.6} --prior_eps {json_dict['prior_eps']} `# 优先缓冲器的超参数(hyperparameters for PrioritizedReplayBuffer)` \
--v_min {json_dict['v_min']} --v_max {json_dict['v_max']} --atom_size {json_dict['atom_size']} `# C51算法的超参数(hyperparameters for C51)` \
--seed {json_dict['seed']} `# 随机种子(random seed)` \
--is_on_server {True} --is_test {False} `# 此次执行是否为测试功能(is test or not)`"
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

        if isinstance(json_dict['env'], list):
            env = ' '.join(json_dict['env'])
        else:
            env = json_dict['env']
        sh_train = rf"python ../../../../valueBase/main_run.py `# 训练执行文件(train main file)` \
--feature {json_dict['feature']} `# 此次训练的自定义特征名称：如使用rainbow方法采用rewardv1奖励时[rainbow_v1](The custom feature for this training)` \
--method {json_dict['method']} `# 此次训练的方法(Algorithm for this training)` \
--process_raw_state_cmbd {json_dict['process_raw_state_cmbd']} `# 正则化处理state的函数(normalized state function)` \
--window_len {35} `# state的滑动窗口大小(state window length)` \
--n_step {json_dict['n_step']} `# 多步算法可以提高训练收敛速度(multi-step algorithm for faster training)` \
--forecast_len {0} `# 用于给state增加预测项(forecast length for state)` \
--lr {json_dict['lr']} `# 梯度下降的学习率(learning rate for gradient descent)` \
--adam_eps {json_dict['adam_eps']} `# 梯度下降的探索率(epsilon for gradient descent)` \
--history_size {json_dict['history_size']} `# 训练开始的步数(training start step)` \
--noise_net_std {json_dict['noise_net_std']} `# 噪声值(noise net std)` \
--hidden_size {128} `# 隐藏层的大小(hidden layer size)` \
--env {env} `# 训练环境的名称(training environment name)` \
--device cpu `# 训练使用的设备名称(training device name)` \
--num_frames {json_dict['num_frames']} `# 总的训练步数(total training steps)` \
--reward_func {json_dict['reward_func']} `# 奖励函数(reward function)` \
--metric_func {json_dict['metric_func']} `# 评价函数(metric function)` \
--eval_action_func {json_dict['eval_action_func']} `# 测试动作函数(eval action function)` \
--action_space {json_dict['action_space']} `# 动作的大小(action space)` \
--test_env {test_env} `# 测试环境名称(test environment name)` \
--train_action_func {json_dict['train_action_func']} `# 训练动作函数(train action function)` \
--raw_state_process_func {json_dict['raw_state_process_func']} `# state处理函数(state process function)` \
--state_dim {json_dict['state_dim']} `# 状态大小(state dimension)` \
--e_weight {json_dict['e_weight']} `# 奖励函数中能耗的比重(energy weight in reward function)` \
--p_weight {json_dict['p_weight']} `# 奖励函数中舒适度的比重(comfort weight in reward function)` \
--rewardArgs {json_dict['rewardArgs']} `# 奖励函数中的参数(reward function parameters)` \
--memory_size {json_dict['memory_size']} `# 缓冲器的存储大小(Replaybuffer memory size)` \
--batch_size {json_dict['batch_size']} `# 训练batch的大小(batch size)` \
--target_update {json_dict['target_update']} `# 目标网络的更新间隔(update interval for target network)` \
--gamma {json_dict['gamma']} `# 折扣率(decaying rate)` \
--alpha {0.5} --beta {0.6} --prior_eps {json_dict['prior_eps']} `# 优先缓冲器的超参数(hyperparameters for PrioritizedReplayBuffer)` \
--v_min {json_dict['v_min']} --v_max {json_dict['v_max']} --atom_size {json_dict['atom_size']} `# C51算法的超参数(hyperparameters for C51)` \
--seed {json_dict['seed']} `# 随机种子(random seed)` \
--is_on_server {True} --is_test {False} `# 此次执行是否为测试功能(is test or not)`"
        with open(path[0], "w", encoding="utf-8") as file:
            file.write(sh_train)
        print(path[0])

            # print("已经将信息写入"+json_path)
