python ../../../valueBase/main_run.py `# 训练执行文件(train main file)` \
--feature rainbow `# 此次训练的自定义特征名称：如使用rainbow方法采用rewardv1奖励时[rainbow_v1](The custom feature for this training)` \
--method rainbow `# 此次训练的方法(Algorithm for this training)` \
--process_raw_state_cmbd part1_v1 `# 正则化处理state的函数(normalized state function)` \
--window_len 35 `# state的滑动窗口大小(state window length)` \
--n_step 3 `# 多步算法可以提高训练收敛速度(multi-step algorithm for faster training)` \
--forecast_len 0 `# 用于给state增加预测项(forecast length for state)` \
--lr 6.5e-05 `# 梯度下降的学习率(learning rate for gradient descent)` \
--adam_eps 0.00015 `# 梯度下降的探索率(epsilon for gradient descent)` \
--history_size 80000 `# 训练开始的步数(training start step)` \
--noise_net_std 0.5 `# 噪声值(noise net std)` \
--hidden_size 128 `# 隐藏层的大小(hidden layer size)` \
--env Part1-Light-Pit-Train-v1 `# 训练环境的名称(training environment name)` \
--device cpu `# 训练使用的设备名称(training device name)` \
--num_frames 1500000 `# 总的训练步数(total training steps)` \
--reward_func part1_v1 `# 奖励函数(reward function)` \
--metric_func part1_v1 `# 评价函数(metric function)` \
--eval_action_func cslDxActCool_1 `# 测试动作函数(eval action function)` \
--action_space part1_v1 `# 动作的大小(action space)` \
--test_env Part1-Light-Pit-Test-v1 Part1-Light-Pit-Test-v2 Part1-Light-Pit-Test-v3 Part1-Light-Pit-Test-v4 `# 测试环境名称(test environment name)` \
--train_action_func cslDxActCool_1 `# 训练动作函数(train action function)` \
--raw_state_process_func cslDx_1 `# state处理函数(state process function)` \
--state_dim 71 `# 状态大小(state dimension)` \
--e_weight 0.4 `# 奖励函数中能耗的比重(energy weight in reward function)` \
--p_weight 0.6 `# 奖励函数中舒适度的比重(comfort weight in reward function)` \
--rewardArgs 10.0 `# 奖励函数中的参数(reward function parameters)` \
--memory_size 100000 `# 缓冲器的存储大小(Replaybuffer memory size)` \
--batch_size 32 `# 训练batch的大小(batch size)` \
--target_update 2000 `# 目标网络的更新间隔(update interval for target network)` \
--gamma 0.99 `# 折扣率(decaying rate)` \
--alpha 0.5 --beta 0.6 --prior_eps 1e-06 `# 优先缓冲器的超参数(hyperparameters for PrioritizedReplayBuffer)` \
--v_min -10.0 --v_max 10.0 --atom_size 51 `# C51算法的超参数(hyperparameters for C51)` \
--seed 777 `# 随机种子(random seed)` \
--is_on_server True --is_test False `# 此次执行是否为测试功能(is test or not)`