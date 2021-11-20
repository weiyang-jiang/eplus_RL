python ../../../valueBase/main_run.py --method rainbow --window_len 35 --n_step 3 --forecast_len 0 --lr 6.5e-05 --adam_eps 0.00015 --history_size 80000 --noise_net_std 0.5 --hidden_size 128 --env IW-tmy3Weather-v9606 --device cpu --num_frames 1500000 --reward_func 10 --metric_func cslDxCool_1 --eval_action_func cslDxActCool_1 --action_space iw_af5_1 --test_env IW-realWeather-v9606 --train_action_func cslDxActCool_1 --raw_state_process_func cslDx_1 --action_space iw_af5_1 --state_dim 13 --e_weight 0.4 --p_weight 0.6 --rewardArgs 10.0 --memory_size 100000 --batch_size 32 --target_update 2000 --gamma 0.99 --alpha 0.5 --beta 0.6 --prior_eps 1e-06 --v_min -10.0 --v_max 10.0 --atom_size 51 --seed 777 --is_on_server True --is_test False