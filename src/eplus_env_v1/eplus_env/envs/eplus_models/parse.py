"""
   @Author: Weiyang Jiang
   @Date: 2021-11-04 00:48:14
"""
import json
import os
import glob
main_path = "/home/weiyang/"
list = [main_path + "eplus_RL/src/eplus_env_v1/eplus_env/envs/eplus_models/rl_exp_part_1/idf/*",
        main_path + "eplus_RL/src/eplus_env_v1/eplus_env/envs/eplus_models/rl_exp_part_2/idf/*",
        main_path + "eplus_RL/src/eplus_env_v1/eplus_env/envs/eplus_models/rl_exp_part_3/idf/*",
        main_path + "eplus_RL/src/eplus_env_v1/eplus_env/envs/eplus_models/rl_exp_part_4/idf/*",
        main_path + "eplus_RL/src/eplus_env_v1/eplus_env/envs/eplus_models/test/idf/*"]
for j in list:
    path_idf = glob.glob(j)
    for i in path_idf:
        if os.path.isdir(i):
            continue
        with open(i, "r", encoding = 'ISO-8859-1') as file:
            data = file.read()
        data = data.replace("/home/weiyang/", main_path)
        with open(i, "w", encoding = 'ISO-8859-1') as file:
            file.write(data)

