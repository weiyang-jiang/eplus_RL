"""
   @Author: Weiyang Jiang
   @Date: 2021-11-06 12:31:48
"""
import numpy as np

"""
   @Author: Weiyang Jiang
   @Date: 2021-10-30 13:09:01
"""
import json
import os
import glob

dict_baseline = {}
dir_list_main = glob.glob("*")

for dir_list_small in dir_list_main:
    if os.path.isfile(dir_list_small):
       continue
    sh_path = glob.glob(f"{dir_list_small}/1/*.json")[0]

    with open(sh_path, "r") as file:
        dict_data = json.loads(file.read())
    train_env = dict_data["env"]
    test_env = dict_data["test_env"]
    if isinstance(test_env, str):
        test_env = [test_env]
    test_env.insert(0, train_env)
    env_dict = {}
    for i in test_env:
        env_dict[i] = {"Energy": "", "Comfort": ""}
    dict_baseline[dir_list_small] = env_dict


json_ = os.getcwd() + "/baseline.json"
with open(json_, "w", encoding="utf-8") as file:
    file.write(json.dumps(dict_baseline, ensure_ascii=False, indent=4))
    print("已经将信息写入"+json_)
