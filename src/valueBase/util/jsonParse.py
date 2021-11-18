"""
   @Author: Weiyang Jiang
   @Date: 2021-10-28 17:08:40
"""
import json
import os

FD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "run")

def json_parse(json_path):
    """ 处理json文件 传入的文件路径应为 rl_parametric_runs_v1/1/run_value.json """
    with open(os.path.join(FD, json_path), "r", encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


