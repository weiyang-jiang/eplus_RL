"""
   @Author: Weiyang Jiang
   @Date: 2021-11-21 12:38:20
"""
import numpy as np
import pandas as pd

data = pd.read_csv('/home/weiyang/eplus_RL/Q_learning_data/QLearnData1.csv')
key = list(data.columns)
value = list(data.iloc[0, :])
dict_data = {}
for i in range(len(key)):
    dict_data[key[i]] = f"{value[i]}"
values_data = [f"{i}" for i in dict_data.values()]
str_data = ",".join(values_data).replace("nan", "").replace("[[", '"[[').replace("]]", ']]"')
with open("/home/weiyang/eplus_RL/Q_learning_data/QLearnData1.csv", "a") as f:
    f.write("\r\n" + str_data)