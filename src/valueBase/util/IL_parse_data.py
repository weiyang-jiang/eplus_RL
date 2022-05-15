"""
   @Author: Weiyang Jiang
   @Date: 2021-11-16 01:52:25
"""


import numpy as np
import pandas as pd
import glob
import os

class IlParse(object):
    def __init__(self, path):
        env_path_list = glob.glob(path+"/*")
        self.env_list = [i.split("/")[-1] for i in env_path_list]
        self.csv_list = [os.path.abspath(i + "/action.csv") for i in env_path_list]


    def parse_csv(self, csv_file="Train_v1"):
        for csv_path in self.csv_list:
            if csv_file in csv_path:
                action_data = pd.read_csv(csv_path)
                action_data = list(action_data.iloc[2:, -1])
                return action_data


if __name__ == '__main__':
    il = IlParse("/A3C_data")
    data = il.parse_csv("Train_v1")
    print(data)

