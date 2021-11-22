"""
   @Author: Weiyang Jiang
   @Date: 2021-11-21 23:51:15
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class ResultParser(object):
    def __init__(self, working_dir, model_type):
        self.model_type = model_type
        self.working_dir = working_dir
        data = pd.read_csv(self.working_dir)
        self.env_list = str(data.loc[0, "Test Environment"]).split(" ")
        features = list(data.iloc[:, 0])
        self.features = []
        test_list = []
        for index, feature in enumerate(features):
            if self.model_type == "ALL":
                self.features.append(feature)
                test_list.append(list(data.loc[index,
                                      "Part1-Light-Pit-Train-v1/Energy Total":"Part1-Light-Pit-Test-v4/Temperature Not Met Baseline"]))
            else:
                if self.model_type in feature.upper():
                    self.features.append(feature)
                    test_list.append(list(data.loc[index, "Part1-Light-Pit-Train-v1/Energy Total":"Part1-Light-Pit-Test-v4/Temperature Not Met Baseline"]))
        test_data = pd.DataFrame(np.array(test_list), columns=data.loc[:, "Part1-Light-Pit-Train-v1/Energy Total":"Part1-Light-Pit-Test-v4/Temperature Not Met Baseline"].columns)
        self.number = len(test_data.index)
        energy_total = pd.DataFrame()
        energy_baseline = pd.DataFrame()
        comfort_total = pd.DataFrame()
        comfort_baseline = pd.DataFrame()


        for str_name in test_data.columns:
            if "Energy Total" in str_name:
                energy_total = energy_total.append(test_data.loc[:, str_name])
            elif "Energy Baseline" in str_name:
                energy_baseline = energy_baseline.append(test_data.loc[:, str_name])
            elif "Temperature Not Met Baseline" in str_name:
                comfort_baseline = comfort_baseline.append(test_data.loc[:, str_name])
            else:
                comfort_total = comfort_total.append(test_data.loc[:, str_name])
        self.energy_total = energy_total.T
        self.energy_baseline = energy_baseline.T
        self.energy_saving = 100 * (np.array(self.energy_baseline) - np.array(self.energy_total)) / np.array(self.energy_baseline)
        self.comfort_total = comfort_total
        self.comfort_baseline = np.array(comfort_baseline.iloc[:, 0])
        self.comfort_total[self.number] = self.comfort_baseline
        self.comfort_total = np.array(self.comfort_total.T)

        # self.comfort_total["baseline"] = self.comfort_baseline
        # print(self.comfort_total)

    def add_text(self, x, y):
        for a, b in zip(x, y):
            if b < 0:
                new_b = b - float(np.clip(8/self.number, 1, 2.5))
            else:
                new_b = b + 0.05
            plt.text(a, new_b, '%.0f' % b, ha='center', va='bottom', fontsize=float(np.clip(40/self.number, 5, 10)))

    def plot_one_img(self, energy_saving, features, env_list):
        x = [i for i in range(len(env_list))]
        total_width, n = 0.5, self.number
        width = total_width / n
        tick_label_list = [re.search(".+-((Test|Train)-v\d+)", i).group(1) for i in env_list]
        for index, feature_name in enumerate(features):
            if index == 0:
                plt.bar(x, energy_saving[index], width=width, label=feature_name, tick_label=tick_label_list)
                self.add_text(x, energy_saving[index])
                # plt.text(x, [i + 0.05 for i in self.energy_saving[index]], self.energy_saving[index], va='bottom', fontsize=11)
            else:
                plt.bar(x, energy_saving[index], width=width, label=feature_name)
                self.add_text(x, energy_saving[index])
            for i in range(len(env_list)):
                x[i] = x[i] + width

    def plot_result(self):
        img_dir = os.path.join(os.path.dirname(self.working_dir), "Baseline_compare")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
    #
    # #     """Plot the training progresses."""

        plt.figure(figsize=(20, 5), dpi=300)
        plt.subplot(121)
        self.plot_one_img(energy_saving=self.energy_saving, features=self.features, env_list=self.env_list)
        plt.title(f'{self.model_type} Energy Baseline Compare')
        plt.ylabel("Energy saving (%)")
        plt.legend(loc="best")

    #
        plt.subplot(122)
        plt.title(f'{self.model_type} Comfort Baseline Compare')
        comfort_sum = self.comfort_total
        features = self.features + ["Baseline"]
        self.plot_one_img(energy_saving=comfort_sum, features=features, env_list=self.env_list)
        plt.ylabel("Temperature Not Met (hrs)")
        plt.legend(loc="best")
        plt.savefig(img_dir + f"/{self.model_type}_Energy_comfort_compare.png")
        plt.show()

    #
    #     plt.bar(x, comfort_list, fc="g", width=width, label="Comfort Not Met(hrs)", tick_label=env_list)
    #     for i in x:
    #         x[i] = x[i] + width
    #     plt.bar(x, comfort_baseline_list, fc="r", width=width, label="Baseline(hrs)")
    #     plt.legend()


if __name__ == '__main__':
    model_type = "dueling"
    model_type = model_type.upper()
    res = ResultParser("/home/weiyang/eplus_RL/Q_learning_data/DuelingData.csv", model_type)
    res.plot_result()