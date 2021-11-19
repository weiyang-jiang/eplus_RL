"""
   @Author: Weiyang Jiang
   @Date: 2021-11-06 12:24:35
"""
import json
import os.path
import re
import time

from lxml import etree
import glob

from matplotlib import pyplot as plt

from valueBase.customized.baseline import BASELINE


class ResultParser(object):
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.baseline_name = re.search(".+run/(.+)/.+", self.working_dir).group(1).split("/")[0]
        self.baseline = BASELINE[self.baseline_name]

    def main_parser(self):
        dir_list = glob.glob(self.working_dir + "/*")
        energy_dict_data = {}
        Conditioned_area_dict = {}
        comfort_dict_data = {}
        for dir_path in dir_list:
            working_dir_name = dir_path.split("/")[-1]
            if working_dir_name.startswith("Eplus"):
                if len(glob.glob(dir_path + "/*")) > 1:
                    continue
                working_dir_name = "-".join(working_dir_name.split("-")[2:-1])
                html_path = os.path.join(dir_path + "/Eplus-env-sub_run1/output/eplustbl.htm")
                Total_area, Conditioned_area, Energy_Per_Total_Building_Area, Energy_Per_Conditioned_Building_Area = self._html_parser(
                    html_path)
                Conditioned_area_dict[working_dir_name] = Conditioned_area
            if working_dir_name == "test_data":
                test_data_dirs = glob.glob(dir_path + "/*")
                for test_data_dir in test_data_dirs:
                    working_dir_name = test_data_dir.split("/")[-1].split("_")[-2]
                    test_data_json = test_data_dir + "/data.json"
                    with open(test_data_json, "r", encoding="utf-8") as f:
                        json_data = json.loads(f.read())
                    comfort_dict_data[working_dir_name] = json_data["comfort"]
                    energy_dict_data[working_dir_name] = json_data["total energy"]

        return energy_dict_data, comfort_dict_data, Conditioned_area_dict

    def _html_parser(self, path):
        # 将字符串解析为html文档
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        html = etree.HTML(text)
        Total_area = float(html.xpath("//body/table[3]//tr[2]/td[2]")[0].text.strip())
        Conditioned_area = float(html.xpath("//body/table[3]//tr[3]/td[2]")[0].text.strip())
        Energy_Per_Total_Building_Area = float(html.xpath("//body/table[1]//tr[2]/td[3]")[0].text.strip())
        Energy_Per_Conditioned_Building_Area = float(html.xpath("//body/table[1]//tr[2]/td[4]")[0].text.strip())
        return (Total_area, Conditioned_area, Energy_Per_Total_Building_Area, Energy_Per_Conditioned_Building_Area)

    def compare(self):
        energy_dict_data, comfort_dict_data, Conditioned_area_dict = self.main_parser()
        energy_list = []
        comfort_list = []
        env_list = []
        comfort_baseline_list = []
        list_ = sorted(comfort_dict_data.keys())
        list_ = [list_[-1]] + list_[:-1]
        for env_name in list_:
            energy_baseline = float(self.baseline[env_name]["Energy"])
            comfort_baseline_list.append(float(self.baseline[env_name]["Comfort"]))
            energy_total = float(energy_dict_data[env_name])
            Conditioned_area = float(Conditioned_area_dict[env_name])
            energy_rl = energy_total/Conditioned_area
            Energy_saving = 100 * ((energy_baseline - energy_rl) / energy_baseline)
            energy_list.append(Energy_saving)
            env_list.append(re.search(".+-((Test|Train)-v\d+)", env_name).group(1))
            comfort_list.append(comfort_dict_data[env_name])
        return energy_list, comfort_list, comfort_baseline_list, env_list

    # ([-90.70453707119147, -102.27192048278307, -6.423655365791495, -26.86119443686938, -34.94926719278468],
    #  [268.0, 84.0, 804.0, 134.0, 909.0], ['Test-v1', 'Test-v2', 'Test-v3', 'Test-v4', 'Train-v1'])
    # ([-34.94926719278468, -90.70453707119147, -102.27192048278307, -6.423655365791495, -26.86119443686938],
    #  [268.0, 804.0, 909.0, 84.0, 134.0], ['Train-v1', 'Test-v1', 'Test-v2', 'Test-v3', 'Test-v4'])

    def plot_result(self):
        energy_list, comfort_list, comfort_baseline_list, env_list = self.compare()

        img_dir = os.path.join(self.working_dir, "Baseline_compare")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        """Plot the training progresses."""
        x = [i for i in range(len(env_list))]
        plt.figure(figsize=(20, 5), dpi=100)
        plt.subplot(121)
        plt.title(f'{self.baseline_name} Energy Baseline Compare')
        plt.bar(x, energy_list, fc="r", tick_label=env_list)

        plt.subplot(122)
        plt.title(f'{self.baseline_name} Comfort Baseline Compare')

        total_width, n = 0.8, 2
        width = total_width / n

        plt.bar(x, comfort_list, fc="g", width=width, label="Comfort Not Met(hrs)", tick_label=env_list)
        for i in x:
            x[i] = x[i] + width
        plt.bar(x, comfort_baseline_list, fc="r", width=width, label="Baseline(hrs)")
        plt.legend()
        plt.savefig(img_dir + "/Energy_comfort_compare.png")


if __name__ == '__main__':
    praser = ResultParser(
        "/home/weiyang/eplus_RL/src/run/rl_parametric_part1_pit_light/1/Eplus-env-Part1-Light-Pit-Train-v1-NSTEP-res1")
    praser.plot_result()
    # HtmlParser("/home/weiyang/eplus_RL/src/run/rl_parametric_part1_pit_light/1/Eplus-env-Part1-Light-Pit-Train-v1-NSTEP-res1/Eplus-env-Part1-Light-Pit-Test-v1-res1/Eplus-env-sub_run1/output/eplustbl.htm")
    # print(html)
    # 将字符串序列化为html
