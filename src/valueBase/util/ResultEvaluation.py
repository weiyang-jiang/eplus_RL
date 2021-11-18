"""
   @Author: Weiyang Jiang
   @Date: 2021-11-13 21:38:01
"""

import os.path
import re

from lxml import etree
import glob



from valueBase.customized.baseline import BASELINE


class ResultParser(object):
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.baseline_name = re.search(".+run/(.+)/.+", self.working_dir).group(1).split("/")[0]
        self.baseline = BASELINE[self.baseline_name]
        Eplus_path = glob.glob(self.working_dir + "/Eplus-env*")[0]
        sub_run_path = glob.glob(Eplus_path + "/Eplus*")[0]
        self.html_path = os.path.join(sub_run_path, "output/eplustbl.htm")
        self.Conditioned_area = self.main_parser()

    def main_parser(self):
        Total_area, Conditioned_area, Energy_Per_Total_Building_Area, Energy_Per_Conditioned_Building_Area = self._html_parser(self.html_path)
        return Conditioned_area

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

    def compare(self, energy_total, env_name):

        energy_rl = energy_total / self.Conditioned_area
        energy_baseline = float(self.baseline[env_name]["Energy"])
        comfort_baseline = float(self.baseline[env_name]["Comfort"])
        Energy_saving = 100 * ((energy_baseline - energy_rl) / energy_baseline)

        return energy_baseline*self.Conditioned_area, comfort_baseline

if __name__ == '__main__':
    result = ResultParser("/home/weiyang/eplus_RL/src/run/rl_parametric_part1_pit_light/dsada/1/Eplus-env-Part1-Light-Pit-Train-v1-RAINBOW-res1")
    print(result.baseline)