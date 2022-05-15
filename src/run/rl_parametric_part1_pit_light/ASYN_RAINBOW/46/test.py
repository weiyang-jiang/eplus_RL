"""
   @Author: Weiyang Jiang
   @Date: 2022-05-01 13:07:59
"""
import pandas as pd
import numpy as np
test_result = pd.DataFrame(columns=["Test Reward", "Energy Total", "Temperature Not Met"])
data_list = {"Test Reward": 500., "Energy Total": 30.,
                                  "Temperature Not Met": 20.}
print(pd.DataFrame(data_list, index=[0]))
# print(np.array(data_list).reshape(1, -1))
test_result = pd.concat([test_result, pd.DataFrame(data_list, index=[0])])
test_result = pd.concat([test_result, pd.DataFrame(data_list, index=[0])])
test_result = pd.concat([test_result, pd.DataFrame(data_list, index=[0])])
test_result = pd.concat([test_result, pd.DataFrame(data_list, index=[0])])
test_result = test_result.reset_index(drop=True)
# test_result = pd.concat([test_result, pd.DataFrame(np.array(data_list).reshape(1, -1), columns=["Test Reward", "Energy Total", "Temperature Not Met"])])
print(test_result)