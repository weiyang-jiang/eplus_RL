import copy
import numpy as np

from valueBase.util.time import get_time_from_seconds


def process_raw_state_1(simTime, state, start_year, start_mon,
                        start_day, start_weekday):
    """
    Raw state processing 1. Do the following things:
    1. Insert day of the week and hour of the day to start of the state list
    
    Args:
        simTime: python list, 1-D 
            Delta seconds from the start time, each item in the list
            corresponds to a row in the state. 
        state: python list, 1-D or 2-D
            The raw observation from the environment. It can be only one 
            sample (1-D) or multiple samples (2-D, where each row is a sample).
        start_year: int 
            Start year.
        start_mon: int 
            Start month.
        start_day: int.
            The day of the month at the start time. 
        start_weekday: int 
            The start weekday. 0 is Monday and 6 is Sunday.
    Return: python list, 1-D or 2-D
        Deepcopy of the processed state. It can be only one sample (1-D) or 
        multiple samples (2-D, where each row is a sample).
    """
    ret = []
    state = np.array(state)
    # Reshape the state to 2-D if it is 1-D
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    # Manipulate the state
    for i in range(state.shape[0]):
        state_i_list = state[i, :].tolist()
        nowWeekday, nowHour = get_time_from_seconds(simTime[i], start_year,
                                                    start_mon, start_day,
                                                    start_weekday)
        nowWeekday = 1 if nowWeekday <= 4 else 0
        state_i_list.insert(0, nowHour)  # Add weekday and hour infomation
        state_i_list.insert(0, nowWeekday)
        # state_i_list[ZPCT_RAW_IDX + 2] = 1 if state_i_list[ZPCT_RAW_IDX + 2] > 0 \
        #                                   else 0 # Occupancy count --> occupancy
        if state.shape[0] > 1:
            ret.append(state_i_list)
        else:
            ret = state_i_list
    return ret


def process_raw_state_2(state_prcd_1, min_max_limits):
    """
    Raw state processing 2. Do the following things:
    1. Standarderlize the state using the min max normalization
    
    Args:
        state_prcd_1: python list, 1-D or 2-D
            The processed state by process_raw_state_1. It can be only one 
            sample (1-D) or multiple samples (2-D, where each row is a sample)
        min_max_limits: python list, 2-D, 2*m where m is the number of state 
                        features
            The minimum and maximum possible values for each state feature. 
            The first row is the minimum values and the second row is the maximum
            values.
            
    Return: python list, 1-D or 2-D
        Min max normalized state. It can be only one sample (1-D) or multiple 
        samples (2-D, where each row is a sample).
    """
    state_prcd_1 = np.array(state_prcd_1)
    min_max_limits = np.array(min_max_limits)
    # Do min-max normalization
    std_state = (state_prcd_1 - min_max_limits[0, :]) / (min_max_limits[1, :] -
                                                         min_max_limits[0, :])
    return std_state.tolist()


def process_raw_state_cmbd(raw_state, simTime, start_year, start_mon,
                           start_date, start_day, min_max_limits, is_add_time_to_state):
    """
    Process the raw state by calling process_raw_state_1 and process_raw_state_2
    in order.
    
    Args:
        raw_state: python list, 1-D or 2-D
            The raw observation from the environment. It can be only one 
            sample (1-D) or multiple samples (2-D, where each row is a sample).
        simTime: python list, 1-D 
            Delta seconds from the start time, each item in the list
            corresponds to a row in the state. 
        start_year: int 
            Start year.
        start_mon: int 
            Start month.
        start_date: int.
            The day of the month at the start time. 
        start_day: int 
            The start weekday. 0 is Monday and 6 is Sunday.
        min_max_limits: python list, 2-D, 2*m where m is the number of state 
                        features
            The minimum and maximum possible values for each state feature. 
            The first row is the minimum values and the second row is the maximum
            values.
    
    Return: python list, 1-D or 2-D
        Processed min-max normalized (0 to 1) state It can be only one sample 
        (1-D) or multiple samples (2-D, where each row is a sample).
        State feature order:
        
    """
    if is_add_time_to_state:
        state_after_1 = process_raw_state_1(simTime, raw_state, start_year, start_mon,
                                            start_date, start_day)
    else:
        state_after_1 = copy.deepcopy(raw_state)
    state_after_2 = process_raw_state_2(state_after_1, min_max_limits)

    return state_after_2


def process_raw_state_cmbd_state_v1(raw_state, simTime, start_year, start_mon,
                                    start_date, start_day, min_max_limits, is_add_time_to_state):
    """
    Process the raw state by calling process_raw_state_1 and process_raw_state_2
    in order.

    Args:
        raw_state: python list, 1-D or 2-D
            The raw observation from the environment. It can be only one
            sample (1-D) or multiple samples (2-D, where each row is a sample).
        simTime: python list, 1-D
            Delta seconds from the start time, each item in the list
            corresponds to a row in the state.
        start_year: int
            Start year.
        start_mon: int
            Start month.
        start_date: int.
            The day of the month at the start time.
        start_day: int
            The start weekday. 0 is Monday and 6 is Sunday.
        min_max_limits: python list, 2-D, 2*m where m is the number of state
                        features
            The minimum and maximum possible values for each state feature.
            The first row is the minimum values and the second row is the maximum
            values.

    Return: python list, 1-D or 2-D
        Processed min-max normalized (0 to 1) state It can be only one sample
        (1-D) or multiple samples (2-D, where each row is a sample).
        State feature order:

    """

    TIMESTATE_LEN = 0
    ZONE_NUM = 22
    IAT_FIRST_RAW_IDX = 4
    CLGSSP_FIRST_RAW_IDX = 26
    HTGSSP_FIRST_RAW_IDX = 48
    ENERGY_RAW_IDX = 70
    ret = raw_state[TIMESTATE_LEN: IAT_FIRST_RAW_IDX+TIMESTATE_LEN]
    iats = np.array(
        raw_state[TIMESTATE_LEN + IAT_FIRST_RAW_IDX: TIMESTATE_LEN + IAT_FIRST_RAW_IDX + ZONE_NUM])
    clgssp = np.array(
        raw_state[TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX: TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX + ZONE_NUM])
    htgssp = np.array(
        raw_state[TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX: TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX + ZONE_NUM])
    energy = raw_state[TIMESTATE_LEN + ENERGY_RAW_IDX]
    ret.extend([min(iats), float(np.mean(iats)), max(iats)])
    ret.extend([min(clgssp), float(np.mean(clgssp)), max(clgssp)])
    ret.extend([min(htgssp), float(np.mean(htgssp)), max(htgssp)])
    ret.extend([energy])
    TIMESTATE_LEN = 2
    ret_limit = []
    for i in range(2):
        ret_limit_first = list(min_max_limits[i][0:IAT_FIRST_RAW_IDX+TIMESTATE_LEN])
        iats = min_max_limits[i][TIMESTATE_LEN + IAT_FIRST_RAW_IDX + 1:TIMESTATE_LEN + IAT_FIRST_RAW_IDX + 4]
        clgssp = min_max_limits[i][TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX + 1:TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX + 4]
        htgssp = min_max_limits[i][TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX + 1:TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX + 4]
        energy = min_max_limits[i][TIMESTATE_LEN + ENERGY_RAW_IDX]
        ret_limit_first.extend(iats.tolist())
        ret_limit_first.extend(clgssp.tolist())
        ret_limit_first.extend(htgssp.tolist())
        ret_limit_first.extend([energy])
        ret_limit.append(ret_limit_first)
    if is_add_time_to_state:
        state_after_1 = process_raw_state_1(simTime, ret, start_year, start_mon,
                                            start_date, start_day)
    else:
        state_after_1 = copy.deepcopy(ret)

    state_after_2 = process_raw_state_2(state_after_1, ret_limit)

    return state_after_2



def process_simple_raw_state(raw_state, simTime, start_year, start_mon,
                           start_date, start_day, min_max_limits, is_add_time_to_state):
    TIMESTATE_LEN = 0
    ZONE_NUM = 22
    IAT_FIRST_RAW_IDX = 4
    CLGSSP_FIRST_RAW_IDX = 26
    HTGSSP_FIRST_RAW_IDX = 48
    ENERGY_RAW_IDX = 70
    ret = raw_state[TIMESTATE_LEN: IAT_FIRST_RAW_IDX + TIMESTATE_LEN]
    iats = np.array(
        raw_state[TIMESTATE_LEN + IAT_FIRST_RAW_IDX: TIMESTATE_LEN + IAT_FIRST_RAW_IDX + ZONE_NUM])
    clgssp = np.array(
        raw_state[TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX: TIMESTATE_LEN + CLGSSP_FIRST_RAW_IDX + ZONE_NUM])
    htgssp = np.array(
        raw_state[TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX: TIMESTATE_LEN + HTGSSP_FIRST_RAW_IDX + ZONE_NUM])
    energy = raw_state[TIMESTATE_LEN + ENERGY_RAW_IDX]
    max_iats_index = np.array(iats).argmax()
    ret.extend([iats[max_iats_index]])
    ret.extend([clgssp[max_iats_index]])
    ret.extend([htgssp[max_iats_index]])
    ret.extend([energy])

    TIMESTATE_LEN = 2
    ret_limit = []
    for i in range(2):
        ret_limit_first = list(min_max_limits[i][0:IAT_FIRST_RAW_IDX+TIMESTATE_LEN])
        iats = min_max_limits[i][TIMESTATE_LEN + max_iats_index]
        clgssp = min_max_limits[i][TIMESTATE_LEN + max_iats_index]
        htgssp = min_max_limits[i][TIMESTATE_LEN + max_iats_index]
        energy = min_max_limits[i][TIMESTATE_LEN + ENERGY_RAW_IDX]
        ret_limit_first.append(iats)
        ret_limit_first.append(clgssp)
        ret_limit_first.append(htgssp)
        ret_limit_first.extend([energy])
        ret_limit.append(ret_limit_first)

    if is_add_time_to_state:
        state_after_1 = process_raw_state_1(simTime, ret, start_year, start_mon,
                                            start_date, start_day)
    else:
        state_after_1 = copy.deepcopy(ret)
    state_after_2 = process_raw_state_2(state_after_1, ret_limit)

    return state_after_2

process_raw_state_cmbd_map = {
    "part1_v1": process_raw_state_cmbd,
    "part1_state_v1": process_raw_state_cmbd_state_v1,
    "part1_simple_v1": process_simple_raw_state,
}



class HistoryPreprocessor:
    """Keeps the last k states.

    Useful for seeing the trend of the change, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Args:
        history_length: int
            Number of previous states to prepend to state being processed.
        prcdState_dim: int
            If the dim is 1, all stacked state will be flatten ([x0, x1, xt]) if dim is 2,
            the stacked state will not be flatten ([[x_0], [x_1], [x_t]])

    """

    def __init__(self, history_length, forecast_dim, prcdState_dim=1):
        self._history_length = history_length
        self._flag_start_net = True
        self._stacked_return_net = None
        self._forecast_dim = forecast_dim
        self._prcdState_dim = prcdState_dim

    def process_state_for_network(self, state):
        """Take the current state and return a stacked state with current
        state and the history states. 
        
        Args:
            state: python 1-D list.
                Expect python 1-D list representing the current state.
            prcdState_dim: int
                If dim == 1, 
        
        Return: np.ndarray, dim = 1*m where m is the state_dim * history_length
            Stacked states.
        """
        forecast_state = None
        # Avoid repeated forecast information if prcdState_dim is 1
        if self._forecast_dim > 0 and self._prcdState_dim == 1:
            ob_state = state[0: len(state) - self._forecast_dim]  # Delete the forecast states
            forecast_state = state[-self._forecast_dim:]  # Get the forecast state
            state = ob_state
        state = np.array(state).reshape(1, -1)
        state_dim = state.shape[-1]
        if self._flag_start_net:
            self._stacked_return_net = np.zeros((self._history_length, state_dim))
            self._stacked_return_net[-1, :] = state
            self._flag_start_net = False
        else:
            for i in range(self._history_length - 1):
                self._stacked_return_net[i, :] = \
                    self._stacked_return_net[i + 1, :]
            self._stacked_return_net[-1, :] = state
        # Determine the final state dim
        if self._prcdState_dim == 1:
            ret = np.copy(self._stacked_return_net.flatten().reshape(1, -1))  # Reshape makes the 1-d array to 2-d
        elif self._prcdState_dim == 2:
            orgShape = self._stacked_return_net.shape
            ret = np.copy(self._stacked_return_net.reshape(
                (-1,) + orgShape))  # Reshape makes the 2-d array to 3-d (new axis added to dim 1)
        # Based on prcdState_dim, append forecast info
        if self._forecast_dim > 0 and self._prcdState_dim == 1:
            ret = np.append(ret, np.array(forecast_state).reshape(1, -1), 1)
        return ret

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self._flag_start_net = True

    def get_config(self):
        return {'history_length': self._history_length}


class SteadyHistoryPreprocessor:
    """Keeps the last k states.

    Useful for seeing the trend of the change, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Args:
        history_length: int
            Number of previous states to prepend to state being processed.
        prcdState_dim: int
            If the dim is 1, all stacked state will be flatten ([x0, x1, xt]) if dim is 2,
            the stacked state will not be flatten ([[x_0], [x_1], [x_t]])

    """

    def __init__(self, history_length, forecast_dim, prcdState_dim=1, thre_list=[0.02]):
        self._history_length = history_length
        self._flag_start_net = True
        self._stacked_return_net = None
        self._forecast_dim = forecast_dim
        self._prcdState_dim = prcdState_dim
        self._thre_list = thre_list

    def process_state_for_network(self, state, pre=False):
        """Take the current state and return a stacked state with current
        state and the history states.

        Args:
            state: python 1-D list.
                Expect python 1-D list representing the current state.
            prcdState_dim: int
                If dim == 1,

        Return: np.ndarray, dim = 1*m where m is the state_dim * history_length
            Stacked states.
        """
        forecast_state = None
        # Avoid repeated forecast information if prcdState_dim is 1
        state = np.array(state).reshape(1, -1)
        state_dim = state.shape[-1]
        if self._flag_start_net:
            self._stacked_return_net = np.zeros((self._history_length, state_dim))
            self._stacked_return_net[-1, :] = state
            self._flag_start_net = False
        else:
            for i in range(self._history_length - 1):
                self._stacked_return_net[i, :] = \
                    self._stacked_return_net[i + 1, :]
            self._stacked_return_net[-1, :] = state
        # Determine the final state dim
        if self._prcdState_dim == 1:
            steady_array = self._stacked_return_net[:, 6:7]
            data_process = Pre_process(steady_array)
            steady_flag = data_process.WindowMean(self._thre_list)
            if steady_flag:
                return np.copy(state), True
            else:
                if pre:
                    return np.copy(state), True
                else:
                    return np.copy(state), False

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self._flag_start_net = True

    def get_config(self):
        return {'history_length': self._history_length}


class Pre_process:

    def __init__(self, data):
        self.data = data

    def Normalization(self):
        data_fea = self.data.iloc[:, :]  # 取数据中指标所在的列
        self.data_normalized = data_fea.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  # 归一化


    # 异常值处理
    # 四分位法
    def OutlierQuantile(self):
        cols = self.data.columns
        for i in cols:
            a = self.data[i].quantile(0.75)
            b = self.data[i].quantile(0.25)
            c = self.data[i]
            c[(c >= (a - b) * 1.5 + a) | (c <= b - (a - b) * 1.5)] = np.nan
            self.data.dropna()
            print('四分位法有效数据量：', len(self.data))
            self.lower_quartile = b - (a - b) * 1.5
            self.upper_quartile = (a - b) * 1.5 + a
            print('四分位法阈值:', self.lower_quartile, '~', self.upper_quartile, '\n', '-' * 60)
            return self.data  # 删除后的有效数据

    # 四倍标准差法
    def OutlierStd(self):
        cols = self.data.columns
        for i in cols:
            self.lower_std = self.data[i].mean() + self.data[i].std() * 4
            self.upper_std = self.data[i].mean() - self.data[i].std() * 4
            qua_std = self.data[i]
            qua_std[(qua_std >= self.lower_std) | (qua_std <= self.upper_std)] = np.nan
            self.data.dropna()
            print('四倍标准差法有效数据量：', len(self.data))
            print('四倍标准差法阈值:', self.lower_std, '~', self.upper_std, '\n', '-' * 60)

    #         self.data.to_csv(u'C:\\Users\\Administrator\\Desktop\\1\\四倍标准差筛选.csv')

    # 滑动窗口法
    def WindowMean(self, thre_list=[0.02, 0.02]):
        window_mean = self.data.mean(axis=1)
        a = np.abs((self.data - window_mean) / self.data)
        for i in range(self.data.shape[1]):
            if (a[:, i] > thre_list[i]).sum() > 0:
                return False
        return True
        # self.data_result.to_csv(u'C:\\Users\\Administrator\\Desktop\\7.21\\滑动窗口筛选.csv')

    def WindowStd(self, thre_list=[0.02, 0.02]):
        window_std = np.std(self.data, axis=1)
        for i in range(self.data.shape[1]):
            if (window_std[:, i] > thre_list[i]).sum() > 0:
                return False
        return True
