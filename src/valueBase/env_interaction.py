"""
This file contains the wrapper classes for environment interactions. 
"""
import random

import gym
import numpy as np


class ActionSpace(object):

	def __init__(self, action_space):
		self._action_space = action_space
		self.n = len(self._action_space)

	def sample(self):
		return random.choice(self._action_space)



class IWEnvInteract(object):

	def __init__(self, env, ob_state_process_func, action_space, state_dim):
		self.action_space = ActionSpace(action_space)
		self._env = env
		self._ob_state_process_func = ob_state_process_func
		self.observation_space = np.zeros(int(state_dim))


	def reset(self):
		
		return self._interact(mode = 'reset', actions = None) 

	def step(self, actions):

		return self._interact(mode = 'step', actions = actions)


	def end_episode(self):
		return self._env.end_episode() 


	def _interact(self, mode, actions = None):
		ret = [] 
		# Reset the env
		forecast = None 
		env_get = None  
		time = None 
		ob_raw = None 
		is_terminal = None 
		if mode == 'reset':
			env_get = self._env.reset()
		elif mode == 'step':
			actions = [int(np.squeeze(actions))]
			env_get = self._env.step(actions)
		if len(env_get) == 4:
			time, ob_raw, forecast, is_terminal = env_get 
		elif len(env_get) == 3:
			time, ob_raw, is_terminal = env_get 
        # Process and normalize the raw observation
		ob_raw = self._ob_state_process_func(ob_raw) 
		if forecast is not None:
        	# Add forecast info to ob_this_raw so they can be normalized
			ob_raw.extend(forecast) 
		ret.append(time) 
		ret.append(ob_raw) 
		ret.append(is_terminal) 
		return ret

	@property
	def min_max_limits(self):
		"""
        Return the min_max_limits for all state features.

        Return: python list of tuple.
            In the order of the state features, and the index 0 of the tuple
            is the minimum value, index 1 is the maximum value.
        """
		return self._env.min_max_limits

	@property
	def start_year(self):
		"""
        Return the EnergyPlus simulaton year.

        Return: int
        """
		return self._env.start_year

	@property
	def start_mon(self):
		"""
        Return the EnergyPlus simulaton start month.

        Return: int
        """
		return self._env.start_mon

	@property
	def start_day(self):
		"""
        Return the EnergyPlus simulaton start day of the month.

        Return: int
        """
		return self._env.start_day

	@property
	def start_weekday(self):
		"""
        Return the EnergyPlus simulaton start weekday. 0 is Monday, 6 is Sunday.

        Return: int
        """
		return self._env.start_weekday

	@property
	def env_name(self):
		return self._env.env_name

	def set_max_res_to_keep(self, num):
		self._env.set_max_res_to_keep(num)

	def close(self):
		"""
        This method must be called after finishing using the environment
        because EnergyPlus runs on a different process. EnergyPlus process
        won't terminating until this method is called.
        """
		self._env.end_env()

	@property
	def model_path(self):
		return self._env.model_path