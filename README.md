# eplus_RL
RL for EnergyPlus based on Rainbow.

This is a project from DQN to Rainbow to control based on EnergyPlus. 

Run
=======
1. Modify src/eplus_env_v1/eplus_env/envs/eplus_models/\*/idf/\* schedule:file path to your own path. I create a "src/eplus_env_v1/eplus_env/envs/eplus_models/parse.py" file to do the multitasks(only need to change
   "main_path" to your own path.
2. Use src/run to run the shell file of different environment.
3. Each run contains train and test with different gym env.
4. "src/run/parse_json.py" and "src/run/parse_sh.py" are used to do multitasks for modifying run_value.json and run.sh 
5. "src/valueBase" contains DQN, DDQN, PQN, Target network, Dueling network, n step network, C51, rainbow which are value-based RL algorithms

Update
======
1. Adding Asyncio for training and testing (cuda is better for training)
2. Update ensemble reinforcement learning for training. Three training environment run concurrently to update one replaybuffer. This enhances robust determination for agent to
   meet balance among three training environment. Details are in Asyn_agent_v1.py.




Rainbow
=======

Rainbow: Combining Improvements in Deep Reinforcement Learning [[1]](#references).


- [x] DQN [[2]](#references)
- [x] Double DQN [[3]](#references)
- [x] Prioritised Experience Replay [[4]](#references)
- [x] Dueling Network Architecture [[5]](#references)
- [x] Multi-step Returns [[6]](#references)
- [x] Distributional RL [[7]](#references)
- [x] Noisy Nets [[8]](#references)


Data-efficient Rainbow [[9]](#references) can be run using the following options (note that the "unbounded" memory is implemented here in practice by manually setting the memory capacity to be the same as the maximum number of timesteps):

Requirements
------------
- python 3.8
- numpy
- gym
- -e eplus-env
- pandas
- matplotlib
- IPython
- collections
- typing


References
----------

[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)  
[2] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[3] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[4] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[5] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[6] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[7] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[8] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
[9] [When to Use Parametric Models in Reinforcement Learning?](https://arxiv.org/abs/1906.05243)  
### Related Papers
1. Zhiang Zhang and Khee Poh Lam. 2018. Practical implementation and evaluation of deep reinforcement learning control for a radiant heating system. In Proceedings of the 5th Conference on Systems for Built Environments (BuildSys '18). ACM, New York, NY, USA, 148-157. DOI: https://doi.org/10.1145/3276774.3276775
2. Zhang Z. A Reinforcement Learning Approach for Whole Building Energy Model Assisted HVAC Supervisory Control[D]. Carnegie Mellon University, 2019.
