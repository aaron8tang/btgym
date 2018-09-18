from btgym import BTgymEnv, BTgymBaseStrategy
import os
import backtrader as bt
import numpy as np
from gym import spaces

from btgym import BTgymEnv, BTgymDataset, BTgymRandomDataDomain
from btgym.algorithms import Launcher

from gym import spaces

env = BTgymEnv(filename='/home/aaron8tang/projects/btgym/examples/data/sh601318.csv',
						 parsing_params=dict(
							 header=None,
						),
						trial_params=dict(
							start_weekdays={0, 1, 2, 3, 4, 5, 6},
							sample_duration={'days': 5, 'hours': 0, 'minutes': 0},
							start_00=False,
							time_gap={'days': 1, 'hours': 0},
							test_period={'days': 5, 'hours': 0, 'minutes': 0},
						),
						episode_params=dict(
							start_weekdays={0, 1, 2, 3, 4, 5, 6},
							sample_duration={'days': 5, 'hours': 0, 'minutes': 0},
							start_00=False,
							time_gap={'days': 1, 'hours': 0},
						),
                         verbose=1, )

done = False

o = env.reset()

while not done:
	action = env.action_space.sample()  # random action
	obs, reward, done, info = env.step(action)
	print('ACTION: {}\nREWARD: {}\nINFO: {}'.format(action, reward, info))

env.close()