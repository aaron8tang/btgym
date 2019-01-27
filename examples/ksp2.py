import itertools
import random
import os

import sys

sys.path.insert(0, '..')

import IPython.display as Display
import PIL.Image as Image

from gym import spaces
from btgym import BTgymEnv


def show_rendered_image(rgb_array):
	"""
	Convert numpy array to RGB image using PILLOW and
	show it inline using IPykernel.
	"""
	Display.display(Image.fromarray(rgb_array))


def render_all_modes(env):
	"""
	Retrieve and show environment renderings
	for all supported modes.
	"""
	for mode in env.metadata['render.modes']:
		print('[{}] mode:'.format(mode))
		show_rendered_image(env.render(mode))


def take_some_steps(env, some_steps):
	"""Just does it. Acting randomly."""
	for step in range(some_steps):
		rnd_action = env.action_space.sample()
		o, r, d, i = env.step(rnd_action)
		if d:
			print('Episode finished,')
			break
	print(step + 1, 'steps made.\n')


'''
env = BTgymEnv(
    filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
    state_shape={'raw': spaces.Box(low=-100, high=100,shape=(30,4))},
    skip_frame=5,
    timeframe=1440,
    start_cash=100,
    render_ylabel='Price Lines',
    render_size_episode=(12,8),
    render_size_human=(8, 3.5),
    render_size_state=(10, 3.5),
    render_dpi=75,
    verbose=0,
)
'''

env = BTgymEnv(
	filename='./data/sh601318.csv',
	start_weekdays={0, 1, 2, 3, 4},
	episode_duration={'days': 90, 'hours': 0, 'minutes': 0},
	# Want to start every episode at the begiining of the day:
	start_00=True,
	time_gap={'days': 1},

	start_cash=100,

	render_ylabel='Price Lines',
	render_size_episode=(12, 8),
	render_size_human=(8, 3.5),
	render_size_state=(10, 3.5),
	render_dpi=75,

	verbose=1,
)

o = env.reset()
take_some_steps(env, 10000)
render_all_modes(env)