from collections import OrderedDict

from btgym import BTgymEnv
from forex import ForexGenius
from gym import spaces
from btgym import BTgymEnv, BTgymBaseStrategy, BTgymDataset

import itertools
import random
import os

import numpy as np

import sys
sys.path.insert(0,'..')

import IPython.display as Display
import PIL.Image as Image

import matplotlib
import matplotlib.pyplot as plt


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
    print(step+1, 'steps made.\n')


class TradingG(BTgymEnv):
    def reset(self, **kwargs):
        observation = super(TradingG, self).reset(**kwargs)
        render_all_modes(self)
        return observation['raw']
    def step(self, action):
        observation, reward, done, info = super(TradingG, self).step(OrderedDict([('default_asset', action)]))
        print(info)
        return observation['raw'], reward, done, info

env = TradingG(filename='btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
                   episode_duration={'days': 2, 'hours': 23, 'minutes': 55},
                    render_ylabel='Price Lines',
    render_size_episode=(12,8),
    render_size_human=(8, 3.5),
    render_size_state=(10, 3.5),
    render_dpi=75,
                         drawdown_call=50,
                         state_shape=dict(raw=spaces.Box(low=0,high=1,shape=(30,4))),
                         port=5555,
                         verbose=1,)
print(env.observation_space)


agent = ForexGenius(actions=4,weights='files/forex_weights.h5f')

agent.fit(env)

env.close()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# After training is done, we save the final weights.

# Finally, evaluate our algorithm for 5 episodes.
# agent.test(env)