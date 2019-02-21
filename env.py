"""
From TradingGym Repository

"""
import os
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from colour import Color
from gym import Env
from gym.spaces import Dict
from mpl_finance import candlestick_ohlc

from btgym import BTgymEnv
from utils.gym import spaces
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc, volume_overlay3
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from keras.applications.inception_v3 import preprocess_input
from collections import OrderedDict

class TradingG(BTgymEnv):
    def __init__(self, **kwargs):
        super(TradingG, self).__init__(**kwargs)
        self.dts = []
    def reset(self, show=False, **kwargs):
        observation = super(TradingG, self).reset(**kwargs)
        #render_all_modes(self)
        return self.getImageArray(observation['my'],show=show)
    def step(self, action, show=False):
        observation, reward, done, info = super(TradingG, self).step(OrderedDict([('default_asset', action)]))

        if done:
            print("Step: {} {} Broker Cash: {} Broker Value: {} Reward: {} Action: {} Message: {}".format(info[0]["step"], info[0]["time"], info[0]["broker_cash"], info[0]["broker_value"], reward, info[0]["action"], info[0]["broker_message"]))

        return self.getImageArray(observation['my'],action,show=show), reward, done, info
    def datetime_range(self,start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta

    def getImageArray(self,observation, action=0, show=False):
        """
            Get the time range
        """
        if len(self.dts) == 0:
            dts = [mdates.date2num(dt) for dt in self.datetime_range(datetime(2016, 9, 1, 7), datetime(2016, 9, 1, 9), timedelta(minutes=1))]
            dts = dts[:len(observation)]
            self.dts = np.reshape(np.array([dts]),(30,1))
        observation = np.append(self.dts,observation,axis=1)
        # observation = np.append(self.dts,observation,axis=1)

        # dpi = 60
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [3,1,1,0.2]},figsize=(384/dpi, 288/dpi),dpi=dpi)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [3,1,1,0.2]})

        ax1.xaxis_date()
        candlestick_ohlc(ax1, observation, width=0.0005, colorup='green', colordown='red')

        """
        Set the Volume
        """

        def default_color(index, open_price, low, high,close_price):
            return 'r' if open_price[index] > close_price[index] else 'g'
        x = np.arange(len(observation))
        candle_colors = [default_color(i, observation[:,1], observation[:,2], observation[:,3], observation[:,4]) for i in x]
        ax2.bar(observation[:,0],observation[:,5],0.0005, color=candle_colors)

        """
        Set the Unrealized PNL
        """
        ax3.fill_between(observation[:,0],observation[:,6])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        """
        Set the actions
        """

        try:
            self.actions
        except:
            self.actions = np.zeros((30,))
        else:
            self.actions = np.append(self.actions,action)
            self.actions = np.delete(self.actions,0)

        def action_color(index,a):
            if a[index] == 0:
                return 'c'
            elif a[index] == 1:
                return 'g'
            elif a[index] == 2:
                return 'r'
            elif a[index] == 3:
                return 'b'
            print(a[index])

        bar_colors  = [action_color(i,self.actions) for i in x]
        # print("Colors: {}".format(bar_colors))
        ax4.bar(observation[:,0],np.ones((30,)),0.0005, color=bar_colors)
        # plt.xticks(rotation=90)
        # fig.tight_layout()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if show == True:
            plt.show()
        plt.close(fig)
        return preprocess_input(data)


