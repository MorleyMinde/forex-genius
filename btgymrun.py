from collections import OrderedDict

from forex import ForexGenius
from gym import spaces
from btgym import BTgymEnv, BTgymDataset

import backtrader as bt
import sys

from rl.callbacks import Callback
from strategy import MyStrategy
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc, volume_overlay3
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0,'..')


class TradingG(BTgymEnv):
    def __init__(self, **kwargs):
        super(TradingG, self).__init__(**kwargs)
        self.dts = []
    def reset(self, **kwargs):
        observation = super(TradingG, self).reset(**kwargs)
        #render_all_modes(self)
        data = self.getImageArray(observation['my'])
        return data
    def datetime_range(self,start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta

    def getImageArray(self,observation, action=0):
        """
            Get the time range
        """
        if len(self.dts) == 0:
            dts = [mdates.date2num(dt) for dt in self.datetime_range(datetime(2016, 9, 1, 7), datetime(2016, 9, 1, 9), timedelta(minutes=1))]
            dts = dts[:len(observation)]
            self.dts = np.reshape(np.array([dts]),(30,1))
        observation = np.append(self.dts,observation,axis=1)
        # observation = np.append(self.dts,observation,axis=1)

        dpi = 60
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [3,1,1,0.2]},figsize=(384/dpi, 288/dpi),dpi=dpi)

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
        # plt.show()
        plt.close(fig)
        return data/255
    def step(self, action):
        observation, reward, done, info = super(TradingG, self).step(OrderedDict([('default_asset', action)]))

        if done:
            print("Step: {} {} Broker Cash: {} Broker Value: {} Reward: {} Action: {} Message: {}".format(info[0]["step"], info[0]["time"], info[0]["broker_cash"], info[0]["broker_value"], reward, info[0]["action"], info[0]["broker_message"]))

        return self.getImageArray(observation['my'],action), reward, done, info

class History(Callback):
	def on_train_begin(self, logs={}):
            print('on_train_begin Logs: {}'.format(logs))
            return

	def on_train_end(self, logs={}):
         print('on_train_end Logs: {}'.format(logs))
         return

	def on_episode_begin(self, epoch, logs={}):
         print('on_epoch_begin Logs: {}'.format(logs))
         return

	def on_episode_end(self, epoch, logs={}):
         print('on_epoch_end Logs: {}'.format(logs))
         return

	def on_batch_begin(self, batch, logs={}):
            return

	def on_batch_end(self, batch, logs={}):
            return

params = dict(
        # CSV to Pandas params.
        parsing_params=dict(
        # CSV to Pandas params.
            sep=';',
            header=0,
            index_col=0,
            parse_dates=True,
            names=['open', 'high', 'low', 'close', 'volume'],

            # Pandas to BT.feeds params:
            timeframe=1,  # 1 minute.
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
        ),

        # Random-sampling params:
        start_weekdays=[0, 1, 2, 3, ],  # Only weekdays from the list will be used for episode start.
        start_00=True,  # Episode start time will be set to first record of the day (usually 00:00).
        episode_duration={'days': 0, 'hours': 2, 'minutes': 0},
        # time_gap={'days': 0, 'hours': 5, 'minutes': 55},
    )

MyDataset = BTgymDataset(
    filename='data/EURUSD/EURUSD_Candlestick_1_M_2005.csv',
    # filename='btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
    **params,
)

MyCerebro = bt.Cerebro()
MyCerebro.addstrategy(MyStrategy,
                      state_shape={
                          'raw': spaces.Box(low=0,high=1,shape=(30,4)),
                          'my': spaces.Box(low=0,high=2,shape=(30,6))
                      },

                      state_low=None,
                      state_high=None,
                      drawdown_call=100,
                      )

MyCerebro.broker.setcash(100.0)
MyCerebro.broker.setcommission(commission=0.0)
MyCerebro.addsizer(bt.sizers.SizerFix, stake=10)
MyCerebro.addanalyzer(bt.analyzers.DrawDown)


env = TradingG(
               dataset = MyDataset,
                engine=MyCerebro,
                   episode_duration={'days': 1, 'hours': 0, 'minutes': 0},
                         port=5557,
                        data_port=5002,
                         verbose=1,)
print(env.observation_space)


agent = ForexGenius(actions=4,weights='files/forex_complete_model.h5f')
agent.fit(env,nb_steps=1000000)

# agent.test(env)

env.close()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# After training is done, we save the final weights.

# Finally, evaluate our algorithm for 5 episodes.
# agent.test(env)

"""

tar -cjvf archive.tar.bz2 stuff
rsync -av --progress files/forex_complete_model.h5f vincentminde@72.14.186.65:/home/vincentminde/forex-genius/files/

"""