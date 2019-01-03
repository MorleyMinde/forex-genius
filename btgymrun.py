from collections import OrderedDict

from forex import ForexGenius
from gym import spaces
from btgym import BTgymEnv, BTgymDataset

import backtrader as bt
import sys

from rl.callbacks import Callback
from strategy import MyStrategy

sys.path.insert(0,'..')


class TradingG(BTgymEnv):
    def reset(self, **kwargs):
        observation = super(TradingG, self).reset(**kwargs)
        #render_all_modes(self)
        return observation['my']
    def step(self, action):
        observation, reward, done, info = super(TradingG, self).step(OrderedDict([('default_asset', action)]))
        # print(observation)
        if done:
            # print(self.previous)
            print("Done Step: {} {} Broker Cash: {} Broker Value: {} Drawdown: {} Max Drawdown: {} Action: {} Message: {} Reward: {}".format(info[0]["step"], info[0]["time"], info[0]["broker_cash"], info[0]["broker_value"], info[0]["drawdown"], info[0]["max_drawdown"], info[0]["action"], info[0]["broker_message"], reward))
        self.previous = "Done Step Previous: {} {} Broker Cash: {} Broker Value: {} Drawdown: {} Max Drawdown: {} Action: {} Message: {} Reward: {}".format(info[0]["step"], info[0]["time"], info[0]["broker_cash"], info[0]["broker_value"], info[0]["drawdown"], info[0]["max_drawdown"], info[0]["action"], info[0]["broker_message"], reward)
        return observation['my'], reward, done, info

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
        episode_duration={'days': 2, 'hours': 23, 'minutes': 55},
        time_gap={'days': 0, 'hours': 5, 'minutes': 55},
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
                          'my': spaces.Box(low=0,high=10000000000,shape=(30,5))
                      },

                      state_low=None,
                      state_high=None,
                      drawdown_call=10,
                      )

MyCerebro.broker.setcash(10000.0)
MyCerebro.broker.setcommission(commission=0.001)
MyCerebro.addsizer(bt.sizers.SizerFix, stake=100)
MyCerebro.addanalyzer(bt.analyzers.DrawDown)


env = TradingG(
               dataset = MyDataset,
                engine=MyCerebro,
                   episode_duration={'days': 1, 'hours': 0, 'minutes': 0},
                         port=5557,
                        data_port=5002,
                         verbose=1,)
print(env.observation_space)


agent = ForexGenius(actions=4,weights='files/forex_weights.h5f')
agent.fit(env,nb_steps=200000)

# agent.test(env)

env.close()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# After training is done, we save the final weights.

# Finally, evaluate our algorithm for 5 episodes.
# agent.test(env)