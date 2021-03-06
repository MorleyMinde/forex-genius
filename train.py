
from env import TradingG
from forex import ForexGenius
from gym import spaces
from btgym import BTgymEnv, BTgymDataset

import backtrader as bt
import sys

from strategy import MyStrategy


sys.path.insert(0,'..')


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
        episode_duration={'days': 0, 'hours': 1, 'minutes': 0},
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
                         port=5557,
                        data_port=5002,
                         verbose=1,)
print(env.observation_space)


agent = ForexGenius(actions=4,weights='files/forex_model.h5f')
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
rsync -av --progress files/forex_model.h5f vincentminde@72.14.186.65:/home/vincentminde/forex-genius/files/

"""