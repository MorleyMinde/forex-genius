from collections import OrderedDict

from btgym import BTgymEnv
from forex import ForexGenius
from gym import spaces
from btgym import BTgymEnv, BTgymBaseStrategy, BTgymDataset

import numpy as np

import sys
sys.path.insert(0,'..')


class TradingG(BTgymEnv):
    def reset(self, **kwargs):
        observation = super(TradingG, self).reset(**kwargs)
        #render_all_modes(self)
        return observation['raw']
    def step(self, action):
        observation, reward, done, info = super(TradingG, self).step(OrderedDict([('default_asset', action)]))
        print(observation)
        print("Step: {} Broker Cash: {} Broker Value: {} Drawdown: {} Max Drawdown: {} Action: {} Message: {} Time: {}"
              .format(info[0]["step"], info[0]["broker_cash"], info[0]["broker_value"], info[0]["drawdown"], info[0]["max_drawdown"], info[0]["action"], info[0]["broker_message"], info[0]["time"]))
        return observation['raw'], reward, done, info

params = dict(
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

        # Random-sampling params:
        start_weekdays=[0, 1, 2, 3, ],  # Only weekdays from the list will be used for episode start.
        start_00=True,  # Episode start time will be set to first record of the day (usually 00:00).
        episode_duration={'days': 2, 'hours': 23, 'minutes': 55},
        time_gap={'days': 0, 'hours': 5, 'minutes': 55},
    )

MyDataset = BTgymDataset(
    filename='data/EURUSD/EURUSD_Candlestick_1_M_2003.csv',
    **params,
)

class MyStrategy(BTgymBaseStrategy):
    """
    Example subclass of BT server inner computation strategy.
    """

    def get_raw_state(self):
        """
        Default state observation composer.

        Returns:
             and updates time-embedded environment state observation as [n,4] numpy matrix, where:
                4 - number of signal features  == state_shape[1],
                n - time-embedding length  == state_shape[0] == <set by user>.

        Note:
            `self.raw_state` is used to render environment `human` mode and should not be modified.

        """
        self.raw_state = np.row_stack(
            (
                np.frombuffer(self.data.open.get(size=self.time_dim)),
                np.frombuffer(self.data.high.get(size=self.time_dim)),
                np.frombuffer(self.data.low.get(size=self.time_dim)),
                np.frombuffer(self.data.close.get(size=self.time_dim)),
                np.frombuffer(self.data.volume.get(size=self.time_dim)),
            )
        ).T

        return self.raw_state

env = TradingG(#filename='data/EURUSD/EURUSD_Candlestick_1_M_2003.csv', #'btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
               dataset = MyDataset,
                strategy = MyStrategy,
                   episode_duration={'days': 2, 'hours': 23, 'minutes': 55},
                         drawdown_call=50,
                         state_shape=dict(raw=spaces.Box(low=0,high=1,shape=(30,5))),
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