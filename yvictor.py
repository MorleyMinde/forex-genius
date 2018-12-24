import pandas as pd

from env import TradingGym
from forex import ForexGenius



for row in range(1, 20):
    print("Started the Iteration: {}".format(row))
    df = pd.read_hdf('files/SGXTW.h5', 'STW')
    env = TradingGym(env_id='training_v1', obs_data_len=256, step_len=128,
                       df=df, fee=0.1, max_position=5, deal_col_name='Price',
                       feature_names=['Price', 'Volume',
                                      'Ask_price','Bid_price',
                                      'Ask_deal_vol','Bid_deal_vol',
                                      'Bid/Ask_deal', 'Updown'])
    agent = ForexGenius(actions=env.action_space.n,weights='files/forex_trade_g_weights.h5f')
    agent.fit(env)
    print("Finished the Iteration: {}".format(row))
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# After training is done, we save the final weights.

# Finally, evaluate our algorithm for 5 episodes.
# agent.test(env)