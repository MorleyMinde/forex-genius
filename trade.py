#!/usr/bin/env python

import argparse

from v20 import V20Timeout, V20ConnectionError
from v20.account import Account

import common.config
import common.args
from datetime import datetime, timezone
import curses
import random
import time
import numpy as np

from forex import ForexGenius
from order.view import print_order_create_response_transactions
import os.path
import dateutil.parser
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

class CandlePrinter():
    def __init__(self):

        self.field_width = {
            'time' : 19,
            'price' : 8,
            'volume' : 6,
        }
        self.prev_observation = np.zeros((30, 5))
        self.weights = "files/forex_model.h5f"
        self.agent = self.load_model()
        self.agent_update_time = os.path.getmtime(self.weights)
        self.action = 3
        self.external_observation = np.zeros((30,3))
        self.final_observation = np.zeros((30,8))
        self.current_trade = None
        self.show_orders()
        self.dts = []

    def load_model(self):
        if self.agent != None:
            self.agent.dispose()
        self.agent = ForexGenius(actions=4,weights=self.weights)

    def set_instrument(self, instrument):
        self.instrument = instrument

    def set_granularity(self, granularity):
        self.granularity = granularity

    def set_candles(self, candles):
        self.candles = candles

    def update_candles(self, candles):
        new = candles[0]
        last = self.candles[-1]

        # Candles haven't changed
        if new.time == last.time and new.volume == last.time:
            return False

        # Update last candle
        self.candles[-1] = candles.pop(0)

        # Add the newer candles
        self.candles.extend(candles)

        # Get rid of the oldest candles
        self.candles = self.candles[-self.max_candle_count():]


        self.use_agent()
            # print(observation)
        return True
    def update_external_data(self):
        api = self.get_api()

        response = api.account.get(self.args.config.active_account)

        #
        # Extract the Account representation from the response and use
        # it to create an Account wrapper
        #
        account = response.get("account", 200)

        #
        # Extract the Account representation from the response and use
        # it to create an Account wrapper
        #
        # account = Account(response.get("account", 200))
        response = api.account.changes(
            self.args.config.active_account,
            sinceTransactionID=int(account.lastTransactionID)
        )

        res = response.get("state",200)
        action = 0.0
        if self.action == 1:
          action = 1.0
        if self.action == 2:
          action = -1.0
        pos_period = 0.0
        self.current_trade = None
        for trade in account.trades:
            self.current_trade = trade
        if self.current_trade != None:
            a = datetime.now(timezone.utc)
            b = dateutil.parser.parse(self.current_trade.openTime)
            c = a - b
            pos_period = divmod(c.days * 86400 + c.seconds, 60)
            pos_period = pos_period[0]
        return [res.unrealizedPL,action,pos_period]
        #return [res.unrealizedPL]
    def datetime_range(self,start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta
    def getImageArray(self,observation, action):
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
        # plt.show()
        plt.close(fig)
        return preprocess_input(data)

    def use_agent(self):
        observation = np.zeros((30, 5))
        i = 0;
        # print("Candle Size: {}".format(observation))
        for candle in self.candles:
            c = getattr(candle, "mid", None)
            observation[i][0] = c.o
            observation[i][1] = c.h
            observation[i][2] = c.l
            observation[i][3] = c.c
            observation[i][4] = candle.volume
            i += 1
        self.external_observation = np.delete(self.external_observation, 1, 0)
        if self.external_observation.shape[0] < 29:
            self.external_observation = np.append(np.zeros((29 - self.external_observation.shape[0],3)),self.external_observation,axis=0)
        # self.external_observation = np.insert(self.external_observation, 30, self.update_external_data(),axis=-1)
        self.external_observation = np.append(self.external_observation, [self.update_external_data()], axis = 0)
        self.prev_observation = observation
        final_observation = np.concatenate((observation,self.external_observation), axis = 1)

        self.final_observation = final_observation
        if self.agent_update_time < os.path.getmtime(self.weights):
            print("Updating Model On Action: {}".format(self.action))
            self.agent = self.load_model()
            self.agent_update_time = os.path.getmtime(self.weights)
        image_data = self.getImageArray(final_observation[:,:6],self.action)
        image_data = image_data.reshape((1,)+image_data.shape)
        predicted_matrix = self.agent.act(image_data)
        action = np.argmax(predicted_matrix)

        if self.action != action:
            print("Changing Action:{} to {}".format(self.action,action))
            if action != 0:
                self.close_orders()

            if action == 1:
                self.buy()
            if action == 2:
                self.sell()
            self.action = action

    def show_orders(self):
        api = self.get_api()
        response = api.trade.list_open(self.args.config.active_account)

        for trade in reversed(response.get("trades", 200)):
            self.current_trade = trade
            if trade.currentUnits < 0:
                self.action = 2
            if trade.currentUnits > 0:
                self.action = 1

        return

    def close_orders(self):
        api = self.get_api()
        response = api.trade.list_open(self.args.config.active_account)

        for trade in reversed(response.get("trades", 200)):
            print(trade)
            print("-" * 80)
            response = api.trade.close(
                self.args.config.active_account,
                trade.id
            )

            print(
                "Response: {} ({})\n".format(
                    response.status,
                    response.reason
                )
            )

            print_order_create_response_transactions(response)
        self.current_trade = None
        return

    def buy(self):
        self.order(5000)
    def sell(self):
        self.order(-5000)
    def order(self,units):
        api = self.get_api()
        response = api.order.market(
            self.args.config.active_account,instrument='EUR_USD',units=units
        )
        print("Response: {} ({})".format(response.status, response.reason))
        print("")
        print_order_create_response_transactions(response)
        self.show_orders()
    def get_api(self):
        parser = argparse.ArgumentParser()

        #
        # The config object is initialized by the argument parser, and contains
        # the REST APID host, port, accountID, etc.
        #
        common.config.add_argument(parser)

        parser.add_argument(
            "instrument",
            type=common.args.instrument,
            help="The instrument to get candles for"
        )

        parser.add_argument(
            "--granularity",
            default=None,
            help="The candles granularity to fetch"
        )

        self.args = parser.parse_args()

        account_id = self.args.config.active_account

        #
        # The v20 config object creates the v20.Context for us based on the
        # contents of the config file.
        #
        return self.args.config.create_context()
    def max_candle_count(self):
        return 30
    def last_candle_time(self):
        return self.candles[-1].time
    def render(self):
        y = 3

        for candle in self.candles:
            # print("{}: {}".format(y,candle.time.split(".")[0]));
            time = candle.time.split(".")[0]
            volume = candle.volume

            for price in ["mid", "bid", "ask"]:
                c = getattr(candle, price, None)

                if c is None:
                    continue

                candle_str = (
                    "{:>{width[time]}} {:>{width[price]}} "
                    "{:>{width[price]}} {:>{width[price]}} "
                    "{:>{width[price]}} {:>{width[volume]}}"
                ).format(
                    time,
                    c.o,
                    c.h,
                    c.l,
                    c.c,
                    volume,
                    width=self.field_width
                )
                # print("{} {} {} {} {} {}".format(time,c.o,c.h,c.l,c.c,volume))

                y += 1

                break
    def poll_candles(self):
        api = self.get_api()
        kwargs = {}

        if self.args.granularity is not None:
            kwargs["granularity"] = self.args.granularity



        #
        # The printer decides how many candles can be displayed based on the size
        # of the terminal
        #
        kwargs["count"] = self.max_candle_count()

        response = api.instrument.candles(self.args.instrument, **kwargs)

        if response.status != 200:
            print(response)
            print(response.body)
            return

        #
        # Get the initial batch of candlesticks to display
        #
        instrument = response.get("instrument", 200)

        granularity = response.get("granularity", 200)

        self.set_instrument(instrument)

        self.set_granularity(granularity)

        self.set_candles(
            response.get("candles", 200)
        )

        self.render()

        #
        # Poll for candles updates every second and redraw
        # the results
        #
        while True:
            time.sleep(58)
            # time.sleep(1)

            kwargs = {
                'granularity': granularity,
                'fromTime': self.last_candle_time()
            }

            try:
                response = api.instrument.candles(self.args.instrument, **kwargs)

                candles = response.get("candles", 200)

                if self.update_candles(candles):
                    self.render()
            except V20Timeout:
                print("Caught V20 Timeout")
            except V20ConnectionError:
                print("Caught V20 Connection Error")

def main():
    """
    Create an API context, and use it to fetch candles for an instrument.

    The configuration for the context is parsed from the config file provided
    as an argumentV
    """

    parser = argparse.ArgumentParser()

    #
    # The config object is initialized by the argument parser, and contains
    # the REST APID host, port, accountID, etc.
    #
    common.config.add_argument(parser)

    parser.add_argument(
        "instrument",
        type=common.args.instrument,
        help="The instrument to get candles for"
    )

    parser.add_argument(
        "--granularity",
        default=None,
        help="The candles granularity to fetch"
    )

    args = parser.parse_args()

    print("Innitial: {} Args: {}".format(common.args.instrument,args))
    account_id = args.config.active_account

    #
    # The v20 config object creates the v20.Context for us based on the
    # contents of the config file.
    #
    api = args.config.create_context()

    printer = CandlePrinter()


    printer.poll_candles()

if __name__ == "__main__":
    main()


"""
Example:

python3 candles_poll.py --granularity M1 EUR_USD

"""