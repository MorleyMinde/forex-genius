#!/usr/bin/env python

import argparse

from v20 import V20Timeout, V20ConnectionError

import common.config
import common.args
from datetime import datetime
import curses
import random
import time
import numpy as np

from forex import ForexGenius
from order.view import print_order_create_response_transactions
import os.path


class CandlePrinter():
    def __init__(self):

        self.field_width = {
            'time' : 19,
            'price' : 8,
            'volume' : 6,
        }
        self.prev_observation = np.zeros((30, 5))
        self.weights = "files/forex_weights.h5f"
        self.agent = ForexGenius(actions=4,weights=self.weights)
        self.agent_update_time = os.path.getmtime(self.weights)
        self.action = 3
        self.show_orders()

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

        if not np.all(np.equal(self.prev_observation, observation)) :
            self.prev_observation = observation
            #print("Act: {}".format(observation))
            action = np.argmax(self.agent.act(np.reshape(observation,(1,1,30,5))))
            if self.agent_update_time < os.path.getmtime(self.weights):
                print("Updating Model On Action: {}".format(self.action))
                self.agent = ForexGenius(actions=4,weights=self.weights)
                self.agent_update_time = os.path.getmtime(self.weights)
            if self.action != action:
                print("Changing Action:{} to {}".format(self.action,action))
                if action == 3:
                    self.close_orders()

                if action == 1:
                    if self.action == 2:
                        self.close_orders()
                    self.buy()
                if action == 2:
                    if self.action == 1:
                        self.close_orders()
                    self.sell()
                self.action = action
    def show_orders(self):
        api = self.get_api()
        response = api.trade.list_open(self.args.config.active_account)

        for trade in reversed(response.get("trades", 200)):
            if trade.currentUnits < 0:
                self.action = 2
            if trade.currentUnits > 0:
                self.action = 1

        return

    def close_orders(self):
        print("Closing Order")
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


        return

    def buy(self):
        self.order(10000)
    def sell(self):
        self.order(-10000)
    def order(self,units):
        api = self.get_api()
        response = api.order.market(
            self.args.config.active_account,instrument='EUR_USD',units=units
        )
        print("Response: {} ({})".format(response.status, response.reason))
        print("")
        print_order_create_response_transactions(response)
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
            time.sleep(1)

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