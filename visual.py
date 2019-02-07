import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc, volume_overlay3
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TradingG:
    def __init__(self, **kwargs):
        super(TradingG, self).__init__(**kwargs)
        self.dts = []
    def datetime_range(self,start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta

    def getImageArray(self,observation, action=None):
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
                return 'w'
            if a[index] == 1:
                return 'g'
            if a[index] == 2:
                return 'r'
            if a[index] == 3:
                return 'b'

        bar_colors  = [action_color(i,self.actions) for i in x]
        ax4.bar(observation[:,0],np.ones((30,)),0.0005, color=bar_colors)
        # plt.xticks(rotation=90)
        # fig.tight_layout()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #plt.show()
        plt.close(fig)
        return data/255


tr = TradingG()
tr.getImageArray(np.array([[1.32127,1.32141,1.32124,1.32139,0.45210001,0.0]
,[1.32141,1.32151,1.32128,1.32131,0.74890002,0.0]
,[1.32128,1.32128,1.3211,1.32114,0.53520001,0.0]
,[1.3213,1.32139,1.32123,1.32135,0.75459998,0.0]
,[1.32131,1.32143,1.32121,1.32143,0.4235,0.0]
,[1.32141,1.32142,1.32125,1.3213,0.677,0.0]
,[1.32127,1.32139,1.32113,1.32122,0.41020001,0.0]
,[1.32139,1.32139,1.32103,1.32132,0.61170001,0.0]
,[1.32123,1.32132,1.32116,1.32132,0.29370001,0.0]
,[1.32127,1.32134,1.32121,1.32126,0.59320001,0.0]
,[1.32119,1.32124,1.32103,1.32107,0.60970001,0.0]
,[1.32106,1.32113,1.32094,1.32102,0.522,0.0]
,[1.32089,1.3211,1.32089,1.32107,0.5865,0.0]
,[1.32096,1.32096,1.3208,1.3208,0.488,0.0]
,[1.3209,1.32107,1.32076,1.32106,0.6515,0.0]
,[1.32094,1.32105,1.32089,1.32094,0.34239999,0.0]
,[1.32102,1.32109,1.32081,1.32085,0.1945,0.0]
,[1.32106,1.32113,1.32097,1.32101,0.81620001,0.0]
,[1.32119,1.32119,1.32102,1.32112,0.3265,0.0]
,[1.32106,1.32115,1.32074,1.32103,0.44179999,0.0]
,[1.32101,1.32101,1.32078,1.32093,0.556,0.0]
,[1.32095,1.32108,1.32085,1.32097,0.742,0.0]
,[1.32109,1.32113,1.3209,1.321,0.665,0.0]
,[1.32093,1.321,1.32088,1.32094,-0.39270001,0.0]
,[1.32086,1.32094,1.32085,1.3209,-0.51729999,0.0]
,[1.32107,1.32107,1.32087,1.32103,-0.37120001,0.0]
,[1.321,1.32104,1.32075,1.32093,0.47179999,0.0]
,[1.32112,1.32119,1.32097,1.32102,0.36,0.0]
,[1.32093,1.32105,1.32085,1.32094,0.42179999,0.0]
,[1.32102,1.32109,1.32087,1.32103,0.45170001,20.0]]))