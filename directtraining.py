import numpy as np
import pandas as pd
import os

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Input,Reshape, Conv1D, MaxPooling1D
from keras.layers.recurrent import GRU

from keras.optimizers import Adam


class ForexGenius(Callback):
    def __init__(self, weights,actions,input):
        self.model = Sequential()
        self.model.add(Reshape((1,4, ), input_shape=input))
        self.model.add(Conv1D(1, (2,2), activation='relu', padding='same'))
        self.model.add(MaxPooling1D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(GRU(512, return_sequences=True, input_shape=input))
        self.model.add(Dropout(0.2))
        self.model.add(GRU(256, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512))

        # self.model.add(Dense(512, input_shape=input))

        self.model.add(Dropout(0.2))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(64))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        # print(self.model.summary())
        # print(self.model.to_json())
        self.model.compile(optimizer ='adam', loss = 'mse', metrics = ['acc'])
        self.weight_backup = weights
        if os.path.isfile(self.weight_backup):
            self.model.load_weights(self.weight_backup)

    def fit(self,X_train, y_train, batch_size = 10, epochs = 100):
        try:
            self.model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs,callbacks=[self])
        finally:
            self.save()

    def save(self):
        self.model.save_weights(self.weight_backup, overwrite=True)
        print("Save Awesomely")
    def test(self,env):
        self.brain.test(env, nb_episodes=5, visualize=True, verbose=2)

    def act(self, x,
                batch_size=None,
                verbose=0,
                steps=None):
        return self.model.predict(x,
                batch_size=batch_size,
                verbose=verbose,
                steps=steps)

    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        self.save()
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return

dataset = pd.read_csv('/home/vincent/gym-trader/data/EURUSD/raw/EURUSD_Candlestick_1_M_2014labeledtr.csv')
dataset = dataset.dropna()
print(dataset.Volume)
dataset = dataset[dataset.Volume != 0]
X = dataset[['Open', 'High', 'Low', 'Close']]
Y = dataset[['Action']]

split = int(len(dataset)*0.9)
X_train, X_test, y_train, y_test = X[:split], X[split:], Y[:split], Y[split:]

# print(np.array(y_test))
forex = ForexGenius(actions=3,weights='files/forex_direct_weights.h5f',input=X_train.shape[1:])

# print("Act:{}".format(forex.act(x= [[1.2323, 1.2454, 1.23234, 1.23242]], batch_size=None)))
forex.fit(X_train, y_train, batch_size = 10, epochs = 100)