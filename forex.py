import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Input,Reshape, GlobalAveragePooling2D
from keras.layers.recurrent import GRU
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.agents.dqn import DQNAgent
from rl.policy import MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy, BoltzmannQPolicy, GreedyQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback
from keras import applications
import os
import subprocess
from keras.models import load_model
from keras.models import Model

class ForexGenius(Callback):
    def __init__(self, weights,actions, training=True):
        #model = None
        if os.path.isfile(weights):
            self.model = load_model(weights)
        else:
            base_model = applications.InceptionV3(weights='imagenet', pooling='avg')
            base_model.trainable = False
            for layer in base_model.layers:
                layer.trainable = False

            self.model = Sequential()
            self.model.add(base_model)
            self.model.add(Dense(1024))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(512))
            self.model.add(Dropout(0.2))
            self.model.add(Activation('relu'))
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))
            self.model.add(Dense(128))
            self.model.add(Activation('relu'))
            self.model.add(Dense(64))
            self.model.add(Activation('relu'))
            self.model.add(Dense(64))
            self.model.add(Activation('relu'))
            self.model.add(Dense(actions))
        # model.add(Activation('softmax'))
        # print(model.summary())
        # print(model.to_json())
        memory = SequentialMemory(limit=50000, window_length=1)
        # policy = MaxBoltzmannQPolicy()
        # policy = BoltzmannGumbelQPolicy()
        # policy = BoltzmannQPolicy()
        # policy = GreedyQPolicy()
        policy = EpsGreedyQPolicy()

        # model = load_model('/home/vincent/gym-trader/files/forex_complete_model.h5f')
        # self.brain = DQNAgent(model=self.model, nb_actions=actions, memory=memory, nb_steps_warmup=32, target_model_update=1e-2, policy=policy, enable_double_dqn=True, enable_dueling_network=True)

        self.brain = SARSAAgent(model=self.model, nb_actions=actions, nb_steps_warmup=10, policy=policy)
        self.brain.compile(Adam(lr=1e-3), metrics=['mae'])
        self.weight_backup = weights
        self.brain.model.summary()
        self.times = 1
        #model.save('files/forex_complete_model.h5f')
        #if os.path.isfile(self.weight_backup):
        #    self.brain.load_weights(self.weight_backup)

    def fit(self,env,nb_steps=5000,callbacks=[]):
        try:
            self.brain.fit(env, nb_steps=nb_steps, visualize=True, verbose=2, callbacks=[self])
        finally:
            try:
                self.save()
            finally:
                self.save()

    def save(self):
        self.times = self.times + 1
        self.brain.model.save(self.weight_backup)
        print("Save Awesomely")

        #if self.times == 100:
        #    ls_output=subprocess.Popen(["rsync", "-av", "--progress", "files/forex_complete_model.h5f", "vincentminde@72.14.186.65:/home/vincentminde/forex-genius/files/"], stdout=subprocess.PIPE)
        #    print("Command Output: {}".format(ls_output))
        #    self.times == 0
        # rsync -av --progress files/forex_complete_model.h5f vincentminde@72.14.186.65:/home/vincentminde/forex-genius/files/
        # ls_output=subprocess.Popen(["rsync", "-av", "--progress", "files/forex_complete_model.h5f", "vincentminde@72.14.186.65:/home/vincentminde/forex-genius/files/"], stdout=subprocess.PIPE)
        # print("Command Output: {}".format(ls_output))
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
        self.save()
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return