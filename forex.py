import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Input,Reshape
from keras.layers.recurrent import GRU
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory

import os
class ForexGenius:
    def __init__(self, weights,actions):
        model = Sequential()
        #model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        #model.add(Flatten(input_shape=(1, 256, 16)))
        # model.add(Flatten(input_shape=(1, 30, 4)))
        model.add(Reshape((30, 4), input_shape=(1, 30, 4)))
        model.add(GRU(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(128, return_sequences=False))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(actions))
        model.add(Activation('linear'))
        print(model.summary())
        print(model.to_json())
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = GreedyQPolicy()
        self.brain = DQNAgent(model=model, nb_actions=actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

        #self.brain = SARSAAgent(model=model, nb_actions=actions, nb_steps_warmup=10, policy=policy)
        self.brain.compile(Adam(lr=1e-3), metrics=['mae'])
        self.weight_backup = weights
        if os.path.isfile(self.weight_backup):
            self.brain.load_weights(self.weight_backup)

    def fit(self,env):
        try:
            self.brain.fit(env, nb_steps=5000, visualize=True, verbose=2)
        finally:
            self.save()

    def save(self):
        self.brain.save_weights(self.weight_backup, overwrite=True)
        print("Save Awesomely")
    def test(self,env):
        self.brain.test(env, nb_episodes=5, visualize=True, verbose=2)