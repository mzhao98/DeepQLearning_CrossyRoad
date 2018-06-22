'''
Deep QLearning Module
'''
import glob, os, sys
import random
import numpy as np
import preprocess as pp
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K


class DQLN:
    def __init__(self, shp_obs, nb_action, reshape_dim):
        self.reshape_dim = reshape_dim
        self.shp_obs = shp_obs # observation space dimensions
        self.nb_action = nb_action # number of possible actions
        self.memory = [] # memory of states encountered so far
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01 # lower bound on exploration rate
        self.epsilon_decay = 0.99 # upper bound on exploration rate
        self.learning_rate = 0.001 # learning rate
        self.model = self.build_model() # the actual NN itself
        self.target_model = self.build_model()
        self.update_target_model()


    def loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis = -1)

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        print(self.shp_obs)
        model.add(Dense(24, input_shape=self.shp_obs, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.nb_action, activation='linear'))
        model.compile(loss=self.loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self, state):
        if(np.random.rand() <= self.epsilon):
            print("random")
            return random.randrange(self.nb_action)
        state = state.reshape(self.reshape_dim)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # state needs to be reshaped to add first dimension
            state = state.reshape(self.reshape_dim)
            # that enumerates the samples (id's how many samples exist)
            target = self.model.predict(state)
            # reshape this too
            if done:
                target[0][action] = reward
            else:
                next_state = next_state.reshape(self.reshape_dim)
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma + np.amax(t)

            self.model.fit(state, target, epochs = 1, verbose = 0)
        if (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)