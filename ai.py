'''
Starter code for OpenAIGym Bot that plays Freeway
'''
import glob, os, sys
import numpy as np
import preprocess as pp
import matplotlib.pyplot as plt
import time
import keras
from keras.layers import Dense
from dqln import DQLN
import gym

###############################################################################
# def remember(self, state, action, reward, next_state, done):
#     self.memory.append((state, action, reward, next_state, done))

###############################################################################
# Training phase

if __name__ == "__main__":
    ENV_NAME = 'Freeway-v0'
    # Set up one instance of Freeway game environment
    env = gym.make(ENV_NAME)
    # initialize seed
    env.seed(123)
    # number of available actions
    nb_actions = env.action_space.n
    # shp_obs = env.observation_space.shape
    # we did some preprocessing, so the new input shape is 100 x 100 x 1
    shp_obs = (100, 100, 1)

    reshape_dim = (1,100,100,1)

    # set up a that dank learning agent
    chicken = DQLN(shp_obs, nb_actions, reshape_dim)

    batch_size = 32

    episodes = 10

    # perform training for episodes
    for e in range(episodes):
        # reset game state at beginning of each game
        state = env.reset()
        # preprocess the game state
        state = pp.preprocess(state)

        # game is not done until time ends
        done = False

        # current epsiode's game score
        curr_score = 0

        while not done:
            env.render()
            # decide on an action
            action = chicken.act(state)
            # update the game based off of action
            next_state, reward, done, info = env.step(action)
            # preprocess new game state
            next_state = pp.preprocess(next_state)
            # remember what just happened
            chicken.remember(state, action, reward, next_state, done)

            # reward is only one if chicken crossed the road, 0 otherwise
            curr_score += reward

            # update state
            state = next_state

        # once the game finishes (time runs out), output the game score
        print("episode: {}/{}, score: {}".format(e, episodes, curr_score))
        # train agent with experience of episode
        chicken.replay(32)
