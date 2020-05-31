# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:12:03 2020

@author: Administrator
"""

import numpy as np
import gym
import cv2
from keras.models import load_model

env = gym.make('Breakout-v0')
print('action space :',env.action_space)
print('observation space :',env.observation_space)

RL = load_model('model.h5')

reward_sum = 0

observation = env.reset()
observation = cv2.resize(src=observation,dsize=(84,84))
observation = np.expand_dims(observation,axis=0)

while True:
    # fresh env
    env.render()

    # RL choose action based on observation
    action_values = RL.predict(observation)
    action =  np.argmax(action_values[0])
    
    # RL take action and get next observation and reward
    observation_, reward, done, info = env.step(action)
    observation_ = cv2.resize(src=observation_,dsize=(84,84))
    observation_ = np.expand_dims(observation_,axis=0)
    
    reward_sum += reward
    
    # swap observation
    observation = observation_
    
    print('reward:',reward)
    print('sum of reward:', reward_sum)
    
    # break while loop when end of this episode
    if done:
        break

# end of game
print('game over')
#env.destroy()
print('game score',reward_sum)






