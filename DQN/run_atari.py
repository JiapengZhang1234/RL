# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:01:52 2020

@author: Administrator
"""

import numpy as np
from Agent_human_level import DeepQNetwork
# from prepross import imgbuffer_process
import gym
import cv2

env = gym.make('Breakout-v0')
print('action space :',env.action_space)
print('observation space :',env.observation_space)
# print('observation space max :',env.observation_space.high)
# print('observation space min :',env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n)



for episode in range(2):
    print('episode:',episode)
    observation = env.reset()
    observation = cv2.resize(src=observation,dsize=(84,84))
    observation = np.expand_dims(observation,axis=0)
    
    while True:
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)
            observation_ = cv2.resize(src=observation_,dsize=(84,84))
            observation_ = np.expand_dims(observation_,axis=0)
            
            
            
            # RL store transition
            RL.store_transition_in_memory(observation,action,reward,observation_,done)

            # RL learn from this transition
            if len(RL.memory) > RL.batch_size:
                RL.learn()
            
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
# end of game
print('game over')
# env.destroy()


    