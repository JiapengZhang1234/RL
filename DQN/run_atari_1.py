# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:46:45 2020

@author: Administrator
"""


import numpy as np
from Agent_hl_ver import DeepQNetwork
# from prepross import imgbuffer_process
import gym
import cv2
import matplotlib.pyplot as plt



env = gym.make('Breakout-v0')
print('action space :',env.action_space)
print('observation space :',env.observation_space)
# print('observation space max :',env.observation_space.high)
# print('observation space min :',env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n)

loss_hist = []
loss_batch = 0

score_hist = []

for episode in range(20):
    print('episode:',episode)
    observation = env.reset()
    observation = cv2.resize(src=observation,dsize=(84,84))
    observation = np.expand_dims(observation,axis=0)
    
    loss_sum = 0
    loss_counter = 0
    score_sum = 0
    score_counter = 0
    
    while True:
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)
            observation_ = cv2.resize(src=observation_,dsize=(84,84))
            observation_ = np.expand_dims(observation_,axis=0)
            
            score_sum = score_sum + reward
            score_counter += 1
            
            # RL store transition
            RL.store_transition_in_memory(observation,action,reward,observation_,done)

            # RL learn from this transition
            if len(RL.memory) > RL.batch_size:
                loss_batch = RL.learn()
                loss_sum = loss_sum + loss_batch
                loss_counter += 1
                
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                loss_hist.append(loss_sum / loss_counter)
                score_hist.append(score_sum / score_counter)
                break
# end of game
print('game over')
# env.destroy()


plt.figure()
plt.plot(loss_hist)
plt.ylabel('average loss')
plt.xlabel('episode')
plt.show()

plt.figure()
plt.plot(score_hist)
plt.ylabel('average score')
plt.xlabel('episode')
plt.show()


