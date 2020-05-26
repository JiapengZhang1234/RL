# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:46:45 2020

@author: Administrator
"""


import numpy as np
from Agent_atari_2 import DeepQNetwork
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

q_value_hist = []
score_hist = []

for episode in range(20):
    print('episode:',episode)
    observation = env.reset()
    observation = cv2.resize(src=observation,dsize=(84,84))
    observation = np.expand_dims(observation,axis=0)
    
    q_sum = 0
    learn_counter = 0
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
                q_eval_in_batch = RL.learn()
                q_sum += q_eval_in_batch
                learn_counter += 1

                
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                score_hist.append(score_sum / score_counter)
                q_value_hist.append(q_sum / learn_counter)
                break


RL.save_trained_model()

# end of game
print('game over')
# env.destroy()


plt.figure()
plt.plot(q_value_hist)
plt.ylabel('average q value')
plt.xlabel('episode')
plt.show()

plt.figure()
plt.plot(score_hist)
plt.ylabel('average score')
plt.xlabel('episode')
plt.show()


