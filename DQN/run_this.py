# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:03:51 2020

@author: Administrator
"""

import numpy as np

import gym
env = gym.make('CartPole-v0')
env = env.unwrapped


print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

from Agent import DeepQNetwork
#act = [num for num in range(env.action_space.n)]

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0])




for episode in range(50):
        # initial observation
        observation = env.reset()
        observation = np.reshape(observation, [1, RL.n_features])
        
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)
            observation_ = np.reshape(observation_, [1, RL.n_features])
            
            # RL store transition
            RL.store_transition_in_mermory(observation,action,reward,observation_,done)

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

RL.save_trained_model()
         









