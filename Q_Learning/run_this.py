# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:33:25 2020

@author: Administrator
"""

import gym
env = gym.make('CartPole-v1')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

from Agent import QLearningTable
act = [num for num in range(env.action_space.n)]
RL = QLearningTable(actions = act)


for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

# end of game
print('game over')
# env.destroy()
         
 








