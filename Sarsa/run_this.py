# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:17:16 2020

@author: Administrator
"""


import gym
env = gym.make('CartPole-v1')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


from Agent import SarsaTable
act = [num for num in range(env.action_space.n)]
RL = SarsaTable(actions = act)



for episode in range(100):
        # initial observation
        observation = env.reset()
        
        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)
            
            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_),action_)

            # swap observation
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

# end of game
print('game over')
# env.destroy()






