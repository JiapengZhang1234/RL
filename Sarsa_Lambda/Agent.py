# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:11:23 2020

@author: Administrator
"""
import numpy as np
import pandas as pd


class SarsaLambdaTable:
    def __init__(self,action_space,learning_rate=0.01,reward_decay=0.9,
                 e_greedy=0.9,trace_decay=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.lambda_ = trace_decay
        
        
        self.q_table = pd.DataFrame(columns=self.actions,
                                    dtype=np.float64)
        self.eligibility_trace = self.q_table.copy()
    
    
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
    
    
    def choose_action(self,observation):
        self.check_state_exist(observation)
        
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action
    
    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_,a_]
        else:
            q_target = r
        
        error = q_target - q_predict
        
        self.eligibility_trace.loc[s, a] += 1
        
        self.q_table +=  self.lr * error * self.eligibility_trace
        
        self.eligibility_trace *= self.lambda_
            
        



    
        

