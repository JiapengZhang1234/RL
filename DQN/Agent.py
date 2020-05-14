# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:05:34 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import keras
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense




class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replay_target_iter=200,
            memory_size=500,
            batch_size=32
            ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon= e_greedy 
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        #build eval_net and target net
        self.eval_net = self._build_net()
        self.target_net = self._build_net()
        
        #build replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        self.learning_step_counter = 0
        self.replay_target_iter = replay_target_iter
        
        
    
       
    def _build_net(self):
        #build model
        model = Sequential()
        model.add(Dense(10,input_dim=self.n_features,activation='relu'))
        model.add(Dense(self.n_actions,activation='linear'))
        
        #loss function and optimization method
        model.compile(loss='mse', optimizer='sgd')
        
        return model
    
    
    def target_net_para_setting(self,eval_n,target_n):
        target_n.set_weights(
            eval_n.get_weights()
            )
       
    

    def store_transition_in_mermory(self,s,a,r,s_,done):
        self.memory.append((s,a,r,s_,done))
    
    
    def choose_action(self,state):
        if np.random.rand() < self.epsilon:
            action_values = self.eval_net.predict(state)
            action =  np.argmax(action_values[0]) #返回最大值的列索引，即动作名
        else:
            action = np.random.choice(self.n_actions)
        return action
    
    
    
    def learn(self):
        # Data Sample        
        minibatch = random.sample(self.memory, self.batch_size)
        
        # eval network update
        for state, action, reward, next_state, done in minibatch:
            if done:
                q_target_s_a = reward
            else:
                q_target_s_a = reward + self.gamma * np.max(
                    self.target_net.predict(next_state))
            
            q_eval = self.eval_net.predict(state)
            q_predict_s_a = q_eval[0][action]
            q_predict_s_a += self.lr * (q_target_s_a - q_predict_s_a)
            q_eval[0][action] =  q_predict_s_a
            
            self.eval_net.fit(state,q_eval)
            self.learning_step_counter += 1
        
        # target network update
        if self.learning_step_counter % self.replay_target_iter == 0:
            self.target_net_para_setting(self.eval_net,self.target_net)
            
            
    def save_trained_model(self):
        self.eval_net.save('RL.h5')
    
     
    def plot_cost(self):
        pass
    
    
    
    
    
        



        
        
        
        






