# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:44:51 2020

@author: Administrator
"""


import numpy as np
# import pandas as pd

import random
from collections import deque


from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,Flatten

#from keras.layers import MaxPooling2D

class DeepQNetwork:
    def __init__(
                 self,
                 n_actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replay_target_iter=300,
                 memory_size=500,
                 batch_size=32
                 ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        #bulid eval network and target network
        self.eval_net = self._build_net()
        self.target_net = self._build_net()
        
        #build replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        #frequency of updating target network
        self.learning_step_counter = 0
        self.replay_target_iter = replay_target_iter
        
        
    
    # build network fun
    def _build_net(self):
        #build sequential model
        model = Sequential()
        
        #add convolution layers
        model.add(Conv2D(32,(8,8),strides=(4,4),activation='relu',
                 input_shape=(84,84,3)))
        model.add(Conv2D(64,(4,4),strides=(2,2),activation='relu'
                  ))
        model.add(Conv2D(64,(3,3),strides=(1,1),activation='relu'
                  ))
        
        #data compression
        model.add(Flatten())
        
        #build first dense layer
        model.add(Dense(512))
        model.add(Activation('relu'))
        
        #bulid output layer
        model.add(Dense(self.n_actions,activation='linear'))
        
        #model optimization
        model.compile(loss='mse', optimizer='RMSprop')
        
        return model

    
    # target network para update fun
    def target_net_para_setting(self,eval_n,target_n):
        target_n.set_weights(
            eval_n.get_weights()
            )
        
    
    # store transition fun
    def store_transition_in_memory(self,s,a,r,s_,done):
        self.memory.append((s,a,r,s_,done))
    
    # choose action according state fun
    def choose_action(self,state):
        if np.random.rand() < self.epsilon:
            action_values = self.eval_net.predict(state)
            action =  np.argmax(action_values[0]) #返回最大值的列索引，即动作名
        else:
            action = np.random.choice(self.n_actions)
        return action
    
    # learn fun
    def learn(self):
        # Data Sample        
        minibatch = random.sample(self.memory, self.batch_size)
        
        # loss compute
        loss = 0
        loss_sum=0

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
                            
            history = self.eval_net.fit(state,q_eval)
            loss = float(history.history['loss'][0])            
            loss_sum = loss_sum + loss
            print(loss)
            
            self.learning_step_counter += 1
        
        loss_batch_average = loss_sum / 32
        
        print('one minibatch is over')
        
        # target network update
        if self.learning_step_counter % self.replay_target_iter == 0:
            self.target_net_para_setting(self.eval_net,self.target_net)
        
        return loss_batch_average
    
    
    def save_trained_model(self):
        self.eval_net.save('model.h5')
    