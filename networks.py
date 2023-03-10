#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[3]:


class RobotActorNetwork(tf.keras.Model):
    
    def __init__(self, state_size, action_size):
        super(RobotActorNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = tf.keras.layers.Dense(500, input_shape = (self.state_size,), activation = 'ReLU') 
        self.layer2 = tf.keras.layers.Dense(400, activation = 'ReLU')
        self.layer3 = tf.keras.layers.Dense(300, activation = 'ReLU')
        self.layer4 = tf.keras.layers.Dense(self.action_size, activation = 'tanh')
        
    def call(self, state):
        output = self.layer1(state)
        output = self.layer2(output)
        output = self.layer3(output)
        actor_output = self.layer4(output)
        return actor_output


# In[3]:


class RobotCriticNetwork(tf.keras.Model):
    
    def __init__(self, state_action_size):
        super(RobotCriticNetwork, self).__init__()
        self.state_action_size = state_action_size
        self.layer1 = tf.keras.layers.Dense(500, input_shape = (self.state_action_size,), activation = 'ReLU') 
        self.layer2 = tf.keras.layers.Dense(400, activation = 'ReLU')
        self.layer3 = tf.keras.layers.Dense(300, activation = 'ReLU')
        self.layer4 = tf.keras.layers.Dense(1, activation = 'ReLU')
       
    def call(self, state_action):
        output = self.layer1(state_action)
        output = self.layer2(output)
        output = self.layer3(output)
        critic_output = self.layer4(output)
        return critic_output

