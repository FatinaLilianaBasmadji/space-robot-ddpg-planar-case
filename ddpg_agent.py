#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from networks import RobotActorNetwork, RobotCriticNetwork
from replaybuffer import RobotReplayBuffer


# In[2]:


class Agent():
    
    def __init__(self, actor_lr, critic_lr, gamma, tau, state_size, action_size, state_action_size, replay_buffer_size, mean, 
                 standard_deviation, min_action, max_action, batch_size, random_sample):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.state_action_size = state_action_size
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.min_action = min_action
        self.max_action = max_action
        self.batch_size = batch_size
        self.random_sample = random_sample 
        self.reply_buffer = RobotReplayBuffer(replay_buffer_size)
        self.actor_main = RobotActorNetwork(state_size, action_size)
        self.actor_target = RobotActorNetwork(state_size, action_size)
        self.critic_main = RobotCriticNetwork(state_action_size)
        self.critic_target = RobotCriticNetwork(state_action_size)
        self.actor_main.compile(optimizer=tf.keras.optimizers.Adam(actor_lr))
        self.actor_target.compile(optimizer=tf.keras.optimizers.Adam(actor_lr))
        self.critic_main.compile(optimizer=tf.keras.optimizers.Adam(critic_lr))
        self.critic_target.compile(optimizer=tf.keras.optimizers.Adam(critic_lr))
        self.actor_target.set_weights(self.actor_main.get_weights())
        self.critic_target.set_weights(self.critic_main.get_weights())
        
    def take_action(self, state):
        state_tf = tf.convert_to_tensor([state], dtype=tf.float32) 
        action_tf = self.actor_main(state_tf) * self.max_action
        noise = np.random.normal(self.mean, self.standard_deviation, self.action_size)
        action_tf = action_tf + noise
        action_tf = tf.reshape(action_tf, [3])
        action_tf = tf.clip_by_value(action_tf, clip_value_min = self.min_action, clip_value_max = self.max_action)
        return action_tf
    
    def store_transition(self, state, action, nextstate, reward):
        self.reply_buffer.save(state, action, nextstate, reward)
        
    def sample_minibatch(self):
        states, actions, next_states, rewards = self.reply_buffer.sample(self.batch_size, self.random_sample)
        return states, actions, next_states, rewards
    
    def training(self):
        states, actions, next_states, rewards = self.sample_minibatch()
        states_tf = tf.convert_to_tensor([states], dtype=tf.float32)
        states_tf = tf.reshape(states_tf, [self.batch_size,self.state_size])
        actions_tf = tf.convert_to_tensor([actions], dtype=tf.float32)
        actions_tf = tf.reshape(actions_tf, [self.batch_size,3])
        next_states_tf = tf.convert_to_tensor([next_states], dtype=tf.float32)
        next_states_tf = tf.reshape(next_states_tf, [self.batch_size,self.state_size])
        
        with tf.GradientTape() as tape1:
            next_actions_tf = self.actor_target(next_states_tf)
            next_actions_tf = tf.reshape(next_actions_tf, [self.batch_size,self.action_size])
            next_states_actions = tf.concat([next_states_tf,next_actions_tf], axis = 1)
            y_alg = rewards + self.gamma * tf.squeeze(self.critic_target(next_states_actions),1)
            states_actions = tf.concat([states_tf,actions_tf], axis = 1) 
            Q_alg = tf.squeeze(self.critic_main(states_actions),1)
            L_alg = keras.losses.MSE(y_alg,Q_alg) 
        gradient1 = tape1.gradient(L_alg, self.critic_main.trainable_variables)
        critic_opt = tf.keras.optimizers.Adam(self.critic_lr)
        critic_opt.apply_gradients(zip(gradient1,self.critic_main.trainable_variables))
        
        with tf.GradientTape() as tape2:
            actions_u_tf = self.actor_main(states_tf)
            actions_u_tf = tf.reshape(actions_u_tf, [self.batch_size,self.action_size])
            states_actions_u = tf.concat([states_tf,actions_u_tf], axis = 1) 
            G_u = - self.critic_main(states_actions_u)
            G_u = tf.math.reduce_mean(G_u)
        gradient2 = tape2.gradient(G_u, self.actor_main.trainable_variables)
        actor_opt = tf.keras.optimizers.Adam(self.actor_lr)  
        actor_opt.apply_gradients(zip(gradient2,self.actor_main.trainable_variables)) 
        
        new_weights = []
        old_weights = self.critic_target.weights
        for i, weight in enumerate(self.critic_main.weights):
            new_weights.append(weight * self.tau + old_weights[i]*(1-self.tau))
        self.critic_target.set_weights(new_weights)
        
        new_weights = []
        old_weights = self.actor_target.weights
        for i, weight in enumerate(self.actor_main.weights):
            new_weights.append(weight * self.tau + old_weights[i]*(1-self.tau))
        self.actor_target.set_weights(new_weights)

