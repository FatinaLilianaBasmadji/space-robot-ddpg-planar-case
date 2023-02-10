#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


class RobotReplayBuffer():
    def __init__(self, MaxSize, InputShape, NrOfActions):
        self.MaxSize = MaxSize
        self.Counter = 0
        self.States = []
        self.Actions = [] 
        self.NextStates = []
        self.Rewards = []
        
    def save(self, State, Action, NextState, Reward):
        if self.Counter == 0:
            self.States = State
            self.Actions = Action
            self.NextStates = NextState
            self.Rewards = Reward
        elif self.Counter < self.MaxSize:
            self.States = np.concatenate((self.States, State), axis=0)
            self.Actions = np.concatenate((self.Actions, Action), axis=0)
            self.NextStates = np.concatenate((self.NextStates, NextState), axis=0)
            self.Rewards = np.concatenate((self.Rewards, Reward), axis=0)
        else:
            i = self.Counter % self.MaxSize
            idx = range(i*np.size(self.States[0]),np.size(self.States[0])+i*np.size(self.States[0]))
            np.put(self.States, idx, State)
            np.put(self.NextStates, idx, NextState)
            idx = range(i*np.size(self.Actions[0]),np.size(self.Actions[0])+i*np.size(self.Actions[0]))
            np.put(self.Actions, idx, Action)
            idx = range(i*np.size(self.Rewards[0]),np.size(self.Rewards[0])+i*np.size(self.Rewards[0]))
            np.put(self.Rewards, idx, Reward)
        self.Counter =  self.Counter + 1
        
    def sample(self, BatchSize=64, RandomSample=1):
        if RandomSample == 1:
            sample = np.random.choice(self.Counter, BatchSize, replace = False)
        else:
            samplelist = np.arange(0,self.Counter)
            sample = np.random.choice(samplelist[-BatchSize:],BatchSize, replace = False)
        SampledStates = self.States[sample]
        SampledStates = SampledStates.reshape(BatchSize,12,1)
        SampledActions = self.Actions[sample]
        SampledNextStates = self.NextStates[sample]
        SampledNextStates = SampledNextStates.reshape(BatchSize,12,1)
        SampledRewards = self.Rewards[sample]
        return SampledStates, SampledActions, SampledNextStates, SampledRewards

