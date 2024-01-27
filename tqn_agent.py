from collections import defaultdict
import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random 
import math
from TestEnv import Electric_Car
from dqn import DQN, TemporalDQN



class TabularQLearningAgent():
    def __init__(self, env: Electric_Car, gamma=0.99, lr=0.01, epsilon=1, epsilon_decay=0.99):
        self.env = env
        #discount factor
        self.gamma = gamma
        #learning rate
        self.lr = lr
        #greedy strategy parameter
        self.epsilon = epsilon
        #Define our action space
        self.action_space = np.array([-1, 0, 1])
        self.action_index_mapping = {-1: 0, 0: 1, 1: 2}
        self.min_epsilon = 0.15
        self.epsilon_decay = epsilon_decay
        #Initialize the Q-table
        action_space_size = len(self.action_space)
        self.Qtable = defaultdict(lambda: np.zeros(action_space_size))
     

    def act(self, state):
        #Greedy strategy implementation
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            discretized_state = self.select_and_discretize_state(state)
            #argmax returns the index of the action with the highest Q-value
            action = self.action_space[np.argmax(self.Qtable[discretized_state])]
            #print(np.argmax(self.Qtable[discretized_state]))
        
            return action
        
    def update_epsilon(self, episode, exponential_decay=False):
        if exponential_decay:
            self.epsilon = max(self.min_epsilon, np.exp(-episode * self.epsilon_decay))
        else:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    def learn(self, state, action, reward, next_state, done):
        #Calculate the Temporal Difference target
        discretized_state = self.select_and_discretize_state(state)
        discretized_next_state = self.select_and_discretize_state(next_state)
        #Get the action index instead of the action
        action_index = np.where(self.action_space == action)[0][0]
        new_q_value = (1 - self.lr) * self.Qtable[discretized_state][action_index] + self.lr * (reward + self.gamma * np.max(self.Qtable[discretized_next_state]))
        #Update the Q-table
        self.Qtable[discretized_state][action_index] = new_q_value

    def select_and_discretize_state(self, state):
        #Discretize the price space
        price = np.digitize(state[1], bins=np.arange(0, 100, 10))
        #Discretize the battery level space
        battery_level = np.digitize(state[0], bins=np.arange(0, 50, 5))
        #discretized_state = (battery_level, price, state[2], state[3], state[4], state[5], state[6])
        discretized_state = (battery_level, price, state[2], state[3])
        return discretized_state