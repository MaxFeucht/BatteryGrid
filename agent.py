import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random 
import math

from dqn import DQN, TemporalDQN
from experience_replay import ReplayBuffer



class DDQNAgent:
    
    def __init__(self, env, device, epsilon_decay, 
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, price_horizon = 96, hidden_dim = 128, action_classes = 7, seed = 2705):
        """
        Params:
        env = environment that the agent needs to play
        device = set up to run CUDA operations
        epsilon_decay = Decay period until epsilon start -> epsilon end
        epsilon_start = starting value for the epsilon value
        epsilon_end = ending value for the epsilon value
        discount_rate = discount rate for future rewards
        lr = learning rate
        buffer_size = max number of transitions that the experience replay buffer can store
        seed = seed for random number generator for reproducibility
        """
        
        self.env = env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        
        #self.replay_memory = ReplayBuffer(self.env, self.buffer_size, seed = seed)
        self.dqn_index = 0
        self.dqn_predict = DQN(self.learning_rate, price_horizon=price_horizon, hidden_dim=hidden_dim, action_classes = action_classes).to(self.device)
        self.dqn_target = DQN(self.learning_rate, price_horizon=price_horizon, hidden_dim=hidden_dim, action_classes = action_classes).to(self.device)
        self.replay_memory = ReplayBuffer(self.env, self.buffer_size, seed = seed)
        
        
        
    def choose_action(self, step, observation, greedy=False):
        """
        Function to choose an action based on the epsilon-greedy policy

        Input:
            step: current iteration step for epsilon decay
            observation: current observation
            greedy: boolean that indicates whether the action should be chosen greedily or not
        
        Returns:
            action: action that the agent takes
        """
        
        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        if not greedy and random.random() <= epsilon:
            action = self.env.action_space.sample()
        else:
            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            q_values = self.dqn_predict(obs_t.unsqueeze(0))
            action = torch.argmax(q_values, dim=1).item()        

        return action
    


    def DQNstep(self):
        """
        Function that switches the DQN from the predictDQN to the targetDQN after 250 steps
        """
        self.dqn_index += 1
        
        if self.dqn_index == 250:
            self.dqn_target.load_state_dict(self.dqn_predict.state_dict())
            self.dqn_index = 0
        
        
        
    def optimize(self, batch_size):
            
        """
        Function that optimizes the DQN based on predicted q-values for the present and the next state, along with the reward received
        
        Params: 
        new_obs = new observation
        obs = old observation
        action = action taken
        reward = reward received
        done = boolean that indicates whether the episode is done or not
        
        Returns:
        loss = loss of the DQN
        """
       
        # Sample from replay buffer
        obs, actions, rewards, terminateds, new_obs = self.replay_memory.sample(batch_size)
        
        # Compute Loss: 
        # First compute DQN output for current state        
        q_values = self.dqn_predict(obs) #Predict q-values for the current state
        action_q_values = torch.gather(input=q_values, dim=1, index=actions) # Select the q-value for the action that was taken
        
        # Then: Compute DQN output for next state, and build the targets based on reward and the max q-value of the next state 
        target_q_values = self.dqn_target(new_obs) # Predict q-values for the next state
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0] # Select the max q-value of the next state
        #new_rewards = torch.tensor(np.where(rewards < 0, rewards / 2, rewards)) # Penalize negative rewards
        targets = rewards + self.discount_rate * (1-terminateds) * max_target_q_values # Compute the target q-value based on the reward and the max q-value of the next state
        
        #Loss
        #loss = F.smooth_l1_loss(action_q_values, targets.detach()) #Compute the loss between the predicted q-value for the action taken and the target q-value based on the next observation
        loss = F.mse_loss(action_q_values, targets.detach()) #Compute the loss between the predicted q-value for the action taken and the target q-value based on the next observation

        #Gradient descent
        self.dqn_predict.optimizer.zero_grad()
        loss.backward()
        self.dqn_predict.optimizer.step()
        
        #Switch DQN step
        self.DQNstep()
        
        return loss.item()
    


class TemporalDDQNAgent(DDQNAgent):
    
    def __init__ (self, 
                 lin_hidden_dim, temp_hidden_dim, kernel_size, dropout, 
                 price_horizon = 96, action_classes = 7, *args, **kwargs):
        super().__init__(hidden_dim=lin_hidden_dim, *args, **kwargs)
        
        num_layers = math.ceil(math.log2(price_horizon/kernel_size) + 1)
        
        self.dqn_predict = TemporalDQN(self.learning_rate, price_horizon=price_horizon, action_classes = action_classes, lin_hidden_dim=lin_hidden_dim, temp_hidden_dim = temp_hidden_dim, kernel_size = kernel_size, num_layers = num_layers, dropout=dropout).to(self.device)
        self.dqn_target = TemporalDQN(self.learning_rate, price_horizon=price_horizon, action_classes = action_classes, lin_hidden_dim=lin_hidden_dim, temp_hidden_dim = temp_hidden_dim, kernel_size = kernel_size, num_layers = num_layers, dropout = dropout).to(self.device)