import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random 

from dqn import DQN
from experience_replay import ReplayBuffer

class DDQNAgent:
    
    def __init__(self, env_name, df, device, epsilon_decay, 
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, price_horizon = 96, hidden_dim = 128, seed = 2705):
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
        
        self.env_name = env_name
        self.env = gym.make(self.env_name, disable_env_checker=True)
        self.env.setup(df, price_horizon = price_horizon)
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        
        #self.replay_memory = ReplayBuffer(self.env, self.buffer_size, seed = seed)
        self.dqn_index = 0
        self.dqn_predict = DQN(self.env, self.learning_rate, price_horizon=price_horizon, hidden_dim=hidden_dim).to(self.device)
        self.dqn_target = DQN(self.env, self.learning_rate, price_horizon=price_horizon, hidden_dim=hidden_dim).to(self.device)
        self.replay_memory = ReplayBuffer(self.env, self.buffer_size, seed = seed)
        
        
        
    def choose_action(self, step, observation, greedy = False):
        
        """Function to choose an action based on the epsilon-greedy policy

        Input:
            step: current iteration step for epsilon decay
            observation: current observation
            greedy: boolean that indicates whether the action should be chosen greedily or not
        
        Returns:
            action: action that the agent takes
        """
        
        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        random_sample = random.random()
    
        if (random_sample <= epsilon) and not greedy:
            action = self.env.action_space.sample()
        
        else:
            #Greedy action
            obs_t = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
            q_values = self.dqn_predict(obs_t)
        
            #max_q_index = torch.argmax(q_values, dim = 1)[0]
            max_q_index = torch.argmax(q_values)
    
            action = max_q_index.detach().item()
        
        return action
    
    
    
    def tensorize(self, obs, new_obs, action, reward, done):

        obs_t = torch.as_tensor(obs, dtype = torch.float32, device=self.device)
        new_obs_t = torch.as_tensor(new_obs, dtype = torch.float32, device=self.device)
        action_t = torch.as_tensor(action, dtype = torch.long, device=self.device)
        reward_t = torch.as_tensor(reward, dtype = torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype = torch.float32, device=self.device)
        
        return obs_t, new_obs_t, action_t, reward_t, done_t



    def DQNstep(self):
        """
        Function that switches the DQN from the predictDQN to the targetDQN after 1000 steps
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
        loss = F.smooth_l1_loss(action_q_values, targets.detach()) #Compute the loss between the predicted q-value for the action taken and the target q-value based on the next observation
        
        #Gradient descent
        self.dqn_predict.optimizer.zero_grad()
        loss.backward()
        self.dqn_predict.optimizer.step()
        
        #Switch DQN step
        self.DQNstep()
        
        return loss.item()