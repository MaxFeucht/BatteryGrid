import numpy as np
import random
import torch
from collections import deque

class ReplayBuffer:
    
    def __init__(self, env, buffer_size, min_replay_size = 1000, seed = 2705):
        
        '''
        Params:
        env = environment that the agent needs to play
        buffer_size = max number of transitions that the experience replay buffer can store
        min_replay_size = min number of (random) transitions that the replay buffer needs to have when initialized
        seed = seed for random number generator for reproducibility
        '''
        
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        #Initialize replay buffer with random transitions (transitions based on random actions)      
        obs, _ = self.env.reset(seed=seed)
        
        for _ in range(self.min_replay_size):
            
            action = env.action_space.sample()
            new_obs, r, terminated, _ = env.step(action)

            transition = (obs['tensor'], action, r, terminated, new_obs['tensor'])
            self.replay_buffer.append(transition)
            obs = new_obs
    
            if terminated:
                obs, _ = env.reset(seed=seed)
              
          
    def add_data(self, data): 
        '''
        Params:
        data = relevant data of a transition, i.e. action, new_obs, reward, done
        '''
        self.replay_buffer.append(data)



    def sample(self, batch_size):
        
        '''
        Params:
        batch_size = number of transitions that will be sampled
        
        Returns:
        tensor of observations, actions, rewards, dones (boolean) and new_observations 
        '''
        
        if len(self.replay_buffer) < batch_size:
            raise ValueError("Not enough data in the replay buffer.")
        
        # Sample random transitions
        transitions = random.sample(self.replay_buffer, batch_size)
        
        # Add last transition to the sample to make sure that the last two (current) transitions are included in the sample --> MIXED TRAINING / PRIORITIZED EXPERIENCE REPLAY
        transitions[-1] = self.replay_buffer[-1] 
        
        # Shuffle transitions
        random.shuffle(transitions)
                
        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4] for t in transitions])
        
        #PyTorch needs these arrays as tensors!, don't forget to specify the device! (cpu/GPU) 
        observations_t = torch.as_tensor(observations, dtype = torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype = torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype = torch.float32, device=self.device).unsqueeze(-1)
        terminated_t = torch.as_tensor(dones, dtype = torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype = torch.float32, device=self.device)
        
        return observations_t, actions_t, rewards_t, terminated_t, new_observations_t
    