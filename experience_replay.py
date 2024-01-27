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


class PrioritizedReplayBuffer:

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
            #Initialize with a non-zero td_error, 
            #TODO should we make it random?

            abs_td_error = 0.1

            transition = (obs['tensor'], action, r, terminated, new_obs['tensor'], abs_td_error)
            self.replay_buffer.append(transition)
            obs = new_obs
    
            if terminated:
                obs, _ = env.reset(seed=seed)

    def add_data(self, data): 
        '''
        Params:
        data = relevant data of a transition, i.e. action, new_obs, reward, done
        '''
        #ensure that the td_error is never 0
        data['abs_td_error'] = data['abs_td_error'] + 0.1
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
        
        # Calculate priorities
        priorities = [transition['abs_td_error'] for transition in self.replay_buffer]

        probabilities = priorities / np.sum(priorities)

        # Sample transitions based on priorities
        transitions = np.random.choice(self.replay_buffer, size=batch_size, p=probabilities)
        
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
    
class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)