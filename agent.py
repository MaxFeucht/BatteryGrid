import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from collections import deque
import datetime

import gymnasium as gym
import numpy as np
import random 
import math




class DDQNAgent:
    
    def __init__(self, env, features, epsilon_decay, 
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, price_horizon, hidden_dim, action_classes, seed = 2705, normalize = True, verbose = False):
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
        self.features = np.array(features)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        self.price_horizon = price_horizon
        self.action_classes = action_classes
        self.normalize = normalize
        self.verbose = verbose

        # Set up price_history queue
        self.price_history = deque(maxlen=price_horizon)

        # Normalize features per column (feature) but not datetime and price in the first two columns
        self.grads = self.features[:,[11, 13, 15, 17, 19, 21, 23]] * 1e-3 # Get gradient features for reward shaping, divide by 1000 to bring gradient on same level as reward
        self.features[:,2:] = self.normalize_data(self.features[:,2:], axis=0) if self.normalize else self.features[:,2:] # Normalize features per column (feature) 

        feature_dim = features.shape[1] - 2 # -2 because we don't want to include the datetime and price columns
        print("Number of engineered features: ", feature_dim)
        
        self.state_dim = price_horizon + feature_dim + 1 + 1 # price history + engineered features + battery level + car availability
        print("State dimension: ", self.state_dim)
        
        self.dqn_index = 0
        self.dqn_predict = DQN(self.learning_rate, feature_dim=feature_dim, price_horizon=self.price_horizon, hidden_dim=hidden_dim, action_classes = action_classes).to(self.device)
        self.dqn_target = DQN(self.learning_rate, feature_dim=feature_dim, price_horizon=self.price_horizon, hidden_dim=hidden_dim, action_classes = action_classes).to(self.device)
        self.dqn_target.load_state_dict(self.dqn_predict.state_dict())  # Initialize target network with the same parameters as the predict network
        self.replay_memory = ReplayBuffer(self.env, state_dim = self.state_dim, buffer_size = self.buffer_size, action_classes = action_classes, seed = seed, state_func=self.obs_to_state, action_func=self.action_to_cont, reward_func = self.shape_reward) # State function is used to transform the observation to the state of the environment
        



    def normalize_data(self, var, axis=None):
            """
            Helper function to normalize data between 0 and 1
            """
            
            if not self.normalize:
                return var
            
            if axis is None:
                return (var - np.min(var)) / (np.max(var) - np.min(var))
        
            else:
                return (var - np.min(var, axis=axis)) / (np.max(var, axis=axis) - np.min(var, axis=axis))
        
    
    

    def obs_to_state(self, obs):
        """
        Matches the observation to the state of the environment. A state is defined as the price history with a determined horizon, the battery level, 
        the car availability and the engineered features corresponding to the current observation.
        
        Args:
            obs (np.array): Observation of the environment
        
        Returns:
            state (np.array): State of the environment
        """
        
        # Get observation
        battery_level = obs[0]
        price = obs[1]
        car_is_available = obs[7]
        
        # Fill price history
        self.price_history.append(price)

        # Match to engineered features by price, hour, day, month, year
        features = self.features[self.env.counter]
                
        # Get date and prices for assertions between features and observation price
        feature_date = features[0]
        features = np.array(features[1:], dtype=np.float32)
        feature_price = features[0]
        features = features[1:] #by doing two times features[1:] we remove the date and price from the features array (not elegant but works)
        
        date = datetime.datetime(int(obs[6]), 1, 1) + datetime.timedelta(days=int(obs[4])-1, hours=int(obs[2])) # Needed to get the correct date from day of year and hour of day
        
        if self.verbose:
            print("Year, Day of Year, Hour of Day:", obs[6], obs[4], obs[2])
            print("Fabricated Date:", date)

        assert round(float(price), 2) == round(float(feature_price), 2), f"Price ({round(float(price), 2)}) and price ({round(float(feature_price), 2)}) do not match, with {obs[2]} hour, {obs[3]} day of week, {obs[4]} day of year, {obs[5]} month and {obs[6]} year, and {feature_date}"
        
        # Normalize data - WATCH OUT; self.price_history is a deque, not an array, the normal price_history is an array!
        price_history = np.array(self.price_history)
        
        if self.normalize:
            price_history = self.normalize_data(price_history)
            battery_level /= 50
            # features are already normalized in the setup function
            
        grads = self.grads[self.env.counter] # Get gradient features for reward shaping
        
        if self.verbose:
            print("Date:", date, "Grads:", grads)
        
        # Concatenate price history, battery level, car availability and features
        state = np.concatenate((price_history, np.array([battery_level, car_is_available]), features))   
        
        return state, grads
    


    def action_to_cont(self, action):
        """
        Function that maps a discrete action to the continuous action space between -1 and 1
        
        Params:
            action (int) = discrete action
        
        Returns:
            rescaled_action (float) = continuous action
        """
        
        if self.action_classes % 2 == 0:
            
            # Map action to the action space from -1 to 1 (even number of actions, where we have only one charge option)
            no_action = self.action_classes - 2
            rescaled_action = (action - no_action) / no_action
            rescaled_action = 1 if rescaled_action > 0 else rescaled_action
            
        else:
            # Map action to the action space from -1 to 1
            middle_action = (self.action_classes - 1) / 2 # Action at which the agent does not charge or discharge
            rescaled_action = (action - middle_action) / middle_action 
        
        
        return rescaled_action



 
        
    def shape_reward(self, reward, action, grads, factor=1, apply=True):
        """
        Function to apply a penalty to the reward based on the gradient of the price.

        Args:
            reward (float): The original reward.
            action (int): The chosen action.
            grads (tuple): Tuple of gradients (grad1, grad2, grad4, grad6, grad8, grad12, grad18).
            factor (float): The factor to control the scaling of the penalty.
            apply (bool): Whether to apply the penalty or not.

        Returns:
            shaped_reward (float): The scaled reward with penalty applied.
        """

        if not apply:
            return reward

        grad1, grad2, grad4, grad6, grad8, grad12, grad18 = grads


        ## Shaping for Sell Action: Penalize buying when price is still increasing, don't penalize when on top
        if action < 0:
            if grad1 > 0:
                penalty = max(grad1, grad2)  # Ensured positive penalty
            else:
                penalty = 0

            if grad1 > 0 and grad2 > 0 and grad4 > 0 and grad6 > 0 and grad8 > 0 and grad12 > -2 and grad18 > 0:
                penalty = 0

            penalized_reward = reward - penalty * np.abs(action) * 25 # 25 is the max power of the battery


        ## Shaping for Buy Action
        elif action > 0:

            if (grad1 > 0) and grad6 < 0 and grad8 < 0 and grad18 < 0: # Aims at points where the price is increasing shortly after a dip (price valley)
                penalty = max((grad6, grad8))
            else:
                penalty = 0           
            
            penalized_reward = reward - penalty * action * 25 * 1 # Enforce two consecutive charge actions by reducing the negative reward for charging (subtracting a negative number increases the reward, and penalty is negative here bc of negative gradient)
        
        
        ## No Shaping for Neutral Action
        else:
            penalized_reward = reward
            
            
        # Scale the reward based on the factor
        shaped_reward = reward * (1 - factor) + penalized_reward * factor

        return shaped_reward
        



    def DQNstep(self):
        """
        Function that switches the DQN from the predictDQN to the targetDQN after 2500 steps
        """
        self.dqn_index += 1
        
        if self.dqn_index == 2500:
            self.dqn_target.load_state_dict(self.dqn_predict.state_dict())
            self.dqn_index = 0




    def choose_action(self, step, state, greedy=False):
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
            action = random.randint(0, self.action_classes-1)
        else:
            obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            q_values = self.dqn_predict(obs_t.unsqueeze(0))
            action = torch.argmax(q_values, dim=1).item() 
        
        # If the price history is not full, the agent is not taking any action
        if len(self.price_history) != self.price_history.maxlen:
            action = (self.action_classes - 1) / 2 # Middle action in which agents doesn't do anything

        return action 
        
        
        

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
        states, actions, rewards, terminateds, new_states = self.replay_memory.sample(batch_size)        
        
        # Compute Loss: 
        # First compute DQN output for current state        
        q_values = self.dqn_predict(states) #Predict q-values for the current state
        action_q_values = torch.gather(input=q_values, dim=1, index=actions) # Select the q-value for the action that was taken
        
        # Then: Compute DQN output for next state, and build the targets based on reward and the max q-value of the next state 
        # target_q_values = self.dqn_target(new_states) # Predict q-values for the next state
        # max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0] # Select the max q-value of the next state
        # new_rewards = torch.tensor(np.where(rewards < 0, rewards / 2, rewards)) # Penalize negative rewards
        # targets = rewards + self.discount_rate * max_target_q_values # Compute the target q-value based on the reward and the max q-value of the next state 
        
        ### ACTUAL DOUBLE DQN IMPLEMENTATION ###
        action_selection = self.dqn_predict(new_states).argmax(dim=1)
        target_q_values = self.dqn_target(new_states)
        targets = rewards + (1 - terminateds) * self.discount_rate * torch.gather(input = target_q_values, dim = 1, index = action_selection.unsqueeze(1)).squeeze(1).view(-1,1)
                
        #Loss
        loss = F.smooth_l1_loss(action_q_values, targets.detach()) #Compute the loss between the predicted q-value for the action taken and the target q-value based on the next observation
        #loss = F.mse_loss(action_q_values, targets.detach()) #Compute the loss between the predicted q-value for the action taken and the target q-value based on the next observation

        #Gradient descent
        self.dqn_predict.optimizer.zero_grad()
        loss.backward()
        self.dqn_predict.optimizer.step()
        
        #Switch DQN step
        self.DQNstep()
        
        return loss.item()
    


class TemporalDDQNAgent(DDQNAgent):
    
    def __init__ (self, env,
                 lin_hidden_dim, conv_hidden_dim, target_dim, kernel_size, dropout, tcn_path,
                 price_horizon, action_classes, *args, **kwargs):
        super().__init__(env = env, hidden_dim=lin_hidden_dim, price_horizon = price_horizon, action_classes = action_classes, *args, **kwargs)
        
        num_layers = math.ceil(math.log2(price_horizon/kernel_size) + 1)
        
        feature_dim = self.features.shape[1] - 2 # -2 because we don't want to include the datetime and price columns
        print("Number of engineered features: ", feature_dim)
        
        self.dqn_predict = TemporalDQN(self.learning_rate, feature_dim=feature_dim, price_horizon=price_horizon, action_classes = action_classes, lin_hidden_dim=lin_hidden_dim, conv_hidden_dim = conv_hidden_dim, target_dim = target_dim, kernel_size = kernel_size, num_layers = num_layers, dropout=dropout, tcn_path = tcn_path).to(self.device)
        self.dqn_target = TemporalDQN(self.learning_rate,feature_dim=feature_dim, price_horizon=price_horizon, action_classes = action_classes, lin_hidden_dim=lin_hidden_dim, conv_hidden_dim = conv_hidden_dim, target_dim = target_dim, kernel_size = kernel_size, num_layers = num_layers, dropout = dropout, tcn_path = tcn_path).to(self.device)


                

#########################################
########### Experience Replay ###########
#########################################


class ReplayBuffer:
    
    def __init__(self, env, state_dim, buffer_size, action_classes, min_replay_size = 1000, seed = 2705, state_func = None, action_func = None, reward_func = None):
        
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
        self.state_dim = state_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_classes = action_classes
        
        
        #Initialize replay buffer with random transitions (transitions based on random actions)      
        obs, r, terminated, _ , _ = env.step(random.uniform(-1,1))
        state, grads = state_func(obs)
        
        while True:
            
            #action = env.continuous_action_space.sample()
            action = random.randint(0, self.action_classes-1) # Sample random action from action space
            cont_action = action_func(action) # Map discrete action to continous space for environment
            
            new_obs, r, terminated, _ , _ = env.step(cont_action)

            # Get state from observation
            new_state, new_grads = state_func(new_obs)
            
            #Reward Shaping
            new_reward = reward_func(r, cont_action, grads)
            
            if state.shape[0] == state_dim and new_state.shape[0] == state_dim:
            
                transition = (state, action, new_reward, terminated, new_state)
                self.replay_buffer.append(transition)
            
            state = new_state
            grads = new_grads
    
            if len(self.replay_buffer) >= self.min_replay_size:
                break
            
            if terminated:
               print("Terminated")
              
    
    
    def add_data(self, data): 
        '''
        Params:
        data = relevant data of a transition, i.e. action, new_obs, reward, done
        '''
        
        if data[0].shape[0] != self.state_dim and data[4].shape[0] != self.state_dim : # Check if the state dimension of the data matches the state dimension of the replay buffer
            raise ValueError("The state dimension of the data does not match the state dimension of the replay buffer.")
        
        self.replay_buffer.append(data)



    def sample(self, batch_size):
        
        '''
        Params:
        batch_size = number of transitions that will be sampled
        
        Returns:
        tensor of states, actions, rewards, dones (boolean) and new_states 
        '''
        
        if len(self.replay_buffer) < batch_size:
            raise ValueError("Not enough data in the replay buffer.")
        
        # Sample random transitions
        transitions = random.sample(self.replay_buffer, batch_size)
        
        # Add last transition to the sample to make sure that the last two (current) transitions are included in the sample --> MIXED TRAINING / PRIORITIZED EXPERIENCE REPLAY
        transitions[-1] = self.replay_buffer[-1] 
        
        # Shuffle transitions
        random.shuffle(transitions)
            
        states = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_states = np.asarray([t[4] for t in transitions])
        
        #PyTorch needs these arrays as tensors!, don't forget to specify the device! (cpu/GPU) 
        states_t = torch.as_tensor(states, dtype = torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype = torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype = torch.float32, device=self.device).unsqueeze(-1)
        terminated_t = torch.as_tensor(dones, dtype = torch.float32, device=self.device).unsqueeze(-1)
        new_states_t = torch.as_tensor(new_states, dtype = torch.float32, device=self.device)
        
        return states_t, actions_t, rewards_t, terminated_t, new_states_t
    







        
        

###########################
########### DQN ###########
###########################

class DQN(nn.Module):
    
    def __init__(self, learning_rate, feature_dim, price_horizon, hidden_dim, action_classes):
        
        '''
        Params:
        learning_rate = learning rate used in the update
        hidden_dim = number of hidden units in the hidden layer
        action_classes = number of actions that the agent can take
        '''
        
        super(DQN,self).__init__()
        input_features = price_horizon + feature_dim + 1 + 1  #battery charge, price, presence, day, hour
                
        self.linear1 = nn.Linear(input_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, action_classes)
        
        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        #Here we use ADAM, but you could also think of other algorithms such as RMSprob
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
    
    
    
    def forward(self, x):
        
        '''
        Params:
        x = observation
        '''
        
        '''
        ToDo: 
        Write the forward pass! You can use any activation function that you want (ReLU, tanh)...
        Important: We want to output a linear activation function as we need the q-values associated with each action
    
        '''
        
        x = self.leakyReLU(self.linear1(x))
        x = self.leakyReLU(self.linear2(x))
        x = self.leakyReLU(self.linear3(x))
        x = self.linear4(x)
        
        return x
    
    
    
###################################
########### TemporalDQN ###########
###################################
    
class TemporalDQN(nn.Module):
    
    def __init__(self, learning_rate, feature_dim, price_horizon, action_classes, lin_hidden_dim, conv_hidden_dim, target_dim, kernel_size,  num_layers, dropout, tcn_path = None):
        
        '''
        Params:
        learning_rate = learning rate used in the update
        hidden_dim = number of hidden units in the hidden layer
        action_classes = number of actions that the agent can take
        '''
        
        super(TemporalDQN, self).__init__()
        self.price_horizon = price_horizon
        
        # TCN Branch        
        tcn_channels = [conv_hidden_dim] * num_layers # First layer has price_horizon channels, second layer has conv_hidden_dim channels to match the input of dimension price_horizon
        tcn_channels[-1] = int(conv_hidden_dim / 8) # Last layer has conv_hidden_dim/8 channels for efficiency and to reduce variance
        self.tcn = TCN(seq_len = price_horizon, num_inputs = 1, num_channels=tcn_channels, out_channels=target_dim, kernel_size=kernel_size, dropout=dropout) # 3 layers with 128 hidden units each

        # Load TCN
        self.tcn.load_state_dict(torch.load(tcn_path, map_location=torch.device('cpu')))
        
        # Cutting and Freezing the TCN:
        self.tcn = torch.nn.Sequential(*(list(self.tcn.children())[:-1])) # Cut off the last layer of the TCN
        
        # Freeze all layers except the last remaining dense layer:
        for param in self.tcn.parameters():
            param.requires_grad = False
            
        # for param in self.tcn[-1].parameters():
        #     param.requires_grad = True
            
        # Get dimensions of the output of the TCN
        with torch.no_grad():
            temp_out = self.tcn(torch.randn(1,1,self.price_horizon))
            temp_out_dim = temp_out.flatten(start_dim = 1).shape[1]
        
        self.lin_features = feature_dim + 1 + 1 # price history + engineered features

        self.linear1 = nn.Linear(self.lin_features, int(lin_hidden_dim/2))
        self.linear2 = nn.Linear(int(lin_hidden_dim/2), int(lin_hidden_dim/2))
        
        ## Concatenate the TCN output and the linear output
        self.linear3 = nn.Linear(temp_out_dim + int(lin_hidden_dim/2), lin_hidden_dim)
        self.linear4 = nn.Linear(lin_hidden_dim, lin_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(lin_hidden_dim, action_classes)

        self.leakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        #Here we use ADAM, but you could also think of other algorithms such as RMSprob
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
    
    
    
    def forward(self, x):
        
        '''
        Params:
        x = observation
        '''
        
        '''
        ToDo: 
        Write the forward pass! You can use any activation function that you want (ReLU, tanh)...
        Important: We want to output a linear activation function as we need the q-values associated with each action
    
        '''
        #Temporal Branch
        temp = self.tcn(x[:,:self.price_horizon].view(-1,1,self.price_horizon)).flatten(start_dim = 1)
        
        # Linear Branch
        lin = self.leakyReLU(self.linear1(x[:,self.price_horizon:]))
        lin = self.leakyReLU(self.linear2(lin))
        
        # Concatenate
        x = torch.cat((temp, lin), dim=1)
        x = self.dropout(self.leakyReLU(self.linear3(x)))
        x = self.out(x)
        
        return x
    
    
#################
###### TCN ######
#################

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):
    """
    TemporalBlock is a module that represents a single temporal block in a Temporal Convolutional Network (TCN).
    It consists of two convolutional layers with residual connections and dropout.

    Args:
        n_inputs (int): Number of input channels.
        n_outputs (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolutional operation.
        dilation (int): Dilation rate of the convolutional operation.
        padding (int): Padding size for the convolutional operation.
        dropout (float, optional): Dropout probability. Default is 0.2.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()


    def init_weights(self):
        """
        Initializes the weights of the convolutional layers.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        """
        Performs forward pass through the temporal block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
    
    

class DepthwiseSeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block. Applies a depthwise convolution followed by a pointwise convolution to reduce the number of parameters.

    Args:
        n_inputs (int): Number of input channels.
        n_outputs (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride value for the convolution.
        dilation (int): Dilation value for the convolution.
        padding (int): Padding value for the convolution.
        dropout (float, optional): Dropout probability. Default is 0.2.
        downsample (bool, optional): Whether to apply downsampling. Default is False.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1, downsample=False):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise_conv1 = weight_norm(nn.Conv1d(n_inputs, n_inputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation, groups=n_inputs))
        self.pointwise_conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, 1))
        self.chomp1 = Chomp1d(padding)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.depthwise_conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation, groups=n_outputs))
        self.pointwise_conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, 1))
        self.chomp2 = Chomp1d(padding)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.depthwise_conv1, self.pointwise_conv1, self.chomp1, self.elu1, self.dropout1,
                                 self.depthwise_conv2, self.pointwise_conv2, self.chomp2, self.elu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None        
        self.elu = nn.ELU()
        self.init_weights()


    def init_weights(self):
        """
        Initializes the weights of the convolutional layers.
        """
        self.depthwise_conv1.weight.data.normal_(0, 0.01)
        self.pointwise_conv1.weight.data.normal_(0, 0.01)
        self.depthwise_conv2.weight.data.normal_(0, 0.01)
        self.pointwise_conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass of the Depthwise Separable Convolution Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.elu(out + res)




class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels) # num_channels is a list of the number of channels for each layer 
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)



class TCN(nn.Module):
    def __init__(self, seq_len, num_inputs, num_channels, out_channels, kernel_size, dropout=0.1):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(
            num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(seq_len*num_channels[-1], out_channels) #seq_len*num_channels[-1])
        #self.dense2 = nn.Linear(seq_len*num_channels[-1], out_channels)

    def forward(self, x):
        tcn_output = self.tcn(x).flatten(start_dim=1) #Flatten over the features and timestep dimensions, preserve batch dimension (dim=0)
        x = self.dense(self.dropout(tcn_output))
        #x = self.dense2(self.dropout(x))
        return x
    
