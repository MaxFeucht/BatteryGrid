import torch
import torch.nn as nn
import torch.optim as optim
from tcn import TCN


class DQN(nn.Module):
    
    def __init__(self, learning_rate, price_horizon = 96, hidden_dim = 128, action_classes = 7):
        
        '''
        Params:
        learning_rate = learning rate used in the update
        hidden_dim = number of hidden units in the hidden layer
        action_classes = number of actions that the agent can take
        '''
        
        super(DQN,self).__init__()
        input_features = price_horizon + 1 + 1 + 1 + 1 #battery charge, price, presence, day, hour
                
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
    
    
    
class TemporalDQN(nn.Module):
    
    def __init__(self, learning_rate, price_horizon = 96, action_classes = 7, lin_hidden_dim = 128, temp_hidden_dim = 64, kernel_size = 2, dropout = 0.2):
        
        '''
        Params:
        learning_rate = learning rate used in the update
        hidden_dim = number of hidden units in the hidden layer
        action_classes = number of actions that the agent can take
        '''
        
        super(TemporalDQN, self).__init__()
        self.price_horizon = price_horizon
        self.input_features = self.price_horizon + 1 + 1 + 1 + 1 #battery charge, price, presence, day, hour
        
        tcn_channels = [price_horizon, temp_hidden_dim] # First layer has price_horizon channels, second layer has temp_hidden_dim channels to match the input of dimension price_horizon
        self.tcn = TCN(seq_len = price_horizon, num_inputs = 1, num_channels=tcn_channels, out_channels=temp_hidden_dim, kernel_size=kernel_size, dropout=dropout) # 3 layers with 128 hidden units each
        
        self.linear1 = nn.Linear(self.input_features - self.price_horizon, int(lin_hidden_dim/4))
        self.linear2 = nn.Linear(int(lin_hidden_dim/4), int(lin_hidden_dim/2))
        
        ## Concatenate the TCN output and the linear output
        self.linear3 = nn.Linear(temp_hidden_dim + int(lin_hidden_dim/2), lin_hidden_dim)
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
        temp = self.tcn(x[:,:self.price_horizon].view(-1,1,self.price_horizon)) 
        
        # Linear Branch
        lin = self.leakyReLU(self.linear1(x[:,self.price_horizon:]))
        lin = self.leakyReLU(self.linear2(lin))
        
        # Concatenate
        x = torch.cat((temp, lin), dim=1)
        x = self.dropout(self.leakyReLU(self.linear3(x)))
        x = self.out(x)
        
        return x