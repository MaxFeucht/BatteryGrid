import torch.nn as nn
import torch.optim as optim

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