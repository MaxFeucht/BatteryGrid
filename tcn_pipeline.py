import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import pandas as pd
from tqdm import tqdm
import random
import math

from collections import deque
from tcn import TCN

seed = 2705
TRAIN = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load Data
def elongate(df):
    df_long = pd.wide_to_long(df, i = "PRICES", j = "hour", stubnames=["Hour"], sep = " ").reset_index()
    df_long.rename(columns={"Hour": "price", "PRICES": "date"}, inplace = True)
    df_long['datetime'] = pd.to_datetime(df_long['date']) + pd.to_timedelta(df_long['hour'], unit='h')
    df_long.sort_values(['datetime'], ascending=[True], inplace=True)
    df_long['price'] = df_long['price'].astype(float) 
    return df_long.reset_index(drop=True)

train_name = 'data/train.xlsx'
val_name = 'data/validate.xlsx'

train = elongate(pd.read_excel(train_name))
val = elongate(pd.read_excel(val_name))


#%%
## Create input and target tensors

price_horizon = 120
target_dim = 3

history = deque(maxlen=price_horizon)

inputs = torch.empty((1, price_horizon))
targets = torch.empty((1,target_dim))

prices = train.loc[:, 'price'].to_numpy()

for i, price in tqdm(enumerate(prices)):
    
    history.append(price)
    target = prices[i+1:i+1+target_dim] # 3 next prices to predict
    
    if len(history) == price_horizon: # If there are already enough historical prices
        if len(target) == target_dim: # If there are enough prices left to predict
            inputs = torch.cat((inputs, torch.tensor(history).unsqueeze(0)))
            targets = torch.cat((targets, torch.tensor(target).unsqueeze(0)))    
    
    assert inputs.shape[0] == targets.shape[0]


#%%


seed = 2705
rep = 2000000
batch_size = 48
gamma = 0.96
epsilon = 1.0
epsilon_decay = 99999
epsilon_min = 0.1
learning_rate = 5e-5
lin_hidden_dim = 128
temp_hidden_dim = 64
kernel_size = 3
dropout = 0.1
price_horizon = 120



### Training Loop for TCN ###
num_layers = math.ceil(math.log2(price_horizon/kernel_size) + 1)
tcn_channels = [temp_hidden_dim] * num_layers # First layer has price_horizon channels, second layer has temp_hidden_dim channels to match the input of dimension price_horizon
tcn = TCN(seq_len = price_horizon, num_inputs = 1, num_channels=tcn_channels, out_channels=target_dim, kernel_size=kernel_size, dropout=dropout) # 3 layers with 128 hidden units each

optimizer = torch.optim.Adam(tcn.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

epochs = 100


for e in range(epochs):
    
    permute_idx = torch.randperm(inputs.shape[0])
    inputs = inputs[permute_idx]
    targets = targets[permute_idx]
    
    for i in tqdm(range(0, inputs.shape[0], batch_size)):
        optimizer.zero_grad()
        
        # Select random batch
        batch_idx = [idx for idx in range(i, min(i+batch_size, inputs.shape[0]))]
        batch_inputs = inputs[batch_idx]
        batch_targets = targets[batch_idx]
        
        # To Device
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        predictions = tcn(batch_inputs.unsqueeze(1).float())
        
        # Two losses: One for the max value index, one for the min value index
        # max_loss = criterion(predictions, batch_targets.unsqueeze(1).float())
        # min_loss = criterion(1-predictions, batch_targets.unsqueeze(1).float())
        # loss = max_loss + min_loss
        
        # Regression loss of the 3 values
        loss = criterion(predictions, batch_targets.unsqueeze(1).float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {e}, Loss: {loss.item()}')


