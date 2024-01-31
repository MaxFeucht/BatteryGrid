#%%

import pandas as pd
import numpy as np
import torch
import math 
from collections import deque
from tqdm import tqdm
from tcn import TCN
from matplotlib import pyplot as plt
from feature_eng import *

#%% Settings for Feature Engineering

# Dataset names
train_name = 'data/train.xlsx'
val_name = 'data/validate.xlsx'

# Gradient, Fourier and Window Features
gradient_sizes = [1, 2, 4, 6, 8, 12, 18, 24]
fourier_window = 72
window_sizes = [3, 6, 12, 24, 48, 72]

# TCN parameters
seed = 2705
batch_size = 32
lin_hidden_dim = 128
kernel_size = 3
dropout = 0.1
target_dim = 5
variant = 1
price_horizon = 120
num_layers = math.ceil(math.log2(price_horizon/kernel_size) + 1)
temp_hidden_dim = 64
tcn_channels = [temp_hidden_dim] * num_layers # First layer has price_horizon channels, second layer has temp_hidden_dim channels to match the input of dimension price_horizon
tcn_path = f'models/tcn_{target_dim}_horizon_large.pt'
  
# Bring data into the right format
train = elongate(pd.read_excel(train_name))
val = elongate(pd.read_excel(val_name))

# Add Fourier and Gradient Features
features_train = features_pipeline(train, fourier_window, gradient_sizes, window_sizes)
features_val = features_pipeline(val, fourier_window, gradient_sizes, window_sizes)

# Add TCN Features
features_train = predict_next_prices(features_train, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout)
features_val = predict_next_prices(features_val, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout)

# Finalize data
features_train = finalize_data(features_train)
features_val = finalize_data(features_val)

# Save data
features_train.to_csv('data/features_train.csv', index=False)
features_val.to_csv('data/features_val.csv', index=False)




#%% Plotting


## Create input and target tensors
price_horizon = 120
target_dim = 5


def create_tensors(df, price_horizon, target_dim):
    """
    Function to create input and target tensors for the TCN model.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        price_horizon (int): Number of historical prices to use as input
        target_dim (int): Number of prices to predict
    
    Returns:
        inputs (torch.Tensor): Tensor with the input prices
        targets (torch.Tensor): Tensor with the target prices
    """
    
    history = deque(maxlen=price_horizon)
    inputs = torch.zeros((1, price_horizon))
    targets = torch.zeros((1,target_dim))
    prices = df.loc[:, 'price'].to_numpy()

    for i, price in tqdm(enumerate(prices)):

        history.append(price)
        target = prices[i+1:i+1+target_dim] # 3 next prices to predict

        # Normalizing Inputs and targets together in a local manner (min-max normalization)
        chunk = torch.concat((torch.tensor(history, dtype = torch.float32).unsqueeze(0), torch.tensor(target, dtype = torch.float32).unsqueeze(0)), dim = 1)
        chunk -= chunk.min(1, keepdim=True)[0]
        chunk /= chunk.max(1, keepdim=True)[0]
        
        history_t = chunk[:,:price_horizon]
        target = chunk[:,price_horizon:]

        if len(history) == price_horizon: # If there are already enough historical prices
            if target.shape[1] == target_dim: # If there are enough prices left to predict

                inputs = torch.cat((inputs, history_t))
                targets = torch.cat((targets, target))

        assert inputs.shape[0] == targets.shape[0]
    
    return inputs, targets



def plot_predictions(inputs, targets, model, target_dim):
    """
    Function to plot the prediction of the next 5 prices by the pretrained TCN model.
    
    Args:
        inputs (torch.Tensor): Tensor with the input prices
        targets (torch.Tensor): Tensor with the target prices
        model (torch.nn.Module): Pretrained TCN model
        target_dim (int): Number of prices to predict    
    """
    
    plt.figure(figsize=(15,20))

    for i in range(12):
        
        plt.subplot(4,3,i+1)
        rand_idx = np.random.randint(0, inputs.shape[0])
        
        input_subset = inputs[rand_idx]
        target_subset = targets[rand_idx]
        
        actual = torch.concat((input_subset[(inputs.shape[1]-10):], target_subset), dim = 0).detach().numpy()
        
        model.eval()
        with torch.no_grad():
            pred = model(input_subset.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy()
            
        pred_chunk = torch.concat((input_subset[(inputs.shape[1]-10):], torch.tensor(pred, dtype = torch.float32)), dim = 0)    
        
        plt.plot(actual,  linestyle = '--', color = 'grey', label = 'Actual')
        plt.plot(pred_chunk,  linestyle = '-', color = 'red', label = 'Predicted')
        plt.vlines(9, ymin = 0, ymax = 1, color = 'black', linestyle = '--', linewidth = 0.5) # Marker where the prediction starts
        plt.legend(loc = 'upper left')
        plt.xticks([i for i in range(len(pred_chunk) + 2)], [i for i in range(inputs.shape[1]-10, inputs.shape[1] + target_dim + 2)], rotation = 45)
        plt.legend()
        
    

inputs, targets = create_tensors(train, price_horizon, target_dim)
val_inputs, val_targets = create_tensors(val, price_horizon, target_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tcn = TCN(seq_len = price_horizon, num_inputs = 1, num_channels=tcn_channels, out_channels=target_dim, kernel_size=kernel_size, dropout=dropout) # 3 layers with 128 hidden units each
tcn.load_state_dict(torch.load(tcn_path, map_location=torch.device('cpu')))
tcn.eval()

plot_predictions(inputs, targets, tcn, target_dim)
plot_predictions(val_inputs, val_targets, tcn, target_dim)

# %%
