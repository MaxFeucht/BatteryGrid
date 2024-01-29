#%%

import pandas as pd
import numpy as np
import torch
import math 
from collections import deque
from tqdm import tqdm
from tcn import TCN
from matplotlib import pyplot as plt

#%% Load Data

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


#%% Feature Engineering
from feature_eng import *

gradient_sizes = [1, 2, 4, 6, 8, 12, 18, 24]
fourier_window = 72
window_sizes = [3, 6, 12, 24, 48, 72]

features_train = features_pipeline(train, fourier_window, gradient_sizes, window_sizes)
features_val = features_pipeline(val, fourier_window, gradient_sizes, window_sizes)

#%% ## Predict next 5 prices with TCN model

seed = 2705
batch_size = 32
lin_hidden_dim = 128
kernel_size = 3
dropout = 0.1
target_dim = 5
variant = 1

if variant == 1:
    price_horizon = 120
    num_layers = math.ceil(math.log2(price_horizon/kernel_size) + 1)
    temp_hidden_dim = 64
    tcn_channels = [temp_hidden_dim] * num_layers # First layer has price_horizon channels, second layer has temp_hidden_dim channels to match the input of dimension price_horizon
    tcn_path = f'models/tcn_{target_dim}_horizon_large.pt'
else:
    price_horizon = 72
    num_layers = math.ceil(math.log2(price_horizon/kernel_size) + 1)
    temp_hidden_dim = 16
    tcn_channels = [temp_hidden_dim] * num_layers # First layer has price_horizon channels, second layer has temp_hidden_dim channels to match the input of dimension price_horizon
    tcn_channels[-1] = int(temp_hidden_dim / 8)
    tcn_path = f'models/tcn_{price_horizon}_horizon_{target_dim}_future.pt'



#%%

def predict_next_prices(feature_df, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout, cut_layers = False):
    global col_list
    """
    Function that predicts the next 5 prices at each timestep with a pretrained TCN model.
    
    Args:
        feature_df (pd.DataFrame): DataFrame with features
        price_horizon (int): Number of historical prices to use as input
        target_dim (int): Number of prices to predict
        tcn_path (str): Path to pretrained TCN model
        tcn_channels (list): List of number of channels for each layer of the TCN
        kernel_size (int): Kernel size of the TCN
        dropout (float): Dropout rate of the TCN
    
    Returns:
    """
    
    df = feature_df.copy()
    tcn = TCN(seq_len = price_horizon, num_inputs = 1, num_channels=tcn_channels, out_channels=target_dim, kernel_size=kernel_size, dropout=dropout) # 3 layers with 128 hidden units each
    tcn.load_state_dict(torch.load(tcn_path, map_location=torch.device('cpu')))
    tcn.eval()
    
    if cut_layers:
        tcn = torch.nn.Sequential(*(list(tcn.children())[:-2])) # Cut off the last two layers of the TCN
    
    # Get number of output dimensions of TCN
    with torch.no_grad():
        temp_out = tcn(torch.randn(1,1,price_horizon))
        temp_out_dim = temp_out.flatten(start_dim = 1).shape[1]
        col_list = [f'tcn_{i}' for i in range(1, temp_out_dim + 1)]
        print(f'TCN output dimension: {temp_out_dim}')
        
    # Predict next 5 prices with pretrained TCN model
    print('Predicting next 5 prices at each timestep with TCN model...')

    prices = deque(maxlen=price_horizon)
    df[col_list] = 0.0

    for i, price in tqdm(enumerate(df['price'].values)):
        prices.append(price)
        if len(prices) == price_horizon:
            with torch.no_grad():
                prices_tensor = torch.tensor(prices, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                prices_tensor -= prices_tensor.min(-1, keepdim=True)[0]
                prices_tensor /= prices_tensor.max(-1, keepdim=True)[0]
                df.loc[i, col_list] = tcn(prices_tensor).squeeze(0).squeeze(0).detach().numpy()        

    print('Done!')
    
    return df

  
features_train = predict_next_prices(features_train, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout)
features_val = predict_next_prices(features_val, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout)

features_train_cut = predict_next_prices(features_train, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout, cut_layers = True)
features_val_cut = predict_next_prices(features_val, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout, cut_layers = True)


#%% # Finalize data


features_train.fillna(0, inplace=True)
features_val.fillna(0, inplace=True)

features_train = features_train.replace([np.inf, -np.inf], 0)
features_val = features_val.replace([np.inf, -np.inf], 0)

features_train.to_csv('data/features_train.csv', index=False)
features_val.to_csv('data/features_val.csv', index=False)



#%%

features_train_cut = predict_next_prices(features_train, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout, cut_layers = True)
features_val_cut = predict_next_prices(features_val, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout, cut_layers = True)

features_train_cut.fillna(0, inplace=True)
features_val_cut.fillna(0, inplace=True)

features_train_cut = features_train_cut.replace([np.inf, -np.inf], 0)
features_val_cut = features_val_cut.replace([np.inf, -np.inf], 0)

features_train_cut.to_csv('data/features_train_cut.csv', index=False)
features_val_cut.to_csv('data/features_val_cut.csv', index=False)


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
