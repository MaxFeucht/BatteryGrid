import pandas as pd
import numpy as np
import torch
import math
from collections import deque
from tqdm import tqdm
from tcn import TCN

###########################################
#### Settings for Feature Engineering #####
###########################################

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



def engineer_features(data, fourier_window = fourier_window, gradient_window = gradient_sizes, general_window = window_sizes, price_horizon = price_horizon, target_dim = target_dim, tcn_path = tcn_path, tcn_channels = tcn_channels, kernel_size = kernel_size, dropout = dropout):
    """
    Function that engineers features for the test data. Incorporates all functions defined below.
    
    Args:
        data (pd.DataFrame): DataFrame with test data
        fourier_window (int): Number of data points in each segment for Fourier transform
        gradient_window (list): List of number of previous points to use for gradient features
        general_window (list): List of window sizes to use for moving features
        price_horizon (int): Number of historical prices to use as input
        target_dim (int): Number of prices to predict
        tcn_path (str): Path to pretrained TCN model
        tcn_channels (list): List of number of channels for each layer of the TCN
        kernel_size (int): Kernel size of the TCN
        dropout (float): Dropout rate of the TCN
        
    Returns:
        df (pd.DataFrame): DataFrame with engineered features
    """
    
    df = data.copy()
    df = features_pipeline(df, fourier_window, gradient_window, general_window)
    df = predict_next_prices(df, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout)
    df = finalize_data(df)
    
    return df
    



def features_pipeline(data, fourier_window, gradient_window, general_window):
    """
    Function that applies all feature engineering functions except for the TCN features.
    
    Args:
        data (pd.DataFrame): DataFrame with test data
        fourier_window (int): Number of data points in each segment for Fourier transform
        gradient_window (list): List of number of previous points to use for gradient features
        general_window (list): List of window sizes to use for moving features
    
    Returns:
        data_copy (pd.DataFrame): DataFrame with engineered features
    """
    
    data_copy = data.copy() 

    columns_begin = len(data_copy.columns)

    # Ensure 'datetime' is the first column and 'price' is the second column
    column_order = ['datetime', 'price'] + [col for col in data_copy.columns if col not in ['datetime', 'price']]
    data_copy = data_copy[column_order]
    
    data_copy = date_features(data_copy) 
    print('...date features created...')

    data_copy = fourier_top_freq(data_copy, fourier_window)
    print('...fourier features created...')

    for prev_points in gradient_window:
        data_copy = gradient_features(data_copy, prev_points)
        data_copy = second_gradient_features(data_copy, prev_points) # gradient_1 is here always created
    print('...gradient features created...')

    for window_size in general_window:
        data_copy = moving_averages(data_copy, window_size)
        data_copy = moving_std(data_copy, window_size)
        data_copy = moving_min(data_copy, window_size)
        data_copy = moving_max(data_copy, window_size)
    print('...moving features created...')
    
    # options: time = month, week & both ---- Business_hours = true or false
    data_copy = match_peak_hours(data_copy, business_hours=False, time='month')
    print('...peak hour features created...')


    print('Total added columns: ', len(data_copy.columns) - columns_begin)
    
    if 'date' in data_copy.columns:
        data_copy = data_copy.drop(columns=['date'])

    return data_copy


def match_peak_hours(data, business_hours=True, time='month'):
    """
    Function that matches the peak hours of the data with the peak hours of the business.

    Args:
        data (pd.DataFrame): DataFrame with test data
        business_hours (bool): If True, create features for business hours
        time (str): 'month', 'week' or 'both'

    Returns:
        
    """
    data_copy = data.copy()
    data_copy['week'] = data_copy['datetime'].dt.isocalendar().week
    data_copy['month'] = data_copy['datetime'].dt.month
    data_copy['hour'] = data_copy['datetime'].dt.hour

    peak_df = pd.read_csv('data/peak_df.csv')

    # for every match with 'week' and 'hour' in peak_df, set non_business_peak to 1
    if time == 'week' or time == 'both':
        data_copy['non_business_peak'] = 0
        data_copy['non_business_valley'] = 0
        if business_hours == True:
            data_copy['business_valley'] = 0
            data_copy['business_peak'] = 0
        for index, row in peak_df.iterrows():
            data_copy.loc[(data_copy['week'] == row['week']) &
                        (data_copy['hour'] == row['non_business_peak']),
                        'non_business_peak'] = 1
            data_copy.loc[(data_copy['week'] == row['week']) &
                        (data_copy['hour'] == row['non_business_valley']),
                        'non_business_valley'] = 1
        if business_hours == True:
                data_copy.loc[(data_copy['week'] == row['week']) &
                        (data_copy['hour'] == row['business_peak']),
                        'business_peak'] = 1
                data_copy.loc[(data_copy['week'] == row['week']) &
                        (data_copy['hour'] == row['business_valley']),
                        'business_valley'] = 1
    elif time == 'month' or time == 'both':
        data_copy['month_non_business_peak'] = 0
        data_copy['month_non_business_valley'] = 0
        if business_hours == True:
            data_copy['month_business_valley'] = 0
            data_copy['month_business_peak'] = 0
        for index, row in peak_df.iterrows():
            data_copy.loc[(data_copy['month'] == row['month']) &
                        (data_copy['hour'] == row['month_non_business_peak']),
                            'month_non_business_peak'] = 1
            data_copy.loc[(data_copy['month'] == row['month']) &
                        (data_copy['hour'] == row['month_non_business_valley']),
                        'month_non_business_valley'] = 1
        if business_hours == True:
            data_copy.loc[(data_copy['month'] == row['month']) &
                        (data_copy['hour'] == row['month_business_peak']),
                        'month_business_peak'] = 1
            data_copy.loc[(data_copy['month'] == row['month']) &
                        (data_copy['hour'] == row['month_business_valley']),
                        'month_business_valley'] = 1
    return data_copy



def gradient_features(data, num_prev_points=1):
    """
    Function that calculates the gradient of the 'price' data for each data point.
    
    Args:
        data (pd.DataFrame): Input data
        num_prev_points (int): Number of previous points to use for gradient features
    
    Returns:
        data_copy (pd.DataFrame): Input data with gradient features
    """
    
    data_copy = data.copy()  # Create a copy of the input data

    for i in range(len(data_copy) - 1):
        if i == 0:
            data_copy.loc[i, f'gradient_{num_prev_points}'] = 0
        else:
            gradient_sum = 0
            for j in range(num_prev_points):
                location_point_a = max(i - j, 0)
                location_point_b = max(i - j - 1, 1)
                point_a = data_copy.loc[location_point_a, 'price']
                point_b = data_copy.loc[location_point_b, 'price']
                gradient_sum += point_a - point_b

            data_copy.loc[i, f'gradient_{num_prev_points}'] = gradient_sum 
    
    return data_copy


def second_gradient_features(data, num_prev_points=1):
    """
    Function that calculates the second gradient of the 'price' data for each data point.
    
    Args:
        data (pd.DataFrame): Input data
        num_prev_points (int): Number of previous points to use for gradient features
    
    Returns:
        data_copy (pd.DataFrame): Input data with second gradient features
    """
    
    data_copy = data.copy()  # Create a copy of the input data

    # check if gradient_1 column exists
    if 'gradient_1' not in data_copy.columns:
        data_copy = gradient_features(data_copy, num_prev_points=1)
    
    for i in range(len(data_copy) - 1):
        if i == 0:
            data_copy.loc[i, f'second_gradient_{num_prev_points}'] = 0
        else:
            second_gradient_sum = 0

            for j in range(num_prev_points): # for amount of num_prev_points compare to the previous point
                location_point_a = max(i - j, 0)
                location_point_b = max(i - j - 1, 1)
                point_a = data_copy.loc[location_point_a, 'gradient_1']
                point_b = data_copy.loc[location_point_b, 'gradient_1']
                second_gradient_sum += point_a - point_b
            
            data_copy.loc[i, f'second_gradient_{num_prev_points}'] = second_gradient_sum    
    
    return data_copy


def fourier_top_freq(data, segment_size=72):
    '''
    Applies Fourier transform to segments of the 'price' data and extracts the top 3 frequencies.
    
    Args:
        data (DataFrame): Input data.
        segment_size (int): Number of data points in each segment for Fourier transform.
    
    Returns:
        DataFrame: The input data with top 3 Fourier frequencies for each segment.
    '''
    data_copy = data.copy()  # Create a copy of the input data

    # Create new columns for the top 3 fourier frequencies
    for i in range(3):
        data_copy[f'fourier_freq_{i + 1}'] = np.nan

    # For each range of data points, calculate the Fourier transform
    for i in range(segment_size, len(data_copy), 1): # Start at <segment_size>
        # Fourier transform of the last <segment_size> data points
        segment = data_copy['price'][i - segment_size:i]
        fourier_coeffs = np.fft.fft(segment)
        freqs = np.fft.fftfreq(segment_size, d=1)  # Assuming hourly data, hence d=1

        # Get indices of top 3 frequencies based on magnitude of Fourier coefficients
        indices = np.argsort(np.abs(fourier_coeffs))[::-1][1:4] # ::-1 to sort in descending order

        for j in range(3):
            column_name = f'fourier_freq_{j + 1}'
            data_copy.loc[i, column_name] = freqs[indices[j]]

    return data_copy


def moving_averages(data, window_size=72):
    '''
    Calculates the moving average of the 'price' data for each data point.
    '''
    data_copy = data.copy()  # Create a copy of the input data
    data_copy[f'moving_average_{window_size}'] = data_copy['price'].rolling(window=window_size, min_periods=1).mean()
    return data_copy


def moving_std(data, window_size=72):
    """
    Function that calculates the moving standard deviation of the 'price' data for each data point.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Window size to use for moving standard deviation
    
    Returns:
        data_copy (pd.DataFrame): Input data with moving standard deviation features
    """
    data_copy = data.copy()  # Create a copy of the input data

    data_copy[f'moving_std_{window_size}'] = data_copy['price'].rolling(window=window_size, min_periods=1).std()
    return data_copy


def moving_min(data, window_size=72):
    """
    Function that calculates the moving minimum of the 'price' data for each data point.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Window size to use for moving standard deviation
    
    Returns:
        data_copy (pd.DataFrame): Input data with moving standard deviation features
    """
    
    data_copy = data.copy()  # Create a copy of the input data
    data_copy[f'moving_min_{window_size}'] = data_copy['price'].rolling(window=window_size, min_periods=1).min()
    return data_copy


def moving_max(data, window_size=72):
    """
    Function that calculates the moving maximum of the 'price' data for each data point.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Window size to use for moving standard deviation
    
    Returns:
        data_copy (pd.DataFrame): Input data with moving standard deviation features
    """
    
    data_copy = data.copy()  # Create a copy of the input data
    data_copy[f'moving_max_{window_size}'] = data_copy['price'].rolling(window=window_size, min_periods=1).max()
    return data_copy


def date_features(data):
    """
    Function that creates date features from the 'datetime' column.
    
    Args:
        data (pd.DataFrame): Input data
    
    Returns:
        data_copy (pd.DataFrame): Input data with date features
    """
    
    data_copy = data.copy()  # Create a copy of the input data

    data_copy['day_of_week'] = data_copy['datetime'].dt.dayofweek
    data_copy['day_of_month'] = data_copy['datetime'].dt.day
    data_copy['week'] = data_copy['datetime'].dt.isocalendar().week
    data_copy['month'] = data_copy['datetime'].dt.month
    data_copy['year'] = data_copy['datetime'].dt.year
    data_copy['hour'] = data_copy['datetime'].dt.hour
    data_copy['season'] = (data_copy['month'] - 1) // 3 + 1
    return data_copy


def average_date_features(data):
    """
    Function that creates average date features from the 'datetime' column.

    Args:
        data (pd.DataFrame): Input data
    
    Returns:
        data_copy (pd.DataFrame): Input data with average date features
    """
    data_copy = data.copy()  # Create a copy of the input data

    data_copy['average_hour'] = data_copy.groupby(['hour'])['price'].transform('mean')
    data_copy['average_day_of_week'] = data_copy.groupby(['day_of_week', 'hour'])['price'].transform('mean')
    data_copy['average_day_of_month'] = data_copy.groupby(['day_of_month', 'hour'])['price'].transform('mean')
    data_copy['average_week'] = data_copy.groupby(['week', 'hour'])['price'].transform('mean')
    data_copy['average_month'] = data_copy.groupby(['month', 'hour'])['price'].transform('mean')
    data_copy['average_season'] = data_copy.groupby(['season', 'hour'])['price'].transform('mean')

    data_copy['average_hour_x_week'] = data_copy.groupby(['hour', 'week'])['price'].transform('mean')
    data_copy['average_hour_x_month'] = data_copy.groupby(['hour', 'month'])['price'].transform('mean')
    return data_copy



def elongate(df):
    """
    Function that elongates the data from wide to long format.
    
    Args:
        df (pd.DataFrame): Input data
    
    Returns:
        df_long (pd.DataFrame): Elongated data
    """
    
    df_long = pd.wide_to_long(df, i = "PRICES", j = "hour", stubnames=["Hour"], sep = " ").reset_index()
    df_long.rename(columns={"Hour": "price", "PRICES": "date"}, inplace = True)
    df_long['datetime'] = pd.to_datetime(df_long['date']) + pd.to_timedelta(df_long['hour'], unit='h')
    df_long.sort_values(['datetime'], ascending=[True], inplace=True)
    df_long['price'] = df_long['price'].astype(float) 
    return df_long.reset_index(drop=True)



def predict_next_prices(feature_df, price_horizon, target_dim, tcn_path, tcn_channels, kernel_size, dropout, cut_layers = False):
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
    
    # Load pretrained TCN model
    df = feature_df.copy()
    tcn = TCN(seq_len = price_horizon, num_inputs = 1, num_channels=tcn_channels, out_channels=target_dim, kernel_size=kernel_size, dropout=dropout) # 3 layers with 128 hidden units each
    tcn.load_state_dict(torch.load(tcn_path, map_location=torch.device('cpu')))
    tcn.eval()
    
    # Cut off the last two layers of the TCN if needed
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

    # Initialize deque with historical prices
    prices = deque(maxlen=price_horizon)
    df[col_list] = 0.0

    # Predict next 5 prices at each timestep
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



def finalize_data(df):
    """
    Function that finalizes the data by removing the first price_horizon rows and the last target_dim rows.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        price_horizon (int): Number of historical prices to use as input
        target_dim (int): Number of prices to predict
    
    Returns:
        df (pd.DataFrame): Finalized DataFrame with features
    """
    data = df.copy()
    data.fillna(0, inplace=True)
    data = data.replace([np.inf, -np.inf], 0)

    return data