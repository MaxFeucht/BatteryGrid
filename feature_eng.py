import pandas as pd
import numpy as np


def features_pipeline(data, fourier_window, gradient_window, general_window):
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
    
    Parameters:
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
    data_copy = data.copy()  # Create a copy of the input data

    '''
    Calculates the moving average of the 'price' data for each data point.

    '''
    data_copy[f'moving_average_{window_size}'] = data_copy['price'].rolling(window=window_size, min_periods=1).mean()
    return data_copy


def moving_std(data, window_size=72):
    data_copy = data.copy()  # Create a copy of the input data

    data_copy[f'moving_std_{window_size}'] = data_copy['price'].rolling(window=window_size, min_periods=1).std()
    return data_copy


def moving_min(data, window_size=72):
    data_copy = data.copy()  # Create a copy of the input data

    data_copy[f'moving_min_{window_size}'] = data_copy['price'].rolling(window=window_size, min_periods=1).min()
    return data_copy


def moving_max(data, window_size=72):
    data_copy = data.copy()  # Create a copy of the input data

    data_copy[f'moving_max_{window_size}'] = data_copy['price'].rolling(window=window_size, min_periods=1).max()
    return data_copy


def date_features(data):
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
    data_copy = data.copy()  # Create a copy of the input data

    # Lets reconsider to not use this it feels like it would not translate well to the test set
    # time of day
    # combinations
    data_copy['average_hour'] = data_copy.groupby(['hour'])['price'].transform('mean')
    data_copy['average_day_of_week'] = data_copy.groupby(['day_of_week', 'hour'])['price'].transform('mean')
    data_copy['average_day_of_month'] = data_copy.groupby(['day_of_month', 'hour'])['price'].transform('mean')
    data_copy['average_week'] = data_copy.groupby(['week', 'hour'])['price'].transform('mean')
    data_copy['average_month'] = data_copy.groupby(['month', 'hour'])['price'].transform('mean')
    data_copy['average_season'] = data_copy.groupby(['season', 'hour'])['price'].transform('mean')

    data_copy['average_hour_x_week'] = data_copy.groupby(['hour', 'week'])['price'].transform('mean')
    data_copy['average_hour_x_month'] = data_copy.groupby(['hour', 'month'])['price'].transform('mean')
    return data_copy
