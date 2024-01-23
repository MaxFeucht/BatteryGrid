import gymnasium as gym
from gymnasium import spaces
from collections import deque
import numpy as np
import pandas as pd



class States():

    def __init__(self, features, price_horizon = 96, normalize = True, verbose = False):
        """
        Initializes the state of the environment
        
        Args:
            features (pd.DataFrame): Dataframe containing the features
            price_horizon (int): Number of prices in the price history
            normalize (bool): Normalize the data between 0 and 1
            verbose (bool): Print statements for debugging
        """
        
        self.price_horizon = price_horizon
        self.features = np.array(features)
        self.normalize = normalize
        self.verbose = verbose

        # Set up price_history queue
        self.price_history = deque(maxlen=price_horizon)

        # Normalize features per column (feature) but not datetime and price in the first two columns
        self.features[:,2:] = self.normalize_data(self.features[:,2:], axis=0) if self.normalize else self.features[:,2:] # Normalize features per column (feature) 
        
                

    def match_obs_to_state(self, obs):
        """
        Matches the observation to the state of the environment
        
        Args:
            obs (np.array): Observation of the environment
        
        Returns:
            state (np.array): State of the environment
        """
        
        battery_level = obs[0]
        price = obs[1]
        hour = obs[2]
        day_of_week = obs[3]
        day_of_year = obs[4]
        month = obs[5]
        year = obs[6]
        car_is_available = obs[7]
        
        # Fill price history
        self.price_history.append(price)

        # Match to engineered features by price, hour, day, month, year
        features = self.features[self.env.counter-1]
                
        # Get date and prices for assertions between features and observation price
        feature_date = features[0]
        features = np.array(features[1:], dtype=np.float32)
        feature_price = features[0]
        features = features[1:] #by doing two times features[1:] we remove the date and price from the features array (not elegant but works)
        
        assert hour == feature_date.hour, "Hour and features do not match"
        assert day_of_week == feature_date.dayofweek, "Day of week and features do not match"
        assert day_of_year == feature_date.dayofyear, "Day of year and features do not match"
        assert month == feature_date.month, "Month and features do not match"
        assert year == feature_date.year, "Year and features do not match"
        assert price == feature_price, "Price history ({price_history[-1]}) and price (feature_price)do not match"
        
        # Normalize data
        if self.normalize:
            self.price_history = self.normalize_data(self.price_history)
            battery_level /= 50
            # features are already normalized in the setup function
        
        # Concatenate price history, battery level, car availability and features
        state = np.concatenate((self.price_history, np.array([battery_level, car_is_available]), features))   
        
        return state
        


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



    
    

