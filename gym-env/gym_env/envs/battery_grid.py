import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd



class BatteryGridEnv(gym.Env):

    def __init__(self):
        self.index = 0
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "battery": spaces.Box(0, 50, dtype=float),
                "prices": spaces.Box(0, np.inf, dtype=float),
                "presence": spaces.Discrete(2),
                "hour": spaces.Discrete(24),
                "day": spaces.Discrete(365),
                "tensor": spaces.Box(0, np.inf, shape=(100,), dtype=float)
            }
        )

        # We have 3 actions, corresponding to selling, buying and doing nothing
        self.action_space = spaces.Discrete(11)
        # If continous action space: action is simply a number between -1 and 1 where -1 indicates selling and 1 indicates buying at 25 kWh - everything in between is selling / buying at a fraction of 25 kWh
        # self.action_space = spaces.Box(-1, 1, dtype=float)


    def setup(self, df, price_horizon = 96):
        self.prices = np.array(df['price'])  # The size of the square grid
        self.datetime = list(df['datetime'])  # The size of the PyGame window
        self.price_horizon = price_horizon
    

    def normalize(self, data):
        """
        Helper function to normalize data between 0 and 1
        """
        return (data - np.min(data)) / (np.max(data) - np.min(data))



    def _get_obs(self, normalize = True):
        """Function to get the observation of the environment

        Args:
            normalize (bool, optional): Normalizes data between 0 and 1. Defaults to True.

        Returns:
            obs_dict: dictionary of observations
        """
        
        price_history = self.prices[:self.index+1]
        battery_charge = self.battery_charge 
        hour = self.datetime[self.index].hour + 6 # Shift to 6 am, such that the day is not "split" in the middle of the night
        day = self.datetime[self.index].day
        
        if len(price_history) < self.price_horizon:
            price_history = np.concatenate((np.zeros(self.price_horizon-len(price_history)), price_history))

        elif len(price_history) > self.price_horizon:
            price_history = price_history[self.index-self.price_horizon:self.index] 
            
        if normalize: # Normalize data
            price_history = self.normalize(price_history)
            battery_charge = battery_charge / 50
            hour = hour / 24 
            day = day / 365
            
        obs_dict = {"battery": battery_charge, "hour": hour, "day": day, "presence": self.presence, "prices": price_history}
        obs_dict["tensor"] = self.dict_to_tensor(obs_dict)
        
        return obs_dict



    def dict_to_tensor(self, obs_dict):
        
        tensor = np.array([])
        for k,v in obs_dict.items():
            if k == "prices":
                tensor = np.concatenate((tensor, v))
            else:
                tensor = np.concatenate((tensor, np.array([v])))
                            
        return tensor
            

    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.battery_charge = 0
        self.index = 0
        self.presence = 1
        
        obs_dict = self._get_obs()

        return obs_dict
        
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        current_price = self.prices[self.index]
        current_hour = self.datetime[self.index].hour        
                
        # Check charge of battery for next day
        if current_hour == 7:
            if self.battery_charge < 20:
                action = np.ceil(((self.battery_charge)/0.9) / 5) # Charge 0 (= 25kwH) when battery is empty, 1 (= 20 kWh) when battery has 5 kWh, 2 (= 15 kWh) when battery has 10 kWh, etc.
                # TODO: add penalty for trying to do anything else than charge when it is 7 and battery is not ready
            
        
        # Check availability of car during the day
        if current_hour == 8:
            draw = np.random.uniform(0,1)
            self.presence = 1 if draw < 0.5 else 0
        elif current_hour == 18:
            if self.presence == 0:
                self.battery_charge -= 20 # Discharge of 20 kWh if car was gone the whole day
            self.presence = 1

        # If car present, take action
        if self.presence == 1:
            
            if action < 5:  # No if statement, because we can always charge
                kwH = (5 - action) * 5 # Discretize, such that action 0 means most discharge, i.e., kWh = (5 - 0)* 5 = 25
                self.battery_charge = np.clip(self.battery_charge + kwH*0.9, 0, 50)
                reward = -current_price * kwH * 2  
                
            elif action > 5: 
                kwH = (action - 5) * 5 # Discretize, such that action 10 means most discharge, i.e., kWh = (10 - 5)* 5 = 25
                discharge = min(self.battery_charge , kwH) # Discharge at most action * 25, but less if battery is has less than 25 kWh
                self.battery_charge = np.clip(self.battery_charge - discharge, 0, 50)
                reward = current_price * discharge * 0.9
                
            elif action == 5:
                reward = 0
                
            else:
                raise ValueError(f"Invalid action {action}")      

        else:
            if action == 0 or action == 1:
                reward = -5 # penalty for trying to charge when car is not available
            else:
                reward = 0


        # Obtain observation
        observation = self._get_obs(normalize = True)
        info = {"price_history" : self.prices[:self.index+1]}
        
        # Increase index
        self.index += 1
        
        # Check termination
        try:
            p = self.prices[self.index] # check if there is a next price
            terminated = False
        except:
            terminated = True

        return observation, reward, terminated, info
    
    
    
    
    
    
    
#### Continous action space ####
    
class BatteryGridEnvCont(gym.Env):
    
    def __init__(self):
        self.index = 0
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "battery": spaces.Box(0, 50, dtype=float),
                "prices": spaces.Box(0, np.inf, dtype=float),
                "presence": spaces.Discrete(2),
                "hour": spaces.Discrete(24),
                "day": spaces.Discrete(365)
            }
        )

        # We have 3 actions, corresponding to selling, buying and doing nothing
        #self.action_space = spaces.Discrete(3)
        # If continous action space: action is simply a number between -1 and 1 where -1 indicates selling and 1 indicates buying at 25 kWh - everything in between is selling / buying at a fraction of 25 kWh
        self.action_space = spaces.Box(-1, 1, dtype=float)


    def set_data(self, df):
        self.prices = np.array(df['price'])  # The size of the square grid
        self.datetime = df['datetime']  # The size of the PyGame window


    def _get_obs(self):
        return {"battery": self.battery_charge, "prices": self.prices[:self.index+1], "hour": self.datetime[self.index].hour, "day": self.datetime[self.index].day, "presence": self.presence}
    
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.battery_charge = 0
        self.price = 0
        self.presence = 1
        self.index = 0
        
        observation = self._get_obs()

        return observation
        
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        current_price = self.prices[self.index]
        current_hour = self.datetime[self.index].hour        
                
        # Check charge of battery for next day
        if current_hour == 7:
            if self.battery_charge < 20:
                action = 0.8 # Equals 20 kWh when normalized
                # TODO: add penalty for trying to do anything else than charge when it is 7 and battery is not ready
            
        # Check availability of car during the day
        if current_hour == 8:
            draw = np.random.uniform(0,1)
            self.presence = 1 if draw < 0.5 else 0
        elif current_hour == 18:
            if self.presence == 0:
                self.battery_charge -= 20 # Discharge of 20 kWh if car was gone the whole day
            self.presence = 1


        if self.presence == 1:
            
            if action < 0:  # No if statement, because we can always charge
                self.battery_charge = np.clip(self.battery_charge + 25*0.9, 0, 50)
                reward = current_price * 25 * action * 2 
                    
            elif action > 0: 
                discharge = min(self.battery_charge / 25, action) # Discharge at most action * 25, but less if battery is has less than 25 kWh
                self.battery_charge = np.clip(self.battery_charge - 25 * discharge, 0, 50)
                reward = current_price * 25 * discharge * 0.9

            elif action == 0:
                reward = 0
                
            else:
                raise ValueError(f"Invalid action {action}")    
        
        else:
            if action < 0: 
                reward = 10*action # penalty for trying to charge when car is not available
            elif action > 0:
                reward = -10*action

        # An episode is done iff the agent has reached the target
        observation = self._get_obs()
            
        self.index += 1
        
        info = {}
        
        # Check termination
        try:
            p = self.prices[self.index] # check if there is a next price
            info['p'] = p
            terminated = False
        except:
            terminated = True

        return observation, reward, terminated, info