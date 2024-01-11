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
                "time": spaces.Discrete(24)
            }
        )

        # We have 3 actions, corresponding to selling, buying and doing nothing
        self.action_space = spaces.Discrete(3)
        # If continous action space: action is simply a number between -1 and 1 where -1 indicates selling and 1 indicates buying at 25 kWh - everything in between is selling / buying at a fraction of 25 kWh
        # self.action_space = spaces.Box(-1, 1, dtype=float)

    def set_data(self, df):
        self.prices = np.array(df['price'])  # The size of the square grid
        self.datetime = df['datetime']  # The size of the PyGame window
        

    def _get_obs(self):
        return {"battery": self.battery_charge, "prices": self.prices[:self.index+1], "time": self.datetime[self.index].hour, "presence": self.presence}
    
    
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
        current_day = self.datetime[self.index].day
        
                
        # Check charge of battery for next day
        if current_hour == 7:
            if self.battery_charge < 20:
                action = 0
                # TODO: add penalty for trying to do anything else than charge when it is 7 and battery is not ready
            
        
        # Check availability of car
        if current_hour == 8:
            draw = np.random.uniform(0,1)
            self.presence = 1 if draw < 0.5 else 0
        elif current_hour == 18:
            self.presence = 1

        if self.presence == 1:
            
            if action == 0:  # No if statement, because we can always charge
                self.battery_charge = np.clip(self.battery_charge + 25*0.9, 0, 50)
                reward = -current_price * 25 * 2 
                    
            elif action == 1:
                if self.battery_charge >= 25:
                    self.battery_charge = self.battery_charge - 25
                    reward = current_price * 25 * 0.9
                else:
                    reward = -5 # penalty for trying to sell when battery is empty

            elif action == 2:
                reward = 0
                
            else:
                raise ValueError(f"Invalid action {action}")    
        
        else:
            if action == 0 or action == 1:
                reward = -5 # penalty for trying to charge when car is not available
            else:
                reward = 0

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