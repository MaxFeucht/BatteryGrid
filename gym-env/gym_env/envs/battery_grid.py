import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd



class BatteryGridEnv(gym.Env):

    def __init__(self):
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "battery": spaces.Box(0, 1, dtype=float),
                "prices": spaces.Box(0, 1, dtype=float),
                "presence": spaces.Discrete(2),
                "hour": spaces.Box(0,1, dtype=float),
                "day": spaces.Box(0,1, dtype=float),
                "tensor": spaces.Box(0, np.inf, shape=(100,), dtype=float)
            }
        )

        # We have 3 actions, corresponding to selling, buying and doing nothing
        self.action_space = spaces.Discrete(11)
        # If continous action space: action is simply a number between -1 and 1 where -1 indicates selling and 1 indicates buying at 25 kWh - everything in between is selling / buying at a fraction of 25 kWh
        # self.action_space = spaces.Box(-1, 1, dtype=float)


    def setup(self, df, price_horizon = 96, future_horizon = 0, action_classes = 7, verbose = False, extra_penalty = False):
        self.prices = np.array(df['price'])  
        self.datetime = list(df['datetime'])  
        self.price_horizon = price_horizon
        self.index = price_horizon
        self.action_space = spaces.Discrete(action_classes)
        
        # Calculate steps for discretization           
        self.no_action = np.floor(self.action_space.n / 2) # Action where we do nothing
        self.kWh_step = np.ceil(27.77 / self.no_action) # KWh that is charged / discharged per step
        self.rest = np.abs(27.77 - self.no_action * self.kWh_step) # Rest KWh that is subtracted from the maximum actions to reach 27.77 kWh
            
        self.future_horizon = future_horizon
        self.extra_penalty = extra_penalty
        self.verbose = verbose
        
        print("Setup with price horizon: ", self.price_horizon, " and future horizon: ", self.future_horizon, " and action space: ", self.action_space.n)
        
        if verbose:
            print("Action space: ", self.action_space, "with no action: ", self.no_action, ", kWh step: ", self.kWh_step, " and rest: ", self.rest)

    

    def normalize(self, var):
        """
        Helper function to normalize data between 0 and 1
        """
        return (var - np.min(var)) / (np.max(var) - np.min(var))



    def _get_obs(self, normalize = True):
        """Function to get the observation of the environment

        Args:
            normalize (bool, optional): Normalizes data between 0 and 1. Defaults to True.

        Returns:
            obs_dict: dictionary of observations
        """
        
        price_history = self.prices[(self.index - self.price_horizon + self.future_horizon) : (self.index + self.future_horizon)]
        battery_charge = self.battery_charge 
        hour = self.datetime[self.index].hour
        day = self.datetime[self.index].day
        
        non_normalized_price = price_history[-1]
        
        # Normalize data
        if normalize: 
            price_history = self.normalize(price_history)
            battery_charge = (battery_charge) / 50
            hour = (hour) / 24 
            day = (day) / 365
            
        obs_dict = {"battery": battery_charge,"prices": price_history, "hour": hour, "day": day, "presence": self.presence}
        obs_dict["tensor"] = self.dict_to_tensor(obs_dict)
        obs_dict["non_normalized_price"] = non_normalized_price
        
        return obs_dict



    def dict_to_tensor(self, obs_dict):
        
        tensor = np.array([])
        for k,v in obs_dict.items():
            if k == "prices":
                tensor = np.concatenate((tensor, v))
            else:
                tensor = np.concatenate((tensor, np.array([v])))
                            
        return tensor
            

    
    def reset(self, seed=None, options = None):
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.battery_charge = 0
        self.index = self.price_horizon
            
        self.presence = 1
        
        obs_dict = self._get_obs()

        info = {"price_history" : self.prices[:self.index]}
        
        return obs_dict, info
    

    def reward_shaping(self, balance):
        
        # Relative balance
        # short_term_window = self.prices[self.index - 24 : self.index]
        # rel_balance = balance / np.mean(balance_history)
        
        # Combining balance and relative price
        # reward = balance * 0.5 + rel_balance
        
        reward = balance
        
        return reward
        
    
    def step(self, action):
        
        # Obtain current price and hour
        current_price = self.prices[self.index]
        current_hour = self.datetime[self.index].hour        
        
        if self.verbose:
            print(f"Current price: {current_price}, current hour: {current_hour}, current battery charge: {self.battery_charge}, current presence: {self.presence}, current index: {self.index}\n")
        
        # Check charge of battery for next day
        if current_hour == 7:
            if self.battery_charge < 20:
                action = self.no_action - np.ceil(((20 - self.battery_charge)/(0.9*self.kWh_step)))
                
                if self.verbose:
                    print(f"FORCED RECHARGE: Charge: {self.battery_charge}, need to charge {20 - self.battery_charge}, Charging {(self.no_action - action) * self.kWh_step} kWh with action {action}\n")
            
        
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

            if action < self.no_action:  # No if statement, because we can always charge
                kWh = (self.no_action - action) * self.kWh_step # Discretize, such that action 0 means most discharge, i.e., kWh = (5 - 0)* 5 = 25
                kWh  -= self.rest if action == 0 else 0 # max = 27.77 kWh
                charge = min((50 - self.battery_charge) / 0.9, kWh) # Discharge at most action * 25, but less if battery is has less than 25 kWh
                self.battery_charge = np.clip(self.battery_charge + charge*0.9, 0, 50)
                balance = -current_price * charge * 2 
                reward = self.reward_shaping(balance)
                
                if self.extra_penalty:
                    # Penalty for charging when battery is full, equal to the price of charging 1 kWh
                    if reward == 0:
                        reward = -current_price * np.abs(action)

                if self.verbose:
                    print(f"Action {action}, Charging {kWh} kWh, balance: {balance}\n")
                
                
            elif action > self.no_action and action <= (self.action_space.n - 1): 
                kWh = (action - self.no_action) * self.kWh_step # Discretize, such that action 10 means most discharge, i.e., kWh = (10 - 5)* 5 = 25
                kWh  -= self.rest if action == (self.action_space.n - 1) else 0 # max = 27.77 kWh
                discharge = min(self.battery_charge , kWh) # Discharge at most action * 25, but less if battery is has less than 25 kWh
                self.battery_charge = np.clip(self.battery_charge - discharge, 0, 50)
                balance = current_price * discharge * 0.9
                reward = self.reward_shaping(balance)

                if self.extra_penalty:
                    # Penalty for discharging when battery is empty equal to the price of charging 1 kWh
                    if reward == 0:
                        reward = -current_price * np.abs(action)
                    
                if self.verbose:
                    print(f"Action {action}, Discharging {kWh} kWh, balance: {balance}\n")


            elif action == self.no_action:
                balance = 0
                reward = self.reward_shaping(balance)
                if self.verbose:
                    print(f"Action {action}, balance: {balance}\n")
                
            else:
                raise ValueError(f"Invalid action {action}") 

        else:
            balance = 0
            reward = self.reward_shaping(balance)
            
            if self.extra_penalty:
                # Penalty for trying to buy or sell electricity when car is not available
                if action != self.no_action:
                    reward = -current_price * np.abs(action)


        # Obtain observation
        observation = self._get_obs(normalize = True)
        info = {"price_history" : self.prices[:self.index],
                "balance": balance}
        
        # Increase index
        self.index += 1
        
        # Check termination
        try:
            p = self.prices[self.index + self.future_horizon] # check if there is a next  (within horizon)
            terminated = False
        except:
            terminated = True

        return observation, reward, terminated, info
    
    
    
    
    
