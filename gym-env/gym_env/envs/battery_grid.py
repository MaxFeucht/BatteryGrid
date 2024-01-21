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


    def get_season(self, date):
        month = date.month
        if month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        elif month in [9, 10, 11]:
            return 3
        else:
            return 4


    def _get_obs(self, normalize=True):
        """Function to get the observation of the environment

        Args:
            normalize (bool, optional): Normalizes data between 0 and 1. Defaults to True.

        Returns:
            obs_dict: dictionary of observations
        """

        price_history = self.prices[(self.index - self.price_horizon + self.future_horizon):(self.index + self.future_horizon)]
        battery_charge = self.battery_charge
        hour = self.datetime[self.index].hour
        day = self.datetime[self.index].day
        weekday = self.datetime[self.index].weekday()
        season = self.get_season(self.datetime[self.index])

        non_normalized_price = price_history[-1]

        # Normalize data
        if normalize:
            price_history = self.normalize(price_history)
            battery_charge /= 50
            hour /= 24
            day /= 31
            weekday /= 7
            season /= 4

        obs_dict = {
            "battery": battery_charge,
            "prices": price_history,
            "hour": hour,
            "day": day,
            "weekday": weekday,
            "season": season,
            "presence": self.presence,
            "tensor": np.concatenate((price_history, np.array([battery_charge, hour, day, weekday, season, self.presence]))),
            "non_normalized_price": non_normalized_price
        }

        return obs_dict

            

    
    def reset(self, seed=None, options = None):
        """
        Resets the environment to its initial state.
        
        Parameters:
            seed (int): The random seed for reproducibility.
            options (dict): Additional options for resetting the environment.
        
        Returns:
            obs_dict (dict): The initial observation of the environment.
            info (dict): Additional information about the environment state.
        """
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.battery_charge = 0
        self.index = self.price_horizon
            
        self.presence = 1
        
        obs_dict = self._get_obs()

        info = {"price_history" : self.prices[:self.index]}
        
        return obs_dict, info
    
    
    
    def perform_action(self, action, current_price):
        """
        Perform the specified action on the battery grid environment.

        Parameters:
            action (int): The action to be performed.
            current_price (float): The current price of electricity.

        Returns:
            reward (float): The reward obtained from performing the action.
            balance (float): The balance obtained from performing the action.
        """
        
        kWh = (abs(self.no_action - action) * self.kWh_step) # kWh that is charged / discharged
        kWh -= self.rest if action == 0 or action == (self.action_space.n - 1) else 0 # Subtract rest if we are at the maximum or minimum action

        if action < self.no_action:  # Charging
            charge = min((50 - self.battery_charge) / 0.9, kWh)
            self.battery_charge = np.clip(self.battery_charge + charge*0.9, 0, 50)
            balance = -current_price * charge * 2 
            action_type = "Charging"
        elif action > self.no_action and action <= (self.action_space.n - 1):  # Discharging
            discharge = min(self.battery_charge , kWh)
            self.battery_charge = np.clip(self.battery_charge - discharge, 0, 50)
            balance = current_price * discharge * 0.9
            action_type = "Discharging"
        elif action == self.no_action:  # No action
            balance = 0
            action_type = "No action"
        else:
            raise ValueError(f"Invalid action {action}") 

        reward = self.reward_shaping(action, balance, current_price)

        if self.verbose:
            print(f"Action {action}, {action_type} {kWh if action_type != 'No action' else ''} kWh, balance: {balance}\n")
            
        return reward, balance


    def reward_shaping(self, action, balance, current_price):
        """
        Calculates the reward for a given action in the battery grid environment.

        Parameters:
            action (int): The action taken in the environment.
            balance (float): The current balance in the environment after taken action.
            current_price (float): The current price in the environment.
            extra_penalty (bool): Flag indicating whether to apply an extra penalty.

        Returns:
            float: The calculated reward.
        """
        
        reward = balance
        if self.extra_penalty and reward == 0 and action != self.no_action:
            reward = -current_price 
        return reward
    
    
    def determine_availibility(self, current_hour):
        """
        Determines the availability of the car at the given hour.

        Args:
            current_hour (int): The current hour of the day.

        Returns:
            None
        """
        
        if current_hour == 8: # Car leaves at 8
            draw = np.random.uniform(0,1)
            self.presence = 1 if draw < 0.5 else 0
            
        elif current_hour == 18: # Car arrives at 18
            if self.presence == 0:
                self.battery_charge -= 20 # Discharge of 20 kWh if car was gone 
            self.presence = 1
    

    def step(self, action):
        """
        Executes a single step in the environment.

        Args:
            action (float): The action to take in the environment.

        Returns:
            observation (numpy.ndarray): The current observation of the environment.
            reward (float): The reward obtained from the action.
            terminated (bool): Whether the episode is terminated or not.
            info (dict): Additional information about the step.
        """
    
        # Obtain current price and hour
        current_price = self.prices[self.index]
        current_hour = self.datetime[self.index].hour

        if self.verbose:
            print(f"Current price: {current_price}, current hour: {current_hour}, current battery charge: {self.battery_charge}, current presence: {self.presence}, current index: {self.index}\n")

        # Check charge of battery for next day
        if current_hour == 7 and self.battery_charge < 20:
            action = self.no_action - np.ceil(((20 - self.battery_charge) / (0.9 * self.kWh_step)))

            if self.verbose:
                print(f"FORCED RECHARGE: Charge: {self.battery_charge}, need to charge {20 - self.battery_charge}, Charging {(self.no_action - action) * self.kWh_step} kWh with action {action}\n")

        # Determine availability of car
        self.determine_availibility(current_hour)

        # If car present, take action
        if self.presence == 1:
            reward, balance = self.perform_action(action, current_price)
        else:
            balance = 0
            reward = self.reward_shaping(action, balance, current_price)

        # Obtain observation
        observation = self._get_obs(normalize=True)
        info = {
            "price_history": self.prices[:self.index],
            "balance": balance
        }

        # Increase index
        self.index += 1

        # Check termination
        terminated = False
        if self.index + self.future_horizon >= len(self.prices):
            terminated = True

        return observation, reward, terminated, info
        
        
        
    
    
