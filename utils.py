import numpy as np
import pandas as pd
import gymnasium as gym
import random
from collections import deque
import datetime

import matplotlib
import matplotlib.pyplot as plt




class RuleEvaluation():
    
    def __init__(self, env, price_horizon, verbose = False):
        
        self.env = env
        self.price_horizon = price_horizon
        self.verbose = verbose
        
        # Set up price_history queue
        self.price_history = deque(maxlen=price_horizon)
        
        self.dates = []
        self.current_price = []
        self.battery_charge = []
        self.presence = []
        self.balance = []
        self.actions = []
                
            
            
    def rule_agent(self, price, low_quantile, high_quantile, null_action = False):
        
        if null_action:
            return 0
            
        if len(self.price_history) == self.price_horizon:
            if price <= np.quantile(self.price_history, low_quantile):
                action = 1
            elif price > np.quantile(self.price_history, low_quantile) and price < np.quantile(self.price_history, high_quantile):
                action = 0
            elif price >= np.quantile(self.price_history, high_quantile):
                action = -1
        else:
            action = 0
            
        return action



    def evaluate(self, low_quantile = 0.25, high_quantile = 0.75, null_action = False):
        """
        Iterate through data and take actions based on price quantiles as a function of the price horizon
        """
        
        obs, reward, terminated, truncated, _ = self.env.step(random.uniform(-1,1)) # Reset environment and get initial observation
        
        # Iterate through data 
        while True:
            
            self.battery_charge.append(obs[0])
            self.current_price.append(obs[1])
            self.price_history.append(obs[1])
            self.presence.append(obs[7])
            date = datetime.datetime(int(obs[6]), 1, 1) + datetime.timedelta(days=int(obs[4]), hours=int(obs[2])) # Needed to get the correct date from day of year and hour of day
            self.dates.append(date) 

            action = self.rule_agent(obs[1], low_quantile, high_quantile, null_action= null_action) # Take action based on rule agent
            
            obs,reward, terminated, truncated, _ = self.env.step(action)
            self.actions.append(action)
            self.balance.append(reward)
            
            if terminated or truncated:
                print("Absolute Balance: ", np.sum(self.balance))
                break
        
        
        
class DDQNEvaluation():

    def __init__(self, price_horizon, verbose = False):
        
        self.verbose = verbose
            
        # Set up price_history queue
        self.price_history = deque(maxlen=price_horizon)
        
        self.dates = []
        self.current_price = []
        self.battery_charge = []
        self.presence = []
        self.balance = []
        self.actions = []
        


    def evaluate(self, agent = None):
        """Function to iterate through data and take actions based on agent policy

        Args:
            iterations (int, optional): _description_. Defaults to 1000.
            agent (_type_, optional): _description_. Defaults to None.
        """
        
        assert agent is not None, "Agent must be defined"

        obs,reward, terminated, _, _ = agent.env.step(random.randint(-1,1)) # Reset environment and get initial observation from random action
                
        # Iterate through data 
        while True:
            
            self.battery_charge.append(obs[0])
            self.current_price.append(obs[1])
            self.price_history.append(obs[1])
            self.presence.append(obs[7])
            date = datetime.datetime(int(obs[6]), 1, 1) + datetime.timedelta(days=int(obs[4]), hours=int(obs[2])) # Needed to get the correct date from day of year and hour of day
            self.dates.append(date) 
            
            # State from observation
            state = agent.obs_to_state(obs)
            
            action = agent.choose_action(0, state, greedy = True) # 0 is the step number for epsilon decay, not used here
            cont_action = agent.action_to_cont(action)
            obs, reward, terminated, truncated, _ = agent.env.step(cont_action)
            self.actions.append(action)
            self.balance.append(reward)
                
            if terminated or truncated:
                print("Absolute Balance: ", np.sum(self.balance))
                break




class Plotter():
    
    def __init__(self, evaluator, range = None):
        
        self.evaluator = evaluator
        
        self.dates = list(self.evaluator.dates)[range[0]:range[1]]
        self.balance = self.evaluator.balance[range[0]:range[1]]
        self.battery_charge = self.evaluator.battery_charge[range[0]:range[1]]
        self.current_price = self.evaluator.current_price[range[0]:range[1]]
        self.actions = self.evaluator.actions[range[0]:range[1]]
        self.presence = self.evaluator.presence[range[0]:range[1]]
        
        self.agent_name = '\n' + self.evaluator.__class__.__name__.replace('Evaluation','') + '-Agent: '


    def normalize(self, data):
        """
        Helper function to normalize data
        """
        return ((data - np.min(data)) / (np.max(data) - np.min(data)))
    
    
    
    def plot_all(self, cum = False, normalize = True):
        
        """ 
        Plots battery charge, reward, price, and cumulative reward (if cum = True) in one combined plot
        """
        
        plt.figure(figsize=(15,5))
        
        vars = [self.battery_charge, self.balance, self.current_price]
        var_names = ['Battery Charge', 'Reward', 'Price']
        
        if cum: 
            self.cumulative_balance = np.cumsum(self.balance)
            vars.append(self.cumulative_balance)
            var_names.append('Cumulative Reward')
      

        for var in vars:
            
            # Normalize data only for the time window of interest (not the entire dataset)
            if normalize:
                var = self.normalize(np.array(var))
                
            plt.plot(self.dates, var)
            
        plt.xticks(rotation=45)
        plt.legend(var_names, loc = 'lower right')
        plt.title(self.agent_name + 'Price, Reward, and Battery Charge\n', size = 14)
        plt.show()
        
        
    
    def plot_single(self, normalize = True):
            
        """ 
        Plots battery charge, balance, price, and cumulative balance in single subplots
        """

        plt.figure(figsize=(15,10))
        self.cumulative_balance = np.cumsum(self.balance)
        vars = [self.battery_charge, self.balance, self.current_price, self.cumulative_balance]
        var_names = ['Battery Charge', 'Reward', 'Price', 'Cumulative Reward']
        cols = ['blue', 'red', 'green', 'orange']
        
        for i, var in enumerate(vars):
            
            # Normalize data only for the time window of interest (not the entire dataset)
            if normalize:
                if var_names[i] != 'Cumulative Reward':
                    var = self.normalize(np.array(var))
                
            plt.subplot(2,2,i+1)
            plt.plot(self.dates, var, color = cols[i])
            plt.title(self.agent_name + var_names[i])
            plt.xticks(rotation=45)
                
        plt.show()
        

    def plot_actions(self, balance = False, battery = False, absence = False):
        
        """ 
        Plots actions taken by agent
        """

        plt.figure(figsize=(15,5))        

        var_names = []

        scatter = plt.scatter(self.dates, self.normalize(self.current_price), c = self.actions, cmap = 'coolwarm')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Action',rotation=270)
        plt.plot(self.dates, self.normalize(self.current_price),  linestyle = '--', color = 'grey')
        var_names.append('Actions')
        var_names.append('Price')
        
        title = self.agent_name + 'Price & Actions'
        
        if balance:
            plt.plot(self.dates, self.normalize(self.balance), color = 'orange', alpha = 0.8)
            var_names.append('Balance')
            title += ' + Balance'
        
        if battery:
            plt.plot(self.dates, self.normalize(self.battery_charge), color = 'green', alpha = 0.8)
            var_names.append('Battery')
            title += ' + Battery Charge' 
        
        if absence:
            for i in range(len(self.presence)):
                if self.presence[i] == 0:
                    plt.hlines(y=0.01, xmin=self.dates[i-1], xmax=self.dates[i], color='red', linewidth = 2.5)
            
            var_names.append('Absence')
            title += ' + Absence'
        
        plt.legend(var_names, loc = 'lower right')
        plt.xticks(rotation=45)
        plt.title(title + '\n', size = 14)
        plt.show()




