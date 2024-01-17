import numpy as np
import pandas as pd
import gymnasium as gym

import matplotlib
import matplotlib.pyplot as plt

import gym_env


def elongate(df):
    df_long = pd.wide_to_long(df, i = "PRICES", j = "hour", stubnames=["Hour"], sep = " ").reset_index()
    df_long.rename(columns={"Hour": "price", "PRICES": "date"}, inplace = True)
    df_long['datetime'] = pd.to_datetime(df_long['date']) + pd.to_timedelta(df_long['hour'], unit='h')
    df_long.sort_values(['datetime'], ascending=[True], inplace=True)
    df_long['price'] = df_long['price'].astype(float) / 1000 # Convert price per MWh to price per KWh
    return df_long.reset_index(drop=True)

train = elongate(pd.read_excel('data/train.xlsx'))
val = elongate(pd.read_excel('data/validate.xlsx'))

env = gym.make('gym_env/BatteryGrid-v0')
env.setup(train, price_horizon=24)




class Evaluation():
    
    def __init__(self, df, env, low_quantile = 0.25, high_quantile = 0.75, price_horizon = 96, verbose = False, start = 0, stop = 1000):
        
        first_index = max(price_horizon, start)
        
        if price_horizon > start:
            print(f"Warning: start index should be at least {price_horizon}")
            
        self.data = df[first_index:stop]
        self.env = env
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.price_horizon = price_horizon
        self.verbose = verbose
        
        self.current_price = []
        self.battery_charge = []
        self.balance = []
        self.actions = []
                
    
    def rule_agent(self):
        """
        Iterate through data and take actions based on price quantiles as a function of the price horizon
        """
                
        obs = self.env.reset()        

        for price in self.data['price']:
            
            if obs['prices'][-1] < np.quantile(obs['prices'][-self.price_horizon:], self.low_quantile):
                action = 0
            elif obs['prices'][-1] > np.quantile(obs['prices'][-self.price_horizon:], self.low_quantile) and obs['prices'][-1] < np.quantile(obs['prices'][-self.price_horizon:], self.high_quantile):
                action = 6
            elif obs['prices'][-1] >  np.quantile(obs['prices'][-self.price_horizon:], self.high_quantile):
                action = 12
            else:
                action = np.random.randint(0,12)
            
            self.current_price.append(price)
            obs,r,t,info =self.env.step(action)
            
            self.battery_charge.append(obs['battery'])
            self.balance.append(r)
            
            if t:
                break
        
        
        
    
    def ddqn_agent(self, agent = None):
        """Function to iterate through data and take actions based on agent policy

        Args:
            iterations (int, optional): _description_. Defaults to 1000.
            agent (_type_, optional): _description_. Defaults to None.
        """
        
        assert agent is not None, "Agent must be defined"
        
        obs = self.env.reset()        

        for i, price in enumerate(self.data['price']):
            
            action = agent.choose_action(i, obs['tensor'], greedy = True)
            self.current_price.append(price)
            obs,r,t,info = self.env.step(action)
            
            self.actions.append(action)
            self.battery_charge.append(obs['battery'])
            self.balance.append(info['balance'])
                
            if t:
                self.end_index = i
                break

    
    
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
            if normalize and var != self.current_price:
                var = self.normalize(np.array(var))
            plt.plot(self.data['datetime'][:self.end_index], var)
            
        plt.xticks(rotation=45)
        plt.legend(var_names)
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
            if normalize:
                var = self.normalize(np.array(var))
            plt.subplot(2,2,i+1)
            plt.plot(self.data['datetime'], var, color = cols[i])
            plt.title(var_names[i])
            plt.xticks(rotation=45)
            
        plt.show()
        