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





class RuleEvaluation():
    
    def __init__(self, df, env, verbose = False):
        
        self.df = df
        self.env = env
        self.verbose = verbose
        
        self.current_price = []
        self.battery_charge = []
        self.presence = []
        self.balance = []
        self.actions = []
                
            
    def rule_agent(self, obs, low_quantile, high_quantile):
        
        
        if obs['prices'][-1] < np.quantile(obs['prices'][-self.env.price_horizon:], low_quantile):
            action = 0
        elif obs['prices'][-1] > np.quantile(obs['prices'][-self.env.price_horizon:], low_quantile) and obs['prices'][-1] < np.quantile(obs['prices'][-self.env.price_horizon:], high_quantile):
            action = 6
        elif obs['prices'][-1] >  np.quantile(obs['prices'][-self.env.price_horizon:], high_quantile):
            action = 12
        else:
            action = np.random.randint(0,12)
            
        return action


    def evaluate(self, low_quantile = 0.25, high_quantile = 0.75):
        """
        Iterate through data and take actions based on price quantiles as a function of the price horizon
        """

        self.data = self.df[self.env.price_horizon:] # Select data based on start index (price horizon)
        
        obs, info = self.env.reset() # Reset environment and get initial observation

        # Iterate through data 
        for i, price in enumerate(self.data['price']):
            
            self.battery_charge.append(obs['battery'])
            self.current_price.append(obs['non_normalized_price'])
            
            action = self.rule_agent(obs, low_quantile, high_quantile) # Take action based on rule agent
            
            obs,r,t,info = self.env.step(action)
            self.actions.append(action)
            self.presence.append(obs['presence'])
            self.balance.append(info['balance'])
            
            if t:
                self.data = self.data[:i+1] # Cut data to match length of episode for plotting
                assert len(self.data) == len(self.balance), f"Data and balance should have same length, {len(self.data)} != {len(self.balance)}"
                print("Absolute Balance: ", np.sum(self.balance))
                break
        
        
        
class DDQNEvaluation():

    def __init__(self, df, env, verbose = False):
            
            self.df = df
            self.env = env
            self.verbose = verbose
            
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

            
        self.data = self.df[self.env.price_horizon:] # Select data based on start and stop index
        
        obs, _ = self.env.reset() # Reset environment and get initial observation

        # Iterate through data
        for i, price in enumerate(self.data['price']):
            
            self.current_price.append(obs['non_normalized_price'])
            self.battery_charge.append(obs['battery'])

            action = agent.choose_action(i, obs['tensor'], greedy = True)

            obs,r,t,info = self.env.step(action)
            self.actions.append(action)
            self.presence.append(obs['presence'])
            self.balance.append(info['balance'])
                
            if t:
                self.data = self.data[:i+1] # Cut data to match length of episode for plotting
                assert len(self.data) == len(self.balance), f"Data and balance should have same length, {len(self.data)} != {len(self.balance)}"
                print("Absolute Balance: ", np.sum(self.balance))
                break




class Plotter():
    
    def __init__(self, evaluator, range = None):
        
        self.evaluator = evaluator
        
        self.dates = list(self.evaluator.data['datetime'])[range[0]:range[1]]
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




