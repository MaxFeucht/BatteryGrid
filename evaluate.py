import numpy as np
import pandas as pd
import gymnasium as gym
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


import matplotlib
import matplotlib.pyplot as plt

import gym_env





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
        self.model: LinearRegression
                            
    def simple_mean_reversal_strategy(self, obs, short_term, long_term):
        print(f'prices: {obs["prices"][-short_term:]}')
        short_sma = np.average(obs['prices'][max(0, len(obs['prices']) - short_term):])
        long_sma = np.average(obs['prices'][max(0, len(obs['prices']) - long_term):])
        #short_sma = np.average(obs['prices'][-short_term:])
        print(f'short: {short_sma}')
        #long_sma = np.average(obs['prices'][-long_term:])
        print(f'long: {long_sma}')
        if short_sma > long_sma:
            action = 12
        else:
            action = 0
        return action
    
    def train_model(self, train, val):
        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # Winter
            elif month in [3, 4, 5]:
                return 1  # Spring
            elif month in [6, 7, 8]:
                return 2  # Summer
            elif month in [9, 10, 11]:
                return 3  # Fall
        # Get the data
       #weekday = train['datetime'].dt.weekday

        season = train['datetime'].dt.month.apply(get_season)

        y = train['price']
        #Normalize between 0 and 1
        hours_normalized = (train['hour'] - 12)/24

        #Can fill with zeroes or drop the first 7 rows from target and hours normalized
        day_one = train['price'].shift(1).fillna(0)
        day_two = train['price'].shift(2).fillna(0)
        day_three = train['price'].shift(3).fillna(0)
        day_four = train['price'].shift(4).fillna(0)
        day_five = train['price'].shift(5).fillna(0)
        day_six = train['price'].shift(6).fillna(0)
        day_seven = train['price'].shift(7).fillna(0)

        x = np.array([
            hours_normalized,
            day_one,
            day_two,
            day_three,
            day_four,
            day_five,
            day_six,
            day_seven]
        ).reshape(-1, 8)
    
        #x = np.array([train['hour'], weekday, season]).reshape(-1, 3)
        #Get validation data 
        #season_val = val['datetime'].dt.month.apply(get_season)
        #weekday_val = val['datetime'].dt.weekday
        day_one_val = val['price'].shift(1).fillna(0)
        day_two_val = val['price'].shift(2).fillna(0)
        day_three_val = val['price'].shift(3).fillna(0)
        day_four_val = val['price'].shift(4).fillna(0)
        day_five_val = val['price'].shift(5).fillna(0)
        day_six_val = val['price'].shift(6).fillna(0)
        day_seven_val = val['price'].shift(7).fillna(0)
       # x_val =  np.array([val['hour'], weekday_val, season_val]).reshape(-1, 3)
        hours_val_normalized = (val['hour'] - 12)/24
        x_val = np.array([
            hours_val_normalized, 
            day_one_val,
            day_two_val,
            day_three_val,
            day_four_val,
            day_five_val,
            day_six_val,
            day_seven_val
                          ]).reshape(-1, 8)
        y_val = val['price']

        model = LinearRegression()
        model.fit(x, y)
        print("Coefficients: ", model.coef_)
        print("Intercept: ", model.intercept_)

        y_pred = model.predict(x_val)

        mse = mean_squared_error(y_val, y_pred)
        print("Mean squared error: ", mse)
        r2 = r2_score(y_val, y_pred)
        print("R2 score: ", r2)
        self.model = model
        return model

        
    
    def regression_strategy(self, obs):
        model = self.model
        price_pred = model.predict(np.reshape(obs['hour'], (-1,1)))
        # If the current price is smaller than predicted price we buy
        if obs['prices'][-1] < price_pred:
            #future_grad = price_pred - obs['prices'][-1]
            #grad = np.gradient(obs['prices'])
            #avg_grad_last_5_hours = np.mean(grad[-5:])      
            action = 12
        # If the current price is higher than predicted price we sell
        elif obs['prices'][-1] > price_pred:
            action = 0
        else:
            action = 6
        return action
    
    def percentile_strategy(self, obs):
        price = obs['prices'][-1]
        price_history = obs['prices'][-self.env.price_horizon:]
        percentiles = [np.quantile(price_history, q) for q in np.arange(0.1, 1.1, 0.1)]

        if price < percentiles[0]:
            action = 0
        elif percentiles[0] <= price < percentiles[1]:
            action = 2
        elif percentiles[1] <= price < percentiles[2]:
            action = 4
        elif percentiles[3] <= price < percentiles[6]:
            action = 6
        elif percentiles[6] <= price < percentiles[7]:
            action = 8
        elif percentiles[7] <= price < percentiles[8]:
            action = 10
        elif price >= percentiles[9]:
            action = 12
        else:
            action = np.random.randint(0, 12)

        return action
 


    def quantile_strategy(self, obs, low_quantile, high_quantile):

        if obs['prices'][-1] < np.quantile(obs['prices'][-self.env.price_horizon:], low_quantile):
            action = 0
        elif obs['prices'][-1] > np.quantile(obs['prices'][-self.env.price_horizon:], low_quantile) and obs['prices'][-1] < np.quantile(obs['prices'][-self.env.price_horizon:], high_quantile):
            action = 6
        elif obs['prices'][-1] >  np.quantile(obs['prices'][-self.env.price_horizon:], high_quantile):
            action = 12
        else:
            action = np.random.randint(0,12)


        return action
            
    def rule_agent(self, obs, func, **kwargs):
        short_term = 30
        long_term = 90
        
        action = func(obs=obs, **kwargs)
        #action = self.quantile_strategy(obs, low_quantile, high_quantile)
        #action = self.regression_strategy(obs)
        #action = self.percentile_strategy(obs)
    

        return action


    def evaluate(self, strategy, **kwargs):
        """
        Iterate through data and take actions based on price quantiles as a function of the price horizon
        """

        self.data = self.df[self.env.price_horizon:] # Select data based on start index (price horizon)
        
        obs, info = self.env.reset() # Reset environment and get initial observation
    
        # Iterate through data 
        for i, price in enumerate(self.data['price']):
            
            self.battery_charge.append(obs['battery'])
            self.current_price.append(obs['non_normalized_price'])
            
            action = self.rule_agent(obs, strategy, **kwargs) # Take action based on rule agent and strategy
            
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




