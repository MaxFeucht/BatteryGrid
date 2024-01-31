import numpy as np
import pandas as pd
import random
from collections import deque
import datetime
from tqdm import tqdm 
import torch

import matplotlib
import matplotlib.pyplot as plt


class Training():
    
    def __init__(self, env, agent, rep, batch_size, price_horizon):
        
        super().__init__()
        
        self.env = env
        self.agent = agent
        self.rep = rep
        self.batch_size = batch_size
        self.price_horizon = price_horizon
    
    
    
    def train(self, range = None):
        
        # Advance environment to the first step of the finetuning range if given
        if range is not None:
            
            assert range[0] < range[1], "Range must be a tuple with the first element smaller than the second"
            
            for i in range(range[0]):
                obs, _, _, _, _ = self.agent.env.step(1) # To advance the environment to the first step of the finetuning range
                _,_ = self.agent.obs_to_state(obs) # To have the price history ready for the first step of the finetuning
            
            assert self.agent.env.counter == range[0] + self.price_horizon + self.agent.replay_memory.min_replay_size, "Environment not advanced to the first step of the finetuning range"

        
        episode_balance = 0
        episode_loss = 0
        episode_counter = 0
        episode_reward = 0

        obs, r, terminated, _, _ = self.env.step(random.randint(-1,1)) # Reset environment and get initial observation
        state, grads = self.agent.obs_to_state(obs)

        for i in tqdm(range(self.rep)):

            action, q = self.agent.choose_action(i, state, greedy = False) # Choose action (discrete)
            cont_action = self.agent.action_to_cont(action) # Convert to continuous action
            
            new_obs, r, t, _, _ = self.env.step(cont_action)
            new_state, new_grads = self.agent.obs_to_state(new_obs)
            
            # Reward Shaping            
            new_reward = self.agent.shape_reward(r, cont_action, grads)

            # Fill replay buffer - THIS IS THE ONLY THING WE DO WITH THE CURRENT OBSERVATION - LEARNING IS FULLY PERFORMED FROM THE REPLAY BUFFER
            if state.shape[0] == self.agent.state_dim and new_state.shape[0] == self.agent.state_dim:
                self.agent.replay_memory.add_data((state, action, new_reward, t, new_state))

            #Update DQN
            loss = self.agent.optimize(self.batch_size)
            
            # Update values
            episode_balance += r
            episode_reward += r
            episode_loss += loss

            # New observation
            state = new_state
            grads = new_grads # Gradients for reward shaping
            
            # Reset environment if end of tuning range is reached
            if range is not None:
                if self.agent.env.counter == range[1]:
                    t = True
            
            if t:
                # Reset Environment
                self.env.counter = 0
                self.env.hour = 1
                self.env.day = 1
                episode_counter += 1
                print('Episode ', episode_counter, 'Balance: ', episode_balance, 'Reward: ', episode_reward, 'Loss: ', episode_loss) # Add both balance and reward to see how training objective and actually spent money differ
                episode_loss = 0
                episode_balance = 0
                episode_reward = 0
                
                
                if episode_counter % 4 == 0:
                    # Evaluate DQN
                    train_dqn = DDQNEvaluation(price_horizon = self.price_horizon)
                    train_dqn.evaluate(agent = self.agent)
                    
                    # Reset Environment
                    self.env.counter = 0
                    self.env.hour = 1
                    self.env.day = 1


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
        self.shape_balance = []
                
            
            
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
            self.shape_balance.append(reward)
            
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
        self.shaped_balance = []
        self.q_values = []
        


    def evaluate(self, agent = None):
        """Function to iterate through data and take actions based on agent policy

        Args:
            iterations (int, optional): _description_. Defaults to 1000.
            agent (_type_, optional): _description_. Defaults to None.
        """
        
        assert agent is not None, "Agent must be defined"

        obs,reward, terminated, _, _ = agent.env.step(random.randint(-1,1)) # Reset environment and get initial observation from random action
        
        # Set to evaluation mode:
        agent.dqn_predict.eval()
        
        with torch.no_grad():
            
            # Iterate through data 
            while True:
                
                self.battery_charge.append(obs[0])
                self.current_price.append(obs[1])
                self.price_history.append(obs[1])
                self.presence.append(obs[7])
                date = datetime.datetime(int(obs[6]), 1, 1) + datetime.timedelta(days=int(obs[4]), hours=int(obs[2])) # Needed to get the correct date from day of year and hour of day
                self.dates.append(date) 
                
                # State from observation
                state, grads = agent.obs_to_state(obs)
                action, q = agent.choose_action(0, state, greedy = True) # 0 is the step number for epsilon decay, not used here
                
                cont_action = agent.action_to_cont(action)
                obs, reward, terminated, _, _ = agent.env.step(cont_action)
                shaped_reward = agent.shape_reward(reward, cont_action, grads)
                
                self.actions.append(action)
                self.balance.append(reward)
                self.shaped_balance.append(shaped_reward)
                self.q_values.append(q)
                
                if terminated:
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
        self.shaped_balance = self.evaluator.shaped_balance[range[0]:range[1]]
        self.q_values = self.evaluator.q_values[range[0]:range[1]]
        
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
        

    def plot_actions(self, balance = False, battery = False, absence = False, shaped = False, q = False):
        
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
        
        if shaped:
            plt.plot(self.dates, self.normalize(self.shaped_balance), color = 'purple', alpha = 0.8)
            var_names.append('Shaped Balance')
            title += ' + Shaped Balance'
        
        if battery:
            plt.plot(self.dates, self.normalize(self.battery_charge), color = 'green', alpha = 0.8)
            var_names.append('Battery')
            title += ' + Battery Charge' 
        
        if q:
            plt.plot(self.dates, self.normalize(self.q_values), color = 'black', alpha = 0.8)
            var_names.append('Q-Values')
            title += ' + Q-Values'
        
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




