from TestEnv import Electric_Car
import argparse
import matplotlib.pyplot as plt
import numpy as np

from feature_eng import *
from agent import DDQNAgent

# Make the excel file as a command line argument, so that you can do: " python3 main.py --excel_file validate.xlsx "
parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='validate.xlsx') # Path to the excel file with the test data
args = parser.parse_args()

env = Electric_Car(path_to_test_data=args.excel_file)

total_reward = []
cumulative_reward = []
battery_level = []

## Create feature dataframe
df = env.test_data

## ENGINEER FEATURES (Settings for feature engineering are given in features.py)
feature_df = engineer_features(df)

## Load Agent ##

seed = 2705
rep = 105000 * 10
batch_size = 48
gamma = 0.965
epsilon = 1.0
epsilon_decay = 199999
epsilon_min = 0.1
learning_rate = 1e-3
price_horizon = 48
future_horizon = 0
hidden_dim = 96
num_layers = 4
positions = False
action_classes = 3
reward_shaping = True
factor = 0.75
verbose = False
normalize = True

agent = DDQNAgent(env = env,
                features = feature_df,
                epsilon_decay = epsilon_decay,
                epsilon_start = epsilon,
                epsilon_end = epsilon_min,
                discount_rate = gamma,
                lr = learning_rate,
                buffer_size = 100000,
                price_horizon = price_horizon,
                hidden_dim=hidden_dim,
                num_layers = num_layers,
                positions = positions,
                action_classes = action_classes, 
                reward_shaping = reward_shaping,
                shaping_factor = factor,
                normalize = normalize,
                verbose = verbose)
        
agent.dqn_predict.load_state_dict(torch.load(f'models/final_agent_{num_layers}_gamma_{gamma}.pt'))
agent.dqn_predict.eval()

observation = env.observation()
for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
    
    if i == 0: # First action is 0
        action = 0
    else: # Then: choose action based on our agent
        action, q = agent.choose_action(i, state, greedy = True) # Choose action (discrete)
    
    # Convert to action from -1 to 1
    cont_action = agent.action_to_cont(action) # Convert to continuous action
        
    # Get the next observation, reward, terminated, truncated and info
    next_observation, reward, terminated, truncated, info = env.step(cont_action)
    
    # Transform the observation to a state. This entails adding the price history and the  
    new_state, _ = agent.obs_to_state(next_observation)

    # Update observation and state
    observation = next_observation
    state = new_state        
    
    # Append data
    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))
    done = terminated or truncated

    if done:
        print('Total reward: ', sum(total_reward))
        # Plot the cumulative reward over time
        plt.plot(cumulative_reward)
        plt.xlabel('Time (Hours)')
        plt.ylabel('Cumulative reward (€)')
        plt.show()




