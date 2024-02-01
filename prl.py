#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import pandas as pd
from tqdm import tqdm
import random

from utils import RuleEvaluation, DDQNEvaluation, Plotter
from agent import DDQNAgent, TemporalDDQNAgent

from TestEnv import Electric_Car

seed = 2705
TRAIN = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### LOAD DATA ###

def elongate(df):
    df_long = pd.wide_to_long(df, i = "PRICES", j = "hour", stubnames=["Hour"], sep = " ").reset_index()
    df_long.rename(columns={"Hour": "price", "PRICES": "date"}, inplace = True)
    df_long['datetime'] = pd.to_datetime(df_long['date']) + pd.to_timedelta(df_long['hour'], unit='h')
    df_long.sort_values(['datetime'], ascending=[True], inplace=True)
    df_long['price'] = df_long['price'].astype(float) 
    return df_long.reset_index(drop=True)

train_name = 'data/train.xlsx'
val_name = 'data/validate.xlsx'

train = elongate(pd.read_excel(train_name))
val = elongate(pd.read_excel(val_name))

features_train = pd.read_csv('data/features_train.csv')
features_val = pd.read_csv('data/features_val.csv')

# del year, month_non_business_peak, month_non_business_valley
# del month_business_peak, month_business_valley, month_non_business_peak
# features_train = features_train.drop(columns=['year', 'month_non_business_peak', 'month_non_business_valley'])
# features_val = features_val.drop(columns=['year', 'month_non_business_peak', 'month_non_business_valley'])
#%%

# Define the intervals for gamma and reward shaping factor
gamma_interval = [0.954, 0.960]
reward_shaping_interval = [0.77, 0.78]
#battery_factor_interval = [0.0, 0.15]
gamma = 0.954
reward_shaping_factor = 0.772
# Define the number of iterations for the random search
num_iterations = 2

for i in range(num_iterations):
    
    # # Generate random values within the intervals
    # gamma = round(np.random.uniform(*gamma_interval), 4)
    # reward_shaping_factor = round(np.random.uniform(*reward_shaping_interval), 4)
    battery_factor = None #round(np.random.uniform(*battery_factor_interval), 4)
    
    # print(f'\nIteration: {i+1}, Gamma: {gamma}, Reward Shaping Factor: {reward_shaping_factor}')
        
    seed = 2705
    rep = 1_000 * 3
    batch_size = 48
    gamma = gamma
    epsilon = 1.0
    epsilon_decay = 99999
    epsilon_min = 0.1
    learning_rate = 5e-5
    price_horizon = 48
    future_horizon = 0
    hidden_dim = 128
    num_layers = 4
    positions = False
    action_classes = 3
    reward_shaping = True
    factor = reward_shaping_factor
    verbose = False
    normalize = True
    df = train_name


    # Initialize Environment
    env = Electric_Car(path_to_test_data=df)
    val_env = Electric_Car(path_to_test_data=val_name)

    #Initialize DQN
    agent = DDQNAgent(env = env,
                    features = features_train,
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

    val_agent = DDQNAgent(env = val_env,
                    features = features_val,
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


    episode_balance = 0
    episode_loss = 0
    episode_counter = 0
    episode_reward = 0


    obs, r, t, _, _ = env.step(random.randint(-1,1)) # Reset environment and get initial observation
    state, grads = agent.obs_to_state(obs)

    for i in tqdm(range(rep)):

        action, q = agent.choose_action(i, state, greedy = False) # Choose action (discrete)
        cont_action = agent.action_to_cont(action) # Convert to continuous action
        
        new_obs, r, t, _, _ = env.step(cont_action)
        new_state, new_grads = agent.obs_to_state(new_obs)
        
        # Reward Shaping            
        new_reward = agent.shape_reward(r, cont_action, grads, battery_factor = battery_factor)

        # Fill replay buffer - THIS IS THE ONLY THING WE DO WITH THE CURRENT OBSERVATION - LEARNING IS FULLY PERFORMED FROM THE REPLAY BUFFER
        if state.shape[0] == agent.state_dim and new_state.shape[0] == agent.state_dim:
            agent.replay_memory.add_data((state, action, new_reward, t, new_state))

        #Update DQN
        loss = agent.optimize(batch_size)
        
        # Update values
        episode_balance += r
        episode_reward += r
        episode_loss += loss

        # New observation
        state = new_state
        grads = new_grads # Gradients for reward shaping
        

        if t:
            # Reset Environment
            env.counter = 0
            env.hour = 1
            env.day = 1
            episode_counter += 1
            
            if episode_counter % 4 == 0:
                print('Episode ', episode_counter, 'Balance: ', episode_balance, 'Reward: ', episode_reward, 'Loss: ', episode_loss) # Add both balance and reward to see how training objective and actually spent money differ
            
            # Scheduler Step
            agent.scheduler.step(episode_loss)
            
            episode_loss = 0
            episode_balance = 0
            episode_reward = 0
            
            
            if episode_counter % 4 == 0:
                # Evaluate DQN
                print("Training Evaluation")
                train_dqn = DDQNEvaluation(price_horizon = price_horizon)
                train_dqn.evaluate(agent = agent)
                
                # Evaluate DQN
                print("Validation Evaluation")
                val_agent.dqn_predict.load_state_dict(agent.dqn_predict.state_dict())
                val_dqn = DDQNEvaluation(price_horizon = price_horizon)
                val_dqn.evaluate(agent = val_agent)
                
                # Reset Environment
                env.counter = 0
                env.hour = 1
                env.day = 1
                val_env.counter = 0
                val_env.hour = 1
                val_env.day = 1
            
                
    # Save agent
    torch.save(agent.dqn_predict.state_dict(), f'models/reverse_gamma_{gamma}factor{reward_shaping_factor}_no_features.pt')


### END TRAINING ###


#%%
### EVALUATION ###

# gamma = 0.954
# reward_shaping_factor = 0.772
seed = 2705
rep = 1_000 * 3
batch_size = 48
gamma = gamma
epsilon = 1.0
epsilon_decay = 99999
epsilon_min = 0.1
learning_rate = 5e-5
price_horizon = 48
future_horizon = 0
hidden_dim = 128
num_layers = 4
positions = False
action_classes = 3
reward_shaping = True
factor = reward_shaping_factor
verbose = False
normalize = True
df = train_name
# reward_shaping_factor = 0.7805
# gamma = 0.9592
# factor = reward_shaping_factor


## Evaluation ##
TRAIN = False

if TRAIN:
    df = train_name
    features = features_train
else:
    df = val_name
    features = features_val


# Initialize Environment
env = Electric_Car(path_to_test_data=df)

#Initialize DQN
agent = DDQNAgent(env = env,
                features = features,
                epsilon_decay = epsilon_decay,
                epsilon_start = epsilon,
                epsilon_end = epsilon_min,
                discount_rate = 0.9471,
                lr = learning_rate,
                buffer_size = 100000,
                price_horizon = price_horizon,
                hidden_dim=hidden_dim,
                num_layers = num_layers,
                positions = positions,
                action_classes = action_classes, 
                reward_shaping = reward_shaping,
                shaping_factor = 0.7348,
                normalize = normalize,
                verbose = False)


agent.dqn_predict.load_state_dict(torch.load(f'models/chargesearch_gamma_{gamma}_factor_{reward_shaping_factor}.pt'))

# Evaluate Rule-Based Agent
eval_ddqn = DDQNEvaluation(price_horizon=price_horizon)
eval_ddqn.evaluate(agent = agent)


#Visualize DDQN Agent
#plot_range = [14000, 14300]
# plot_range = [9750, 9820]
plot_range = [0, 110000]

plotter = Plotter(eval_ddqn, range = plot_range)
#plotter.plot_actions(battery = False, balance=False, absence = False)
# plotter.plot_actions(battery = False, balance=False, absence = True)
# plotter.plot_actions(battery = False, balance=True, absence = True)
plotter.plot_actions(battery = True, balance=True, absence = True, shaped = True)

plotter.plot_single()




#%%

### FEATURE IMPORTANCE ###
def permutation_importance(features, baseline_performance):
    #Initialize DQN
    print("<<< VALIDATION SET BASELINE PERFORMANCE: ", baseline_performance, " >>>")
    importances = {}

    crafted_columns = features.columns[4:]

    for column in crafted_columns:
        # for 10 times
        for i in range(10):
            shuffled_features = features.copy()
            shuffled_features[column] = np.random.permutation(shuffled_features[column])

            temp_env = Electric_Car(path_to_test_data=df)

            temp_agent = DDQNAgent(env = temp_env,
                        features = features,
                        epsilon_decay = epsilon_decay,
                        epsilon_start = epsilon,
                        epsilon_end = epsilon_min,
                        discount_rate = 0.9471,
                        lr = learning_rate,
                        buffer_size = 100000,
                        price_horizon = price_horizon,
                        hidden_dim=hidden_dim,
                        num_layers = num_layers,
                        positions = positions,
                        action_classes = action_classes, 
                        reward_shaping = reward_shaping,
                        shaping_factor = 0.7348,
                        normalize = normalize,
                        verbose = False)

            temp_agent.dqn_predict.load_state_dict(torch.load(f'models/chargesearch_gamma_{gamma}_factor_{reward_shaping_factor}.pt'))

            temp_eval_ddqn = DDQNEvaluation(price_horizon=price_horizon)
            temp_eval_ddqn.evaluate(agent = temp_agent)
            # Compute performance decrease
            shuffled_performance =  np.sum(temp_eval_ddqn.balance)
            # importance = -400 - (-300) = -100
            importance = baseline_performance - shuffled_performance
            if column in importances:
                importances[column] += importance
            else:
                importances[column] = importance
        importances[column] /= 10
    # Sort and display the importances
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    # round up
    sorted_importances = [(feature, round(importance)) for feature, importance in sorted_importances]
    for feature, importance in sorted_importances:
        # print(f"Feature: {feature:30} {importance:>8 }")
        print(f"Feature: {feature}: {importance}")

    return importances

# Compute baseline performance
baseline_performance = np.sum(eval_ddqn.balance)  # Replace with your baseline metric

# Compute feature importances
importances = permutation_importance(features_val, baseline_performance)

### END FEATURE IMPORTANCE ###


# %%
### GRAPH IMPORTANCE ###

import seaborn as sns
# make negative importances positive
def plot_importances(importances):
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=False)
    sorted_importances = [(feature, abs(importance)) for feature, importance in sorted_importances]

    # take top 10
    # sorted_importances = sorted_importances[:40]

    # Create a custom rainbow color palette
    rainbow_colors = sns.color_palette("Spectral", len(sorted_importances))

    # feature list should be on y axis
    y = [x[0] for x in sorted_importances]

    # feature importance should be on x axis
    x = [x[1] for x in sorted_importances]

    # Plot with custom colors
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    sns.barplot(x=x, y=y, palette=rainbow_colors)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

plot_importances(importances)

plotter.shaped_balance
plotter.balance

diff = np.array(plotter.shaped_balance) - np.array(plotter.balance)

# Number of difference
plt.figure(figsize =(10,5))
nonzero_diff = diff[np.nonzero(diff)]
plt.hist(nonzero_diff, bins = 100)


# %%
