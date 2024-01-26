import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import pandas as pd
from tqdm import tqdm
import random

from utils import DDQNEvaluation, Plotter
from agent import ConvDDQNAgent

from TestEnv import Electric_Car

seed = 2705
TRAIN = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
# Load Data

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



# # Feature Engineering
# from feature_eng import *

# gradient_sizes = [1, 2, 4, 6, 8, 12, 18, 24]
# fourier_window = 72
# window_sizes = [3, 6, 12, 24, 48, 72]

# features_train = features_pipeline(train, fourier_window, gradient_sizes, window_sizes)
# features_val = features_pipeline(val, fourier_window, gradient_sizes, window_sizes)

# features_train.fillna(0, inplace=True)
# features_val.fillna(0, inplace=True)

# features_train = features_train.replace([np.inf, -np.inf], 0)
# features_val = features_val.replace([np.inf, -np.inf], 0)

# features_train.to_csv('data/features_train.csv', index=False)
# features_val.to_csv('data/features_val.csv', index=False)


features_train = pd.read_csv('data/features_train.csv')
features_val = pd.read_csv('data/features_val.csv')


#%%

seed = 2705
rep = 1000000
batch_size = 32
gamma = 0.965
epsilon = 1.0
epsilon_decay = 99999
epsilon_min = 0.1
learning_rate = 5e-5
price_horizon = 72
lin_hidden_dim = 64
conv_hidden_dim = 16
num_layers = 2
kernel_size = 3
dropout = 0.1
action_classes = 3
reward_shaping = True
factor = 1
verbose = False
TRAIN = True
df = train_name

#%%

# Initialize Environment
env = Electric_Car(path_to_test_data=df)

#Initialize DQN
agent = ConvDDQNAgent(env = env,
                            features = features_train,
                            epsilon_decay = epsilon_decay,
                            epsilon_start = epsilon,
                            epsilon_end = epsilon_min,
                            discount_rate = gamma,
                            lr = learning_rate,
                            buffer_size = 100000,
                            price_horizon = price_horizon,
                            lin_hidden_dim=lin_hidden_dim,
                            conv_hidden_dim=conv_hidden_dim,
                            kernel_size = kernel_size,
                            dropout = dropout,
                            num_layers = 2,
                            action_classes = action_classes, 
                            verbose = verbose)

obs, r, terminated, _, _ = env.step(random.randint(-1,1)) # Reset environment and get initial observation
state, grads = agent.obs_to_state(obs)

    
def count_parameters(model):
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, unfrozen

frozen, unfrozen = count_parameters(agent.dqn_predict)
print(f'Number of frozen parameters: {frozen}')
print(f'Number of unfrozen parameters: {unfrozen}')



#%%

episode_balance = 0
episode_loss = 0
episode_counter = 0
episode_reward = 0

with tqdm(total=rep) as pbar:
    for i in range(rep):

        action = agent.choose_action(i, state, greedy = False) # Choose action (discrete)
        cont_action = agent.action_to_cont(action) # Convert to continuous action
        
        new_obs, r, t, _, _ = env.step(cont_action)
        new_state, new_grads = agent.obs_to_state(new_obs)
        
        # Reward Shaping            
        new_reward = agent.shape_reward(r, cont_action, grads, apply = reward_shaping, factor = factor)

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
        
        pbar.update(1)

        if t:
            # Reset Environment
            env.counter = 0
            env.hour = 1
            env.day = 1
            episode_counter += 1
            print('Episode ', episode_counter, 'Balance: ', episode_balance, 'Reward: ', episode_reward, 'Loss: ', episode_loss) # Add both balance and reward to see how training objective and actually spent money differ
            episode_loss = 0
            episode_balance = 0
            episode_reward = 0
            
            
            if episode_counter % 4 == 0:
                # Evaluate DQN
                train_dqn = DDQNEvaluation(price_horizon = price_horizon)
                train_dqn.evaluate(agent = agent)
                
                # Reset Environment
                env.counter = 0
                env.hour = 1
                env.day = 1
                

# Save agent
torch.save(agent.dqn_predict.state_dict(), f'models/tempagent_{action_classes}_gamma{gamma}.pt')
pbar.close



#%%

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
agent = TemporalDDQNAgent(env = env,
                            features = features_train,
                            epsilon_decay = epsilon_decay,
                            epsilon_start = epsilon,
                            epsilon_end = epsilon_min,
                            discount_rate = gamma,
                            lr = learning_rate,
                            buffer_size = 100000,
                            price_horizon = price_horizon,
                            lin_hidden_dim=lin_hidden_dim,
                            conv_hidden_dim=conv_hidden_dim,
                            target_dim = target_dim,
                            kernel_size = kernel_size,
                            dropout = dropout,
                            tcn_path = tcn_path,
                            action_classes = action_classes, 
                            verbose = verbose)

agent.dqn_predict.load_state_dict(torch.load(f'models/tempagent_{action_classes}_gamma{gamma}.pt'))

# Evaluate Rule-Based Agent
eval_ddqn = DDQNEvaluation(price_horizon=price_horizon)
eval_ddqn.evaluate(agent = agent)

#Visualize DDQN Agent
plot_range = [8000, 8200]

plotter = Plotter(eval_ddqn, range = plot_range)
plotter.plot_actions(battery = False, balance=False, absence = False)
plotter.plot_actions(battery = False, balance=False, absence = True)
plotter.plot_actions(battery = False, balance=True, absence = True)
plotter.plot_actions(battery = True, balance=True, absence = True)

plotter.plot_single()
### Current (last) problem: Agent only charges the battery half way. I need to incentivize charging the battery to 50kWh.

