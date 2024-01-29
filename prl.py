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

features_train = pd.read_csv('data/features_train.csv')
features_val = pd.read_csv('data/features_val.csv')


#%%

for factor in [0.1, 0.25, 0.5, 1]:
        
    print(f'Factor: {factor}')
        
    seed = 2705
    rep = 500000
    batch_size = 48
    gamma = 0.96
    epsilon = 1.0
    epsilon_decay = 99999
    epsilon_min = 0.1
    learning_rate = 5e-5
    price_horizon = 48
    future_horizon = 0
    hidden_dim = 112
    num_layers = 4
    positions = False
    action_classes = 3
    reward_shaping = True
    factor = factor
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

        action = agent.choose_action(i, state, greedy = False) # Choose action (discrete)
        cont_action = agent.action_to_cont(action) # Convert to continuous action
        
        new_obs, r, t, _, _ = env.step(cont_action)
        new_state, new_grads = agent.obs_to_state(new_obs)
        
        # Reward Shaping            
        new_reward = agent.shape_reward(r, cont_action, grads)

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
            episode_loss = 0
            episode_balance = 0
            episode_reward = 0
            
            
            if episode_counter % 4 == 0:
                # Evaluate DQN
                print("Training Evaluation")
                train_dqn = DDQNEvaluation(price_horizon = price_horizon)
                train_dqn.evaluate(agent = agent)
                
                # Reset Environment
                env.counter = 0
                env.hour = 1
                env.day = 1
            
                # Evaluate DQN
                print("Validation Evaluation")
                val_agent.dqn_predict.load_state_dict(agent.dqn_predict.state_dict())
                val_dqn = DDQNEvaluation(price_horizon = price_horizon)
                val_dqn.evaluate(agent = val_agent)
                
                # Reset Environment
                env.counter = 0
                env.hour = 1
                env.day = 1
            
                
    # Save agent
    torch.save(agent.dqn_predict.state_dict(), f'models/agent_layers{num_layers}_gamma{gamma}_{price_horizon}_shaped_bat_val_{factor}.pt')




#%%

### Retraining Loop ###

# Initialize Environment
env = Electric_Car(path_to_test_data=df)

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

agent.dqn_predict.load_state_dict(torch.load(f'models/agent_layers{num_layers}_gamma{gamma}___.pt'))


finetune_range = [13500 - 1048, 20000] # 1048 because of min_buffer_size and price_horizon
finetune_rep = 300000

## Now: Normal Training Loop
obs, r, terminated, _, _ = env.step(random.randint(-1,1)) # Reset environment and get initial observation
state, grads = agent.obs_to_state(obs)

# Advance environment to the first step of the finetuning range if given
if finetune_range is not None:
    
    assert finetune_range[0] < finetune_range[1], "Range must be a tuple with the first element smaller than the second"
    
    for i in range(finetune_range[0]):
        obs, _, _, _, _ = agent.env.step(1) # To advance the environment to the first step of the finetuning range
        _,_ = agent.obs_to_state(obs) # To have the price history ready for the first step of the finetuning
    
    assert agent.env.counter == finetune_range[0] + agent.price_horizon + agent.replay_memory.min_replay_size + 1, "Environment not advanced to the first step of the finetuning range"
    print("Environment advanced to the first step of the finetuning range")
    
    
for i in tqdm(range(finetune_rep)):

    action = agent.choose_action(i, state, greedy = False) # Choose action (discrete)
    cont_action = agent.action_to_cont(action) # Convert to continuous action
    
    new_obs, r, t, _, _ = env.step(cont_action)
    new_state, new_grads = agent.obs_to_state(new_obs)
    
    # Reward Shaping            
    new_reward = agent.shape_reward(r, cont_action, grads)

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
    
    # Check if we are at the end of the finetuning range
    if finetune_range is not None:
        if agent.env.counter == finetune_range[1]:
            t = True
            print("End of finetuning range reached")
                    
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
        
        if finetune_range is not None:
                    
            for i in range(finetune_range[0]):
                obs, _, _, _, _ = agent.env.step(1) # To advance the environment to the first step of the finetuning range
                _,_ = agent.obs_to_state(obs) # To have the price history ready for the first step of the finetuning
            
            assert agent.env.counter == finetune_range[0], "Environment not advanced to the first step of the finetuning range"
            print("Environment advanced to the first step of the finetuning range")
        
        
torch.save(agent.dqn_predict.state_dict(), f'models/agent_layers{num_layers}_gamma{gamma}_{price_horizon}_shaped_bat_val.pt')


#%%

## Evaluation ##
TRAIN = True

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

agent.dqn_predict.load_state_dict(torch.load(f'models/agent_layers{num_layers}_gamma{gamma}_{price_horizon}_shaped_bat_val.pt'))

# Evaluate Rule-Based Agent
eval_ddqn = DDQNEvaluation(price_horizon=price_horizon)
eval_ddqn.evaluate(agent = agent)

#Visualize DDQN Agent
plot_range = [0, 27000]

plotter = Plotter(eval_ddqn, range = plot_range)
#plotter.plot_actions(battery = False, balance=False, absence = False)
# plotter.plot_actions(battery = False, balance=False, absence = True)
# plotter.plot_actions(battery = False, balance=True, absence = True)
#plotter.plot_actions(battery = True, balance=True, absence = True)

plotter.plot_single()
### Current (last) problem: Agent only charges the battery half way. I need to incentivize charging the battery to 50kWh.
# %%
