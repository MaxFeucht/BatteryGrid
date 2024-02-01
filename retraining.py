
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

    action, q = agent.choose_action(i, state, greedy = False) # Choose action (discrete)
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
