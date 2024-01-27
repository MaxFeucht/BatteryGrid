from TestEnv import Electric_Car
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqn_agent import TabularQLearningAgent


# Make the excel file as a command line argument, so that you can do: " python3 main.py --excel_file validate.xlsx "
parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='./data/train.xlsx') # Path to the excel file with the test data
args = parser.parse_args()

def setup():
    epochs = 10  # Define the number of epochs
    total_rewards_per_epoch = []  # List to store the total reward for each epoch

    env = Electric_Car(path_to_test_data=args.excel_file)
    tabular_q_agent = TabularQLearningAgent(env, epsilon_decay=0.9)


    
    for epoch in range(epochs):
        #reset the environment
        env.counter = 0
        env.hour = 1
        env.day = 1

        observation = env.observation()
        total_reward = []
        cumulative_reward = []
        battery_level = []
        done = False
        if(epoch!=0):
            tabular_q_agent.update_epsilon(epoch, exponential_decay=False)
            print(tabular_q_agent.epsilon)
      
        
        while not done: # Loop through 2 years -> 730 days * 24 hours
            # Choose a random action between -1 (full capacity sell) and 1 (full capacity charge)
            # action = env.continuous_action_space.sample()
            # Only choose randomly 1 or -1 or 0
            #action = np.random.choice([-1, 0, 1])
            # Choose action through the tabular q-learning agent, state is only the battery level, price and hour of day
            action = tabular_q_agent.act(observation)
            # The observation is the tuple: [battery_level, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
            next_observation, reward, terminated, truncated, info = env.step(action)
            tabular_q_agent.learn(observation, action, reward, next_observation, terminated)
            total_reward.append(reward)
            cumulative_reward.append(sum(total_reward))
            done = terminated or truncated
            observation = next_observation

        #append the cumulative award for the epcoh
        total_rewards_per_epoch.append(sum(cumulative_reward))
        print(f'Epoch: {epoch} done!, Reward: {sum(total_reward)}')

         
    if done:
        print('Total reward: ', sum(total_reward))
        print(dict(tabular_q_agent.Qtable))
        # Plot the cumulative reward over time
        plt.plot(range(epochs), total_rewards_per_epoch)
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative reward (€)')
        #plt.ylim(-5000, 500)
        plt.show()

setup()