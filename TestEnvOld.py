import gym
import numpy as np
import pandas as pd


class Electric_Car(gym.Env):

    def __init__(self, path_to_test_data=str):
        # Define a continuous action space, -1 to 1. (You can discretize this later!)
        self.continuous_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Define the test data
        self.test_data = pd.read_excel(path_to_test_data)
        self.price_values = self.test_data.iloc[:, 1:25].to_numpy()
        self.timestamps = self.test_data['PRICES']
        self.state = np.empty(7)

        # Battery characteristics
        self.battery_capacity = 50  # kWh
        self.max_power = 25 / 0.9  # kW
        self.charge_efficiency = 0.9  # -
        self.discharge_efficiency = 0.9  # -
        self.battery_level = self.battery_capacity / 2  # kWh (start at 50%)
        self.minimum_morning_level = 20  # kWh
        self.car_use_consumption = 20  # kWh

        # Time Tracking
        self.counter = 0
        self.hour = 1
        self.day = 1
        self.car_is_available = True



    def step(self, action):

        action = np.squeeze(action)  # Remove the extra dimension

        if action <-1 or action >1:
            raise ValueError('Action must be between -1 and 1')

        # Calculate if, at 7am and after the chosen action, the battery level will be below the minimum morning level:
        if self.hour == 7:
            if action > 0 and (self.battery_level < self.minimum_morning_level):
                if (
                        self.battery_level + action * self.max_power * self.charge_efficiency) < self.minimum_morning_level:  # If the chosen action will not charge the battery to 20kWh
                    action = (self.minimum_morning_level - self.battery_level) / (
                                self.max_power * self.charge_efficiency)  # Charge until 20kWh
            elif action < 0:
                if (self.battery_level + action * self.max_power) < self.minimum_morning_level:
                    if self.battery_level < self.minimum_morning_level:  # If the level was lower than 20kWh, charge until 20kWh
                        action = (self.minimum_morning_level - self.battery_level) / (
                                    self.max_power * self.charge_efficiency)  # Charge until 20kWh
                    elif self.battery_level >= self.minimum_morning_level:  # If the level was higher than 20kWh, discharge until 20kWh
                        action = (self.minimum_morning_level - self.battery_level) / (
                            self.max_power)  # Discharge until 20kWh
            elif action == 0:
                if self.battery_level < self.minimum_morning_level:
                    action = (self.minimum_morning_level - self.battery_level) / (
                                self.max_power * self.charge_efficiency)

        # There is a 50% chance that the car is unavailable from 8am to 6pm
        if self.hour == 8:
            self.car_is_available = np.random.choice([True, False])
            if not self.car_is_available:
                self.battery_level -= self.car_use_consumption
        if self.hour == 18:
            self.car_is_available = True
        if not self.car_is_available:
            action = 0

        # Calculate the costs and battery level when charging (action >0)
        if (action > 0) and (self.battery_level <= self.battery_capacity):
            if (self.battery_level + action * self.max_power * self.charge_efficiency) > self.battery_capacity:
                action = (self.battery_capacity - self.battery_level) / (self.max_power * self.charge_efficiency)
            charged_electricity_kW = action * self.max_power
            charged_electricity_costs = charged_electricity_kW * self.price_values[self.day - 1][
                self.hour - 1] * 2 * 1e-3
            reward = -charged_electricity_costs
            self.battery_level += charged_electricity_kW * self.charge_efficiency

        # Calculate the profits and battery level when discharging (action <0)
        elif (action < 0) and (self.battery_level >= 0):
            if (self.battery_level + action * self.max_power) < 0:
                action = -self.battery_level / (self.max_power)
            discharged_electricity_kWh = action * self.max_power  # Negative discharge value
            discharged_electricity_profits = abs(discharged_electricity_kWh) * self.discharge_efficiency * \
                                             self.price_values[self.day - 1][self.hour - 1] * 1e-3
            reward = discharged_electricity_profits
            self.battery_level += discharged_electricity_kWh
            # Some small numerical errors causing the battery level to be 1e-14 to 1e-17 under 0 :
            if self.battery_level < 0:
                self.battery_level = 0

        else:
            reward = 0

        self.counter += 1  # Increase the counter
        self.hour += 1  # Increase the hour

        if self.counter % 24 == 0:  # If the counter is a multiple of 24, increase the day, reset hour to first hour
            self.day += 1
            self.hour = 1

        terminated = self.counter == len(
            self.price_values.flatten()) - 1  # If the counter is equal to the number of hours in the test data, terminate the episode
        truncated = False

        info = action  # The final action taken after all constraints! For debugging purposes

        self.state = self.observation()  # Update the state

        return self.state, reward, terminated, truncated, info



    def observation(self):  # Returns the current state
        battery_level = self.battery_level
        price = self.price_values[self.day - 1][self.hour - 1]
        hour = self.hour
        day_of_week = self.timestamps[self.day - 1].dayofweek  # Monday = 0, Sunday = 6
        day_of_year = self.timestamps[self.day - 1].dayofyear  # January 1st = 1, December 31st = 365
        month = self.timestamps[self.day - 1].month  # January = 1, December = 12
        year = self.timestamps[self.day - 1].year
        self.state = np.array(
            [battery_level, price, int(hour), int(day_of_week), int(day_of_year), int(month), int(year),
             int(self.car_is_available)])

        return self.state