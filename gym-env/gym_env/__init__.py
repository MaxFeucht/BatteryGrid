# Register the environment with OpenAI Gym
from gymnasium.envs.registration import register

register(
     id="gym_env/BatteryGrid-v0",
     entry_point="gym_env.envs:BatteryGridEnv",
)

