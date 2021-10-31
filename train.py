import gym
from gym.envs.box2d import CarRacing

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

if __name__ == '__main__':
    env = lambda :  CarRacing(
        grayscale=1,
        show_info_panel=0,
        discretize_actions="hard",
        frames_per_state=4,
        num_lanes=1,
        num_tracks=1,
        )
    
    env = DummyVecEnv([env])

    model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log='tensor_logs/ppo')
    model.learn(total_timesteps=200000)
    model.save('learned_models/car_racing_weights_200k')
