import gymnasium as gym
from stable_baselines3 import PPO, DDPG, HerReplayBuffer
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env_name = "FetchPickAndPlace-v2"
# env = gym.make(env_name, render_mode='human')
env = gym.make(env_name)
env = Monitor(env)

model = TQC.load("./model/FetchPushToPick_tqc.pkl", env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
env.close()
print(mean_reward, std_reward)
