import datetime
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "FetchPush-v2"
env = gym.make(env_name)
env2 = gym.make("FetchReach-v2")

model = TQC.load("./model/FetchReach_tqc_her.pkl", env=env2)

env = Monitor(env)
model.tensorboard_log = "./tensorboard/FetchPushTest_TQC_HER-v2/"
model.policy.observation_space = env.observation_space
model.actor.latent_pi[0] = torch.nn.Linear(31, 256)
env.num_envs = 1
env = DummyVecEnv([lambda: env])
model.env = env
model.observation_space = env.observation_space
model.critic.q_networks[0][0] = torch.nn.Linear(35, 256)
model.critic.q_networks[1][0] = torch.nn.Linear(35, 256)
model.critic_target.q_networks[0][0] = torch.nn.Linear(35, 256)
model.critic_target.q_networks[1][0] = torch.nn.Linear(35, 256)
tmp = np.squeeze(model.replay_buffer.observations['observation'])
tmp = np.pad(tmp,((0,0),(0,15)))
tmp = torch.from_numpy(tmp)
tmp = torch.unsqueeze(tmp, 1)
model.replay_buffer.observations['observation'] = tmp.numpy()
tmp2 = np.squeeze(model.replay_buffer.next_observations['observation'])
tmp2 = np.pad(tmp2,((0,0),(0,15)))
tmp2 = torch.from_numpy(tmp2)
tmp2 = torch.unsqueeze(tmp2, 1)
model.replay_buffer.next_observations['observation'] = tmp2.numpy()
model.verbose = 1
model.learn(total_timesteps=1e6)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
env.close()
print(mean_reward, std_reward)
model.save("./model/FetchPushTest_tqc_her2.pkl")
print(datetime.datetime.now())
