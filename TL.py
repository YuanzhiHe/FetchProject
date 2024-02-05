import gymnasium as gym
import torch
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env_name1 = "FetchSlide-v2"
env1 = gym.make(env_name1)

env_name2 = "FetchPush-v2"
env2 = gym.make(env_name2)

env1 = Monitor(env1)
env2 = Monitor(env2)

model = TQC.load("./model/FetchPush_tqc_her.pkl", env=env2)
# model2 = TQC.load("./model/FetchPush_tqc_her.pkl", env=env2)


# model.policy.observation_space = env2.observation_space
# model.actor.latent_pi[0] = torch.nn.Linear(31, 512)
# model.actor.mu = model2.actor.mu

mean_reward, std_reward = evaluate_policy(model, env1, n_eval_episodes=10)

env1.close()
env2.close()
print(mean_reward, std_reward)
