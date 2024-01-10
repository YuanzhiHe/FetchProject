import gymnasium as gym
import torch
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env_name1 = "FetchReach-v2"
env1 = gym.make(env_name1)

env_name2 = "FetchPush-v2"
env2 = gym.make(env_name2, render_mode='human')

env1 = Monitor(env1)
env2 = Monitor(env2)

model = TQC.load("./model/FetchReach_tqc_her2.pkl", env=env1)
model2 = TQC.load("./model/FetchPush_tqc_her.pkl", env=env2)


model.policy.observation_space = env2.observation_space
model.actor.latent_pi = model2.actor.latent_pi
model.actor.mu = model2.actor.mu

mean_reward, std_reward = evaluate_policy(model, env2, n_eval_episodes=10)

# episode_rewards = []
# episode_lengths = []
#
# # Divides episodes among different sub environments in the vector as evenly as possible
#
# current_rewards = 0
# current_lengths = 0
# observations = env2.reset()
# states = None
# episode_starts = np.ones((1,), dtype=bool)
# actions, states = model2.predict(
#     observations,  # type: ignore[arg-type]
#     state=states,
#     episode_start=episode_starts,
# )
# episode_counts = 0
# episode_count_targets = 10
# while episode_counts < episode_count_targets:
#     new_observations, reward, done, info = env2.step(actions)
#     current_rewards += reward
#     current_lengths += 1
#     if done:
#         episode_rewards.append(current_rewards)
#         episode_lengths.append(current_lengths)
#         episode_counts += 1
#         current_rewards = 0
#         current_lengths = 0
#
#     observations = new_observations
#
# mean_reward = np.mean(episode_rewards)
# std_reward = np.std(episode_rewards)

env1.close()
env2.close()
print(mean_reward, std_reward)
