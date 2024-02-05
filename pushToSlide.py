import datetime
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import HerReplayBuffer

env_name = "FetchSlide-v2"
env = gym.make(env_name)
env = Monitor(env)
env2 = gym.make("FetchPush-v2")

model = TQC.load("./model/FetchPush_tqc_her.pkl", env=env2)
model2 = TQC(
    env=env,
    policy='MultiInputPolicy',
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
    ),
    tau=0.05,
    buffer_size=1000000,
    batch_size=256,
    gamma=0.95,
    learning_rate=0.001,
    verbose=1,  # output log
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    tensorboard_log="./tensorboard/FetchSlideTest_TQC_HER-v2/"
)


original_first_layer = [model.critic.q_networks[0][0], model.critic.q_networks[1][0]]
original_first_layer2 = [model.critic_target.q_networks[0][0], model.critic_target.q_networks[1][0]]
model2.critic.q_networks[0][0].weight.data = original_first_layer[0].weight.data
model2.critic.q_networks[1][0].weight.data = original_first_layer[1].weight.data
model2.critic_target.q_networks[0][0].weight.data = original_first_layer2[0].weight.data
model2.critic_target.q_networks[1][0].weight.data = original_first_layer2[1].weight.data
model2.critic.q_networks[0][0].bias.data = original_first_layer[0].bias.data
model2.critic.q_networks[1][0].bias.data = original_first_layer[1].bias.data
model2.critic_target.q_networks[0][0].bias.data = original_first_layer2[0].bias.data
model2.critic_target.q_networks[1][0].bias.data = original_first_layer2[1].bias.data

model2.learn(total_timesteps=1e6)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
env.close()
print(mean_reward, std_reward)
model.save("./model/FetchSlideTest_tqc_her.pkl")
print(datetime.datetime.now())
