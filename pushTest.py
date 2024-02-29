import datetime
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import HerReplayBuffer

env_name = "FetchPushDense-v2"
env = gym.make(env_name)
env2 = gym.make("FetchReachDense-v2")
env = Monitor(env)
model = TQC.load("./model/FetchReachDense_tqc.pkl", env=env2)
model2 = TQC(
    env=env,
    policy='MultiInputPolicy',
    # replay_buffer_class=HerReplayBuffer,
    # # Parameters for HER
    # replay_buffer_kwargs=dict(
    #     n_sampled_goal=4,
    #     goal_selection_strategy='future',
    # ),
    tau=0.05,
    buffer_size=1000000,
    # batch_size=2048,
    batch_size=256,
    gamma=0.95,
    learning_rate=0.001,
    verbose=1,  # output log
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    tensorboard_log="./tensorboard/FetchPushTestDense_TQC_HER-v2/"
)

# model.tensorboard_log = "./tensorboard/FetchPushTestDense_TQC_HER-v2/"
# model.policy.observation_space = env.observation_space
# model.tensorboard_log = None
original_layer = [model.actor.latent_pi[0], model.actor.mu]

# model2.actor.latent_pi[0].weight.data[:,:16] = original_layer[0].weight.data

# model2.actor.latent_pi[0].weight.data[:,:3] = original_layer[0].weight.data[:,:3]
# model2.actor.latent_pi[0].weight.data[:,9:11] = original_layer[0].weight.data[:,3:5]
# model2.actor.latent_pi[0].weight.data[:,20:25] = original_layer[0].weight.data[:,5:10]
# model2.actor.latent_pi[0].weight.data[:,25:] = original_layer[0].weight.data[:,10:]
# model2.actor.latent_pi[0].bias.data = original_layer[0].bias.data
# model2.actor.mu.weight.data = original_layer[1].weight.data
# model2.actor.mu.bias.data = original_layer[1].bias.data

# env.num_envs = 1
# env = DummyVecEnv([lambda: env])
# model.env = env
# model.observation_space = env.observation_space

original_first_layer = [model.critic.q_networks[0][0], model.critic.q_networks[1][0]]

# model2.critic.q_networks[0][0].weight.data[:,:20] = original_first_layer[0].weight.data
# model2.critic.q_networks[1][0].weight.data[:,:20] = original_first_layer[1].weight.data

model2.critic.q_networks[0][0].weight.data[:,:3] = original_first_layer[0].weight.data[:,:3]
model2.critic.q_networks[0][0].weight.data[:,9:11] = original_first_layer[0].weight.data[:,3:5]
model2.critic.q_networks[0][0].weight.data[:,20:25] = original_first_layer[0].weight.data[:,5:10]
model2.critic.q_networks[0][0].weight.data[:,25:] = original_first_layer[0].weight.data[:,10:]
model2.critic.q_networks[1][0].weight.data[:,:3] = original_first_layer[1].weight.data[:,:3]
model2.critic.q_networks[1][0].weight.data[:,9:11] = original_first_layer[1].weight.data[:,3:5]
model2.critic.q_networks[1][0].weight.data[:,20:25] = original_first_layer[1].weight.data[:,5:10]
model2.critic.q_networks[1][0].weight.data[:,25:] = original_first_layer[1].weight.data[:,10:]

model2.critic.q_networks[0][0].bias.data = original_first_layer[0].bias.data
model2.critic.q_networks[1][0].bias.data = original_first_layer[1].bias.data
# model2.critic_target.q_networks[0][0].bias.data = original_first_layer2[0].bias.data
# model2.critic_target.q_networks[1][0].bias.data = original_first_layer2[1].bias.data
# original = model.policy.actor.latent_pi[0]
# model2.policy.actor = model.policy.actor
# model2.policy.actor.latent_pi[0] = torch.nn.Linear(31, 512)
# model2.policy.actor.latent_pi[0].weight.data[:,:16] = original.weight.data
# model2.policy.actor.latent_pi[0].bias.data = original.bias.data

# tmp = np.squeeze(model.replay_buffer.observations['observation'])
# tmp = np.pad(tmp,((0,0),(0,15)))
# tmp = torch.from_numpy(tmp)
# tmp = torch.unsqueeze(tmp, 1)
# model.replay_buffer.observations['observation'] = tmp.numpy()
# tmp2 = np.squeeze(model.replay_buffer.next_observations['observation'])
# tmp2 = np.pad(tmp2,((0,0),(0,15)))
# tmp2 = torch.from_numpy(tmp2)
# tmp2 = torch.unsqueeze(tmp2, 1)
# model.replay_buffer.next_observations['observation'] = tmp2.numpy()
# model.replay_buffer.observations['observation'] = np.zeros((1000000,1,25))
# model.replay_buffer.next_observations['observation'] = np.zeros((1000000,1,25))

# model.verbose = 1
# model.learning_rate = 0.001
# model.batch_size = 2048
model2.learn(total_timesteps=1e6)
mean_reward, std_reward = evaluate_policy(model2, env, n_eval_episodes=10, render=False)
env.close()
print(mean_reward, std_reward)
model2.save("./model/FetchPushTestDense_tqc.pkl")
print(datetime.datetime.now())
