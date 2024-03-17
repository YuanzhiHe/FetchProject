import datetime
import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import HerReplayBuffer

env_name = "FetchPickAndPlace-v2"
env = gym.make(env_name)
env = Monitor(env)
env2 = gym.make("FetchReach-v2")

model = TQC.load("./model/FetchReach_tqc_her.pkl", env=env2)
model2 = TQC(
    env=env,
    policy='MultiInputPolicy',
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
    ),
    tau=0.05,
    buffer_size=1000000,
    batch_size=256,
    gamma=0.95,
    learning_rate=0.001,
    verbose=1,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    tensorboard_log="./tensorboard/FetchPick_TQC_HER/"
)

original_first_layer = [model.critic.q_networks[0][0], model.critic.q_networks[1][0]]
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

model2.learn(total_timesteps=1e6)
mean_reward, std_reward = evaluate_policy(model2, env, n_eval_episodes=10, render=False)
env.close()
print(mean_reward, std_reward)
model2.save("./model/FetchPick_tqc_her.pkl")
print(datetime.datetime.now())